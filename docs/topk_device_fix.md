# topk SmoothTop1SVM + DDP device 불일치 문제

## 1. 현상

`--inst_loss svm` 으로 학습할 때, **DDP**(`torchrun --nproc_per_node=2`) 사용 시 아래 에러가 난다.

```
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

에러 위치:

- `model_clam.py` → `inst_eval()` → `self.instance_loss_fn(logits, all_targets)`
- `topk/svm.py` (SmoothTop1SVM.forward) → `topk/functional.py` (fun) → **`topk/utils.py` 15행**  
  `delta = torch.ne(y[:, None], labels[None, :]).float()`

즉, **topk 라이브러리 내부**에서 `y`는 GPU, `labels`는 CPU에 있어서 `torch.ne`에서 device 불일치가 발생한다.

---

## 2. 원인

### 2.1 우리가 넘기는 텐서는 이미 GPU

- `model_clam.py`의 `inst_eval()`에서:
  - `logits` = classifier 출력 → GPU
  - `all_targets = all_targets.to(logits.device)` 로 이미 **logits와 같은 device(GPU)**로 맞춰서  
    `instance_loss_fn(logits, all_targets)` 에 넘긴다.

### 2.2 문제는 topk **내부**에서 만드는 `labels`

- SmoothTop1SVM은 내부에서 **클래스 인덱스**를 쓰기 위해 `labels` 같은 텐서를 만든다.
- 이때 `torch.arange(...)` 등으로 새 텐서를 만들면 **기본 device는 CPU**이다.
- 그래서 `delta(y, labels, alpha)` 호출 시:
  - `y`: 우리가 넘긴 logits에서 유래 → **GPU**
  - `labels`: topk 내부에서 CPU에 생성  
  → `torch.ne(y[:, None], labels[None, :])` 에서 **cuda vs cpu** 에러 발생.

즉, **원인은 “우리가 넘기는 target”이 아니라, topk가 내부적으로 만드는 `labels`가 CPU에 있다는 것**이다.

---

## 3. 해결 방향

- topk 패키지 코드(site-packages)를 직접 수정하지 않고,
- **`delta(y, labels, alpha)`가 호출될 때만** `labels`를 `y`와 같은 device로 옮기면 된다.

그래서 **monkey-patch**로 `topk.utils.delta`를 다음처럼 바꾼다:

- 원래: `delta(y, labels, alpha)` 그대로 호출
- 패치: `labels = labels.to(y.device)` 한 뒤, 원래 `delta(y, labels, alpha)` 호출

이렇게 하면 topk 내부에서 CPU로 만든 `labels`도 항상 `y`와 같은 device에서 연산된다.

---

## 4. 패치가 적용돼야 하는 위치

- `delta`를 **실제로 호출하는 쪽**은 `topk.functional.fun()` 이다.
- `topk.functional`은 보통 `from topk.utils import delta` 로 **import 시점의 delta 참조**를 갖는다.
- 따라서:
  - **`topk.utils.delta`** 를 우리 wrapper로 바꾸고,
  - **`topk.functional`이 이미 로드된 뒤**라면 **`topk.functional.delta`** 도 같은 wrapper로 덮어줘야,  
    `fun()` 안에서 호출되는 `delta`가 우리 패치된 버전이 된다.

패치 시점은 **topk가 한 번이라도 import된 직후, 그리고 SmoothTop1SVM.forward()가 호출되기 전**이면 된다.

---

## 5. 코드에서의 적용

- **main.py 맨 위**: 다른 모듈 import 전에 topk를 import하고 `topk.utils.delta`(및 필요 시 `topk.functional.delta`)를 패치해 두었다.  
  → 이론상 topk가 **처음** import될 때부터 패치가 들어가야 한다.
- **서버/다른 환경**에서는 import 순서가 달라서, `core_utils` 등이 먼저 topk를 로드할 수 있다.  
  그 경우 main.py 패치보다 **먼저** topk가 로드되면, 그때는 아직 패치가 없어서 `fun()`이 옛 `delta`를 쓰게 된다.
- 그래서 **SmoothTop1SVM을 쓰는 직전**에 한 번 더 패치하는 것이 안전하다:
  - **utils/core_utils.py** 에서 `from topk.svm import SmoothTop1SVM` 한 직후,
  - `topk.utils.delta`와 `topk.functional.delta`를 같은 wrapper로 덮어쓴다.

이렇게 하면:

1. **문제**: topk 내부 `delta(y, labels, alpha)` 에서 `labels`가 CPU, `y`가 GPU라 device 불일치.
2. **해결**: `delta`를 wrapper로 교체해, 호출 시 `labels = labels.to(y.device)` 후 원래 `delta` 호출.
3. **적용**: main.py 상단 패치 + core_utils에서 SmoothTop1SVM 사용 직전 패치로, 어떤 import 순서에서도 패치가 적용되도록 함.

이후 `torchrun --nproc_per_node=2 ... --inst_loss svm` 으로 다시 실행하면 해당 RuntimeError는 사라져야 한다.
