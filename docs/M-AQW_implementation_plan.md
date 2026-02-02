# M-AQW 구현 계획 (라플라시안 단일 축, 좌/우 독립 τ·k)

## 1. 목표

- **단일 품질 축**: 라플라시안 점수만 사용.
- **독립 파라미터**: 좌측(blur)용 (τ_L, k_L), 우측(artifact)용 (τ_R, k_R) 총 4개.
- **학습 방식**: 슬라이드별 품질 통계 및 분포(shape) → Meta-MLP → 4개 파라미터 예측 → Double-sigmoid로 타일별 가중치 W_i → **특징 보정** f_i' = f_i × W_i → 기존 CLAM attention.

---

## 2. 수식 정의

### 2.1 품질 축 정규화 (per-slide)

- 한 슬라이드 내 라플라시안 점수 벡터 `q_raw` ∈ ℝ^N.
- 정규화:
  ```text
  q_min = min(q_raw),  q_max = max(q_raw)
  q_i = (q_raw_i - q_min) / (q_max - q_min + ε)   ∈ [0, 1]
  ```
- 이후 τ_L, τ_R도 [0, 1] 구간으로 해석 (같은 스케일).

### 2.2 Double-sigmoid 가중치 W(q)

- **좌측 (blur, q 낮음)**: q가 τ_L보다 작으면 가중치 감소.
  - `W_left(q) = sigmoid( k_L * (q - τ_L) )`
  - q ≪ τ_L → W_left → 0,  q ≫ τ_L → W_left → 1.

- **우측 (artifact, q 높음)**: q가 τ_R보다 크면 가중치 감소.
  - `W_right(q) = sigmoid( k_R * (τ_R - q) )`
  - q ≫ τ_R → W_right → 0,  q ≪ τ_R → W_right → 1.

- **최종**:
  - `W(q) = W_left(q) × W_right(q)`
  - τ_L, τ_R는 **제약 없이** [0, 1] 구간에서 자유롭게 학습. (τ_L < τ_R일 때는 중간 구간이 plateau, 그렇지 않아도 모델이 스스로 곡선 형태를 학습.)

---

## 3. Meta-MLP 설계

- **입력**: 슬라이드 내 정규화된 라플라시안 분포 정보 (forward 시 해당 bag에서 계산).
  - **통계 6차원**:  
    `[mean(q_norm), std(q_norm), min(q_norm), max(q_norm), p25(q_norm), p75(q_norm)]`
  - **히스토그램 10차원**:  
    q_norm ∈ [0, 1]을 10개 구간으로 균등 분할한 bin별 밀도값  
    (각 bin b는 구간 `[0.1*(b-1), 0.1*b)`에 속하는 패치 비율 `count_b / N`)
  - ⇒ **최종 입력 dim = 16** (통계 6 + hist 10)
- **출력**: 4개 스칼라 → (τ_L, k_L, τ_R, k_R).
- **출력 활성화** (제약 없이 각각 독립):
  - τ_L, τ_R ∈ (0, 1): `sigmoid` 사용.
  - k_L, k_R > 0: `softplus(x) + 0.1` (최소 기울기 보장).
- **구조 예**:  
  `Linear(16 → 32) → ReLU → Linear(32 → 4)`

---

## 4. 데이터 파이프라인

### 4.1 Feature 추출 단계 (`extract_features_fp.py`)

- **저장 필드**:
  - 기존: `features`, `coords`.
  - 추가: **`laplacian_scores`** (dtype float32, shape [N]).
- **조건**: blur 관련 모드가 켜져 있을 때 (예: `--blur_mode maqw`) 라플라시안 계산 후 **원점수 그대로** 저장 (threshold/이진 가중치 미적용).

### 4.2 Dataset (`dataset_generic.py` / `dataset_h5.py`)

- **H5 로드**: `features`, `coords`, **`laplacian_scores`** (있으면).
- **반환**:  
  - `(features, label, coords, laplacian_scores)` (없으면 4번째는 `None`으로 반환).
  - `laplacian_scores`는 M-AQW에서만 사용하며, 나머지 파이프라인은 기존과 동일하게 동작.
- **collate**: `laplacian_scores`는 길이가 슬라이드마다 다르므로 list of 1D tensors → padding 또는 list로 유지 후 모델에서 하나의 bag씩 처리.

### 4.3 DataLoader / batch

- 배치 형태: `(data, label, coords, laplacian_scores)`.
- `data`: [B, N_max, D] padded 또는 list of [N_i, D].  
  CLAM은 보통 **bag 단위**이므로 B=1, N=패치 수인 경우가 많음.  
  그대로 두고 `laplacian_scores`도 [N] 또는 list of [N_i].

---

## 5. 모델 통합 (CLAM + M-AQW)

### 5.1 M-AQW 모듈 (신규)

- **입력**:
  - `h`: [N, D] 패치 특징.
  - `q`: [N] 라플라시안 (원점수 또는 이미 정규화된 값).
- **내부**:
  1. q를 per-slide로 [0,1] 정규화 (min/max 또는 percentile).
  2. slide 통계 및 히스토그램 계산:  
     `q_norm`에서 통계 6차원 + hist 10차원 → `[1, 16]` 벡터.
  3. Meta-MLP(stats_concat) → (τ_L, k_L, τ_R, k_R), 각각 sigmoid/softplus 적용.
  4. `W_i = W_left(q_norm_i) * W_right(q_norm_i)` for all i.
  5. `h_out = h * W_i` (W_i를 [N,1]로 브로드캐스트).
- **출력**: `h_out` [N, D].

### 5.2 CLAM forward 수정

- **현재**: `forward(h, label=..., weights=None, ...)`  
  - `weights`가 있으면 attention logit에 곱함.
- **변경**:
  - `forward(h, label=..., weights=None, laplacian_scores=None, ...)`.
  - `laplacian_scores`가 있으면:
    1. **먼저** M-AQW로 `h_mod = M_AQW(h, laplacian_scores)`.
    2. 이후 **h 대신 h_mod**로 attention 및 classifier 진행 (weights 인자 무시 또는 제거).
  - `laplacian_scores`가 없으면 기존과 동일: `weights`만 있으면 attention에 곱하고, 없으면 그대로.

### 5.3 학습

- 품질 레이블 없이 **분류 손실만** 사용.
- M-AQW + Meta-MLP 파라미터는 CLAM과 함께 end-to-end 학습.
- (선택) τ_L, τ_R, k_L, k_R 로깅/시각화로 해석 가능성 확보.

---

## 6. 파일별 작업 목록

| 파일 | 작업 |
|------|------|
| `extract_features_fp.py` | `blur_mode=maqw` 시 라플라시안 **원점수**를 `laplacian_scores`로 H5 저장. |
| `dataset_generic.py` | H5에서 `laplacian_scores` 로드; 반환에 추가. M-AQW 시에만 품질 가중치 계산에 사용. |
| `dataset_h5.py` (필요 시) | H5 키 `laplacian_scores` 읽기. |
| **신규** `models/maqw.py` (또는 `utils/maqw.py`) | Meta-MLP, Double-sigmoid W(q), 정규화, M-AQW forward 구현. |
| `models/model_clam.py` | M-AQW 모듈 추가; `forward`에서 `laplacian_scores` 인자 받고, 있으면 h → M-AQW(h,q) → attention. |
| `core_utils.py` | 배치에서 `laplacian_scores` 꺼내서 `model(..., laplacian_scores=...)` 로 전달. |

---

## 7. 구현 순서 제안

1. **수식 고정**: 위 Double-sigmoid를 코드로 한 번 구현 (단위 테스트용 스크립트 권장).
2. **M-AQW 모듈**: `maqw.py`에 Meta-MLP + W(q) + 정규화 구현, 입력 (h, q) → h' 출력까지.
3. **Feature 추출**: `laplacian_scores` 저장 로직 추가.
4. **Dataset**: `laplacian_scores` 로드 및 반환.
5. **CLAM 연동**: forward에 laplacian 경로 추가, M-AQW 호출 및 h_mod 사용.
6. **학습 루프**: 배치에 laplacian 넘기고, 기존 분류 loss로만 학습.
7. (선택) τ, k 로깅/시각화 및 해석 가능성 정리.

---

## 8. τ_L, τ_R 학습 방식

- τ_L, τ_R은 **제약 없이** 각각 `sigmoid`로 (0, 1) 구간에서 자유롭게 학습.
- τ_L < τ_R이 아니어도 됨. 필요하면 모델이 스스로 순서와 곡선 형태를 학습.

이 계획대로 진행하면 “라플라시안 단일 축 + 좌/우 독립 (τ, k)” M-AQW를 현재 CLAM 파이프라인 위에 올릴 수 있습니다.

# M-AQW 구현 계획 (라플라시안 단일 축, 좌/우 독립 τ·k)

## 1. 목표

- **단일 품질 축**: 라플라시안 점수만 사용.
- **독립 파라미터**: 좌측(blur)용 (τ_L, k_L), 우측(artifact)용 (τ_R, k_R) 총 4개.
- **학습 방식**: 슬라이드별 품질 통계 → Meta-MLP → 4개 파라미터 예측 → Double-sigmoid로 타일별 가중치 W_i → **특징 보정** f_i' = f_i × W_i → 기존 CLAM attention.

---

## 2. 수식 정의

### 2.1 품질 축 정규화 (per-slide)

- 한 슬라이드 내 라플라시안 점수 벡터 `q_raw` ∈ ℝ^N.
- 정규화:
  ```text
  q_min = min(q_raw),  q_max = max(q_raw)
  q_i = (q_raw_i - q_min) / (q_max - q_min + ε)   ∈ [0, 1]
  ```
- 이후 τ_L, τ_R도 [0, 1] 구간으로 해석 (같은 스케일).

### 2.2 Double-sigmoid 가중치 W(q)

- **좌측 (blur, q 낮음)**: q가 τ_L보다 작으면 가중치 감소.
  - `W_left(q) = sigmoid( k_L * (q - τ_L) )`
  - q ≪ τ_L → W_left → 0,  q ≫ τ_L → W_left → 1.

- **우측 (artifact, q 높음)**: q가 τ_R보다 크면 가중치 감소.
  - `W_right(q) = sigmoid( k_R * (τ_R - q) )`
  - q ≫ τ_R → W_right → 0,  q ≪ τ_R → W_right → 1.

- **최종**:
  - `W(q) = W_left(q) × W_right(q)`
  - τ_L, τ_R는 **제약 없이** [0, 1] 구간에서 자유롭게 학습. (τ_L < τ_R일 때는 중간 구간이 plateau, 그렇지 않아도 모델이 스스로 곡선 형태를 학습.)

---

## 3. Meta-MLP 설계

- **입력**: 슬라이드 내 라플라시안 통계 (forward 시 해당 bag에서 계산).
  - 후보: `[mean, std, min, max, p25, p75]` (6차원) 또는
  - `[mean, std, p10, p25, p50, p75, p90]` (7차원).
- **출력**: 4개 스칼라 → (τ_L, k_L, τ_R, k_R).
- **출력 활성화** (제약 없이 각각 독립):
  - τ_L, τ_R ∈ (0, 1): `sigmoid` 사용.
  - k_L, k_R > 0: `softplus(x) + 0.1` (최소 기울기 보장).
- **구조 예**: Linear(6 or 7 → 32) → ReLU → Linear(32 → 4).

---

## 4. 데이터 파이프라인

### 4.1 Feature 추출 단계 (`extract_features_fp.py`)

- **저장 필드**:
  - 기존: `features`, `coords`.
  - 추가: **`laplacian_scores`** (dtype float32, shape [N]).
- **조건**: blur 관련 모드가 켜져 있을 때 (예: `--blur_mode maqw` 또는 `--blur_mode weight` 확장) 라플라시안 계산 후 **원점수 그대로** 저장 (threshold/이진 가중치 미적용).
- **호환**: 기존 `blur_mode=weight`는 이진 가중치만 저장하므로, M-AQW용으로는 별도 모드(예: `maqw`)를 두거나, `weight`일 때 **추가로** `laplacian_scores`도 저장하도록 확장.

### 4.2 Dataset (`dataset_generic.py` / `dataset_h5.py`)

- **H5 로드**: `features`, `coords`, **`laplacian_scores`** (있으면).
- **반환**:  
  - 기존: `(features, label, coords, weights)`.  
  - M-AQW: `(features, label, coords, laplacian_scores)`  
  - 또는 기존 `weights` 자리에 Laplacian을 넘기고, 모델에서 “타입”으로 구분할 수도 있으나, 명확성을 위해 **별도 필드 `laplacian_scores`** 권장.
- **collate**: `laplacian_scores`는 길이가 슬라이드마다 다르므로 list of 1D tensors → padding 또는 list로 유지 후 모델에서 하나의 bag씩 처리.

### 4.3 DataLoader / batch

- 배치 형태: `(data, label, coords, laplacian_scores)`.
- `data`: [B, N_max, D] padded 또는 list of [N_i, D].  
  CLAM은 보통 **bag 단위**이므로 B=1, N=패치 수인 경우가 많음.  
  그대로 두고 `laplacian_scores`도 [N] 또는 list of [N_i].

---

## 5. 모델 통합 (CLAM + M-AQW)

### 5.1 M-AQW 모듈 (신규)

- **입력**:
  - `h`: [N, D] 패치 특징.
  - `q`: [N] 라플라시안 (원점수 또는 이미 정규화된 값).
- **내부**:
  1. q를 per-slide로 [0,1] 정규화 (min/max 또는 percentile).
  2. slide 통계 계산: `stats = [mean(q), std(q), p25, p75, ...]` → [1, S].
  3. Meta-MLP(stats) → (τ_L, k_L, τ_R, k_R), 각각 sigmoid/softplus 적용.
  4. `W_i = W_left(q_i) * W_right(q_i)` for all i.
  5. `h_out = h * W_i` (W_i를 [N,1]로 브로드캐스트).
- **출력**: `h_out` [N, D].

### 5.2 CLAM forward 수정

- **현재**: `forward(h, label=..., weights=None, ...)`  
  - `weights`가 있으면 attention logit에 곱함.
- **변경**:
  - `forward(h, label=..., weights=None, laplacian_scores=None, ...)`.
  - `laplacian_scores`가 있으면:
    1. **먼저** M-AQW로 `h_mod = M_AQW(h, laplacian_scores)`.
    2. 이후 **h 대신 h_mod**로 attention 및 classifier 진행 (weights 인자 무시 또는 제거).
  - `laplacian_scores`가 없으면 기존과 동일: `weights`만 있으면 attention에 곱하고, 없으면 그대로.

### 5.3 학습

- 품질 레이블 없이 **분류 손실만** 사용.
- M-AQW + Meta-MLP 파라미터는 CLAM과 함께 end-to-end 학습.
- (선택) τ_L, τ_R, k_L, k_R 로깅/시각화로 해석 가능성 확보.

---

## 6. 파일별 작업 목록

| 파일 | 작업 |
|------|------|
| `extract_features_fp.py` | `blur_mode=maqw`(또는 확장) 시 라플라시안 **원점수**를 `laplacian_scores`로 H5 저장. |
| `dataset_generic.py` (jw_weight) | H5에서 `laplacian_scores` 로드; 반환에 추가. `weights`와 별도 필드로 두거나, M-AQW 시에만 laplacian 넘기기. |
| `dataset_h5.py` (필요 시) | H5 키 `laplacian_scores` 읽기. |
| **신규** `models/maqw.py` (또는 `utils/maqw.py`) | Meta-MLP, Double-sigmoid W(q), 정규화, M-AQW forward 구현. |
| `models/model_clam.py` | M-AQW 모듈 추가; `forward`에서 `laplacian_scores` 인자 받고, 있으면 h → M-AQW(h,q) → attention. |
| `core_utils.py` | 배치에서 `laplacian_scores` 꺼내서 `model(..., laplacian_scores=...)` 로 전달. |

---

## 7. 구현 순서 제안

1. **수식 고정**: 위 Double-sigmoid를 코드로 한 번 구현 (단위 테스트용 스크립트 권장).
2. **M-AQW 모듈**: `maqw.py`에 Meta-MLP + W(q) + 정규화 구현, 입력 (h, q) → h' 출력까지.
3. **Feature 추출**: `laplacian_scores` 저장 로직 추가.
4. **Dataset**: `laplacian_scores` 로드 및 반환.
5. **CLAM 연동**: forward에 laplacian 경로 추가, M-AQW 호출 및 h_mod 사용.
6. **학습 루프**: 배치에 laplacian 넘기고, 기존 분류 loss로만 학습.
7. (선택) τ, k 로깅/시각화 및 해석 가능성 정리.

---

## 8. τ_L, τ_R 학습 방식

- τ_L, τ_R은 **제약 없이** 각각 `sigmoid`로 (0, 1) 구간에서 자유롭게 학습.
- τ_L < τ_R이 아니어도 됨. 필요하면 모델이 스스로 순서와 곡선 형태를 학습.

이 계획대로 진행하면 “라플라시안 단일 축 + 좌/우 독립 (τ, k)” M-AQW를 현재 jw_weight/CLAM 파이프라인 위에 올릴 수 있습니다.
