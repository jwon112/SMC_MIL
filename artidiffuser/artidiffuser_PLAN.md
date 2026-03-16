## ArtiDiffuser 복원 모델 학습 및 CLAM 전처리 통합 계획

### 1. 데이터셋 구조 정리

- **COAD-Artifact** (`/home/jupyter/data/image_team/COAD-Artifact`)
  - 폴더: `normal/`, `marking/`, `out_of_focus/`, `tattoo/`, `tissue_folding/`
  - 각 폴더 안에 artifact 타입별 PNG 패치 존재.
  - 클래스 라벨 예시: `0=normal`, `1=marking`, `2=out_of_focus`, `3=tattoo`, `4=tissue_folding`.
- **ArtiDiffuser-Synth** (`/home/jupyter/data/image__team/ArtiDiffuser-Synth`)
  - 폴더: `marking/`, `out_of_focus`, `tattoo`, `tissue_folding` 각각 아래에 `ori/`, `inpainted/` 존재.
  - `ori` = 깨끗한 조직 이미지, `inpainted` = 공식 ArtiDiffuser 모델로 합성된 artifact 이미지.

---

### 2. 마스크 생성 파이프라인

- **Synth용 마스크 (ori vs inpainted 차이 기반)**
  - 두 이미지를 `uint8 RGB`로 읽고 `diff = |ori - inpainted|` 계산.
  - 채널 평균 또는 max로 `[H,W]` 스칼라 맵을 만든 뒤, threshold `T`(예: 20~40)로 이진 마스크 생성.
  - 마스크를 PNG 또는 NPY/HDF5로 저장하고, `masks/` 하위 폴더 구조를 정리.

- **COAD-Artifact용 마스크 (real artifact)**
  - 각 artifact 타입별로 간단한 heuristics로 rough mask 생성:
    - `marking`/`tattoo`: HSV/색 기반 threshold + morphological ops.
    - `out_of_focus`: 라플라시안/에지 강도 기반 저주파 영역.
    - `tissue_folding`: gradient + intensity 변화 기반 fold edge 주변 영역.
  - 완전하지 않아도 되며, DDPM 학습 특성상 다소 noisy한 마스크도 허용.

---

### 3. 공통 데이터셋/로더 인터페이스

- **배치 딕셔너리 형태**
  - `img`: Tensor `[B, 3, H, W]` (정상 또는 artifact 이미지).
  - `cls`: LongTensor `[B]` (0~4 클래스 라벨).
  - `mask`: Tensor `[B, 1, H, W]` (artifact 영역 마스크, normal은 0).
  - `src`: 선택적 문자열 (`"coad"` / `"synth"`)로 데이터 출처 구분.

- **로더 설계**
  - COAD-only, COAD+Synth 실험 모두 동일 인터페이스를 유지해, 학습 루프는 그대로 두고 데이터 구성만 바꿀 수 있게 설계.

---

### 4. DDPM 학습 루프 (ArtiDiffuser 원리 반영)

- **네트워크**: UNet 기반 noise 예측 모델 (`mini/model/unet.py` 또는 ArtiDiffuser UNet 구조 참고).

- **정상 패치 학습 (cls == 0)**
  - DDPM forward: `x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * eps`.
  - 모델: `eps_hat = model(x_t, t, cls=0, mask=None)`.
  - 손실: `loss = MSE(eps_hat, eps)` (전체 픽셀 평균).

- **artifact 패치 학습 (cls > 0, mask > 0)**
  - 동일하게 `x_t` 생성 후 `eps_hat = model(x_t, t, cls=c, mask=mask)`.
  - 마스크 가중치 손실:
    - `w = 10` 등으로 설정.
    - `w_map = 1 + (w-1) * mask`로 픽셀별 가중치 맵 생성.
    - `per_pixel = (eps_hat - eps)**2`, `loss = (per_pixel * w_map).mean()`.

---

### 5. 실험 1: COAD-only 학습

- **데이터**: COAD-Artifact (normal + 4 artifact 타입), Synth는 사용하지 않음.
- **학습**:
  - 위 DDPM 루프를 COAD 데이터만으로 수행.
  - 배치 구성에서 normal:artifact 비율을 적절히 조정(예: 1:1 ~ 1:3)해 안정적으로 학습.
- **추론 & CLAM 연동**:
  - coords-only H5 + WSI → 패치 생성(기존 CLAM WSI 로직 재사용).
  - 학습된 모델로 artifact 패치 + mask 영역을 normal class 조건으로 inpainting-style 복원.
  - 복원된 패치를 `output_dir/patches/slide_id.h5`의 `imgs`로 저장하고, `coords`/attrs는 그대로 복사.
  - CLAM 실행 시 `patch_h5_dir`를 새로운 복원 H5 디렉터리로 바꿔 성능 비교.

---

### 6. 실험 2: COAD + Synth (전략 A: 데이터 추가)

- **데이터**: COAD-Artifact + ArtiDiffuser-Synth (ori, inpainted, mask).
- **매핑**:
  - Synth `ori` → 정상 패치 (cls=0).
  - Synth `inpainted` → artifact 패치 (cls>0, Synth 마스크).
- **학습**:
  - COAD-only와 동일한 DDPM loss 구조 유지.
  - 배치에 COAD와 Synth를 섞어 사용 (데이터 소스 비율은 1:1 등으로 실험적으로 조정).
- **추론 & CLAM 연동**:
  - 실험 1과 동일한 전처리/CLAM 파이프라인.
  - COAD-only vs COAD+Synth(A) 모델의 CLAM 성능 및 복원 이미지 품질 비교.

---

### 7. 실험 3: COAD + Synth (전략 B: 약한 증류)

- **아이디어**:
  - Synth는 `(input = inpainted, target = ori)` 페어가 있으므로, DDPM 학습에 **추가 재구성 loss**를 얹어 teacher처럼 활용.

- **추가 loss (Synth 배치에만)**:
  - DDPM noise prediction loss는 동일.
  - 별도로, 샘플링된 결과 `x0_hat`와 `ori` 사이의 mask 영역 MSE를 계산:
    - `recon_loss = ((x0_hat - x0_ori)**2 * mask).mean()`.
    - `loss_total = loss_ddpm + lambda_recon * recon_loss` (lambda_recon은 0.1~1.0 정도로 작은 값부터).

- **추론 & CLAM 연동**:
  - 실험 1, 2와 동일하게 복원 → H5 → CLAM.
  - 세 모델(COAD-only, COAD+Synth-A, COAD+Synth-B)의 성능/품질 비교.

---

### 8. 평가 및 후속 작업

- **정량 평가**:
  - 세 설정에 대해 동일한 CLAM 학습/추론 세팅에서 AUC, ACC 등 비교.

- **정성 평가**:
  - COAD-Artifact real 패치에 대한 복원 전/후 시각 비교.

- **후속 작업 후보**:
  - 마스크 생성 개선, loss 가중치/스케줄 튜닝, UNet 용량 조정.
  - 필요 시 LatentArtiFusion-style latent space 확장 또는 Stable Diffusion inpainting과의 조합도 고려.

