## mini: segmentation ops playground

이 폴더는 CLAM-master 레포 안에서 **세그멘테이션(U-Net) 구조/모듈/연산을 빠르게 교체**하며 실험하기 위한 최소 프로젝트입니다.

### 빠른 시작 (Pascal VOC 2012)

- 기본은 `torchvision.datasets.VOCSegmentation`을 사용하며, 처음 실행 시 자동 다운로드가 가능합니다.
- Windows/서버 환경 모두에서 동작하도록 경로는 인자로 받습니다.

```bash
python mini/train.py --dataset voc --data_root ./_data --run_dir ./mini/runs/voc_unet_baseline --epochs 20 --batch_size 8
```

### 폴더 구조

- `mini/data/dataloader.py`: 세그멘테이션 데이터셋/전처리/로더
- `mini/model/unet.py`: 교체 가능한 U-Net 뼈대
- `mini/module/blocks.py`: Conv 블록/업샘플/정규화 등 교체 가능한 모듈
- `mini/train.py`: 학습 루프, 저장, 간단한 검증(mIoU)

### 다음 단계(당신이 하려는 방향)

- `mini/module/blocks.py`에 새로운 블록을 추가하고
- `mini/model/unet.py`에서 해당 블록을 주입(인자)하거나 Registry로 선택
- `mini/train.py` 실행 인자로 블록/정규화/업샘플 방식을 바꿔 성능/학습 안정성 비교

