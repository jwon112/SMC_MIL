# WSI 정량 특징 추출 — 서버 실행 안내

이 패키지는 **기존 WSI 원본 + 기존 패치 좌표**에서 수식 기반 특징을 계산합니다.
UNI2를 다시 학습하거나 증강하지 않으며 기존 PT/H5/원본/분할 파일을 수정하지 않습니다.
CPU 작업입니다. GPU 번호를 지정할 필요가 없습니다.

## 1. 이번 구현의 범위

- 사용자 `pathology_feature_extractor_v2.py`의 핵 형태/밀도, 림프구 유사 heuristic,
  H/E·RGB·회색 강도, GLCM, LBP, Gabor, wavelet, FFT, 핵 공간 graph를 재사용.
- 원본 WSI를 기존 **source 좌표**로 읽고 주변 halo(기본 8 µm)에서 핵 분할.
  핵 중심이 해당 core 안에 있는 경우에만 개수에 포함. context 바깥 경계에 닿은 핵은 제외.
- header MPP를 확인하고 0.25 µm/px로 resample. 조직/핵 면적은 이 표준 해상도로 계산.
  x/y MPP가 달라도 처리하며 core 물리크기 반올림 오차가 1%를 넘으면 중단.
- positive-area overlap patch를 원래 인덱스 순서로 제거. 인접 core가 접하는 것은 허용.
  분석 면적은 선택된 **비중복 core 면적**이며 전체 WSI 면적이 아님.
- 패치별 밀도 p90 등 hotspot 관련 분포 요약, 접하는 patch끼리의 rook Moran's I 추가.
  Moran은 slide 내 기술통계이며 p-value/임상 판정이 아님. tissue 내부 경계효과가 남음.
- 추가 reviewed mask가 있으면 심근/기질 면적비, 기질 형태, 기질 skeleton 기반
  area/length, annotated lymphocyte 수·심근 내부 밀도·심근 경계 근접도를 계산.
- 기질 feature는 연결된 mask 조각의 측정이며 논문에서 정의한 '섬유'를 자동 식별한 것이 아님.

현재 **구현하지 않은 것**: 자동 심근/기질 세포모델 학습, 검증된 림프구 classifier,
혈관 내 CD68/C4d 정량, 전역 림프구 초점, 경계보정 Ripley/permutation enrichment,
재이식 episode가 확인된 종적 변화량. 해당 값은 기존 색 수식으로 가짜 대체하지 않습니다.
HoVer-Net/CellViT 등의 segmentation 모델도 이번 패키지에 자동 다운로드/실행하지 않습니다.

H&E에서만 수식을 실행합니다. IHC/Others에 H&E 정규화를 적용하지 않습니다.
`manual_only`가 기본입니다. 현 목록이 그대로라면 H&E 84장 전후지만 audit에서 다시 셉니다.
PLHE는 stain CSV에서 HE로 확정된 경우만 포함됩니다. 모델 예측만으로 HE를 확정하지 않습니다.

## 2. 올릴 파일

ZIP의 파일들을 서버 `/home/jupyter/image_team/projects/SMC_MIL` 안에 풉니다.
기존 `pathology_feature_extractor_v2.py`가 있다면 먼저 비교/백업 후 동봉 버전을 배치하세요.
다른 기존 CLAM/UNI2 코드는 교체하지 않습니다. 패키지에는 다음 파일이 있습니다.

```
run_wsi_pathomics.py
pathomics_core.py
pathology_feature_extractor_v2.py
pathomics_config.json
requirements_pathomics.txt
export_pathomics.mjs
README_PATHOMICS.md
```

config의 데이터 경로는 이전 실행 로그 기준입니다. 실제 존재 여부와 source reader
인터페이스는 audit로 확인합니다. 기존 source reader는 서버에 있어야 합니다:
`extract_features_dicom.py`, `extract_features_mrxs.py`, `dicom_pyramid.py`, `mrxs_pyramid.py`.

별도의 환자 엑셀을 다시 올릴 필요는 없습니다. 서버의 gold CSV에
`slide_id,case_id,event_id,biopsy_date,source_dataset`가 필요합니다.
날짜가 없는 future cohort CSV 대신 기본 설정의 원래 gold CSV를 사용하세요.
예시 ECG 엑셀의 **표 형태**만 참고했으며, ECG 값이나 ECG label을 복사하지 않습니다.

## 3. 서버 실행

프롬프트 `(clam_latest) root@...#`는 복사하지 말고 명령만 실행합니다.

```bash
cd /home/jupyter/image_team/projects/SMC_MIL
python -c "import numpy,pandas,scipy,skimage,cv2,pywt,h5py,PIL; print('pathomics dependencies OK')"
python -u run_wsi_pathomics.py audit
```

import가 실패한 경우에만 환경 담당자와 확인하고 누락 패키지를 설치합니다.
현재 환경을 무조건 업데이트하지 마세요. 요구 목록은 `requirements_pathomics.txt`입니다.

audit는 모든 대상에서 source 좌표, 기존 PT 존재/H5 row 수, MPP,
중복 core 수를 확인합니다. 40x 폴더명만 보고 MPP를 0.25로 추정하지 않습니다.
MPP가 없다면 헤더/검사기록으로 확인한 값을 다음 CSV에 넣고 config에
`"mpp_overrides_csv": "/path/reviewed_mpp.csv"`를 추가하세요.

```text
slide_id,mpp_x,mpp_y,review_source
SAMPLE_SLIDE,0.25,0.25,scanner_record_checked
```

MRXS baseline H5의 오래된 좌표 불일치는 별도 flag로 기록하고 source 좌표만 사용합니다.
원본 UNI 재인코딩 동일성을 이 도구가 입증하는 것은 아닙니다. PT feature와 직접 concat하기
전에는 `slide_id+patch_index+좌표+MPP`와 기존 encoder parity를 별도 검증해야 합니다.

먼저 포맷별 1장, 장당 8개 패치를 검사합니다.

```bash
python -u run_wsi_pathomics.py smoke --workers 1
```

`/home/jupyter/data/image_team/pathomics_v1/smoke/slides/<slide_id>/preview_*.png`:
왼쪽 원본 core / 오른쪽 핵 경계(하늘색)·림프구 유사 경계(노란색).
이는 검증된 림프구 annotation이 아닙니다. 병리 검토로 분할을 확인한 후 전체 실행합니다.

```bash
nohup python -u run_wsi_pathomics.py all --workers 2 > pathomics_full.log 2>&1 &
tail -f pathomics_full.log
```

`Ctrl+C`는 tail만 끝냅니다. 학습이 아니라 특징 추출 작업입니다.
CPU/RAM 부담이 크면 `--workers 1`로 시작하세요. GPU 학습과 병행 시 CPU/I/O 경합 가능.
처리속도는 패치 100개마다 출력됩니다. 전체 소요시간은 smoke가 아닌 실제 full-slide
처리속도와 남은 patch 수를 기준으로 추정하세요. 미리 몇 시간이라고 보장하지 않습니다.

중단 후 같은 명령 재실행: checksum·config가 같은 완료 slide는 재사용,
미완료 slide만 처음부터 다시 계산합니다. source/config/코드가 바뀌면 출력 경로를 새로 설정해야 합니다.
실패 slide가 있으면 성공한 것만 정상 전체 결과인 척 내보내지 않고 exit 1입니다.
의도적으로 중간 현황만 보고 싶다면 `python run_wsi_pathomics.py report`;
불완전 event의 특징은 NA이며 `feature_status=incomplete`입니다.

## 4. 결과 — 예시 Excel처럼 한 행 = 생검 1건

기본 저장 위치: `/home/jupyter/data/image_team/pathomics_v1/full/`

| 파일 | 내용 |
|---|---|
| `biopsy_features_core.csv` | 핵심 후보 특징, examid·case_id·event_id·날짜 이후 mean/median/std/min/max/p10/p90/valid_n 열 |
| `biopsy_features_all.csv` | 수백 개 patch 특징을 전부 집계한 넓은 표 |
| `slide_features_all.csv` | WSI별 집계 + slide 내 Moran |
| `slides/<slide_id>/patch_features.csv.gz` | 각 source patch의 모든 수치·원래 index·좌표·QC |
| `workbook_data.json` | 핵심 생검 표·WSI QC·사전·감사 정보를 한 XLSX로 바꾸는 입력 |
| `feature_dictionary.csv` | 특징 이름·단위·정의·주의사항 |
| `cohort_metadata.csv` | 원래 label 등 연결 메타데이터; feature 계산과 분리 |
| `candidate_feature_columns.json` | 후보 숫자 열 목록; 전부 모델에 투입하라는 의미는 아님 |
| `exclusions.csv`, `run_config.json` | 미포함 이유·설정/코드 hash·라이브러리 버전 |

CSV는 UTF-8 BOM으로 저장되어 Excel에서 바로 열 수 있습니다. 수십만 patch를 한 XLSX에
전부 넣지는 않습니다. 생검 표는 모든 gold event를 유지하여 HE가 없는 event도 남깁니다.
미래 예측 input은 해당 날짜의 행만 사용하며 환자 전체 생검을 평균하지 않습니다.
양성/음성 결과를 보고 feature/QC 설정을 고르지 마세요.

한 생검에 여러 WSI가 있으면 QC 통과 core들을 직접 모아 분포를 집계합니다.
`__mean`은 patch 값 평균, `__p90`은 patch 분포 90분위수,
`__std`는 표본 SD(유효 patch 1개면 NA), `__valid_n`은 해당 수치가 유효한 patch 수입니다.
`pooled_density`는 핵 수 합계 / 분석 면적 합계로, patch 밀도의 단순 평균과 다릅니다.
연속절편이 같은 3D 세포를 재관측할 수 있어 슬라이드 pooled 수를 독립 세포 수로 해석하면 안 됩니다.

## 5. 진짜 .xlsx 파일 생성

Python 서버 추출은 Excel 설치나 Node가 없어도 끝납니다. XLSX 작성은 동봉 Node exporter로
분리했습니다. **Node와 `@oai/artifact-tool`가 없는 서버에서 .xlsx까지 자동 생성된다고
가정하지 마세요.** 현재 로컬 Codex 환경에는 이 의존성이 있습니다.

서버에 이미 exporter 의존성이 구성된 경우:

```bash
node export_pathomics.mjs /home/jupyter/data/image_team/pathomics_v1/full/workbook_data.json
```

구성되지 않은 경우에는 `workbook_data.json` **한 파일만 로컬로 내려받아 전달**하면
`생검_WSI_정량특징.xlsx`로 변환할 수 있습니다. 서버 CSV 작업을 다시 할 필요는 없습니다.
로컬 개발환경의 실행 예:

```bash
/Users/zangzoo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node /Users/zangzoo/vscode/biopsy/pathomics_server/export_pathomics.mjs /Users/zangzoo/Downloads/workbook_data.json /Users/zangzoo/vscode/biopsy/outputs/pathomics_생검특징.xlsx
```

첫 시트 `Biopsy_Features`는 ECG 예시처럼 첫 행에 feature명, 이후 한 생검씩 배치합니다.
추가 시트: Slide_QC, Feature_Dictionary, Exclusions, Extraction_Audit, Methods.
엑셀에는 핵심 후보 세트, 전체 특징은 `biopsy_features_all.csv`에 있습니다.

## 6. 외부 분할 마스크 연결 (선택)

config에 `"mask_root": "/path/reviewed_masks"` 추가. 각 core마다:

```
reviewed_masks/<slide_id>/00000000.npz
reviewed_masks/<slide_id>/00000000.json
```

파일명 숫자는 source coordinate의 **원래 patch_index**. NPZ는 pickle이 없는 정수 배열:
`myocardium`(0/1), `stroma`(0/1), `lymphocyte_instances`(0=배경, 1..N=객체).
필요한 key만 넣어도 됩니다. 배열 크기는 halo 없는 source core `(patch_size,patch_size)`.
myocardium/stroma가 겹치면 중단. 외부 lymphocyte core 경계 객체는 제외하므로 경계 편향이 남습니다.
분할 provenance sidecar에는 다음 값을 넣습니다.

```json
{
  "slide_id": "SAMPLE_SLIDE",
  "patch_index": 0,
  "x_source": 1024,
  "y_source": 2048,
  "patch_size": 256,
  "coordinate_sha256": "audit에서 나온 source 좌표파일 SHA256",
  "review_source": "model_version_and_pathologist_QC_reference"
}
```

분할 마스크가 없으면 해당 특징은 NA. 0은 실제 면적/수가 0일 때만 사용합니다.
마스크 추가/변경 시 새 output 경로에서 실행하세요.

## 7. 선행연구와 해석

- CACHE-Grader: https://pubmed.ncbi.nlm.nih.gov/33982079/
- CARE: https://pmc.ncbi.nlm.nih.gov/articles/PMC10940208/
- 기질 재형성: https://doi.org/10.1016/j.jhlto.2024.100202
- 공간 분석 개념: https://www.nature.com/articles/s41592-021-01358-2

구획·공간 특징이라는 개념을 참고한 탐색 구현이지 위 논문의 모델/파라미터 재현이 아닙니다.
기본 H/E 강도는 정규화하지 않은 원본 HED projection입니다. `macenko_fixed`는 별도 옵션이고
H&E만 허용됩니다. 세 번째 HED 축이 실제 DAB 발현이라는 뜻은 아닙니다.
닫힌 흰 영역은 혈관 내강/부종 확정값이 아니며 `qc_enclosed_white_fraction`으로 명명했습니다.
핵/조직 면적비는 N/C ratio가 아닙니다. 특징 수가 많다고 성능이 좋아지는 것은 아닙니다.
추후 UNI2와 결합할 때는 train-only 결측대치/정규화/feature selection과 patient split을 유지하세요.

## 8. 검증 범위

로컬 합성 RGB/마스크 테스트로 수식·생검 집계·일부 리더 연계 계약을 검증합니다.
실제 서버 DICOM/MRXS 전체 처리와 병리 분할 정확도는 여기서 실행/검증하지 않았습니다.
먼저 audit와 포맷별 smoke 결과를 확인하세요. 동봉 예시 XLSX는 **합성 테스트 값**입니다.
