# 2026-10-07: 문헌 보강과 WSI 실험 종합

## 작업 목적

누적된 염색·patch sampling·임계값·불균형 처리 결과를 연결하고, 설명 가능한 범위와
남은 질문을 구분한다. [종합 문서](../docs/smc_wsi_research_synthesis_20261007.md)를
현재 연구 해석의 진입점으로 추가했다. 새 학습이나 원본 결과 변경은 수행하지 않았다.

## 문헌 등록

기존 reviewed evidence를 먼저 검색한 후 필요한 방법론 문헌 7편을 확보했다.
공식 PMC/JMLR, 저자 arXiv, 대학 repository의 공개 PDF를 사용했다.
각각 page-preserving text를 추출하고 아래 페이지를 이미지로 확인했다.

| RAG paper ID | 확인 페이지 / 범위 | 주 연결 |
|---|---|---|
| `goorbergh2022_imbalance` | PDF p.1, Abstract | 재균형·discrimination·calibration |
| `carriero2025_imbalance_ml` | PDF p.1, Abstract | ML 재균형의 확률 해석 |
| `chicco2020_mcc` | PDF p.1, Abstract | MCC와 F1의 보완 |
| `lipton2014_f1_threshold` | PDF p.1, Abstract; arXiv v2 | F1 임계값의 조건 |
| `cawley2010_selection_bias` | PDF p.1, Abstract | 선택 과정과 평가 분리 |
| `varoquaux2018_small_sample_cv` | PDF p.1, Abstract; 2017 arXiv v1 | 소표본 CV 불확실성 |
| `riley2020_sample_size` | PDF p.1 표지 + p.2 Summary points/도입부 | 환자·event·복잡도와 표본 적절성 |

전문 전체를 검토한 것으로 표시하지 않았다. 근거 atom은 모두 `qualified`로 등록하여
심장이식 WSI로의 적용 한계를 드러냈다. 메타데이터, PDF SHA256, source URL,
선택 페이지/추출 줄 locator, card, claim, topic, facet를 함께 갱신했다.

- 등록 문헌 **25→32편**, claim **27→34개**, reviewed atom **29→36개**.
- topic **14→18개**: `imbalance_handling`, `rare_positive_evaluation`,
  `threshold_selection`, `small_sample_validation` 추가.
- 기존 후보 queue는 검토한 논문과 별개이므로 유지했다.
- 원문 PDF와 full extracted text는 로컬에 보관한다. 구조화된 RAG 메타데이터와
  검증·검색 도구는 Git으로 공유하도록 ignore 범위를 좁혔다.

## 종합 해석

1. 불균형 보정이 항상 개선책이라는 가정은 문헌과 이번 대응 결과 모두 지지하지 않는다.
   다만 tabular/회귀 근거로 우리 sampling의 인과 효과를 설명하지 않는다.
2. AP/AUROC, cutoff에서의 TP/FP/F1/F2/MCC, calibration은 서로 다른 질문이다.
   F1을 추가했다고 이 구분이 사라지는 것은 아니다.
3. seed 평균과 ensemble을 구분하고 환자 단위 조건부 bootstrap을 전체 학습
   불확실성으로 표현하지 않는다. 기존의 높은 성능과 독립 평가 결과를 섞지 않는다.
4. 다음 작업은 오류 사례의 기존 자료 대조와 보류 염색 검토의 연결이다. 제외 양성
   3건은 ACR 재판정이 아니라 염색 보류 해결이 목적이다. 학습 길이 대조는 보류 유지.

## 확인 방법

```bash
python reference/tools/validate_reference.py --strict-sources
python reference/tools/coverage.py
python -m unittest discover -s reference/tests -v
python reference/tools/search_reference.py "class imbalance threshold small sample validation" --top-k 7
python reference/tools/search_reference.py "불균형 양성 부족 평가지표 임계값" --top-k 7
```

`--strict-sources`는 PDF/fulltext가 있는 로컬에서 실행한다. Git으로 받기만 한 서버는
기본 validation과 structured-evidence 검색을 사용할 수 있으며 원문 검증은 manifest의
출처를 확보한 후 수행한다. 검토 locators는 이번에 확보한 파일 버전에 해당한다.

실행 결과: strict validation PASS(32 papers / 34 claims / 18 topics / 36 reviewed atoms),
retrieval 회귀 검사 8개 PASS. 영문 원래 질문은 불균형·소표본·calibration 근거를,
한국어 질문은 불균형·MCC·F1 임계값·PR 근거를 검색했다. coverage도 재생성했다.
PDF/fulltext를 제외한 별도 metadata-only 사본에서도 기본 validation과 JSON 검색이
통과했고, 종합 문서와 이 노트의 로컬 링크가 모두 존재함을 확인했다.

## 같은 날짜 후속 보강: 앞으로의 연구 방향

사용자 요청에 따라 종합 문서 7절에 연구 방향과 실행 전 판단 기준을 추가했다.
권장 주제는 희귀 양성에서 다중 염색 이미지가 H&E와 염색 구성·취득 과정 정보
이상의 정보를 제공하는지다. 기존 reviewed RAG를 먼저 검색하고 CLAM, MADELEINE,
Howard의 공개 1차 출처를 재확인했다. 새로운 논문 등록이나 학습 실행은 하지 않았다.

- 즉시 가능한 작업: 기존 오류 사례/QC 대조, 염색 비교 가능한 cohort의 양성 환자 수 확인.
- 학습 후보: 작은 frozen-feature 기준선, 메타데이터-only 대조, 동일 event의 염색 추가 정보.
- 보조 분석: 고정 검토량에서 OOF 순위의 양성 포착 정도. 사후 분석이며 임상 유용성 검증과 구분.
- 조건부 확장: ROI pooling, 별도 AMR endpoint, unlabeled adaptation, 임상 endpoint 연결.
- 각 후보에 필요한 자료, 비교 조건 및 중단·보류 기준을 기록했다. 학습 길이 대조 보류는 유지한다.

### 가독성 정리

사용자 요청으로 종합 문서를 처음 읽는 사람 기준으로 다시 편집했다.
검사·양성의 의미를 먼저 설명하고, 영어 용어와 중복 설명을 줄였다.
핵심 결과·평가의 한계·후속 연구 순서는 유지하고 세부 구현 조건은 본문에서 덜어냈다.
