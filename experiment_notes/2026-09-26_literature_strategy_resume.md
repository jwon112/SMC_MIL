# 문헌 기반 연구전략 조사 재개 체크포인트

작성일: 2026-09-26

## 현재 목표

SMC 심장이식 WSI 데이터의 다음 연구방향을 문헌 근거로 결정한다. 단순한 데이터셋 기술을 넘어서, 현재 생검 판정과 임상적으로 의미 있는 경과 및 미래 거부반응 위험을 서로 다른 연구과제로 구분한다.

## 확인된 로컬 데이터 구조

- 환자 136명
- 병리 사건 578건
- WSI bag 1,269개
- 환자당 평균 사건 4.25건, 중앙값 4건, 최대 9건
- 환자의 80.9퍼센트가 반복 생검을 보유
- 사건당 슬라이드 평균 2.2개, 중앙값 1개, 최대 9개
- 사건의 47.8퍼센트가 복수 슬라이드
- 고등급 ACR 양성은 40 bag, 17사건, 14환자
- AMR 양성은 71 bag, 19사건, 14환자
- 일반 3-fold 환자 분할에서는 희귀 과제의 검증 양성 환자가 fold당 4명에서 5명뿐

## 문헌조사에서 확정한 방향

1. 가장 방어 가능한 1차 과제는 H&E 기반 동시점 ACR 판정이다.
2. 슬라이드가 아니라 생검 사건을 분석 단위로 삼고, 사건 안의 복수 슬라이드를 계층적으로 집계해야 한다.
3. ISHLT 등급 재현과 임상적으로 유의한 거부반응 경과 예측은 동일한 목표가 아니다.
4. 미래 거부반응은 현재 등급 분류기를 재사용하는 문제가 아니라, 반복 생검과 임상정보를 결합한 종단 위험예측 문제다.
5. AMR은 H&E만으로 완전한 pAMR 분류를 재현한다고 주장하면 안 된다.
6. GSE290577은 공간전사체 tissue-core 자료이며 routine WSI 외부검증 코호트가 아니다.
7. 136명의 독립 환자 수를 고려하면 복잡한 end-to-end 예후모델보다 frozen image representation과 저차원 임상변수의 결합부터 시작해야 한다.

## 새로 확보하고 검토한 문헌

### Arabayarmohammadi et al., 2024

- 논문: Failing to Make the Grade
- DOI: 10.1161/CIRCHEARTFAILURE.123.010950
- 공개 PMC author manuscript 전체를 HTML에서 PDF로 보존
- PDF: reference/paper_origin/peyster2024_clinical_trajectory.pdf
- 추출 텍스트: reference/extracted_text/peyster2024_clinical_trajectory.md
- 페이지 이미지: reference/page_images/peyster2024_clinical_trajectory/
- SHA-256: 10a9b4990c97f8b9020265bd0fafb8420e19ead7718f67db96bc1550e643dc82
- 검토된 핵심 결과:
  - CARE의 임상 경과 판별 AUC 0.81
  - ISHLT grade 최적화 모델의 임상 경과 AUC 0.48
  - 임상 경과 최적화 CARE의 grade 판별 AUC 0.59
  - 단일기관, 불균형, 임상적으로 evident한 사건 부족이 명시적 한계

### Kim et al., 2026

- 논문: An integrated clinical-histopathologic prediction model for cardiac allograft rejection
- DOI: 10.1016/j.healun.2026.04.031
- PMID: 42070726
- 출판사 PDF 자동 접근은 차단됨
- 공식 NCBI E-utilities PubMed XML 초록을 PDF로 보존했으며, 근거 범위는 초록으로 제한
- PDF: reference/paper_origin/kim2026_integrated_risk_pubmed.pdf
- 추출 텍스트: reference/extracted_text/kim2026_integrated_risk_pubmed.md
- 페이지 이미지: reference/page_images/kim2026_integrated_risk_pubmed/
- SHA-256: 3dc192eeb59c07e3aa6bc1c731ddd2466caa05d10d3dd1ae26793bddd3d49d99
- 검토된 핵심 결과:
  - 484명, EMB encounter 1,188건
  - H&E 형태학 370개와 종단 임상변수 268개 결합
  - 통합 종단모델 AUROC 0.86, AUPRC 0.74
  - IRRI 고위험 HR 6.15, 저위험 HR 0.52
- full text가 아니므로 split, calibration, censoring, 외부검증 세부사항은 미확인 상태

## 아직 완료하지 않은 작업

- 위 두 논문의 card, evidence atom, claim, paper index, topic route, source manifest 등록
- strict validator 실행
- 원래 질의로 재검색하여 skill 회수 여부 확인
- 최종 연구전략 문서 또는 답변 작성

## 등록 예정 ID

- paper_id: arabayarmohammadi2024_clinical_trajectory
- paper_id: kim2026_integrated_risk
- claim_id: claim-clinical-trajectory-morphology-feasible
- claim_id: claim-grade-vs-clinical-trajectory
- claim_id: claim-longitudinal-multimodal-risk

## 재개 순서

1. reference/cards에 두 카드 추가
2. reference/evidence/core_evidence.jsonl에 2024년 근거 2개와 2026년 근거 2개 추가
3. reference/indexes/papers.json과 claims.json 갱신
4. reference/indexes/topics.json의 cardiac_allograft_pathology 및 weak_and_longitudinal_labels에 경로 추가
5. reference/paper_origin/manifest.json에 출처 URL, provenance, access note, SHA-256 추가
6. python reference/tools/validate_reference.py --strict-sources
7. python reference/tools/search_reference.py로 grade trajectory future rejection longitudinal multimodal 질의를 다시 실행
8. 최종 전략을 단계 0 데이터감사, 단계 1 동시점 ACR benchmark, 단계 2 사건단위 계층모델, 단계 3 임상경과, 단계 4 종단 멀티모달 예후, 단계 5 외부검증 순서로 정리

## 도구상 주의

- 기본 apply_patch는 현재 한글 workspace 경로에서 sandbox helper 오류가 발생했다.
- codex hidden apply-patch를 PowerShell 변수로 호출하면 큰 패치 또는 JSON의 큰따옴표 전달이 깨졌다.
- 다음 재개 시에는 Windows native argument 전달을 보존하는 방식으로 작은 패치를 적용하거나, 정상 동작하는 apply_patch 환경에서 등록한다.
- reference와 experiment_notes는 Git에서 제외되어 있으므로 현재 확보 자료와 이 체크포인트는 로컬에만 존재한다.
