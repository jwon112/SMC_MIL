# 2026-10-07: 미검토 문헌 집계와 추가 검토

## 요청과 수행

미검토 논문 수를 확인하고 추가 검토했다. 작업 전 등록 43편과 대기 8편을 대조한 뒤, 대기 목록 중 Peyster의 심장 간질 변화 연구, CONCH, TITAN의 핵심 본문·방법·한계를 검토했다.
각각 5개 PDF 페이지를 추출·렌더링하여 확인하고, 재사용 가능한 근거 6개를 RAG에 등록했다.

작업 후 **등록 46편, 근거 54개, 대기 5편**이다. 등록 문헌 중 **15편은 초록·요약 수준**이다.
`reviewed`를 전체 논문 검증 완료로 오해하지 않도록 검토 범위와 미확인 사항을 [검토 현황](../reference/REVIEW_STATUS_2026-10-07.md)에 정리했다.

## 현재 연구와의 연결

- `claim-peyster2024_stromal_remodeling-longitudinal-stroma`: 현재 중증 거부반응이 없는 생검에도 장기 조직 변화와 예후를 연결하는 별도 연구 질문이 가능하다. SMC에 해당 임상 경과가 있는지 확인이 필요하다.
- `claim-lu2024_conch-frozen-representation`: 고정 특징과 작은 분류기의 비교에 참고한다. 이번 원문은 2023 arXiv v2이며, 2024 출판본과 구분한다.
- `claim-chen2025_titan-slide-representation`: 사전학습된 슬라이드 표현과 mean pooling/ABMIL 비교는 참고할 만하다. 현재 UNI v2 특징을 TITAN 입력으로 재사용할 수 있다고 가정하지 않는다.
- `claim-chen2025_titan-validation-boundaries`: 신장 이식 거부반응 평가는 심장 이식 검증이 아니며, 환자 분리와 생존분석 모델 선택 조건을 별도로 기록했다.

새 학습을 실행하거나 실험 라벨을 수정하지 않았다. 다음 검토 순서는 현황 문서에 기록했다.

## 확인

등록과 대기 목록을 함께 갱신했고 출처 경로·해시·페이지 위치를 검증했다. 검증 명령:

출처 엄격 검증, 기존 검색 테스트 8개, 새 문헌 3편의 관련 질의 상위 8개 검색 확인이 모두 통과했다.

```text
python reference/tools/validate_reference.py --strict-sources
python -m unittest discover -s reference/tests -v
python reference/tools/search_reference.py "TITAN renal allograft rejection" --top-k 8
```
