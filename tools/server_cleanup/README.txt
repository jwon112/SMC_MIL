SMC 서버 정리 — Git 전달 방식 (2026-10-04)

최초 전환: 기존 ZIP 설치 파일을 Git에 맞추기
서버 SMC_MIL 루트에서 실행합니다. 결과·feature 전체를 stash하지 않습니다.

git fetch origin
git show origin/main:tools/sync_server_checkout.py > /tmp/smc_sync_server_checkout.py
python /tmp/smc_sync_server_checkout.py --apply

sync 도구는 원격 커밋의 검토 기록과 일치하는, 업데이트 대상 경로의 로컬 수정만
이름 있는 stash에 보존하고 검토한 커밋으로 fast-forward합니다.
기존 코드와 다른 미검토 변경, 서버 독자 커밋은 자동 덮어쓰지 않고 중단합니다.
보관된 stash에는 기존 ZIP 설치 이력이 있으므로 업데이트 후 stash pop하지 마세요.
기존 ZIP 2차 도구로 결과 이동까지 수행했다면 먼저 그 도구로 rollback하세요.

이후 일반 업데이트
git pull --ff-only origin main

결과 정리
python tools/server_cleanup/manage.py --action plan
python tools/server_cleanup/manage.py --action apply

대상: 초기 gold 3-fold 16개와 weak-label 3-fold 64개, 총 80개.
서버에 없는 항목은 missing으로 보고하고 건너뜁니다.
정확한 목록은 move_plan.json 및 실행 후
.server_cleanup_phase2_20261004/preview.json에 저장됩니다.
보관 위치:
  archive/experiments/early_cv/
  archive/experiments/weak_linkage/

apply는 Git으로 받은 코드를 검증하고 결과 디렉터리와 경로 대응표만 관리합니다.
코드 설치·덮어쓰기는 수행하지 않습니다. 코드 변경은 Git으로 관리합니다.
같은 서버 파일시스템에서 폴더 이름을 바꾸므로 결과 파일을 복제하지 않습니다.
현재 대조실험, 5-fold baseline, augmentation, 외부 검증, future, results_old*,
라벨·split·feature·로그는 현재 위치에서 계속 사용합니다.

완료 출력:
  verified: true
  moved_experiments: 80  (실제 존재한 개수)
  experiment_discovery_unchanged: true
전체 파일 SHA-256 및 집계 대상 실험 이름 집합을 이동 전후 비교합니다.
실제 성능을 재계산하거나 GPU 추론을 실행하지 않습니다.
과거 학습·집계 프로세스가 대상 파일을 사용 중이면 PID를 출력하고 중단합니다.
이동 중 과거 학습·집계 실행을 새로 시작하지 마세요.

재검증과 결과 위치 복구
python tools/server_cleanup/manage.py --action verify
python tools/server_cleanup/manage.py --action rollback

Git 버전의 rollback은 결과 위치와 경로 대응표만 복구합니다.
Git 코드 이력은 유지됩니다. 코드 버전을 되돌릴 때는 먼저 결과 rollback을 하세요.
transaction.json 및 .smc_result_archive.json은 복구·조회에 필요합니다.
강제 종료 후 operation.lock이 남으면 기록된 PID의 종료를 확인하고
.server_cleanup_phase2_20261004/operation.lock만 제거한 뒤 rollback하세요.

조회 호환성
수정된 집계기·외부 평가·비교·eval.py는 보관 실험을 계속 찾습니다.
CV launcher는 보관된 이름을 건너뜁니다. 재학습은 새 실험 이름을 사용하세요.
임의의 과거 노트북에는 아래 명령으로 확인한 실제 경로를 사용하세요.
python tools/smc_result_paths.py results/이전실험이름

테스트
python tools/server_cleanup/test_migration.py
python tools/server_cleanup/test_sync.py
첫 테스트는 synthetic 결과의 이동 전후 지표·OOF·검증 대상 일치와 복구를 확인합니다.
프로세스 검사는 로컬 테스트에서 mock하고 서버 plan/apply에서 실제 /proc를 검사합니다.
