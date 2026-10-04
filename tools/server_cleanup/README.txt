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

3차: 루트 코드·로그·라벨 버전의 추가 검토
python tools/server_cleanup/audit_workspace.py

결과는 .server_cleanup_phase3/<UTC timestamp>/workspace_audit.zip 입니다.
출력의 bundle 경로를 확인해 ZIP을 로컬 Project root로 가져오면 추가 분석에 쓸 수 있습니다.
이 ZIP은 서버에서 수집한 검토 자료입니다. 실행 코드 전달은 계속 Git으로 합니다.

수집 내용: 루트 파일 목록·Git 추적 여부, 실행 프로세스의 프로젝트 경로 참조,
코드 import 및 문자열 참조, 현재/보관 실험 설정의 라벨 참조,
라벨 파일 스키마·행 수·식별자 개수·SHA-256 및 코드 사본.
현재 루트와 mini의 notebook은 코드 셀만 추출하며 출력 셀은 포함하지 않습니다.
라벨 CSV 행, 원본 Excel 내용, feature, WSI, checkpoint, 전체 로그는 복사하지 않습니다.
파일 이동·삭제·라벨 수정은 수행하지 않습니다.
Git에 없는 코드는 먼저 연구 이력으로 보존하고 역할·진입점·의존성을 확인합니다.
단순 참조 미검출이나 파일 시각은 deprecated 판정의 충분한 근거가 아닙니다.
CSV의 case_id 반복은 반복 생검·다중 슬라이드에서 정상일 수 있으며 자동 오류로 판정하지 않습니다.

서버·로컬 Git 통합 (서버의 현재 코드가 기준)
서버에서 최신 관리 도구를 받은 후 실행합니다.

git pull --ff-only origin main
python tools/publish_server_sources.py --action plan
python tools/publish_server_sources.py --action publish

plan은 미추적/수정 파일 중 소스·문서·설정·노트북을 선별하고 경로를 출력합니다.
정확한 목록은 .server_source_sync/plan.json에 있으며 publish는 그 파일 목록과
해시가 여전히 일치하는지 검증한 뒤 해당 경로만 add, commit, push합니다.
기존 staged 작업이 있으면 섞어 커밋하지 않고 중단합니다.
Git 작성자 이름/이메일 및 origin push 권한이 서버에도 필요합니다.
push가 실패해도 생성한 로컬 커밋은 보존됩니다. 오류 해결 후 git push origin main을
재시도하면 됩니다. force push는 사용하지 않습니다.

관리 대상: 소스, 실행 shell, README/문서, 환경 설정, 이름으로 식별되는 JSON 설정,
출력 없는 노트북. 노트북은 원본을 .server_source_sync/notebook_backups/에 보존한 뒤
출력·실행번호·첨부·일시 메타데이터를 제거합니다. 코드/Markdown 셀 내용은 유지합니다.
모호한 JSON/TXT, 2MB 초과 파일, credential 의심 문자열은 HOLD로 남겨 별도 검토합니다.
자동 검사는 모든 비밀정보를 탐지한다는 보장이 없으므로 plan의 경로를 확인하세요.

서버 보존 대상: 원본 데이터, label/split 데이터, feature, 모델 가중치, 학습 결과,
로그, 설치 ZIP, 백업. 기존에 Git으로 추적하던 task CSV나 split의 이력은 지우지 않으며
이번 도구가 변경된 CSV/데이터를 새로 커밋하지도 않습니다.
논문 원문·추출 캐시와 기존 local reference workspace 제외 규칙도 유지됩니다.

서버가 PUSHED 커밋을 출력하면 로컬에서도 git pull --ff-only origin main으로 받습니다.
이후 같은 브랜치를 번갈아 수정할 때는 작업 전에 pull, 작업 후 commit/push합니다.
이 단계는 코드의 최신 내용을 하나로 모으는 단계이며 root의 파일 위치는 유지합니다.
서버 전용 소스를 받은 뒤 역할·의존성을 함께 검토해 코드 폴더 재배치를 진행합니다.
