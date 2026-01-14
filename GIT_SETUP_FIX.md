# Git 인식 문제 해결 방법

Git이 설치되어 있지만 PowerShell에서 인식되지 않는 경우 해결 방법입니다.

## 해결 방법

### 방법 1: PowerShell 완전히 재시작 (가장 간단)

1. **현재 PowerShell 창을 완전히 닫기**
   - X 버튼으로 닫거나 `exit` 명령어 입력

2. **새 PowerShell 창 열기**
   - Windows 키 + X → "Windows PowerShell" 또는 "터미널" 선택
   - 또는 시작 메뉴에서 "PowerShell" 검색

3. **다시 확인**
   ```powershell
   git --version
   ```

### 방법 2: 현재 세션에서 PATH 새로고침

현재 PowerShell 창에서 다음 명령어 실행:

```powershell
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
git --version
```

### 방법 3: Git 경로 직접 사용 (임시 해결책)

```powershell
& "C:\Program Files\Git\bin\git.exe" --version
```

### 방법 4: PATH 환경 변수 수동 추가 (영구적 해결)

만약 위 방법들이 작동하지 않으면:

1. **시스템 환경 변수 편집**
   - Windows 키 + R → `sysdm.cpl` 입력 → Enter
   - "고급" 탭 → "환경 변수" 클릭

2. **시스템 변수에서 Path 선택 → 편집**

3. **새로 만들기 클릭 후 다음 경로 추가:**
   ```
   C:\Program Files\Git\bin
   C:\Program Files\Git\cmd
   ```

4. **확인 클릭하여 모든 창 닫기**

5. **PowerShell 완전히 재시작**

6. **확인:**
   ```powershell
   git --version
   ```

## 빠른 확인 명령어

현재 PowerShell 창에서 실행:

```powershell
# Git 설치 확인
Test-Path "C:\Program Files\Git\bin\git.exe"

# PATH에 Git 경로가 있는지 확인
$env:PATH -split ';' | Select-String -Pattern 'git'

# Git 경로 직접 실행 테스트
& "C:\Program Files\Git\bin\git.exe" --version
```

## 가장 확실한 방법

**PowerShell을 완전히 닫고 새로 열면 대부분 해결됩니다!**
