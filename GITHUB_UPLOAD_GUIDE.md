# GitHub 업로드 가이드

이 문서는 CLAM 프로젝트를 GitHub에 업로드하는 방법을 안내합니다.

## 사전 준비사항

### 1. Git 설치

#### Git 설치 확인
먼저 Git이 이미 설치되어 있는지 확인하세요:

```powershell
git --version
```

만약 "git이 인식되지 않습니다" 또는 "command not found" 오류가 나오면 Git을 설치해야 합니다.

#### Git 다운로드 및 설치

**방법 1: 공식 웹사이트에서 다운로드 (권장)**

1. **Git 공식 웹사이트 방문**
   - https://git-scm.com/download/win 접속
   - 또는 직접 다운로드: https://github.com/git-for-windows/git/releases/latest

2. **다운로드**
   - "64-bit Git for Windows Setup" 다운로드 (대부분의 경우)
   - 파일명 예: `Git-2.x.x-64-bit.exe`

3. **설치 실행**
   - 다운로드한 `.exe` 파일을 실행
   - 설치 마법사가 나타나면 "Next" 클릭

4. **설치 옵션 선택 (권장 설정)**
   - **Select Components**: 기본 선택 유지 (Git Bash, Git GUI 등)
   - **Choosing the default editor**: 원하는 에디터 선택 (기본값: Vim 또는 Notepad++)
   - **Adjusting your PATH environment**: 
     - **"Git from the command line and also from 3rd-party software"** 선택 (권장)
   - **Choosing HTTPS transport backend**: 기본값 유지
   - **Configuring the line ending conversions**: 
     - **"Checkout Windows-style, commit Unix-style line endings"** 선택 (권장)
   - **Configuring the terminal emulator**: 기본값 유지
   - **Configuring extra options**: 기본값 유지

5. **설치 완료**
   - "Install" 클릭하여 설치 진행
   - 설치 완료 후 "Finish" 클릭

6. **설치 확인**
   - PowerShell 또는 명령 프롬프트를 **새로 열기** (중요!)
   - 다음 명령어 실행:
   ```powershell
   git --version
   ```
   - 버전 정보가 표시되면 설치 성공!

**방법 2: 패키지 관리자 사용 (선택사항)**

- **Chocolatey** 사용 시:
  ```powershell
  choco install git
  ```

- **Winget** 사용 시:
  ```powershell
  winget install Git.Git
  ```

#### Git 사용자 정보 설정 (처음 한 번만)

Git을 처음 설치한 후, 사용자 이름과 이메일을 설정해야 합니다:

```powershell
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

예시:
```powershell
git config --global user.name "홍길동"
git config --global user.email "hong@example.com"
```

설정 확인:
```powershell
git config --global user.name
git config --global user.email
```

### 2. GitHub 계정 준비
   - GitHub 계정이 필요합니다: https://github.com
   - 계정이 없다면 무료로 가입할 수 있습니다

## 단계별 업로드 방법

### 1단계: Git 저장소 초기화

프로젝트 폴더에서 다음 명령어를 실행하세요:

```bash
git init
```

### 2단계: 파일 추가 및 첫 커밋

```bash
# 모든 파일 추가
git add .

# 첫 커밋 생성
git commit -m "Initial commit: CLAM project"
```

### 3단계: GitHub에서 새 저장소 생성

1. GitHub 웹사이트(https://github.com)에 로그인
2. 우측 상단의 "+" 버튼 클릭 → "New repository" 선택
3. 저장소 이름 입력 (예: `CLAM-master` 또는 원하는 이름)
4. 설명 추가 (선택사항)
5. Public 또는 Private 선택
6. **"Initialize this repository with a README" 체크박스는 해제** (이미 README가 있으므로)
7. "Create repository" 클릭

### 4단계: 원격 저장소 연결 및 푸시

GitHub에서 저장소를 생성한 후, 표시되는 명령어를 사용하거나 아래 명령어를 실행하세요:

```bash
# 원격 저장소 추가 (YOUR_USERNAME과 YOUR_REPO_NAME을 실제 값으로 변경)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 기본 브랜치 이름을 main으로 설정 (필요한 경우)
git branch -M main

# GitHub에 푸시
git push -u origin main
```

**참고**: GitHub에서 제공하는 URL을 사용하세요. 예를 들어:
- HTTPS: `https://github.com/username/repository-name.git`
- SSH: `git@github.com:username/repository-name.git`

### 5단계: 인증 (필요한 경우)

최초 푸시 시 GitHub 인증이 필요할 수 있습니다:
- **Personal Access Token (PAT)** 사용 권장
- GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
- "Generate new token" 클릭하여 토큰 생성
- 토큰을 비밀번호 대신 사용

## 추가 팁

### .gitignore 확인
- `.gitignore` 파일이 이미 업데이트되어 있어 불필요한 파일들이 제외됩니다
- 필요시 `.gitignore`를 수정하여 추가 파일/폴더를 제외할 수 있습니다

### 대용량 파일 처리
- `.h5`, `.pt`, `.pth` 등의 모델/데이터 파일은 기본적으로 제외됩니다
- 대용량 파일이 필요하다면 Git LFS를 사용하거나 별도로 관리하세요

### 커밋 메시지 작성 가이드
- 명확하고 간결하게 작성
- 예: "Add feature extraction script", "Fix bug in heatmap generation"

## 문제 해결

### Git이 설치되지 않은 경우
- PowerShell에서 `git --version` 명령어로 확인
- 설치되지 않았다면 위의 "Git 다운로드 및 설치" 섹션을 참조하여 설치하세요
- 설치 후 **새로운 PowerShell 창**을 열어야 합니다 (기존 창에서는 인식되지 않을 수 있음)

### 인증 오류 발생 시
- Personal Access Token 사용 확인
- 또는 SSH 키 설정 고려

### 푸시 거부 시
- 원격 저장소 URL 확인
- GitHub 저장소가 올바르게 생성되었는지 확인

## 다음 단계

업로드가 완료되면:
1. GitHub 저장소 페이지에서 파일들이 올바르게 업로드되었는지 확인
2. README.md가 제대로 표시되는지 확인
3. 필요시 추가 설명이나 문서 작성
4. Issues, Pull Requests 등의 기능 활용

---

**도움이 필요하시면 언제든지 문의하세요!**
