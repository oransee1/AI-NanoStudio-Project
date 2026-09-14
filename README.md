# NanoStudio-3D

> 스케치업(.skp) 기반 생성형 AI 3D 텍스처링 & 시네마틱 디렉팅 파이프라인

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-green.svg)](https://doc.qt.io/qtforpython-6/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 프로젝트 개요

NanoStudio-3D는 **무거운 상용 3D 소프트웨어(Blender, V-Ray 등) 없이** 경량화된 파이썬 파이프라인과 생성형 AI를 결합하여 건축 조감도/투시도 및 인디 게임용 3D 애셋 제작 장벽을 해소하는 데스크톱 애플리케이션입니다.

### 필수 기술 요소
| 기술 요소 | 적용 모듈 | 역할 |
|:---|:---|:---|
| **AI Agent** | Cinematic Director Agent | 3D 카메라 궤적 분석 → Google Flow/Veo 모션 프롬프트 자동 생성 |
| **멀티모달 AI** | Nano Banana (Gemini Image) | 오프스크린 캡처(이미지) + 프롬프트(텍스트) 결합 실사 렌더링 |
| **자동화 워크플로우** | End-to-End Pipeline | SKP 파싱 → UV 언랩 → 그림자 베이크 → AI 합성 → GLB/PDF 출력 |

## 🏗️ 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                  NanoStudio-3D Desktop Client                │
│   [PySide6 GUI]  ←→  Dual Viewport (CAD + Cinematic)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
     ┌─────────────────────┼─────────────────────┐
     ▼                     ▼                     ▼
[3D Geometry Core]   [Baking Engine]     [Generative AI]
• openskp (SKP)      • trimesh.ray       • Gemini 2.5 Flash
• trimesh            • OpenCV             • Nano Banana
• xatlas (UV)        • reportlab (PDF)    • Veo (Flow)
```

## 🚀 설치 및 실행

### 1. 환경 설정
```bash
git clone https://github.com/your-repo/AI-NanoStudio-Project.git
cd AI-NanoStudio-Project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. API 키 설정
```bash
# .env 파일 생성
echo GEMINI_API_KEY=your_api_key_here > .env
```

### 3. 메인 애플리케이션 실행
```bash
python main.py
```

### 4. 피드백 서버 실행 (선택)
```bash
streamlit run feedback_server.py
```

## 📂 프로젝트 구조

```
AI-NanoStudio-Project/
├── main.py                 # 메인 GUI 애플리케이션 (3930+ 라인)
├── viewport_widget.py      # 듀얼 뷰포트 위젯 (사분면 CAD + 시네마틱)
├── skp_loader.py           # openskp 기반 SKP 바이너리 파서
├── uv_unwrapper.py         # xatlas 자동 UV 언랩 & 아틀라스 패킹
├── shadow_baker.py         # trimesh 레이트레이싱 그림자/AO 베이커
├── genai_worker.py         # Gemini AI 비동기 워커 (텍스처 + PBR)
├── material_binder.py      # Multiply 블렌딩 & PBR 바인더
├── cinematic_director.py   # 시네마틱 디렉터 Agent (FR-08/09)
├── conti_pdf_builder.py    # 콘티 PDF 기획서 자동 생성기 (FR-10)
├── camera_manager.py       # 카메라 상태 관리자
├── glb_viewer.py           # GLB 파일 독립 뷰어
├── event_logger.py         # 전역 활동 로깅 시스템
├── feedback_server.py      # Streamlit 피드백 수집 웹 앱
├── test_pipeline.py        # E2E 파이프라인 테스트
├── requirements.txt        # 의존성 명세
├── .env                    # API 키 (git 제외)
├── 3d_model/               # 테스트용 SKP 파일
├── Output/                 # 렌더링 결과물
├── Export/                 # OBJ 내보내기
├── HDRI_환경맵/             # 환경맵 텍스처
├── TextureSample/          # PBR 텍스처 샘플
└── Documents/              # 기획 문서
```

## 🎮 주요 기능

### Part 1: 건축/인테리어 렌더링
- **PBR 실사 렌더링**: 4677×2631 초고해상도 시네마틱 스틸컷 출력
- **시네마틱 디렉터**: 카메라 A/B 시점 등록 → Gemini AI가 Veo 모션 프롬프트 자동 작성
- **콘티 PDF 일괄 추출**: First/Last Frame + 프롬프트 + 기획서 PDF 원클릭 생성

### Part 2: 게임 에셋 제작
- **xatlas 자동 UV 언랩**: 2048×2048 아틀라스 패킹
- **레이트레이싱 그림자 베이크**: trimesh.ray 기반 AO + 직사광 라이트맵
- **AI 텍스처링**: Gemini Image로 스타일 질감 생성 → Multiply 합성 → 자동 착용
- **GLB 원클릭 익스포트**: Unity/Unreal 즉시 호환 게임 에셋

## 📊 피드백 수집

실사용자 테스트(5인 이상) 피드백을 수집하기 위한 Streamlit 기반 웹 설문이 내장되어 있습니다:

```bash
streamlit run feedback_server.py
```

## 📄 라이선스

MIT License

---

> **AI 생성물 고지**: 본 프로젝트에서 생성되는 모든 텍스처 이미지 및 PDF 기획서에는 "Synthesized by Gemini / Nano Banana AI" 워터마크가 포함됩니다.
