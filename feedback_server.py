import streamlit as st
import json
import os
from datetime import datetime

# 디렉토리 설정
DATA_DIR = "feedback_data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def load_feedbacks():
    feedbacks = []
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
                feedbacks.append(json.load(f))
    return feedbacks

st.set_page_config(page_title="NanoStudio-3D 피드백", layout="wide")

st.title("NanoStudio-3D 사용자 피드백 설문")

# 사이드바 통계
feedbacks = load_feedbacks()
st.sidebar.header("피드백 통계")
st.sidebar.write(f"총 응답 수: {len(feedbacks)}")

if feedbacks:
    avg_score_1 = sum(f["score_skp"] for f in feedbacks) / len(feedbacks)
    avg_score_2 = sum(f["score_uv"] for f in feedbacks) / len(feedbacks)
    avg_score_3 = sum(f["score_ai"] for f in feedbacks) / len(feedbacks)
    avg_score_4 = sum(f["score_prompt"] for f in feedbacks) / len(feedbacks)
    
    st.sidebar.subheader("평균 점수")
    st.sidebar.write(f"SKP 처리: {avg_score_1:.2f} / 5")
    st.sidebar.write(f"UV/그림자: {avg_score_2:.2f} / 5")
    st.sidebar.write(f"AI 생성: {avg_score_3:.2f} / 5")
    st.sidebar.write(f"프롬프트: {avg_score_4:.2f} / 5")
    
    st.sidebar.subheader("최근 피드백")
    feedbacks.sort(key=lambda x: x["timestamp"], reverse=True)
    for fb in feedbacks[:5]:
        st.sidebar.text(f"{fb['timestamp']}\n{fb['comment'][:30]}...")

# 설문 폼
with st.form("feedback_form"):
    st.subheader("기능 평가 (1~5점)")
    
    score_skp = st.slider("SKP 파일 로드 및 계층 분할 정확도", 1, 5, 3)
    score_uv = st.slider("자동 UV 언랩 및 그림자 베이킹 품질", 1, 5, 3)
    score_ai = st.slider("AI 텍스처/조감도 생성 품질 만족도", 1, 5, 3)
    score_prompt = st.slider("Google Flow/Veo 카메라 모션 프롬프트 실효성", 1, 5, 3)
    
    st.subheader("자유 피드백")
    comment = st.text_area("개선점이나 버그, 기타 의견을 자유롭게 적어주세요.")
    
    st.subheader("결과물 이미지 (선택사항)")
    uploaded_file = st.file_uploader("관련 스크린샷이나 결과물을 업로드하세요", type=["png", "jpg", "jpeg"])
    
    submitted = st.form_submit_button("피드백 제출하기")

    if submitted:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        image_path = None
        if uploaded_file is not None:
            image_path = os.path.join(IMAGE_DIR, f"{timestamp}_{uploaded_file.name}")
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        feedback_data = {
            "timestamp": timestamp,
            "score_skp": score_skp,
            "score_uv": score_uv,
            "score_ai": score_ai,
            "score_prompt": score_prompt,
            "comment": comment,
            "image_path": image_path
        }
        
        json_path = os.path.join(DATA_DIR, f"feedback_{timestamp}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(feedback_data, f, ensure_ascii=False, indent=4)
            
        st.success("피드백이 성공적으로 제출되었습니다. 감사합니다!")
