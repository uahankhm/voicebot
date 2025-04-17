import streamlit as st
from audiorecorder import audiorecorder
import numpy as np
import openai
import os
from datetime import datetime
from io import BytesIO
from gtts import gTTS
import base64

# STT: Whisper 모델로 오디오 -> 텍스트
def STT(audio, client):
    filename = "input.mp3"
    audio.export(filename, format="mp3")

    with open(filename, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )

    os.remove(filename)
    return transcript.text

# GPT 응답
def TTS(response):
    if st.session_state.get("tts_played"):
        return  # 이미 재생했으면 그냥 넘어감

    #gTTS를 활용하여 음성 파일 생성.
    filename = 'output.mp3'
    tts = gTTS(text=response, lang="ko")
    tts.save(filename)
    
    #음원 파일 자동 재생
    with open(filename, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio autoplay>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True,)
    #파일 삭제
    os.remove(filename) 
    # ✅ 재생한 걸 표시!
    st.session_state["tts_played"] = True
        
def ask_gpt(messages, model, client):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content

# 세션 초기화
def reset_settings():
    st.session_state["openai_api_key"] = ""
    st.session_state["gpt_model"] = "gpt-4-turbo"
    st.session_state["chat"] = []
    st.session_state["messages"] = [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korea"}]
    if "client" in st.session_state:
        del st.session_state["client"]

# 메인
def main():
    st.set_page_config(page_title="음성 비서 프로그램", layout="wide")

    flag_start = False

    # 세션 초기값 설정
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korea"}]

    if "check_audio" not in st.session_state:
        st.session_state["check_audio"] = []

    # 제목
    st.header("음성 비서 프로그램")
    st.markdown("---")

    # 설명
    with st.expander("음성비서 프로그램에 관하여", expanded=True):
        st.markdown(
            """
            - 음성 비서 프로그램의 UI는 Streamlit으로 구성되어 있습니다.<br>
            - STT(Speech To Text)는 OpenAI Whisper를 사용합니다.<br>
            - 답변은 GPT 모델을 통해 생성됩니다.<br>
            - TTS(Text-To-Speech)는 Google Translate TTS를 활용합니다.<br>
            """,
            unsafe_allow_html=True
        )

    # 사이드바
    with st.sidebar:
        st.header("🔐 OpenAI 설정")
        api_key_input = st.text_input("🔑 OpenAI API 키 입력", type="password")

        if api_key_input:
            st.session_state["openai_api_key"] = api_key_input
            if "client" not in st.session_state:
                st.session_state["client"] = openai.OpenAI(api_key=api_key_input)

        model = st.radio("🤖 GPT 모델", ["gpt-3.5-turbo", "gpt-4-turbo"], index=1)

        if st.button(label="초기화"):
            reset_settings()
            st.experimental_rerun()

    # 클라이언트 확인
    if "client" not in st.session_state:
        st.warning("❗ 먼저 OpenAI API 키를 입력해주세요.")
        st.stop()

    client = st.session_state["client"]

    # 본문
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎙️ 질문하기")
        audio = audiorecorder("클릭하여 녹음하기", "녹음 중")

        if len(audio) > 0 and not np.array_equal(audio, st.session_state["check_audio"]):
            # 오디오 재생
            buffer = BytesIO()
            audio.export(buffer, format="mp3")
            buffer.seek(0)
            st.audio(buffer)

            if audio.duration_seconds == 0:
                st.error("🚨 녹음된 오디오가 없습니다. 다시 시도해주세요.")
            else:
                question = STT(audio, client)
                now = datetime.now().strftime("%H:%M")
                st.session_state["chat"].append(("user", now, question))
                st.session_state["messages"].append({"role": "user", "content": question})
                st.session_state["check_audio"] = audio
                flag_start = True
            
    
    with col2:
        st.subheader("🤖 질문 / 답변")
        if flag_start:
            response = ask_gpt(st.session_state["messages"], model, client)
            now = datetime.now().strftime("%H:%M")
            st.session_state["messages"].append({"role": "system", "content": response})
            st.session_state["chat"].append(("bot", now, response))
            
            st.session_state["response"] = response
            st.session_state["tts_played"] = False
            # ✅ return 유도
            st.rerun()
            
            
        # ✅ TTS는 별도로 실행 (렌더링 후 1회)
        if "response" in st.session_state and not st.session_state.get("tts_played", False):
            TTS(st.session_state["response"])
            

        # 채팅 내용 출력
        st.markdown("### 💬 대화 내용")
        for role, time, msg in st.session_state["chat"]:
            icon = "👤" if role == "user" else "🤖"
            if role == "user":
                st.write(f'''
                         <div style = "display: flex; align-items: flex-end; gap: 8px; margin-bottom: 10px;">
                            <div style = "background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:9px;">
                            {icon}{msg}
                            </div>
                            <div style ="font-size:0.8rem;color:grey;">
                            {time}
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                st.write("")
            else:
                st.write(f'''
                        <div style = "display:flex; align-items:flex-end ;justify-content:flex-end;">
                            <div style ="font-size:0.8rem;color:grey;">
                            {time}
                            </div>
                            <div style = "background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:9px;">
                            {icon}{msg}
                            </div>
                        </div>''', unsafe_allow_html=True)
                st.write("")
                         
            # st.markdown(f"**{icon} [{time}]**: {msg}")

            #gTTS를 활용하여 음성 파일 생성 및 재생
        
if __name__ == "__main__":
    main()
