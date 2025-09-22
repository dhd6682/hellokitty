from fastapi import FastAPI, UploadFile, File
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import speech_recognition as sr
import shutil
import uvicorn
import os

# FastAPI 인스턴스 생성
app = FastAPI()

# 모델 로드
encoder = VoiceEncoder()

# 이미 지정된 파일 경로
file1_path = Path("audio/나비.wav")

# 참조 음성 파일 전처리
wav_1 = preprocess_wav(file1_path)
embed_1 = encoder.embed_utterance(wav_1)

# 음성 인식 설정
recognizer = sr.Recognizer()

# 명령어 인식을 위한 함수 정의
def recognize_command(file_path):
    with sr.AudioFile(str(file_path)) as source:
        audio = recognizer.record(source)
        try:
            # 음성을 텍스트로 변환
            text = recognizer.recognize_google(audio, language="ko-KR")
            print(f"인식된 명령어: {text}")
            return text
        except sr.UnknownValueError:
            print("명령어를 인식할 수 없습니다.")
            return ""
        except sr.RequestError as e:
            print(f"음성 인식 서비스 오류: {e}")
            return ""

@app.post("/compare-voice/")
async def compare_voice(file2: UploadFile = File(...)):
    # 파일 저장 경로 설정
    file2_path = Path(f"temp_{file2.filename}")

    # 파일 저장
    with file2_path.open("wb") as buffer:
        shutil.copyfileobj(file2.file, buffer)

    # WAV 파일 로드 및 전처리
    wav_2 = preprocess_wav(file2_path)

    # 음성 임베딩 생성
    embed_2 = encoder.embed_utterance(wav_2)

    # 두 벡터 간 코사인 유사도 계산
    similarity = np.dot(embed_1, embed_2)

    # 명령어 인식
    command_2 = recognize_command(file2_path)

    # 특정 명령어 확인
    target_command = "나비"
    command_match = target_command in command_2

    # 임시 파일 삭제
    if file2_path.exists():
        os.remove(file2_path)

    # 결과 반환
    result = {
        "similarity": float(similarity),
        "is_same_voice": bool(similarity >= 0.60),
        "command_match": command_match,
        "final_result": bool(similarity >= 0.60 and command_match)
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
