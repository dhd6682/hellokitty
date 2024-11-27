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

    # 참조 음성 파일 경로 설정
    reference_paths = {
        "나비": Path(r"C:\Projects\Python_basic\최종프로젝트\나비.wav"),
        "앉아": Path(r"C:\Projects\Python_basic\최종프로젝트\앉아.wav"),
        "이리와": Path(r"C:\Projects\Python_basic\최종프로젝트\이리와.wav")
    }

    # 명령어 인식
    command_2 = recognize_command(file2_path)

    # WAV 파일 로드 및 전처리
    wav_2 = preprocess_wav(file2_path)
    embed_2 = encoder.embed_utterance(wav_2)

    # 결과 초기화
    similarity = 0.0
    recognized_command = "없음"

    for command, reference_path in reference_paths.items():
        # 참조 음성 파일 전처리
        wav_1 = preprocess_wav(reference_path)
        embed_1 = encoder.embed_utterance(wav_1)

        # 두 벡터 간 코사인 유사도 계산
        similarity = np.dot(embed_1, embed_2)

        # 특정 명령어 확인
        if command in command_2:
            recognized_command = command
            break

    # 임시 파일 삭제
    if file2_path.exists():
        os.remove(file2_path)

    # 결과 반환
    result = {
        "similarity": float(similarity),
        "is_same_voice": bool(similarity >= 0.60),
        "command_match": recognized_command,
        "final_result": bool(similarity >= 0.60 and recognized_command != "없음")
    }
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
