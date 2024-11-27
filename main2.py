from fastapi import FastAPI, UploadFile, File
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
import uvicorn
import shutil

# FastAPI 인스턴스 생성
app = FastAPI()

# 모델 로드
encoder = VoiceEncoder()

# 이미 지정된 파일 경로
file1_path = Path(r"C:\Projects\Python_basic\최종프로젝트\냐옹.wav")

# 참조 음성 파일 전처리
wav_1 = preprocess_wav(file1_path)
embed_1 = encoder.embed_utterance(wav_1)

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

    # 임시 파일 삭제
    file2_path.unlink()

    # 결과 반환
    result = {
        "similarity": float(similarity),
        "is_same_voice": bool(similarity >= 0.60)
    }
    return result

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
