# Hellokitty 프로젝트

이 프로젝트는 고양이의 성격을 분석하고, 음성 명령과 특정 목소리의 유사성을 비교하는 두 가지 주요 기능을 제공합니다. Python과 FastAPI를 사용하여 웹 API 형식으로 제공됩니다.

## 주요 기능

### 1. 고양이 행동 예측 (main3.py)
   - 고양이의 다양한 성격 특성을 입력 받아 행동을 예측하는 모델입니다.
   - FastAPI를 이용해 REST API 엔드포인트로 제공됩니다.
### 2. 음성 유사성 비교 및 명령어 인식 (main4.py)
   - 사용자의 음성을 비교하여 두 음성이 같은 사람의 것인지 판단하고, 특정 명령어를 인식하는 기능을 제공합니다.
   - 음성 명령을 사용해 고양이와 상호작용하는 시나리오에 활용할 수 있습니다.



## 주요 기능 설명

### 1. 고양이 행동 예측 모델 (main3.py)
이 기능은 고양이의 성격 데이터를 기반으로 행동을 예측하는 **랜덤 포레스트 분류기**를 사용합니다. 데이터는 `LabelEncoder`로 인코딩되며, 모델은 `train_test_split`을 통해 학습 및 테스트 데이터를 나눠 학습됩니다.

- **모델 학습**: `RandomForestClassifier`를 사용하여 고양이의 행동을 예측하는 모델을 학습시킵니다.
- **API 엔드포인트**: `/predict-behavior/`를 통해 사용자가 성격 특성을 입력하면 예측된 행동을 반환합니다.

### 2. 음성 유사성 비교 및 명령어 인식 (main4.py)
이 기능은 두 개의 음성을 비교하여 같은 사람인지 확인하고, 특정 명령어(`"나비"`와 같은)가 포함되어 있는지 확인합니다.

- **음성 임베딩**: `Resemblyzer`의 `VoiceEncoder`를 사용하여 음성 임베딩을 생성합니다.

![resemblyzer](https://github.com/user-attachments/assets/c45c5fa2-ff4b-4556-b596-29235567c407)
- **음성 인식**: `SpeechRecognition` 라이브러리를 사용하여 한국어 음성을 텍스트로 변환하고 명령어를 인식합니다.

- **유사성 계산**: 두 음성 간의 **코사인 유사도**를 계산하여 두 음성이 같은 사람인지 여부를 판단합니다.


## 설치 방법

### 요구 사항
- Python 3.8+
- FastAPI
- scikit-learn
- pandas, numpy
- joblib
- resemblyzer
- speechrecognition
- uvicorn


```bash
pip install fastapi scikit-learn pandas numpy joblib resemblyzer speechrecognition uvicorn
```

### 요청 값 예시
```
{
  "신경증": 0.5,
  "외향성": 0.7,
  "지배성": 0.3,
  "충동성": 0.6,
  "우호성": 0.8
}
```




