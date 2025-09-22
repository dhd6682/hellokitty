from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

# Load the augmented CSV file
file_path = 'Augmented_Cat_Personality_Traits_Data_Translated (1).csv'
cat_data = pd.read_csv(file_path)

# Encoding the 'Item' column which contains categorical data
label_encoder = LabelEncoder()
cat_data['Item'] = label_encoder.fit_transform(cat_data['Item'])

# Split the data into features and target variable
X = cat_data.drop(columns=['Item'])
y = cat_data['Item']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier to predict cat behavior
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and label encoder to disk
joblib.dump(model, 'cat_behavior_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and label encoder
model = joblib.load('cat_behavior_model.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Define request body model
class CatTraitsInput(BaseModel):
    신경증: float
    외향성: float
    지배성: float
    충동성: float
    우호성: float

# 고양이 행동을 예측하는 엔드포인트 정의
@app.post("/predict-behavior/")
async def predict_behavior(input_data: CatTraitsInput):
    # 예측을 위한 입력 준비
    example_input = np.array([[input_data.신경증, input_data.외향성, input_data.지배성, input_data.충동성, input_data.우호성]])
    # Make prediction
    predicted_behavior = label_encoder.inverse_transform(model.predict(example_input))
    
    return {"predicted_behavior": predicted_behavior[0]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
