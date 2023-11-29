from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

app = FastAPI()

# Load the pre-trained models and preprocessors
model_path_default = '/home/lenovo/Desktop/gaisession/models/loan_default.joblib'
model_path_amount = '/home/lenovo/Desktop/gaisession/loan_amount.joblib'

loaded_data_default = joblib.load(model_path_default)
loaded_data_amount = joblib.load(model_path_amount)

# Extract the models and preprocessors
model_default = loaded_data_default['model']
preprocessor_default = loaded_data_default['preprocessor']

model_amount = loaded_data_amount['model']
preprocessor_amount = loaded_data_amount['preprocessor']

# Define Pydantic model for input validation
class LoanDataInput(BaseModel):
    amount: float 
    interest: float
    grade: str
    years: int
    ownership: str
    income: int
    age: int

# Apply one-hot encoding to a single input instance
def preprocess_input(data: LoanDataInput, preprocessor):
    input_data = pd.DataFrame([data.dict()])
    input_data_transformed = preprocessor.transform(input_data)
    return input_data_transformed

@app.post("/predict_default")
def predict_default(data: LoanDataInput):
    # Preprocess the input data
    input_data_transformed = preprocess_input(data, preprocessor_default)

    # Make predictions using the pre-trained model
    prediction = model_default.predict(input_data_transformed)

    # Return the prediction
    return {"prediction_default": float(prediction[0])}

@app.post("/predict_amount")
def predict_amount(data: LoanDataInput):
    # Preprocess the input data
    input_data_transformed = preprocess_input(data, preprocessor_amount)

    # Ensure the correct order of columns
    input_data_transformed = input_data_transformed[list(pd.get_dummies(df, columns=['grade', 'ownership']).columns)]

    # Standardize numeric features using the same scaler
    input_data_transformed[col[1:]] = scaler.transform(input_data_transformed[col[1:]])

    # Make predictions using the pre-trained model
    prediction = model_amount.predict(input_data_transformed)

    # Return the prediction
    return {"prediction_amount": float(prediction[0])}



