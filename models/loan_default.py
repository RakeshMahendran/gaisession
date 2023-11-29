import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
import joblib

data = pd.read_csv('/home/lenovo/Desktop/gaisession/data/loan_data_nov2023.csv')

# Separate features (X) and target variable (y)
X = data.drop('default', axis=1)
Y = data['default']

categorical_cols = ['grade', 'ownership']

preprocessor = ColumnTransformer(
    transformers=[
        ('onehot', OneHotEncoder(), categorical_cols)
    ],
    remainder='passthrough'
)

X = preprocessor.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, Y_train)


joblib.dump({'model': model, 'preprocessor': preprocessor}, '/home/lenovo/Desktop/gaisession/models/loan_default.joblib')




