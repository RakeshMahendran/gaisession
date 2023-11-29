import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load the data from CSV
df = pd.read_csv('/home/lenovo/Desktop/gaisession/data/loan_data_nov2023.csv')

# Separate features (X) and target variable (y)
col = ['amount', 'interest', 'years', 'income', 'age']

# Apply one-hot encoding to categorical columns
df2 = pd.get_dummies(df, columns=['grade', 'ownership'])

# Ensure that the one-hot encoding columns match those from training
one_hot_cols_train = set(pd.get_dummies(df, columns=['grade', 'ownership']).columns)
one_hot_cols_input = set(df2.columns)

# Add missing columns to the input data
missing_cols = one_hot_cols_train - one_hot_cols_input
for col in missing_cols:
    df2[col] = 0

# Reorder columns to match the order during training
df2 = df2[list(pd.get_dummies(df, columns=['grade', 'ownership']).columns)]

# Separate features (X) and target variable (y)
Y = df2['amount']
X = df2.drop('amount', axis='columns')

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=50)

# Standardize the numeric features (excluding 'amount')
scaler = StandardScaler()
X_train[col[1:]] = scaler.fit_transform(X_train[col[1:]])
X_test[col[1:]] = scaler.transform(X_test[col[1:]])

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, Y_train)

# Save both the model and preprocessor using joblib
joblib.dump({'model': model, 'preprocessor': scaler}, 'loan_amount.joblib')


# Make predictions on the test set
Y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

