import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

# Load the data
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Ensure the models directory exists
os.makedirs('models', exist_ok=True)

# Save the model to a file
model_path = 'models/linear_regression_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model trained and saved to {model_path}")
