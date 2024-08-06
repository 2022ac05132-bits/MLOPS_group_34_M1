from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def load_data():
    iris = load_iris()
    X = iris.data
    y = iris.target
    return train_test_split(X, y, test_size=0.3, random_state=42)

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def save_model(model, model_path):
    import joblib
    joblib.dump(model, model_path)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    save_model(model, "models/model.pkl")
