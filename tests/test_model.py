from src.model import load_data, train_model
from sklearn.metrics import accuracy_score

def test_train_model():
    X_train, X_test, y_train, y_test = load_data()
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    assert accuracy > 0.8  # Expecting accuracy to be above 80%

def test_save_model():
    import os
    from src.model import save_model, train_model, load_data
    model = train_model(*load_data()[:2])
    save_model(model, "models/test_model.pkl")
    assert os.path.exists("models/test_model.pkl")
