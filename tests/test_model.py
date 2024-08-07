import pickle
import numpy as np

def test_model_prediction():
    """
    This method is triggered at test stage in CI-CD via pytest
    
    """
    with open('models/linear_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Adjust the sample data to match the features of the California housing dataset
    sample_data = np.array([8.3252, 41.0, 6.984126984126984, 1.0238095238095237, 322.0, 2.5555555555555554, 37.88, -122.23]).reshape(1, -1)
    prediction = model.predict(sample_data)

    assert prediction is not None
    print("Prediction:", prediction)
