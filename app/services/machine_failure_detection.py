import pickle
import numpy as np

def predict_failure(data_dict):
    # Define class index to name mapping
    class_names = [
        "No Failure",     
        "Failure"         
    ]

    # Load the trained model
    with open('models/MachineFailureDetectionmodel.pkl', 'rb') as f:
        model = pickle.load(f)

    # Predict
    prediction = model.predict(data_dict)
    predicted_class_name = class_names[prediction[0]]
    print("Predicted class:", predicted_class_name)

    return predicted_class_name
