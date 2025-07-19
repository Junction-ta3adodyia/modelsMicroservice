import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

def predict_disease(img_path):
    # Define class index to name mapping
    class_names = [
        "Bacterial Red disease",
        "Bacterial diseases - Aeromoniasis",
        "Bacterial gill disease",
        "Fungal diseases Saprolegniasis",
        "Healthy Fish",
        "Parasitic diseases",
        "Viral diseases White tail disease"
    ]

    # Load the trained model
    model = load_model('models/FishDiseasesDetectionmodel.keras')

    # Example: Load and preprocess a test image
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create batch

    # Predict
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)
    predicted_class_name = class_names[predicted_class[0]]
    print("Predicted class:", predicted_class_name)

    return predicted_class_name
