import numpy as np
import pandas as pd
import regex as re
import joblib
import en_core_web_sm
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses an image for model prediction.

    Parameters:
    - image_path (str): Path to the input image file.
    - target_size (tuple): Target size for resizing the image.

    Returns:
    - numpy array: Preprocessed image as a numpy array.
    """
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def decode_predictions(predictions, top=3):
    """
    Decodes model predictions and returns top classes.

    Parameters:
    - predictions (numpy array): Model predictions.
    - top (int): Number of top predictions to return.

    Returns:
    - list: Top decoded predictions in the format (class, description, probability).
    """
    decoded_predictions = decode_predictions(predictions, top=top)[0]
    return [(label, description, round(prob, 4)) for (_, label, description, prob) in decoded_predictions]

def make_predictions(model, img_array):
    """
    Makes predictions using a pre-trained model.

    Parameters:
    - model: Loaded pre-trained model.
    - img_array: Preprocessed image array.

    Returns:
    - list: Top decoded predictions.
    """
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions)
    return decoded_predictions
