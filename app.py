from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
import numpy as np
import os

app = Flask(__name__)

# Load the pre-trained MobileNet model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

# Specify the input shape based on MobileNet expectations
input_shape = (224, 224, 3)

# Load the pre-trained MobileNet model without the top (classification) layers
Mobile_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# Build your custom model on top of the MobileNet base
x = Mobile_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(classes), activation='softmax')(x)

model1 = Model(inputs=Mobile_model.input, outputs=predictions)

# Load the pre-trained weights of your extended model
model1.load_weights('path/to/your/model_weights.h5')

# Make sure to compile the model after loading weights
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Path to the folder where user uploads will be stored
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=input_shape[:2])
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        # Save the uploaded file to the uploads folder
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Preprocess the uploaded image
        img_array = preprocess_image(file_path)

        # Make predictions
        predictions = model1.predict(img_array)

        # Decode predictions (for example, for ImageNet classes)
        decoded_predictions = decode_predictions(predictions)

        # Return the top predictions
        return jsonify({'predictions': decoded_predictions[0]})

if __name__ == '__main__':
    app.run(debug=True)
