from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
model = pickle.load(open('model.pkl', 'rb'))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('upload.html', prediction="No file part")
    file = request.files['file']
    if file.filename == '':
        return render_template('upload.html', prediction="No selected file")
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(UPLOAD_FOLDER, filename))
        img_path = os.path.join(UPLOAD_FOLDER, filename)
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        return render_template('upload.html', prediction=predicted_class, img_path=img_path)
    else:
        return render_template('upload.html', prediction="Invalid file format")

if __name__ == "__main__":
    app.run(debug=True)
