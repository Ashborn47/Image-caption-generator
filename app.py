from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
import os
import requests
from io import BytesIO

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from PIL import Image
import numpy as np
import pickle

app = Flask(__name__)

# Load the tokenizer and model
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('best_model.h5')
# vgg_model = VGG16(weights='imagenet', include_top=False, pooling='avg')

vgg_model = VGG16()
# restructure the model
vgg_model = Model(inputs=vgg_model.inputs, outputs=vgg_model.layers[-2].output)

max_length = 35  # Set the max length based on your model

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break
    return in_text

def generate_caption_and_display(image_path):
    # Load image
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = preprocess_input(image)
    feature = vgg_model.predict(image, verbose=0)
    #  # Reshape feature vector from (1, 512) to (1, 1, 512) to match expected shape (None, 1, 512)
    # feature = np.expand_dims(feature, axis=1)
    caption = predict_caption(model, feature, tokenizer, max_length)
    return caption


# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    image_url = None
    caption = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return 'No file part'
        
        file = request.files['image']
        
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_url = url_for('static', filename='uploads/' + filename)

        caption = generate_caption_and_display(filepath)
        # Remove 'startseq' and 'endseq' from the generated caption
        caption = caption.replace('startseq', '').replace('endseq', '').strip()
        print(caption)
    return render_template('upload.html', image_url=image_url , cap = caption)

@app.route('/link', methods=['GET', 'POST'])
def link():
    image_url = None
    caption = None
    if request.method == 'POST':
        if 'image_url' in request.form:
            image_url = request.form['image_url']
            try:
                # Download the image
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content)).convert('RGB')
                image = image.resize((224, 224))
                
                # Save the image locally
                filename = secure_filename(image_url.split("/")[-1])
                filepath = os.path.join('static', 'uploads', filename)
                image.save(filepath)
                image_url = url_for('static', filename='uploads/' + filename)

                # Process the image
                image = img_to_array(image)
                caption = generate_caption_and_display(filepath)
                # Remove 'startseq' and 'endseq' from the generated caption
                caption = caption.replace('startseq', '').replace('endseq', '').strip()


            except Exception as e:
                return f"Error processing image: {e}"
        else:
            return 'No image URL provided'
    return render_template('link.html', image_url=image_url, cap=caption)

    return render_template('link.html')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use the port provided by Render or default to 5000
    app.run(host='0.0.0.0', port=port)
