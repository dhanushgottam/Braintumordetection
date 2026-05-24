from flask import Flask, render_template, request
import os

from model.predict import predict_tumor

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():

    if 'image' not in request.files:
        return "No file uploaded"

    file = request.files['image']

    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(
        app.config['UPLOAD_FOLDER'],
        file.filename
    )

    file.save(filepath)

    # Predict tumor
    prediction, confidence = predict_tumor(filepath)

    return render_template(
        'result.html',
        image_path=filepath,
        prediction=prediction,
        confidence=confidence
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)