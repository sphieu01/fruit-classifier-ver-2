from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import base64
from io import BytesIO

app = Flask(__name__)

# Load model đã train
model = load_model('fruit_classifier_model.h5')
class_names = ['apple', 'banana', 'orange']

def prepare_image(img):
    img = img.resize((100, 100))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)
    return img_array

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    label = None
    confidence = None
    img_data = None

    if request.method == 'POST':
        file = request.files.get('file')
        if file:
            img = Image.open(file.stream).convert('RGB')
            img_array = prepare_image(img)
            predictions = model.predict(img_array)
            pred_index = np.argmax(predictions)
            label = class_names[pred_index]
            confidence = f"{predictions[0][pred_index]:.2f}"

            # Chuyển ảnh sang base64 để hiển thị
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_data = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

    return render_template('index.html', label=label, confidence=confidence, img_data=img_data)

if __name__ == "__main__":
    app.run(debug=True)
