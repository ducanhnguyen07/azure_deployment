from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from PIL import Image
import io

# Tải mô hình VGG16 và đặt các lớp dự đoán
model = load_model("model.resnet50.h5")
output_class = ["battery", "glass", "metal", "organic", "paper", "plastic"]

# Khởi tạo Flask app
app = Flask(__name__)

def preprocess_image(img):
    img = img.resize((224, 224))  # Chuyển đổi kích thước
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)  # Dùng preprocess_input của VGG16
    return img

@app.route('/predict', methods=['POST'])
def predict():
    # Kiểm tra xem có file trong yêu cầu không
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']

    # Kiểm tra nếu không có ảnh được gửi
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
    
    try:
        # Đọc và xử lý ảnh
        img = Image.open(io.BytesIO(file.read()))
        preprocessed_img = preprocess_image(img)

        # Dự đoán
        prediction = model.predict(preprocessed_img)
        predicted_class = output_class[np.argmax(prediction)]
        predicted_accuracy = round(np.max(prediction) * 100, 2)

        return jsonify({
            "predicted_class": predicted_class,
            "accuracy": predicted_accuracy
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
