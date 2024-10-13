
from flask import Flask, Response
from PIL import Image
import io
import numpy as np
from flask_cors import CORS
import onnxruntime as ort

app = Flask(__name__)

ort_session = ort.InferenceSession("generator.onnx")


allowed_origins = [
    "http://127.0.0.1:5500",
]

# Enable CORS with specific origins
CORS(app, resources={r"/generate": {"origins": allowed_origins}})




@app.route('/generate', methods=['GET'])
def predict():
    #noise = np.random.randn(1, 512).astype(np.float32)


    noise = np.random.randn(1, 512)


    noise = np.array2string(noise, formatter={'float_kind': lambda x: f'{x:.4e}'})
    img = ort_session.run(None, {"input": noise})


    img = np.squeeze(img)
    img = np.clip(img, 0, 1)

    image = (img * 255).astype(np.uint8)

    image = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(image)


    # Save image to an in-memory buffer
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return Response(buffer.getvalue(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
