import torch
from torch import nn
from flask import Flask, Response
from PIL import Image
import io
import numpy as np
from flask_cors import CORS

app = Flask(__name__)

allowed_origins = [
    "http://127.0.0.1:5500",
]

# Enable CORS with specific origins
CORS(app, resources={r"/generate": {"origins": allowed_origins}})


class Genearator(nn.Module):

    def __init__(self):
        super().__init__()

        self.gen_sequence = nn.Sequential(

            nn.ConvTranspose2d(128, 512, 3, stride=3),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(512, 256, 5, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(256, 128, 4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),

            nn.ConvTranspose2d(128, 3, 4, stride=4),
            nn.Tanh()
        )


    def forward(self, x):
        x = x.view(-1, 128, 2, 2)  # [batch_size, channels, height, width]
        x = self.gen_sequence(x)
        return x
    
generator = Genearator()
generator.load_state_dict(torch.load('generator.pt', map_location=torch.device('cpu')))


@app.route('/generate', methods=['GET'])
def predict():
    noise = torch.randn(1, 512)
    img = generator(noise)  
    
    img = img.squeeze()
    img = img.clamp(0, 1)

    image = img.detach().cpu().numpy()
    image = (image * 255).astype(np.uint8)  # Scale to [0, 255]
    image = np.transpose(image, (1, 2, 0))
    image = Image.fromarray(image)


    # Save image to an in-memory buffer
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    return Response(buffer.getvalue(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
