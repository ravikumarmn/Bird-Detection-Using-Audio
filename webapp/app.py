from flask import Flask, render_template, request, url_for
import os
from src.model import BirdClassifier
from src import config
from src.utils import mel_spectrogram
import torchaudio
import torch
import matplotlib
matplotlib.use('Agg')
from torchvision.transforms import Normalize, Resize, ToPILImage
import torchvision.transforms as transforms


import torch.nn.functional as F


def image_transform(image):
    transform = transforms.Compose([
    ToPILImage(),
    Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image)

app = Flask(__name__)

params = {k:v for k,v in config.__dict__.items() if "__" not in k}

model = BirdClassifier(num_classes=params["NUM_CLASSES"]).to(params['DEVICE'])
model_state_dict = torch.load(params["SAVE_DIR_PATH"])
model.load_state_dict(model_state_dict)
model.eval()

result_dir = r"webapp/static/result"
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        music_file = request.files['music_file']
        if music_file:
            waveform, sample_rate = torchaudio.load(music_file)
            spectrogram = mel_spectrogram(waveform,sample_rate)
            if params["IMAGE_TRANSFORM"]:
                spectrogram = image_transform(spectrogram)
            import matplotlib.pyplot as plt
            
            mel_spectrogram_path = os.path.join("webapp/static/", "mel_spectrogram.png")
            mel_spectrogram_image = spectrogram.squeeze(0).cpu().numpy()  # Convert tensor to numpy array
            plt.figure(figsize=(6, 3))
            plt.imshow(mel_spectrogram_image, aspect='auto')
            plt.axis('off')
            plt.savefig(mel_spectrogram_path, dpi=80)
            plt.close()
            outputs = model(spectrogram.unsqueeze(0))
            probabilities = F.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            prediction = params["MAPPING"].get(predicted_labels.item())
            print("Predicted : ",prediction,predicted_labels)
            return render_template('index.html', prediction=prediction, mel_spectrogram="mel_spectrogram.png")
    return render_template('index.html')

@app.route('/result')
def result():
    result_images = []
    for filename in os.listdir(result_dir):
        if filename.endswith('.png'):
            result_images.append(url_for('static', filename=rf"result/{filename}"))

    
    return render_template('result.html', result_images=result_images)


if __name__ == '__main__':
    app.run(debug=True)