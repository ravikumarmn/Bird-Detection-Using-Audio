import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchaudio
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torchvision.transforms import Normalize, Resize, ToPILImage
from tqdm import tqdm


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join("result", "confusion_matrix.png"))
    plt.close()


import torch
from tqdm import tqdm

def train_fn(model, train_loader, optimizer, loss_fn, device='cpu', l2_lambda=0.01):
    tqdm_train = tqdm(train_loader, total=len(train_loader), postfix="TRAIN")
    model.train()
    correct = 0
    train_loss = 0.0
    total = 0
    for batch in tqdm_train:
        batch_waveform = batch["spectrogram"].to(device)
        batch_labels = batch["label"].to(device)

        optimizer.zero_grad()

        outputs = model(batch_waveform)
        loss = loss_fn(outputs, batch_labels)
        l2_loss = model.l2_regularization_loss()  # Calculate L2 regularization loss
        total_loss = loss + l2_loss  # Add L2 regularization loss to the total loss
        tqdm_train.set_description(f"Loss: {total_loss.item():.3f}")  # Display the total loss
        total_loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        total += batch_labels.size(0)
        correct += predicted.eq(batch_labels).sum().item()

        train_loss += total_loss.item()

    train_loss /= len(train_loader)
    train_accuracy = 100.0 * correct / total

    return train_loss, train_accuracy


def eval_fn(model, val_loader, loss_fn, device='cpu'):
    tqdm_val = tqdm(val_loader, total=len(val_loader), postfix="VAL")
    model.eval()
    eval_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for batch in tqdm_val:
            batch_waveform = batch["spectrogram"].to(device)
            batch_labels = batch["label"].to(device)

            outputs = model(batch_waveform)
            loss = loss_fn(outputs, batch_labels)
            l2_loss = model.l2_regularization_loss()  # Calculate L2 regularization loss
            total_loss = loss + l2_loss  # Add L2 regularization loss to the total loss
            tqdm_val.set_description(f"Loss: {total_loss.item():.3f}")  # Display the total loss
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()

            eval_loss += total_loss.item()

    eval_loss /= len(val_loader)
    eval_accuracy = 100.0 * correct / total

    return eval_loss, eval_accuracy

def image_transform(image):
    transform = transforms.Compose([
    ToPILImage(),
    Resize((224, 224)),
    transforms.ToTensor(),
    # Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image)

def mel_spectrogram(waveform,sample_rate):
    spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_fft=1024,  # Size of the FFT window
            hop_length=512,  # Number of samples between successive frames
            n_mels=128  # Number of MEL bins
        )
    return spectrogram(waveform)

    
