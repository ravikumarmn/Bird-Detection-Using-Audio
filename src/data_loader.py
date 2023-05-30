import os

import torch
import torch.utils.data as data
import torchaudio
from sklearn.model_selection import train_test_split

import config
from utils import image_transform, mel_spectrogram


class CustomDataset(data.Dataset):
    def __init__(self, file_list, labels, transform=None,debug = False):
        self.debug = debug
        self.file_list = file_list
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_list[index]
        label = self.labels[index]
        waveform, sample_rate = torchaudio.load(file_path)
        
        spectrogram = mel_spectrogram(waveform,sample_rate)

        if self.transform is not None:
            spectrogram = self.transform(spectrogram)
        item = {
                "spectrogram" : spectrogram,
                "label" : label
            }
        return item

    def __len__(self):
        if self.debug:
            return  200
        else:
            return len(self.file_list)

dataset_dir = "dataset"
classes = sorted(os.listdir(dataset_dir))

file_list = []
labels = []
for i, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        files = os.listdir(class_dir)
        file_list.extend([os.path.join(class_dir, f) for f in files])
        labels.extend([i] * len(files))

# Split the data into train and test sets
train_files, test_files, train_labels, test_labels = train_test_split(file_list, labels, test_size=0.2, random_state=42,)#stratify=labels,

# Create the custom datasets
train_dataset = CustomDataset(train_files, train_labels, transform=image_transform,debug = config.DEBUG)
test_dataset = CustomDataset(test_files, test_labels, transform=image_transform,debug = config.DEBUG)

batch_size = config.BATCH_SIZE

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

