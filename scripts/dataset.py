import os

from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


resize = transforms.Compose([transforms.Resize((224,224)), # resize spectrogram to 224×224
                             transforms.ToTensor(),        # NumPy array (H×W×C) → tensor (C×H×W)
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],   # normalize greyscale values
                                                  std =[0.229, 0.224, 0.225])])  
    
class SpectrogramDataset(Dataset) :
    def __init__(self, pths, df, transform=resize):
        """
        custom PyTorch Dataset class to load & process image data and labels
        For each input data entry:
        1. load image data
        2. get the corresponding labels
        3. convert labels (NumPy array) into PyTorch tensor
        4. transform image data using transform

        input:
            pths     : paths of spectrograms
            df       : dataset (pandas DataFrame)
            transform: data preprocessing
        """
        self.pths = pths
        self.df = df
        self.transform = transform
        
    # provide size of dataset (number of spectrograms)
    def __len__(self) :
        return len(self.pths)
    
    # specify how to read i-th sample
    def __getitem__(self, idx) :
        # get path of i-th spectrogram
        pth = self.pths[idx]
        # read spectrogram
        img = Image.open(pth)
        # get labels of i-th entry
        filename = os.path.basename(pth)
        label = self.df[self.df['file'] == filename]
        # get only columns of labels
        label = label.iloc[0, 1:].astype('int').values
        # convert label to tensor
        label = torch.tensor(label).float()
        # preprocess image
        img = self.transform(img)
        
        # return input for current entry
        # Each of spectrograms and labels is now a PyTorch tensor.
        return img, label