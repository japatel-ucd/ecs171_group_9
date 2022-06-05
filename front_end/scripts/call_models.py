import torch
import torch.nn as nn
from scripts.ConvNets import GoogLe, Dense
from scripts.dataset import SpectrogramDataset
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
import os.path
import numpy

def inference(filename):
    filename = filename[:2] + ".jpg"

    google = GoogLe()
    dense = Dense([3, 3, 3, 3],
                k=32,
                theta=0.5,
                num_classes=1)

    exc_path = os.path.abspath(os.getcwd())
    #exc_path = ".."
    google.load_state_dict(torch.load(exc_path + "\\models\\google.pth",  map_location=torch.device('cpu')))
    google.eval()
    dense.load_state_dict(torch.load(exc_path + "\\models\\dense.pth",  map_location=torch.device('cpu')))
    dense.eval()

    google_resize = transforms.Compose([transforms.Resize((64,64)),   # resize spectrogram to 64×64
                                #transforms.Resize((224,224)), # resize spectrogram to 224×224
                                transforms.ToTensor(),        # NumPy array (H×W×C) → tensor (C×H×W)
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],   # normalize greyscale values
                                                    std =[0.229, 0.224, 0.225])])


    dense_resize = transforms.Compose([#transforms.Resize((64,64)),   # resize spectrogram to 64×64
                                transforms.Resize((224,224)), # resize spectrogram to 224×224
                                transforms.ToTensor(),        # NumPy array (H×W×C) → tensor (C×H×W)
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],   # normalize greyscale values
                                                    std =[0.229, 0.224, 0.225])])

    exp_label = 0
    if filename[:2] == "en": exp_label = 1
    img_path = exc_path + "\\spectrograms\\" + filename
    test_df = pd.DataFrame({'file' : img_path,
                            'label': [exp_label,]})
    
    
    image = Image.open(img_path)
    image_google = google_resize(image)
    image_google.unsqueeze_(0)  

    image_dense = dense_resize(image)
    image_dense.unsqueeze_(0)  
    
    print(test_df)
    
    with torch.no_grad():
        output_google = google(image_google)
    
    with torch.no_grad():
        output_dense = dense(image_dense)
    

    google_pred = (output_google > 0.5).long().item()
    google_raw = output_google.detach().numpy()[0].item()
    dense_pred = (output_dense > 0.5).long().item()
    dense_raw = (output_dense.detach().numpy()[0].item())

    return [google_pred, google_raw, (google_pred == exp_label)], [dense_pred, dense_raw, (dense_pred == exp_label)]
