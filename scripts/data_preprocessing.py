import os
from glob import glob
import os.path
import numpy as np
import pandas as pd
import soundfile as sf
import scipy.signal as signal
import matplotlib.pyplot as plt
import gc
import IPython.display as ipd
import librosa
import librosa.display
import pylab
import matplotlib
from matplotlib import figure
from keras_preprocessing.image import ImageDataGenerator
import shutil
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


matplotlib.use('Agg')


def create_spectrogram(filename,name):
    name = name.replace('\\', '/')
    if os.path.exists('../images/' + name + '.jpg'): return
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename = '../images/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S

def create_spectrogram_test(filename,name):
    name = name.replace('\\', '/')
    if os.path.exists('../images/' + name + '.jpg'): return
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = '../images/' + name + '.jpg'
    fig.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


Data_dir=np.array(glob("../images/train/*"))


print("=== TRAIN IMAGES ===")
count = 0
for file in Data_dir:
    count = count + 1
    name = file.split('/')[-1]
    create_spectrogram(file,name)

    if count % 5000 == 0 or count == 1:
        print("TRAIN IMAGES:", count, sep='\t')
        gc.collect()
        

Data_dir=np.array(glob("../images/test/*"))


count = 0
print("=== TEST IMAGES ===")
for file in Data_dir:
    name = file.split('/')[-1]

    create_spectrogram_test(file,name)
    count = count + 1
    if count % 100 == 0 or count == 1:
        print("TEST IMAGE:", count, sep='\t')
        gc.collect()
        
gc.collect()


# for setting up the file structure necessary for PyTorch
def copy_images():
    image_path = '../images/'

    image_dir = np.array(glob("../images/train/*"))

    count = 0
    for file in image_dir:
        count = count + 1
        if count % 5000 == 0 or count == 1:
            print("TRAIN IMAGES MOVED:", count)
                
        new_file = file.split('\\')[-1]
        category, new_file = new_file.split('_', 1)
        category_path = image_path + 'processed/' + 'train/' + category
        if not os.path.exists(category_path):
            os.makedirs(category_path)
        new_file = category_path + '/' + new_file
        if os.path.exists(new_file): continue
        shutil.copyfile(file, new_file)


    image_dir = np.array(glob("../images/test/*"))

    count = 0
    for file in image_dir:
        count = count + 1
        if count % 100 == 0 or count == 1:
            print("TEST IMAGES MOVED:", count)
                
        new_file = file.split('\\')[-1]
        category, new_file = new_file.split('_', 1)
        category_path = image_path + 'processed/' + 'test/' + category
        if not os.path.exists(category_path):
            os.makedirs(category_path)
        new_file = category_path + '/' + new_file
        if os.path.exists(new_file): continue
        shutil.copyfile(file, new_file)


if not os.path.exists('../images/processed'):
    print("Copying images")
    copy_images()

