import os
import os.path

import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib import figure
import numpy as np


def create_spectrogram(filename,name):
    print(filename)
    filename = filename.replace(":", "")
    print(os.path.abspath(os.getcwd()))
    img_path = os.path.abspath(os.getcwd()) + '\\audio\\'+filename
    print(img_path, os.path.exists(img_path))
    plt.interactive(False)
    clip, sample_rate = librosa.load(img_path, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    save_path = os.path.abspath(os.getcwd()) + '\\spectrograms\\' +filename[:-4]+'.jpg'
    plt.savefig(save_path, dpi=400, bbox_inches='tight',pad_inches=0)
    os.remove(img_path)
    del img_path,name,clip,sample_rate,fig,ax,S


