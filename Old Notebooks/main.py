# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf
import model

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.python.keras import layers
from tensorflow.python.keras.layers import *
import time
import nibabel as nib
from tqdm import tqdm

from IPython import display

def main():

    iqt_ff = model.iqtModel(2, (145, 174, 145, 8), 2, 6, False)

    b0 = np.array(nib.load("Data/dt_all_2.nii").get_fdata())
    b0_l = np.array(nib.load("Data/dt_all_lowres_2_2.nii").get_fdata())

    phase = (1,0)

    y_pred, cost = iqt_ff.forwardpass(b0_l, b0, phase)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
