#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf

import glob
# import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers, activations
import time
import nibabel as nib
from tqdm import tqdm
import gc

from scipy.ndimage import binary_erosion, zoom

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio

from IPython import display
import datetime
import cv2

from data_loader import load_data_dti_only
from sr_utility import space_to_depth, depth_to_space

import model
import data_utils
from Architectures import ESRGAN


def main():

    e_or_o = 1
    
    p_size_i = 5
    p_size_o = 3
    
    # generator = tf.keras.saving.load_model('Model/espcn{}x_baseline'.format(2*p_size_o + e_or_o))
    # generator = tf.keras.saving.load_model('Model/espcn{}x_no_t1'.format(2*p_size_o + e_or_o))

    # generator = tf.keras.saving.load_model('Model/espcn{}x_baseline_t1_start'.format(2*p_size_o + e_or_o))
    # generator = tf.keras.saving.load_model('Model/espcn{}x_baseline_t2_start'.format(2*p_size_o + e_or_o))
    
    # generator = tf.keras.saving.load_model('Model/espcn{}x_baseline_t1_end'.format(2*p_size_o + e_or_o))
    # generator = tf.keras.saving.load_model('Model/espcn{}x_baseline_t2_end'.format(2*p_size_o + e_or_o))

    # generator = tf.keras.saving.load_model('Model/espcn{}x_t1_end'.format(2*p_size_o + e_or_o))
    # generator = tf.keras.saving.load_model('Model/espcn{}x_t1_start'.format(2*p_size_o + e_or_o))
    
    generator = tf.keras.saving.load_model('Model/espcn{}x_GAN_t1_start'.format(2*p_size_o + e_or_o))
    # generator = tf.keras.saving.load_model('Model/espcn{}x_GAN_t2_start'.format(2*p_size_o + e_or_o))

    
    # generator = tf.keras.saving.load_model('Model/espcn{}x_t2_end'.format(2*p_size_o + e_or_o))
    # generator = tf.keras.saving.load_model('Model/espcn{}x_t2_start'.format(2*p_size_o + e_or_o))
    # generator = tf.keras.saving.load_model('Model/espcn{}x_GAN_t2_end'.format(2*p_size_o + e_or_o))
    
    subj = ['123117', '163129', '211720', '530635', '638049', '765056', '887373', '904044']
    
    
    subjects_lr, subjects_t1, subjects_hr, subjects_masks, [transform_lr, transform_hr] = load_data_dti_only("../HCPData/Test", subj, pads = [(8+16, 7+16), (9+8, 9+8), (8+16, 7+16), (0,0)])
    
    for subj in range(8):

        print("Subject ", subj)
    
        lowres_input = subjects_lr[subj]
        t1_input = subjects_t1[subj]
        hires_input = subjects_hr[subj]
        mask_input = subjects_masks[subj]
        
        
        s = 6
        
        lowres_input_ds = lowres_input[::2,::2,::2,:]
        
        (xsize, ysize, zsize, comp) = lowres_input_ds.shape

        
        result_image = np.zeros((xsize*2, ysize*2, zsize*2, comp))
        
        # result_image = np.zeros((xsize, ysize, zsize, comp))
        
        recon_indx = [(i, j, k) for k in np.arange(p_size_i+1,
                                                   zsize-p_size_i+1,
                                                   2*p_size_o+1)
                                for j in np.arange(p_size_i+1,
                                                   ysize-p_size_i+1,
                                                   2*p_size_o+1)
                                for i in np.arange(p_size_i+1,
                                                   xsize-p_size_i+1,
                                                   2*p_size_o+1)]
        
        for (i, j, k) in tqdm(recon_indx):
        
            if (i - p_size_i < 0 or j - p_size_i < 0 or k - p_size_i < 0):
                continue
        
            if (i + p_size_i >= xsize or j + p_size_i >= ysize or k + p_size_i >= zsize):
                continue
        
            if np.max(mask_input[::2,::2,::2][i - p_size_i : i + p_size_i,
                                              j - p_size_i : j + p_size_i,
                                              k - p_size_i : k + p_size_i]) != 1:
                continue
        
            # lr_patch_samp = np.zeros((2 * p_size_i + e_or_o, 2 * p_size_i + e_or_o, 2 * p_size_i + e_or_o, 70))
        
            lr_patch =  np.copy(lowres_input_ds)[
                                i - p_size_i : i + p_size_i + e_or_o,
                                j - p_size_i : j + p_size_i + e_or_o,
                                k - p_size_i : k + p_size_i + e_or_o,:]
        
            t1_patch =  np.copy(t1_input)[
                                2*(i - p_size_i - e_or_o) : 2*(i + p_size_i),
                                2*(j - p_size_i - e_or_o) : 2*(j + p_size_i),
                                2*(k - p_size_i - e_or_o) : 2*(k + p_size_i),:]
        
            # lr_patch_samp[...,:6] = lr_patch
        
            # lr_patch_samp[...,6:] = space_to_depth(t1_patch)
        
            # lr_patch = lr_patch_samp
            
        
            output_patch = generator([lr_patch[None,...], space_to_depth(t1_patch)[None,...]], training=False)

            # output_patch = generator(lr_patch_samp[None,...,:6], training=False)

            result_image[2*(i - p_size_o) : 2*(i + p_size_o + e_or_o),
                         2*(j - p_size_o) : 2*(j + p_size_o + e_or_o),
                         2*(k - p_size_o) : 2*(k + p_size_o + e_or_o), :] = output_patch[0,...]

        result_image[mask_input == 0] = 0
        
        
        lr_act = np.pad(lowres_input, ((0,0), (0,0), (0,0), (2,0)), mode='constant')
        hr_act = np.pad(hires_input, ((0,0), (0,0), (0,0), (2,0)), mode='constant')
        hr_pred = np.pad(result_image, ((0,0), (0,0), (0,0), (2,0)), mode='constant')

        mse_int = np.zeros(hires_input[...,0].shape)
        mse_gen = np.zeros(hires_input[...,0].shape)
        
        for ch in range(2,8):

            # mse_int.append(np.square(hires_input[...,ch-2] - zoom(lowres_input[::2,::2,::2,ch-2], 2)))
            # mse_gen.append(np.square(hires_input[...,ch-2] - result_image[...,ch-2]))

            mse_int = mse_int + (hires_input[...,ch-2] - zoom(lowres_input[::2,::2,::2,ch-2], 2))**2
            mse_gen = mse_gen + (hires_input[...,ch-2] - result_image[...,ch-2])**2
            
            
            lr_act[...,ch] = zoom(lowres_input[::2,::2,::2,ch-2], 2) * transform_lr[0][ch-2,1] + transform_lr[0][ch-2,0]
            lr_act[mask_input == 0] = 0
            hr_act[...,ch] = hires_input[...,ch-2] * transform_hr[0][ch-2,1] + transform_hr[0][ch-2,0]
            hr_act[mask_input == 0] = 0
            hr_pred[...,ch] = result_image[...,ch-2] * transform_lr[0][ch-2,1] + transform_lr[0][ch-2,0]
            hr_pred[mask_input == 0] = 0

        mse_int = np.sqrt(mse_int)
        mse_gen = np.sqrt(mse_gen)

        mse_int = np.median(mse_int[mask_input != 0])
        mse_gen = np.median(mse_gen[mask_input != 0])

        print("Interp RMSE", (np.array(mse_int)))
        print("Recon RMSE", (np.array(mse_gen)))
        
        # print("Interp median MSE", np.median(np.array(mse_int)))
        # print("Recon median MSE", np.median(np.array(mse_gen)))

        # print("Interp RMSE", np.sqrt(np.mean(np.array(mse_int))))
        # print("Recon RMSE", np.sqrt(np.mean(np.array(mse_gen))))
        
        [XSIZE, YSIZE, ZSIZE, dim] = hr_act.shape
        
        
        md_lr_i, fa_lr_i, cfa_lr_i = data_utils.calc_MD_FA_CFA(lr_act[::2,::2,::2,:], XSIZE//2, YSIZE//2, ZSIZE//2)
        md_lr, fa_lr, cfa_lr = data_utils.calc_MD_FA_CFA(lr_act, XSIZE, YSIZE, ZSIZE)
        md_hr, fa_hr, cfa_hr = data_utils.calc_MD_FA_CFA(hr_act, XSIZE, YSIZE, ZSIZE)
        md_gen, fa_gen, cfa_gen = data_utils.calc_MD_FA_CFA(hr_pred, XSIZE, YSIZE, ZSIZE)
        
        
        # In[52]:
        
        
        print("SSIM MD int: ", ssim(md_hr[:,:,:], md_lr[:,:,:], data_range=np.max(md_hr) - np.min(md_hr)))
        print("SSIM MD gen: ", ssim(md_hr[:,:,:], md_gen[:,:,:], data_range=np.max(md_hr) - np.min(md_hr)))
        
        print("SSIM FA int: ", ssim(fa_hr[:,:,:], fa_lr[:,:,:], data_range=np.max(fa_hr) - np.min(fa_hr)))
        print("SSIM FA gen: ", ssim(fa_hr[:,:,:], fa_gen[:,:,:], data_range=np.max(fa_hr) - np.min(fa_hr)))


if __name__ == "__main__":
    main()

# First Subject
# 
# MDint - 0.9898495633161474# MDgen - 
# 0.9857957730325824FAint - 
# 0.98636613513627FAgen - 1
# 0.9792560044103593

# In[ ]:




