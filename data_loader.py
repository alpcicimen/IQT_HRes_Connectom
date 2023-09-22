import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion, zoom
from data_utils import backward_shuffle_img
import matplotlib.pyplot as plt


def load_data(location, subjects, pads = [(8+16, 7+16), (9+24, 9+24), (8+16, 7+16), (0,0)]):

    subjects_lr = []
    subjects_t1 = []
    subjects_hr = []
    subjects_masks = []
    
    subjects_preproc_values_lr = []
    subjects_preproc_values_hr = []
    
    subjects_t1_preproc_values = []
    
    for subj in subjects:
    
        h4_imgs = []
        h4_imgs_lr = []
        dti_imgs = []
        dti_imgs_lr = []
    
        # DTI normalisation metrics pre-normalisation (storing max and min values in the mask) - Store for HR only
        dti_preprocs_hr = np.zeros((6,2))
        dti_preprocs_lr = np.zeros((6,2))
    
        print("Loading Subject {}".format(subj))
        
        for i in range(1,9):
        
            dti_data = np.array(nib.load("{}/{}/T1w/Diffusion/dt_all_{}.nii".format(location, subj, i)).get_fdata())
            dti_data_lr = np.array(nib.load("{}/{}/T1w/Diffusion/dt_all_lowres_2_{}.nii".format(location, subj, i)).get_fdata())
        
            # if i > 1:
            #     dti_data[dti_imgs[0] < 0] = 0
            #     dti_data_lr[dti_imgs[0] < 0] = 0
        
            dti_imgs.append(dti_data)
            dti_imgs_lr.append(dti_data_lr)
        
            dti_data = None
            dti_data_lr = None
        
        for i in range(1,25):
        
            h4_data = np.array(nib.load("{}/{}/T1w/Diffusion/h4_all_{}.nii".format(location, subj, str(i).zfill(2))).get_fdata())
            h4_data_lr = np.array(nib.load("{}/{}/T1w/Diffusion/h4_all_lowres_2_{}.nii".format(location, subj, str(i).zfill(2))).get_fdata())
        
            # if i > 1:
            #     h4_data[h4_data[0] < 0] = 0
            #     h4_data_lr[h4_data[0] < 0] = 0
            
            h4_imgs.append(h4_data)
            h4_imgs_lr.append(h4_data_lr)
        
            h4_data = None
            h4_data_lr = None
        
        h4_imgs = np.array(h4_imgs).transpose((1,2,3,0))
        h4_imgs_lr = np.array(h4_imgs_lr).transpose((1,2,3,0))
        
        dti_imgs = np.array(dti_imgs).transpose((1,2,3,0))
        dti_imgs_lr = np.array(dti_imgs_lr).transpose((1,2,3,0))
        
        mask = h4_imgs[...,0]
    
        binmask = np.zeros(mask.shape)
    
        binmask[mask >= 0] = 1
    
        mask = binary_erosion(binmask, structure=np.ones((3,3,3),np.uint8),iterations = 1)
    
        binmask = None
    
        # # Calculate DTI ranges and normalise to range [0,1]
        for i in range(2,8):

            dti_preprocs_lr[i-2,:] = np.array([np.mean(dti_imgs_lr[mask>0,i]), np.std(dti_imgs_lr[mask>0,i])])
            dti_imgs_lr[mask>0, i] = (dti_imgs_lr[mask>0, i] - dti_preprocs_lr[i-2,0]) / (dti_preprocs_lr[i-2,1])
            dti_imgs_lr[mask==0, i] = 0
    
            dti_preprocs_hr[i-2,:] = np.array([np.mean(dti_imgs[mask>0,i]), np.std(dti_imgs[mask>0,i])])
            dti_imgs[mask>0, i] = (dti_imgs[mask>0, i] - dti_preprocs_hr[i-2,0]) / (dti_preprocs_hr[i-2,1])
            dti_imgs[mask==0, i] = 0
    
    
        t1w = nib.load("{}/{}/T1w/Diffusion/t1w.nii".format(location, subj))

        # t1w = nib.load("{}/{}/T1w/Diffusion/t2w.nii".format(location, subj))
        
        t1w_data = np.array(t1w.get_fdata())

        t1w_data = np.flip(t1w_data, 0) # No Flipping for t2
        
        t1w_data_format = np.concatenate(
            (t1w_data[0::2,0::2,0::2, None], 
             t1w_data[1::2,0::2,0::2, None],
             t1w_data[0::2,1::2,0::2, None], 
             t1w_data[1::2,1::2,0::2, None],
             t1w_data[0::2,0::2,1::2, None], 
             t1w_data[1::2,0::2,1::2, None],
             t1w_data[0::2,1::2,1::2, None],
             t1w_data[1::2,1::2,1::2, None]),
            axis=-1)

        perc01 = np.percentile(t1w_data_format[mask>0], 0.1)
        perc99 = np.percentile(t1w_data_format[mask>0], 99.9)

        # print(perc01, perc99)
        
        # Clip array with different limits across the z dimension
        t1w_data_format = np.clip(t1w_data_format, a_min=perc01, a_max=perc99)

        t1w_preprocs = np.array([np.mean(t1w_data_format[mask>0,:]), np.std(t1w_data_format[mask>0,:])])
        
        subjects_t1_preproc_values.append(t1w_preprocs)    
        t1w_data_format[mask>0, :] = (t1w_data_format[mask>0, :] - t1w_preprocs[0]) / t1w_preprocs[1]
        t1w_data_format[mask==0, :] = 0
        
        # t1w_preprocs = np.array([np.min(t1w_data_format[mask>0,:]), np.max(t1w_data_format[mask>0, :])])
        
        # subjects_t1_preproc_values.append(t1w_preprocs)    
        # t1w_data_format[mask>0, :] = (t1w_data_format[mask>0, :] - t1w_preprocs[0]) / (t1w_preprocs[1] - t1w_preprocs[0])
        # t1w_data_format[mask==0, :] = 0
        
        subjects_preproc_values_lr.append(dti_preprocs_lr)
        subjects_preproc_values_hr.append(dti_preprocs_hr)
    
        dti_imgs_lr = np.pad(dti_imgs_lr, pads, mode='constant')
        t1w_data_format = np.pad(t1w_data_format, pads, mode='constant')
        dti_imgs = np.pad(dti_imgs, pads, mode='constant')
    
        mask = np.pad(mask, (pads[0], pads[1], pads[2]), mode='constant')
    
        lowres_input = np.copy(dti_imgs_lr[...,2:])
        t1_input = np.copy(t1w_data_format)
        hires_output = np.copy(dti_imgs[...,2:])
    
        # lowres_input = np.pad(lowres_input, ((8+8, 7+8), (9+8, 9+8), (8+8, 7+8), (0,0)), mode='constant')
        # t1_input = np.pad(t1_input, ((8+8, 7+8), (9+8, 9+8), (8+8, 7+8), (0,0)), mode='constant')
        # hires_output = np.pad(hires_output, ((8+8, 7+8), (9+8, 9+8), (8+8, 7+8), (0,0)), mode='constant')
        
        # lowres_input = np.concatenate((h4_imgs_lr[...,2:], dti_imgs_lr[...,2:], t1w_data_format), axis=-1)
        # hires_output = np.copy(h4_imgs[...,2:])
    
        subjects_lr.append(lowres_input)
        subjects_t1.append(t1_input)
        subjects_hr.append(hires_output)
        subjects_masks.append(mask)

    return subjects_lr, subjects_t1, subjects_hr, subjects_masks, [subjects_preproc_values_lr, subjects_preproc_values_hr]

#fromCardiff/FastFile-4AfUcztmNtzTwDXb/procd/04843

def load_data_cardiff(location, subjects, pads = [(8+16, 7+16), (9+24, 9+24), (8+16, 7+16), (0,0)]):

    subjects_lr = []
    subjects_t1 = []
    subjects_masks = []
    
    subjects_preproc_values_lr = []
    
    subjects_t1_preproc_values = []
    
    for subj in subjects:
        
        dti_imgs_lr = []

        dti_preprocs_lr = np.zeros((6,2))
    
        print("Loading Subject {}".format(subj))
        
        for i in range(1,9):
        
            dti_data_lr = np.array(nib.load("{}/{}/dti_output{}.nii".format(location, subj, i)).get_fdata())
        
            dti_imgs_lr.append(dti_data_lr)

            dti_data_lr = None
        
        dti_imgs_lr = np.array(dti_imgs_lr).transpose((1,2,3,0))

        mask = dti_imgs_lr[...,0]
            
        binmask = np.zeros(mask.shape)
    
        binmask[mask >= 0] = 1
    
        mask = binary_erosion(binmask, structure=np.ones((3,3,3),np.uint8),iterations = 1)
    
        binmask = None
    
        # # Calculate DTI ranges and normalise to range [0,1]
        for i in range(2,8):

            dti_preprocs_lr[i-2,:] = np.array([np.mean(dti_imgs_lr[mask>0,i]), np.std(dti_imgs_lr[mask>0,i])])
            dti_imgs_lr[mask>0, i] = (dti_imgs_lr[mask>0, i] - dti_preprocs_lr[i-2,0]) / (dti_preprocs_lr[i-2,1])
            dti_imgs_lr[mask==0, i] = 0

        mask = np.array(nib.load("{}/{}/dti_output1_resampled.nii.gz".format(location, subj, i)).get_fdata())

        plt.imshow(mask[...,120])
        plt.figure()

        print(np.max(mask), np.min(mask))
            
        binmask = np.zeros(mask.shape)
    
        binmask[mask >= 0] = 1

        plt.imshow(binmask[...,120])
    
        mask = binary_erosion(binmask, structure=np.ones((6,6,6),np.uint8),iterations = 1)
    
        binmask = None
        
        t1w = nib.load("{}/{}/t1w.nii".format(location, subj))

        t1w_data = np.array(t1w.get_fdata())
        
        t1w_data_format = np.concatenate(
            (t1w_data[0::2,0::2,0::2, None], 
             t1w_data[1::2,0::2,0::2, None],
             t1w_data[0::2,1::2,0::2, None], 
             t1w_data[1::2,1::2,0::2, None],
             t1w_data[0::2,0::2,1::2, None], 
             t1w_data[1::2,0::2,1::2, None],
             t1w_data[0::2,1::2,1::2, None],
             t1w_data[1::2,1::2,1::2, None]),
            axis=-1)

        # perc01 = np.percentile(t1w_data_format[zoom(mask, 2, mode='nearest')>0], 0.1)
        # perc99 = np.percentile(t1w_data_format[zoom(mask, 2, mode='nearest')>0], 99.9)
        
        # # Clip array with different limits across the z dimension
        # t1w_data_format = np.clip(t1w_data_format, a_min=perc01, a_max=perc99)

        t1w_preprocs = np.array([np.mean(t1w_data_format[mask>0,:]), np.std(t1w_data_format[mask>0,:])])
        
        subjects_t1_preproc_values.append(t1w_preprocs)    
        t1w_data_format[mask>0, :] = (t1w_data_format[mask>0, :] - t1w_preprocs[0]) / t1w_preprocs[1]
        t1w_data_format[mask==0, :] = 0
        
        subjects_preproc_values_lr.append(dti_preprocs_lr)
    
        dti_imgs_lr = np.pad(dti_imgs_lr, pads, mode='constant')
        t1w_data_format = np.pad(t1w_data_format, pads, mode='constant')
    
        mask = np.pad(mask, (pads[0], pads[1], pads[2]), mode='constant')
    
        lowres_input = np.copy(dti_imgs_lr[...,2:])
        t1_input = np.copy(t1w_data_format)
    
        subjects_lr.append(lowres_input)
        subjects_t1.append(t1_input)
        subjects_masks.append(mask)
        
    return subjects_lr, subjects_t1, subjects_masks, subjects_preproc_values_lr

def load_data_dti_only(location, subjects, pads = [(8+16, 7+16), (9+24, 9+24), (8+16, 7+16), (0,0)]):

    subjects_lr = []
    subjects_t1 = []
    subjects_hr = []
    subjects_masks = []
    
    subjects_preproc_values_lr = []
    subjects_preproc_values_hr = []
    
    subjects_t1_preproc_values = []
    
    for subj in subjects:
    
        dti_imgs = []
        dti_imgs_lr = []
    
        # DTI normalisation metrics pre-normalisation (storing max and min values in the mask) - Store for HR only
        dti_preprocs_hr = np.zeros((6,2))
        dti_preprocs_lr = np.zeros((6,2))
    
        print("Loading Subject {}".format(subj))
        
        for i in range(1,9):
        
            dti_data = np.array(nib.load("{}/{}/T1w/Diffusion/dt_all_{}.nii".format(location, subj, i)).get_fdata())
            dti_data_lr = np.array(nib.load("{}/{}/T1w/Diffusion/dt_all_lowres_2_{}.nii".format(location, subj, i)).get_fdata())
        
            dti_imgs.append(dti_data)
            dti_imgs_lr.append(dti_data_lr)
        
            dti_data = None
            dti_data_lr = None
        
        dti_imgs = np.array(dti_imgs).transpose((1,2,3,0))
        dti_imgs_lr = np.array(dti_imgs_lr).transpose((1,2,3,0))
        
        mask = dti_imgs_lr[...,0]
    
        binmask = np.zeros(mask.shape)
    
        binmask[mask >= 0] = 1
    
        mask = binary_erosion(binmask, structure=np.ones((3,3,3),np.uint8),iterations = 1)

        mask = np.copy(binmask)
        
        binmask = None
    
        # # Calculate DTI ranges and normalise to range [0,1]
        for i in range(2,8):

            dti_preprocs_lr[i-2,:] = np.array([np.mean(dti_imgs_lr[mask>0,i]), np.std(dti_imgs_lr[mask>0,i])])
            dti_imgs_lr[mask>0, i] = (dti_imgs_lr[mask>0, i] - dti_preprocs_lr[i-2,0]) / (dti_preprocs_lr[i-2,1])
            dti_imgs_lr[mask==0, i] = 0
    
            dti_preprocs_hr[i-2,:] = np.array([np.mean(dti_imgs[mask>0,i]), np.std(dti_imgs[mask>0,i])])
            dti_imgs[mask>0, i] = (dti_imgs[mask>0, i] - dti_preprocs_hr[i-2,0]) / (dti_preprocs_hr[i-2,1])
            dti_imgs[mask==0, i] = 0
    
    
        # t1w = nib.load("{}/{}/T1w/Diffusion/t1w.nii".format(location, subj))

        t1w = nib.load("{}/{}/T1w/Diffusion/t2w.nii".format(location, subj))
        
        t1w_data = np.array(t1w.get_fdata())

        t1w_data = np.flip(t1w_data, 0)
        
        t1w_data_format = np.concatenate(
            (t1w_data[0::2,0::2,0::2, None], 
             t1w_data[1::2,0::2,0::2, None],
             t1w_data[0::2,1::2,0::2, None], 
             t1w_data[1::2,1::2,0::2, None],
             t1w_data[0::2,0::2,1::2, None], 
             t1w_data[1::2,0::2,1::2, None],
             t1w_data[0::2,1::2,1::2, None],
             t1w_data[1::2,1::2,1::2, None]),
            axis=-1)

        perc01 = np.percentile(t1w_data_format[mask>0], 0.1)
        perc99 = np.percentile(t1w_data_format[mask>0], 99.9)
        
        # Clip array with different limits across the z dimension
        t1w_data_format = np.clip(t1w_data_format, a_min=perc01, a_max=perc99)

        t1w_preprocs = np.array([np.mean(t1w_data_format[mask>0]), np.std(t1w_data_format[mask>0])])
        
        subjects_t1_preproc_values.append(t1w_preprocs)    
        t1w_data_format[mask>0, :] = (t1w_data_format[mask>0, :] - t1w_preprocs[0]) / t1w_preprocs[1]
        t1w_data_format[mask==0, :] = 0
    
        subjects_preproc_values_lr.append(dti_preprocs_lr)
        subjects_preproc_values_hr.append(dti_preprocs_hr)
    
        dti_imgs_lr = np.pad(dti_imgs_lr, pads, mode='constant')
        t1w_data_format = np.pad(t1w_data_format, pads, mode='constant')
        dti_imgs = np.pad(dti_imgs, pads, mode='constant')
    
        mask = np.pad(mask, (pads[0], pads[1], pads[2]), mode='constant')
    
        lowres_input = np.copy(dti_imgs_lr[...,2:])
        t1_input = np.copy(t1w_data_format)
        hires_output = np.copy(dti_imgs[...,2:])
    
        subjects_lr.append(lowres_input)
        subjects_t1.append(t1_input)
        subjects_hr.append(hires_output)
        subjects_masks.append(mask)

    return subjects_lr, subjects_t1, subjects_hr, subjects_masks, [subjects_preproc_values_lr, subjects_preproc_values_hr]

def load_data_h4(location, subjects, pads = [(8+16, 7+16), (9+24, 9+24), (8+16, 7+16), (0,0)]):

    subjects_lr = []
    subjects_t1 = []
    subjects_hr = []
    subjects_masks = []
    
    subjects_preproc_values_lr = []
    subjects_preproc_values_hr = []
    
    subjects_t1_preproc_values = []
    
    for subj in subjects:
    
        h4_imgs = []
        h4_imgs_lr = []
        dti_imgs = []
        dti_imgs_lr = []
    
        # DTI normalisation metrics pre-normalisation (storing max and min values in the mask) - Store for HR only
        h4_preprocs_hr = np.zeros((22,2))
        h4_preprocs_lr = np.zeros((22,2))
    
        print("Loading Subject {}".format(subj))
        
        for i in range(1,9):
        
            dti_data = np.array(nib.load("{}/{}/T1w/Diffusion/dt_all_{}.nii".format(location, subj, i)).get_fdata())
            dti_data_lr = np.array(nib.load("{}/{}/T1w/Diffusion/dt_all_lowres_2_{}.nii".format(location, subj, i)).get_fdata())
        
            # if i > 1:
            #     dti_data[dti_imgs[0] < 0] = 0
            #     dti_data_lr[dti_imgs[0] < 0] = 0
        
            dti_imgs.append(dti_data)
            dti_imgs_lr.append(dti_data_lr)
        
            dti_data = None
            dti_data_lr = None
        
        for i in range(1,25):
        
            h4_data = np.array(nib.load("{}/{}/T1w/Diffusion/h4_all_{}.nii".format(location, subj, str(i).zfill(2))).get_fdata())
            h4_data_lr = np.array(nib.load("{}/{}/T1w/Diffusion/h4_all_lowres_2_{}.nii".format(location, subj, str(i).zfill(2))).get_fdata())
        
            # if i > 1:
            #     h4_data[h4_data[0] < 0] = 0
            #     h4_data_lr[h4_data[0] < 0] = 0
            
            h4_imgs.append(h4_data)
            h4_imgs_lr.append(h4_data_lr)
        
            h4_data = None
            h4_data_lr = None
        
        h4_imgs = np.array(h4_imgs).transpose((1,2,3,0))
        h4_imgs_lr = np.array(h4_imgs_lr).transpose((1,2,3,0))
        
        dti_imgs = np.array(dti_imgs).transpose((1,2,3,0))
        dti_imgs_lr = np.array(dti_imgs_lr).transpose((1,2,3,0))
        
        mask = h4_imgs[...,0]
    
        binmask = np.zeros(mask.shape)
    
        binmask[mask >= 0] = 1
    
        mask = binary_erosion(binmask, structure=np.ones((5,5,5),np.uint8),iterations = 1)
    
        binmask = None

        # # Calculate DTI ranges and normalise to range [0,1]
        for i in range(2,22):

            h4_preprocs_lr[i-2,:] = np.array([np.mean(h4_imgs_lr[mask>0,i]), np.std(h4_imgs_lr[mask>0,i])])
            h4_imgs_lr[mask>0, i] = (h4_imgs_lr[mask>0, i] - h4_preprocs_lr[i-2,0]) / (h4_preprocs_lr[i-2,1])
            h4_imgs_lr[mask==0, i] = 0
    
            h4_preprocs_hr[i-2,:] = np.array([np.mean(h4_imgs[mask>0,i]), np.std(h4_imgs[mask>0,i])])
            h4_imgs[mask>0, i] = (h4_imgs[mask>0, i] - h4_preprocs_hr[i-2,0]) / (h4_preprocs_hr[i-2,1])
            h4_imgs[mask==0, i] = 0
    
    
        t1w = nib.load("{}/{}/T1w/Diffusion/t1w.nii".format(location, subj))
        
        t1w_data = np.array(t1w.get_fdata())

        t1w_data = np.flip(t1w_data, 0)
        
        t1w_data_format = np.concatenate(
            (t1w_data[0::2,0::2,0::2, None], 
             t1w_data[1::2,0::2,0::2, None],
             t1w_data[0::2,1::2,0::2, None], 
             t1w_data[1::2,1::2,0::2, None],
             t1w_data[0::2,0::2,1::2, None], 
             t1w_data[1::2,0::2,1::2, None],
             t1w_data[0::2,1::2,1::2, None],
             t1w_data[1::2,1::2,1::2, None]),
            axis=-1)

        t1w_preprocs = np.array([np.mean(t1w_data_format[mask>0]), np.std(t1w_data_format[mask>0])])
        
        subjects_t1_preproc_values.append(t1w_preprocs)    
        t1w_data_format[mask>0, :] = (t1w_data_format[mask>0, :] - t1w_preprocs[0]) / t1w_preprocs[1]
        t1w_data_format[mask==0, :] = 0
    
        subjects_preproc_values_lr.append(h4_preprocs_lr)
        subjects_preproc_values_hr.append(h4_preprocs_hr)
    
        dti_imgs_lr = np.pad(dti_imgs_lr, pads, mode='constant')
        t1w_data_format = np.pad(t1w_data_format, pads, mode='constant')
        dti_imgs = np.pad(dti_imgs, pads, mode='constant')

        h4_imgs_lr = np.pad(h4_imgs_lr, pads, mode='constant')
        h4_imgs = np.pad(h4_imgs, pads, mode='constant')
    
        mask = np.pad(mask, (pads[0], pads[1], pads[2]), mode='constant')
    
        lowres_input = np.copy(h4_imgs_lr[...,2:])
        t1_input = np.copy(t1w_data_format)
        hires_output = np.copy(h4_imgs[...,2:])
    
        subjects_lr.append(lowres_input)
        subjects_t1.append(t1_input)
        subjects_hr.append(hires_output)
        subjects_masks.append(mask)

    return subjects_lr, subjects_t1, subjects_hr, subjects_masks, [subjects_preproc_values_lr, subjects_preproc_values_hr]