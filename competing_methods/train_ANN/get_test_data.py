import os 
import sys
import glob 

import numpy as np 
import matplotlib.pyplot as plt 
import nrrd 
import nibabel as nb 
from IPython import embed 

import svtools as sv


def load_test_data(name='default'):
    
    if name == 'default':
        return data_generator
    elif name == 'synthetic':
        return data_generator_synthetic
    else:
        sys.exit('the specified test data does not exist')
        

def data_generator(test_dir, threshold):
        
    #files = glob.glob(test_dir + "/*[0-9].nii.gz")
    files = glob.glob(test_dir + "/*.nii.gz")
    assert files, f"no .nii.gz files were found inside {test_dir}"
    
    for file in files:
        
        imo = nb.load(file)
        im = imo.get_fdata()

        # check what is the file 
        if im.ndim == 3 and im.shape[-1] == 32: 
            # insert one more dimension 
            im = np.expand_dims(im, axis=-2)
        
        S0 = im[:,:,:,0]    

        # create a mask to use over the images 
        mask = np.zeros_like(S0)
        mask[S0>threshold] = 1        
        # if not S0[S0>200].size == 0: # check that threshold of 200 yields non empty image (normalized images have S0 already )
        #     mask[S0>200] = 1
        # elif S0[S0==1].size > 1: 
        #     mask[S0==1] = 1
        # else:
        #     embed(header=sv.msg(f"Image appears to be empty {S0[S0==1]}\n{file}"))
        #assert np.nonzero(mask).size > 1

        # check that some values were found 
        if not np.nonzero(mask)[0].__len__()!=0:  
            #embed(header=sv.msg(f"Image appears to be empty. No values about the threshold of {threshold} were found in \n{file}"))
            print(f"Image appears to be empty. No values about the threshold of {threshold} were found in \n{file}. Returning small file")
            #from IPython import embed; embed()
            
            voxel = [1.        , 1.11839712, 0.97449911, 0.96357012, 0.79052824, 0.68123859, 0.60109288, 0.53734064, 0.44444445, 0.41347906, 0.35154828, 0.33515483, 0.30418944, 0.2276867 , 0.24590164, 0.21493624, 0.15300547, 0.15482695, 0.09653916, 0.13661203, 0.140255  , 0.09107468, 0.0856102 , 0.06193078, 0.10928962, 0.04735883, 0.03642987, 0.04007286, 0.06010929, 0.04735883,  0.00546448, 0.03825137]
            voxel = np.array(voxel)
            x,y,z,t=im.shape
            im_final = np.zeros_like(im)

            # small 30x30 patch 
            im_temp = np.tile(voxel, (30,30,z,t,1))[:,:,:,0,:]
            #from IPython import embed; embed()

            # set some values to zero 
            im_final[x//2-30:x//2, y//2-30:y//2,:, :] = im_temp
            im_m = np.moveaxis(im_final,-1,0)

        else:  			

            im_m = [np.multiply(mask,i) for i in np.moveaxis(im,-1,0)]
            im_m = np.array(im_m)

            im_m = normalize_image(im_m)
        
        yield im_m,file


def data_generator_synthetic(test_dir, threshold=0):
    
    # generator for processing synthetic data
        
    files = glob.glob(test_dir + "/*.nii.gz") # Different to 'data_generator'
    assert files, f"no .nii.gz files were found inside {test_dir}"
    
    for file in files:
        
        imo = nb.load(file)
        im = imo.get_fdata()
        S0 = im[:,:,:,0]    
        

        # create a mask to use over the images 
        mask = np.zeros_like(S0)
        mask[S0>0] = 1 # Different to 'data_generator'
        im_m = [np.multiply(mask,i) for i in np.moveaxis(im,-1,0)]
        im_m = np.array(im_m)

        im_m = normalize_image(im_m)
        
        yield im_m,file



# ------------
# Utils
# ------------        
    
def normalize_image(img, normalize_type="norm_by_voxel"): 
    """Normalize input image"""
    
    if normalize_type == "first_echo":            
        S0 = img[0,:,:,:]
        S0mean = np.mean(S0[S0 != 0])
        img = np.divide(img,S0mean)
    elif normalize_type == "0_1":            
        mmin = np.min(img)
        mmax = np.max(img)
        img = (img - mmin) / (mmax-mmin)
    elif normalize_type == "norm_by_voxel":            
        
        sh = img.shape
        img_v = np.reshape(img,(32,sh[1]*sh[2]*sh[3]))
        
        S0 = img_v[0,:]       
        img_v_norm = np.divide(img_v,np.expand_dims(S0,0),out=np.zeros_like(img_v), where=S0!=0)  #c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)            
  
        img = np.reshape(img_v_norm, sh)
        
    return img 
        
