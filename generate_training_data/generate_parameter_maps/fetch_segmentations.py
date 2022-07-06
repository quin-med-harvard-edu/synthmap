"""Fetches 10 adult brain segmentations from default location (provided by Onur) and processes them to fit into the SynthSeg codebase format. 

For example:
- reduces images to be from slice 100 to slice 164 (divisible by 32)
- cuts x and y axis to be divisible by 32 also 

"""

import glob 
import os

import numpy as np 
import nrrd 
import nibabel as nb 

import svtools as sv
path = '/fileserver/motion/onur/MWF/data/tissuesegmentation/'  # path to 10 segmentations of adult brain provided by Onur 
savedir = '/home/ch215616/w/mwf_data/synthetic_data/training_data/segmentation_targets/' # path to save the processed segmentations

subjects = glob.glob(path+'patient*')
seg_name = 'TissueSegmentation.nrrd'

for s in subjects:
    f = s+'/'+seg_name
    
    
    im, header = nrrd.read(f)
    x,y,z = im.shape
    
    xd = (x-160)//2
    yd = (y-192)//2
    zd = (z-192)//2
    
    #imo = nb.Nifti1Image(im[xd:x-xd,yd:y-yd,zd:z-zd],affine=np.eye(4))
    
    
    ydd = y-yd if yd%2==0 else y-yd-1 # some slices end up being 191 instead of 192 - this is a fix 
    imo = nb.Nifti1Image(im[xd:x-xd,yd:ydd,100:164],affine=np.eye(4)) # note that brain 2 will be cut off this way! 
    print(os.path.basename(f))
    imo.shape
    
    debug = False
    if debug:
        
        f = os.path.basename(f)
        sv.plot_slice(imo.get_fdata(),f"{f} middle")
        sv.plot_slice(imo.get_fdata()[0],f"{f} first")
        sv.plot_slice(imo.get_fdata()[-1],f"{f} last")
    
    
    patient_name = s.split('/')[-1]    
    savename = savedir+patient_name+'.nii.gz'
    nb.save(imo,savename)