"""

This function is used by `generate_B1_3D.py`. Copy the sole function used by generate_B1_3D into the file and archive this file. 


The purpose of this file is to fix the dilation caused by zeroing out the negative values of the parameter maps produced by SynthSeg code. In fact this would negatively affect the Unet behaviour too.


Use `close_array` to perform closing on numpy array and `close_nifti` to load and process nifti and save to correct filename. 

"""


import sys

from scipy.ndimage import binary_closing
from scipy.ndimage import grey_closing
import numpy as np
import nibabel as nb 


import svtools as sv



def close_array(im, closing_type):
    # perform closing on the entire image 
    
    imnew = np.zeros_like(im)
    
    if im.ndim == 4: 
        x,y,z,t = im.shape
        
        for echo in range(0,t):            
            for sl in range(0,z):
                imnew[:,:,sl,echo] = close_slice(im[:,:,sl,echo],closing_type)                
    elif im.ndim == 3: 
        x,y,z = im.shape
        
        for sl in range(0,z):
            imnew[:,:,sl] = close_slice(im[:,:,sl],closing_type)
    else:
        
        # special case for parameter maps 
        sys.exit('Image ndims is not 3 or 4. Please check image.')
    
    return imnew
    

def close_nifti(im_path, closing_type='grey'):
    
    # opens a nifti file and performs grey or binary closing operation on it and save it with new suffix to the same location
    
    imo = nb.load(im_path)
    im = imo.get_fdata()
    
    imnew = close_array(im,closing_type)
    
    # save image 
    imonew = nb.Nifti1Image(imnew,affine=imo.affine, header=imo.header)
    savename = im_path.replace('.nii.gz','_closed.nii.gz')
    nb.save(imonew,savename)
    
    print(f"Completed {closing_type} closing on:\n{savename}")

    
    
def close_slice(imslice,closing_type):

    if closing_type == 'grey':
        out = grey_closing(imslice,size=(2,2))
    elif closing_type == 'binary':
        out = binary_closing(imslice)
    return out 






def basic_closing_test():
    
    d = '/home/ch215616/code/SynthSeg/sv407_ismrm/test14_multiple_segmentations_targets/'

    im_path = d + 'params_1.nii.gz'
    B1_path = d + 'params_1_B1.nii.gz'


    imo = nb.load(im_path)
    im = imo.get_fdata()

    imob1 = nb.load(B1_path)
    imb1 = imob1.get_fdata()

    sv.plot_slice(imb1[:,:,50],'before closing')

    imb1_closed = binary_closing(imb1[:,:,50])
    sv.plot_slice(imb1_closed,'after closing - binary')
    
    imb1_closed_grey2 = grey_closing(imb1[:,:,50],size=(2,2))
    sv.plot_slice(imb1_closed_grey2,'after closing - grey - 2-2size')

    imb1_closed_grey3 = grey_closing(imb1[:,:,50],size=(3,3))
    sv.plot_slice(imb1_closed_grey3,'after closing - grey - 3-3size')


def get_neighbours(im_slice,x,y):
    
    
    # very dumb way to do this - do not use it 
    # find mean values of 3x3 neighbours of a given pixel 
    
    
    mean = np.mean(im_slice[x-1:x+2,y-1:y+2])
    
    return mean 
    
    
def fill_zeros_with_mean_filter():
    
    # build a mask around the brain 
    
    # inside the mask only, find the zero values. 
    
    # take the local 3x3 neighbourhood around the voxel and fill with mean values 
    
    f = '/home/ch215616/code/SynthSeg/sv407_ismrm/test19c_not_removing_negatives/params_0.nii.gz'
    import nibabel as nb 
    import numpy as np
    import cv2
    import svtools as sv
    
    mimo = nb.load(f)
    mim = mimo.get_fdata()

    
    
    #### PARTS TO PLUG INTO MY CODE 
    import numpy.ma as ma    
    import cv2
    
    
    x,y,z,param = mim.shape
    
    # pick mwf 
    mwf = mim[:,:,:,3]
    iewf = mim[:,:,:,4]

    # filter of ones 
    kernel = np.ones((3,3),np.float32)/(3**2)
    
    import time 
    def remove_negatives(image,kernel,z):

        # first, find local neighbours of each pixel by convolving with a 3x3 filters of ones. Then replace it 
        start = time.time()
        # convolve the slices with this filter 
        means = np.zeros_like(image)
        for sl in range(0,z):
            means[:,:,sl] = cv2.filter2D(image[:,:,sl],-1,kernel)
        end = time.time()
        
        print(f"{end-start:0.2f}seconds")
        
        # Now pick values that are negative and replace them with the means 
#         image_filtered = np.where(np.where(image<0),means,image)

        negatives = np.where(image<0) 
        
        image[negatives] = means[negatives]
        
#         ma.array(means,mask=np.where(image<0))
#         np.where(np.where(image<0), ma.array(means, mask=np.where(image<0)), image) 
        # option 1: 
        #np.where(np.where(a<0), ma.array(a, mask=np.where(a<0)).mean(axis=0), a) 
        # option 2: 
        # Complete= np.where(np.isnan(partial),replace,partial)
        # source: https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
        
        return image
    
    mwf_filtered = remove_negatives(mwf,kernel,z)
    iewf_filtered = remove_negatives(iewf,kernel,z)
    mim[:,:,:,3] = mwf_filtered
    mim[:,:,:,4] = iewf_filtered
    
    
    imnewo = nb.Nifti1Image(mim,affine=mimo.affine,header=mimo.header)
    f2 = f.replace('.nii.gz','_filtered.nii.gz')
    nb.save(imnewo,f2)
    
    sv.itksnap([f,f2])

    
    
    
    
def test_demonstrate_synthseg_MWF_IEWF_params_with_zeros():
    
    
    # CONCLUSION - only the fractions have values that are below zero INSIDE THE BRAIN (and that should be closed)
    # e.g. im[:,:,:,3]
    # e.g. im[:,:,:,4]
    
    # NO zeros here -> 
    # e.g. im[:,:,:,1]
    # e.g. im[:,:,:,2]
    
    """Test closing strategy on the parameter file that is being generated by SynthSeg. 
    
    Result should be - all holes filled, and zeros interpolated properly. 
    
    
    """


        
        

    # check_old_results_with_holes
    imdir = '/home/ch215616/code/SynthSeg/sv407_ismrm/test17_multiseg_fix_GM_WM_T2_reduce_MWF_variance/'
    check_holes(imdir)

    # check_old_results_with_holes
    imdir = '/home/ch215616/code/SynthSeg/sv407_ismrm/test18_check_morphological_closing/'        
    check_holes(imdir)
    

    # check_old_results_with_holes
    imdir = '/home/ch215616/code/SynthSeg/sv407_ismrm/test19b_confidence_of_095/'         # MARKED THE ZEROS HERE 
    check_holes(imdir, params=2)    

    # check_old_results_with_holes
    imdir = '/home/ch215616/code/SynthSeg/sv407_ismrm/test19_confidence_of_0995/'        
    check_holes(imdir,params=1)    
    
    
    # check_old_results_with_holes
    imdir = '/home/ch215616/code/SynthSeg/sv407_ismrm/test18_check_morphological_closing/'        
    check_holes_v2(imdir)
    
        
    def check_holes(imdir,parameter='mwf',params=0):
        
        if parameter == 'mwf': 
            index = 3 
        elif parameter == 'iewf':
            index = 4 
        elif parameter == 'mu':
            index = 1 
        
        # check if the generated and saved parameter maps contain values less than zero or zero INSIDE the brain

        import nibabel as nb 
        import numpy as np
        import os 
        f = imdir+'params_'+str(params)+'.nii.gz'
        dirname, filename = os.path.split(f)
        im = nb.load(f).get_fdata()

        imclosed = close_array(im,closing_type='grey')

        import svtools as sv

        # generate mask 
        im = im[:,:,:,index]
        mask = np.zeros_like(im)
        mask[im>0] = 1 


        # save all files 

        r= dirname+'/'+'test_'
        savename1 = r+'original'+'.nii.gz'
        imo = nb.Nifti1Image(im,affine = np.eye(4))
        nb.save(imo,savename1)

        savename2 = r+'closed'+'.nii.gz'
        imo = nb.Nifti1Image(imclosed,affine = np.eye(4))
        nb.save(imo,savename2)

        savename3 = r+'mask'+'.nii.gz'
        imo = nb.Nifti1Image(mask,affine = np.eye(4))
        nb.save(imo,savename3)


        sv.itksnap([savename1,savename2,savename3,savename4],remote=True)

        
        
    def check_holes_v2(imdir,parameter='mwf'):
        
        # this is a temporary hack that loads TWO files - before after closing 
        
        if parameter == 'mwf': 
            index = 3 
        elif parameter == 'iewf':
            index = 4 
        elif parameter == 'mu':
            index = 1 
        
        # check if the generated and saved parameter maps contain values less than zero or zero INSIDE the brain

        import nibabel as nb 
        import numpy as np
        import os 
        f = imdir+'params_0.nii.gz'
        dirname, filename = os.path.split(f)
        im = nb.load(f).get_fdata()
        
        f2 = imdir+'params_0_closed.nii.gz'
        im2 = nb.load(f).get_fdata()

        imclosed = close_array(im,closing_type='grey')
        imclosed2 = close_array(im2,closing_type='grey')

        import svtools as sv

        # generate mask 
        im = im[:,:,:,index]
        im2 = im2[:,:,:,index]
        mask = np.zeros_like(im)
        mask[im>0] = 1 
        mask2 = np.zeros_like(im2)
        mask2[im2>0] = 1 


        # save all files 

        r= dirname+'/'+'test_'
        savename1 = r+'original'+'.nii.gz'
        imo = nb.Nifti1Image(im,affine = np.eye(4))
        nb.save(imo,savename1)

        savename1a = r+'original'+'_synthsegclosed.nii.gz'
        imo = nb.Nifti1Image(im2,affine = np.eye(4))
        nb.save(imo,savename1a)        
        
        savename2 = r+'closed'+'.nii.gz'
        imo = nb.Nifti1Image(imclosed,affine = np.eye(4))
        nb.save(imo,savename2)

        savename2a = r+'closed'+'_synthsegclosed.nii.gz'
        imo = nb.Nifti1Image(imclosed2,affine = np.eye(4))
        nb.save(imo,savename2a)        
        
        savename3 = r+'mask'+'.nii.gz'
        imo = nb.Nifti1Image(mask,affine = np.eye(4))
        nb.save(imo,savename3)

        savename3a = r+'mask'+'_synthsegclosed.nii.gz'
        imo = nb.Nifti1Image(mask2,affine = np.eye(4))
        nb.save(imo,savename3a)        

        sv.itksnap([savename1,savename1a, savename2,savename2a,savename3,savename3a])
        