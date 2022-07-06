
"""Generates brains using SynthSeg with custom parameters, that can be specified from the `set_options.py` file or on the fly. 
"""

import os
import numpy as np
import time
import shutil
import sys 
import argparse 
import glob

import nibabel as nb  
import cv2
from yaml2object import YAMLObject
import yaml 

import svtools as sv

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0" 


#SynthSeg
sys.path.append('/home/ch215616/w/code/mwf/ext/SynthSeg/') 
from ext.lab2im import utils
from SynthSeg.brain_generator import BrainGenerator    

# for viewing files
# from view import view_itksnap
from view import view_rview, view_itksnap, fslsplit, view_itksnap_split

# input options 
import set_options,set_options_yaml


def generate_from_single_label(datadir,opt):

    assert os.path.exists(datadir)

    # write results to the same directory 
    result_dir = datadir

    # get number of channels 
    prior_means = np.load(opt["prior_means"]) if isinstance(opt["prior_means"],str) else opt["prior_means"]        
    prior_channels = prior_means.shape[0]//2
    use_specific_stats_for_channel = True if prior_channels>1 else False 


    # instantiate BrainGenerator object
    brain_generator = BrainGenerator(labels_dir=opt['path_label_map'],
                                     extra_options=opt,    
                                     generation_labels=opt['generation_labels'],
                                     prior_distributions=opt['prior_distribution'],
                                     prior_means=opt['prior_means'],
                                     prior_stds=opt['prior_stds'],
                                     output_shape=opt['output_shape'],
                                     use_specific_stats_for_channel=use_specific_stats_for_channel,
                                     n_channels=prior_channels,
                                     apply_bias_field=opt['apply_bias_field'],
                                     apply_linear_trans=opt['apply_linear_trans'],
                                     apply_nonlin_trans=opt['apply_nonlin_trans'],
                                     bias_field_std=opt['bias_field_std'],
                                     bias_shape_factor=opt['bias_shape_factor'], 
                                     scaling_bounds=opt['scaling_bounds'],
                                     rotation_bounds=opt['rotation_bounds'],
                                     shearing_bounds=opt['shearing_bounds'],    
                                     nonlin_std=opt['nonlin_std'],
                                     nonlin_shape_factor=opt['nonlin_shape_factor'],
                                     blur_background=opt['blur_background'],
                                     blur_range=opt['blur_range'],                                                                                                          
                                     flipping=opt['flipping'])    
      
    # create result dir
    utils.mkdir(result_dir)

    for n in range(opt['n_examples']):

        # generate new image and corresponding labels
        start = time.time()
        im, lab = brain_generator.generate_brain()
        end = time.time()
        print('generation {0:d} took {1:.01f}s'.format(n, end - start))

        im = postprocess(im) # rescale volume fraction maps and remove zeros - by sv407 

        # save output image and label map
        utils.save_volume(np.squeeze(im), brain_generator.aff, brain_generator.header,
                    os.path.join(result_dir, 'params_%s.nii.gz' % n),dtype=np.float32,nii_dtype=np.float32) #sv407 - saving images as float32 (range 8d.p.)
        utils.save_volume(np.squeeze(lab), brain_generator.aff, brain_generator.header,
                    os.path.join(result_dir, 'params_labels_%s.nii.gz' % n), dtype=np.uint8, nii_dtype=np.uint8) #sv407 - saving labels as uint16 (range 0-255)
        

def load_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='if specified, parameters are read from yaml config yaml file')    
    parser.add_argument('--view', action='store_true', default=False,help='plot output in itksnap and in rview')
    parser.add_argument('--split_channels', action='store_true', default=False,help='view split files')
    parser.add_argument('--datadir','-d',type=str,default = '', help='path to a previously generated directory that also contains an options file (which may be read once again to re-run the experiment once again)')
    parser.add_argument('--debug',action='store_true')
    args = parser.parse_args()
    
    return args 

def cleanup(datadir):
    # find all nifti files that begin with 'sim_' 
    old_files = glob.glob(datadir+"sim_*.nii.gz")
    print(f"Removing the following files:\n{old_files}")

    # move all files 
    trash = '/home/ch215616/trash/'
    for f in old_files:
        _, f_name = os.path.split(f)
        shutil.move(f,trash+f_name)

    # give an option to return the files
    print(f"To undo, run the following:")
    cmd = ["mv"]
    old_files_moved = [trash+os.path.split(f)[1] for f in old_files]
    cmd.extend(old_files_moved)
    cmd.append(datadir)
    print(' '.join(cmd))


def postprocess(im): # rescale volume fraction maps and remove zeros - by sv407 

    im = np.squeeze(im)
    slices = im.shape[2]

    # filter of ones 
    kernel = np.ones((3,3),np.float32)/(3**2)

    # remove negatives through a mean neighbourhood filter for mwf and iewf 
    im[:,:,:,3] = remove_negatives(im[:,:,:,3],kernel,slices)
    im[:,:,:,4] = remove_negatives(im[:,:,:,4],kernel,slices)

    # old 

    # re-scale file estimates 
    #mult = 100  # multiplication factor used for myelin fraction 

    # remove negatives (these should not exist)
    #im[im<0] = 0 



    # rescaling is turned off - as nifti is not able to store this data properly - just divide by 100 when you have loaded the data from .nii in forward model
    # rescale volume fraction maps 
    #im[:,:,:,:,3] = np.divide(im[:,:,:,:,3],mult)
    #im[:,:,:,:,4] = np.divide(im[:,:,:,:,4],mult)

    # THIS DOES NOT work
    # fill in zero values with neighbouring pixels via closing 
    #im = dilation_closing.close_array(np.squeeze(im),closing_type='grey')


    return im

def remove_negatives(image,kernel,z):

    # convolve the slices with a kernel to find the mean of the local neighbourhood, then replace all negative values with the mean of the neighborhood
    means = np.zeros_like(image)
    for sl in range(0,z):
        means[:,:,sl] = cv2.filter2D(image[:,:,sl],-1,kernel)
    
    negatives = np.where(image<0)     
    image[negatives] = means[negatives]
    
    return image


if __name__ == '__main__':
    
    #load basic args
    args = load_args()
    datadir = args.datadir + '/' if not args.datadir.endswith('/') else args.datadir 


    # generate custom input options
    if args.config is not None:
        # Read options from .yaml file
        opt = sv.read_yaml_as_dict(args.config)
        opt['config'] = args.config
        opt,datadir = set_options_yaml.prepare_opt(opt)    
    else:
        # get options from python script 
        opt,datadir = set_options.generate(datadir)

    # clean up previously generated files 
    #cleanup(datadir)
    print('WARNING:not running `cleanup(datadir)`. Old files may be remaining in dir.')

    # run model     
    generate_from_single_label(datadir,opt)

    
    # show results in itksnap 
    multiseg = True if os.path.isdir(opt["path_label_map"]) else False # our data consists of multiple segmentation examples
    view_files_cmd = [] # write to file variable
    if args.view: 
        cmd = view_itksnap(datadir,remote=False, multiseg=multiseg)    
        view_files_cmd.append(cmd)
    else:
        cmd = view_itksnap(datadir,remote=True, multiseg=multiseg)
        view_files_cmd.append(cmd)

    # show results in rview
    cmd = view_rview(datadir,remote=True)
    view_files_cmd.append(cmd)


    # split 4D images into 3D*N_parameters and show separately in itksnap
    if args.split_channels: 
        fslsplit(datadir,prefix="sim_",suffix=".nii.gz")
        cmd = view_itksnap_split(datadir,remote=True)
        view_files_cmd.append(cmd) 

    with open(datadir+"view_files.txt", "w") as f: 
        out = []
        for cmd in view_files_cmd:
            out.append(' '.join(cmd)) # need to concat each sublist first
        f.writelines('\n'.join(out))
    



    
    
