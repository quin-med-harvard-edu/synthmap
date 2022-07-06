### ADDITIONAL changes over v4 - ies mu and sigma are sampled randomly from uniform distribution (given by ranges)



### ADDITIONAL changes - by using 'anima_hybrid' model we can control the values of mu and sigma set to each parmaeter. 
# for now this is implemented only for ie fraction but can easily be extended. 

# conventionally IEWF set to:     
#    mu_ies_    = np.float64(100)
#    sigma_ies_ = np.sqrt(100)


### ADDITIONAL changes: 
# - input option 'anima_low_mwf_in_gm' which generates VERY small amount of myelin (next to zero) in GM regions... 

# docstring at the bottom (temporarily)

import glob 
import os 
import time  
import sys
import datetime
import argparse

import nibabel as nb 
import numpy as np
from IPython import embed

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import svtools as sv
from generate_B1_field import generate_B1_volume_3D_array


def sample_uniform(low,high):

    return np.random.uniform(low,high,1)[0]

def load_args(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir','-d',type=str,help='path to directory that contains the generated simulation files - with names `sim_...`')
    parser.add_argument('--slices',default=None, nargs='+', help='define a slice range over which signals will be calculated')
    parser.add_argument('--save_volume',action='store_true', help='whether to save all slices into a single volume')
    parser.add_argument('--save_distribution',action='store_true', help='whether to save fulle pT2 file')    
    parser.add_argument('--volume_range',default=None, type=int,nargs='+', help='if specified, we select a subset of volumes specified in the given range from the directory')   
    parser.add_argument('--model',choices=['anima', 'anima_tissues', 'default', 'anima_hybrid', 'anima_sample_mu_ies'],default='default', type=str, help='choose model to use for defining the component parameters. Default model uses gaussian model with priors that are extracted from the brain for mean and std of the components.')   
    parser.add_argument('--low_mwf_in_gm',action="store_true", help='set mwf values in gm to be 10x lower than in the priors')   
    parser.add_argument('--low_mwf_in_gm_and_csf',action="store_true", help='set mwf values in gm and csf to be 10x lower than in the priors')   
    parser.add_argument('--savedir',default='signals', type=str, help='name of savedir')   
    parser.add_argument('--load_B1',default=None, type=str, help='generate from specific B1')   
    parser.add_argument('--donotskip',action="store_true", help='do not skip existing files')   
    
    # define values for ies manually... 
    parser.add_argument('--mu_ies',type=float, default=None, help='define value for mu_ies')   
    parser.add_argument('--sigma_ies',type=float, default=None, help='define value for sigma_ies')   

    parser.add_argument('--mu_ies_range',type=float, nargs='+', default=None, help='define range for mu_ies to sample from (uniform)')   
    parser.add_argument('--sigma_ies_range',type=float,  nargs='+', default=None, help='define range for sigma_ies to sample from (uniform)')   

    #parser.add_argument('--savejson',action="store_true", help='save json file with generated parameters for each slice and volume')
    
    args = parser.parse_args()

    return args 

def get_T2():

    # get T2 range 
    t2rangetype='linspace'
    T2_samples = 2000
    T2range = np.linspace(10, 2000, num=T2_samples)  # T2 must be defined in the same range as pre-generated EPG dictionary        
    batch_size = 1
    T2range = np.tile(T2range,(batch_size,1))  

    return T2range    


def torch_2pi_squared():    
    
    # preallocated 2pi**2 as a torch tensor to save comp time
    pi = torch.FloatTensor([np.pi])
    sqrt_2pi = np.sqrt(2*np.pi)
    sqrt_2pi = torch.FloatTensor([sqrt_2pi])

    return sqrt_2pi


def get_gaussian_torch(T2range,mu,sigma,vf,sqrt_2pi):     
    
    # subtract mu from T2 range n
    res = torch.sub(T2range,mu)

    
    ### BELOW lines cover the following equation
        # gauss = np.exp(-0.5*res**2/sigma**2)/(sigma*np.sqrt(2*np.pi))    
        
    # denominator
    denominator = torch.mul(sigma,sqrt_2pi)
    if denominator.ndim == 1: 
        denominator = denominator.unsqueeze(1)
    if sigma.ndim == 1: 
        sigma = sigma.unsqueeze(1)
    if vf.ndim == 1: 
        vf = vf.unsqueeze(1)        
    
    # numerator
    res_sq = torch.pow(res,2)
    sigma_sq = torch.pow(sigma,2)
    res_sq_minus_half = torch.mul(torch.FloatTensor([-0.5]),res_sq )
    numerator = torch.div(res_sq_minus_half, sigma_sq)
    numerator = torch.exp(numerator)
    
    # combine numerator and denominator 
    gauss = torch.div(numerator, denominator)
    gauss = torch.mul(gauss,vf)
    
    return gauss 


def main(args):

    # init paths 
    test_name = args.savedir  + "/" #'signals'+'/
    datadir = args.datadir + '/' if not args.datadir.endswith('/') else args.datadir
    savedir = datadir+test_name

    # Flip angle related 
    EPG_dict = get_EPG_path()

    # get flip angle range based off the generated epg_dict 
    flip_angle_range = get_flip_angle_range(EPG_dict)    

    # init params
    batch_size = 40000
    background_threshold = 5 # threshold value at which the mask will be defined to calculate the values
    
    # define T2 range. Must be the same as pre-generated EPG dict 
    T2range_v_template = get_T2()



    # load generated file paths 
    param_file_paths = get_param_files(datadir, args.volume_range)

    # load a single image and extract shape 
    im_shape = nb.load(param_file_paths[0]).shape
    slice_shape = (im_shape[0],im_shape[1])
    num_slices = im_shape[2]

    
        
        
    # save key parameter to dict and then json 
    p = {"datadir":datadir, "savedir":savedir, \
         "param_file_paths":param_file_paths, \
         "background_threshold":background_threshold, \
         "batch_size":batch_size}


    if args.mu_ies_range is not None:
        mu_ies_list = []

    if args.sigma_ies_range is not None:
        sigma_ies_list = []

    # saving filenames 
    filenames = []        

    # save model 
    p["model"] = args.model
    sv.save_args(savedir,p)

        
    # process each file individually
    for file in param_file_paths:

        # load image 
        im_obj  = nb.load(file)
        im_volume = im_obj.get_fdata()
        if args.model == 'anima_tissues' or args.low_mwf_in_gm or args.low_mwf_in_gm_and_csf:
            seg_volume = nb.load(file.replace("params_", "params_labels_")).get_fdata()

            # initialize a new volume to save updated params to: 
            im_volume_low_mwf_in_gm = im_volume.copy()

            # get new filename
            os.makedirs(savedir,exist_ok=True)
            file_low_mwf_in_gm = savedir + "/" + os.path.basename(file)


        #generate B1 volume 
        B1vol = generate_B1_volume_3D_array(im_volume,var_sigma_slice2slice=False)
        B1vol = np.multiply(180.,B1vol)
        
        # torch 2pi
        sqrt_2pi = torch_2pi_squared()
        
        # save B1 volume 
        B1savename = file.replace('.nii.gz','_B1_3D.nii.gz')
        B1savename = savedir + os.path.basename(B1savename)
        header = im_obj.header.copy()
        header.set_data_shape(B1vol.shape)
        imnewo = nb.Nifti1Image(B1vol, affine = im_obj.affine, header=header)
        nb.save(imnewo,B1savename)

        if args.load_B1 is not None: 

            assert os.path.exists(args.load_B1)
            B1vol = nb.load(args.load_B1).get_fdata()
            assert B1vol.shape == im_obj.shape[0:3], embed(header=sv.msg("Check B1 shape"))
            #from IPython import embed; embed()

        

        basename = os.path.basename(file)
        basename = basename.replace('.nii.gz','_')
        basename = basename.replace('params_','signals_') if 'params_' in basename else basename.replace('sim_','signals_')  # second type is the old file convention

        # iterate over all the slices unless args.slices had been specified 
        if args.slices is not None:
            assert int(args.slices[1])<=num_slices
            assert int(args.slices[0])>=0  
            assert int(args.slices[1]) >= int(args.slices[0]), f"specified slice range is not correct"
            slices = [int(args.slices[0]),int(args.slices[1])]
            

        iter_range = range(0,num_slices) if args.slices is None else range(slices[0],slices[1])

        for jj,s in enumerate(iter_range):

            print(f"Processing slice {s} for {os.path.basename(file)}")
            #start_init_nonzeros = time.time()                    
            
            # Skipe slice if it already exists
            savename = savedir+basename+'s'+str(s)+'.nii.gz'
            if os.path.exists(savename) and not args.donotskip:
                continue

            # select slice 
            im = im_volume[:,:,s,:]
            flip_angles = B1vol[:,:,s]

            # estimate correct volume fractions 
            vf_myelin, vf_ies, vf_csf = update_volume_fractions_CSF(im)
            
            if args.low_mwf_in_gm or args.low_mwf_in_gm_and_csf:

                if args.low_mwf_in_gm:                 
                    low_mwf = 'gm_only'
                else:
                    low_mwf = 'gm_and_csf'

                vf_myelin, vf_ies, vf_csf = update_volume_fractions_low_mwf_in_gm(vf_myelin, vf_ies, vf_csf, seg_volume[:,:,s], low_mwf=low_mwf)                    


                # save updated fractions to new volume 
                im_volume_low_mwf_in_gm[:,:,s,3] = np.multiply(vf_myelin,100)
                im_volume_low_mwf_in_gm[:,:,s,4] = np.multiply(vf_ies,100)


            # load means 
            if args.model == 'default':
                mu_myelin = im[:,:,0]
                mu_ies    = im[:,:,1]
                mu_csf    = im[:,:,2]
            # load means (if anima model)
            elif args.model == 'anima':
                mu_myelin,mu_ies,mu_csf = load_means(slice_shape)
            elif args.model == 'anima_tissues':
                mu_myelin,mu_ies,mu_csf = load_means_anima_tissues(slice_shape, seg_volume[:,:,s])
            elif args.model == 'anima_sample_mu_ies':
                # load mu myelin and mu_csf as fixed 
                mu_myelin,_,mu_csf = load_means(slice_shape)
                # load mu_ies as defined by synthesis 
                mu_ies    = im[:,:,1]
            
            elif args.model == 'anima_hybrid':
                assert args.mu_ies is not None or args.mu_ies_range is not None 
                # select mu_ies value based on input 
                mu_myelin,mu_ies,mu_csf = load_means_hybrid(slice_shape, model=args.model, mu_ies=args.mu_ies, mu_ies_range=args.mu_ies_range)                
                if args.mu_ies_range is not None:
                    mu_ies_list.append(np.round(mu_ies[0,0],2))
            else:
                sys.exit('not implemented')


            if args.model == 'anima_hybrid':
                assert args.sigma_ies is not None or args.sigma_ies_range is not None
                # select sigma_ies value based on input 
                sigma_myelin,sigma_ies,sigma_csf = load_sigmas_hybrid(slice_shape, model=args.model, sigma_ies=args.sigma_ies, sigma_ies_range=args.sigma_ies_range)        

                if args.sigma_ies_range is not None:
                    sigma_ies_list.append(np.round(sigma_ies[0,0],2))

            else:
                # load sigmas
                sigma_myelin,sigma_ies,sigma_csf = load_sigmas(slice_shape, model=args.model)

                
            # mask all parameter images from background and unroll all into 1D vectors
            nonzeros    = np.where(im[:,:,2]>background_threshold)
            mu_myelin_v = mu_myelin[nonzeros]
            mu_ies_v    = mu_ies[nonzeros]
            mu_csf_v    = mu_csf[nonzeros]
            vf_myelin_v = vf_myelin[nonzeros]
            vf_ies_v    = vf_ies[nonzeros]
            vf_csf_v    = vf_csf[nonzeros]
            sigma_myelin_v = sigma_myelin[nonzeros]
            sigma_ies_v    = sigma_ies[nonzeros]
            sigma_csf_v    = sigma_csf[nonzeros]        
            flip_angles_v  = flip_angles[nonzeros]
            
            # get the indices of the matching flip angles in the EPG dictionary to the B1 field (i.e. `flip_angle` image)
            flip_angles_v_t = torch.Tensor(flip_angles_v).unsqueeze(1)
            indx = find_matching_flip_angles_torch(flip_angles_v_t,flip_angle_range)        
            
            # initialize additional variables
            L = mu_myelin_v.size
            
            pT2_full = np.zeros((L,T2range_v_template.shape[1]))
            signals_full = np.zeros((L,32))  
            iterable = range(0,L)    # for batch generator
            
            # convert to tensors 
            mu_myelin_v = torch.Tensor(mu_myelin_v).unsqueeze(1)
            mu_ies_v = torch.Tensor(mu_ies_v).unsqueeze(1)
            mu_csf_v = torch.Tensor(mu_csf_v).unsqueeze(1)
            sigma_myelin_v = torch.Tensor(sigma_myelin_v).unsqueeze(1)
            sigma_ies_v = torch.Tensor(sigma_ies_v).unsqueeze(1)
            sigma_csf_v = torch.Tensor(sigma_csf_v).unsqueeze(1)
            vf_myelin_v = torch.Tensor(vf_myelin_v).unsqueeze(1)
            vf_ies_v = torch.Tensor(vf_ies_v).unsqueeze(1)
            vf_csf_v = torch.Tensor(vf_csf_v).unsqueeze(1)            
            
            # Estimate pT2 and S(TE) on a batch by batch basis (due to very large memory constraints)        
            for i,(x1,x2) in enumerate(get_batch2(iterable, batch_size)): # yields chunks of batch_size, or smaller (for last batch)


                # update the correct T2range size for last batch
                if x2-x1 <batch_size:
                    T2range_v = np.tile(T2range_v_template,(x2-x1,1))    
                else:
                    T2range_v = np.tile(T2range_v_template,(batch_size,1))        
                T2range_v_torch = torch.FloatTensor(T2range_v)                       
                
                # estimate individual gaussians
                gauss_myelin = get_gaussian_torch(T2range_v_torch,mu_myelin_v[x1:x2],sigma_myelin_v[x1:x2],vf_myelin_v[x1:x2],sqrt_2pi)
                gauss_ies    = get_gaussian_torch(T2range_v_torch,mu_ies_v[x1:x2],sigma_ies_v[x1:x2],vf_ies_v[x1:x2],sqrt_2pi)
                gauss_csf    = get_gaussian_torch(T2range_v_torch,mu_csf_v[x1:x2],sigma_csf_v[x1:x2],vf_csf_v[x1:x2],sqrt_2pi)

                # get T2 density function 
                pT2 = gauss_myelin+gauss_ies+gauss_csf   # for shape (176,220,2000) this single slice array is 590Mb in size 

                # # match EPG dictionary    
                indx_batch = indx[x1:x2]                            
                EPG_batch=EPG_dict[:,indx,:]
                
                # # estimate signal
                signals = estimate_echoes_torch(pT2,EPG_batch)

                # add batch to full array size
                pT2_full[x1:x2,:] = pT2.numpy()
                signals_full[x1:x2,:] = signals.numpy()
                                
            # reshape back into original image size
            echoes = np.zeros((slice_shape[0],slice_shape[1],32))
            echoes_i = np.zeros(slice_shape)  # used for filling in nonzero values 
            for i,echo in enumerate(np.moveaxis(signals_full,-1,0)):
                echoes_i[nonzeros] = echo
                echoes[:,:,i] = echoes_i
                
            if args.save_distribution: # do this only for the first 5 slices
                distributions = np.zeros((slice_shape[0],slice_shape[1],T2range_v.shape[0]))
                distributions_i = np.zeros(slice_shape)
                
            if args.save_distribution:
                for i,dist in enumerate(np.moveaxis(pT2_full,-1,0)):
                    distributions_i[nonzeros] = dist

                    distributions[:,:,i] = distributions_i

            if args.save_distribution:
                # save pT2 
                savename = savedir+basename+'pT2_s'+str(s)+'.nii.gz'
                save_nii(distributions,savename,im_obj)
                del distributions

                

            # save echoes 
            savename = savedir+basename+'s'+str(s)+'.nii.gz'
            save_nii(echoes,savename, im_obj)        

            filenames.append(os.path.basename(savename))
            
            del pT2_full 
            del signals_full

        # save updated volume 
        if args.low_mwf_in_gm or args.low_mwf_in_gm_and_csf:
            im_obj_low_mwf_in_gm=nb.Nifti1Image(im_volume_low_mwf_in_gm, affine=im_obj.affine, header=im_obj.header)
            nb.save(im_obj_low_mwf_in_gm,file_low_mwf_in_gm)

        # concat signal slices into 1 volume (only those that are provided)
        concat_slices_into_volume2(savedir, basename,im_obj, im_shape,args.save_volume)
        
        print('Results saved to:')
        print(savedir)






    # add parameters to save 
    if args.sigma_ies is not None:
        p["sigma_ies_actual"]=args.sigma_ies
    elif args.sigma_ies_range is not None: 
        p["sigma_ies_range_low"]=args.sigma_ies_range[0]
        p["sigma_ies_range_high"]=args.sigma_ies_range[1]
        p["sigma_ies_range_actual"]=sigma_ies_list
        p["filenames"]=filenames
    else: 
        if args.model == 'default':
            p["sigma_ies_actual"]=5
        else:
            p["sigma_ies_actual"]=10



    if args.mu_ies is not None:
        p["sigma_ies_actual"]=args.mu_ies
    elif args.mu_ies_range is not None: 
        p["mu_ies_range_low"]=args.mu_ies_range[0]
        p["mu_ies_range_high"]=args.mu_ies_range[1]
        p["mu_ies_range_actual"]=mu_ies_list
        p["filenames"]=filenames
    else:
        if model == 'anima':
            p["mu_ies_actual"]=100
        else:
            p["mu_ies_actual"]="varied_through_image"
    sv.save_args(savedir,p)
        
        

                    


########################################################################################## FIX THESE FUNCS FOR FULL MODEL


        
def get_flip_angle_range(epgdict):
    # generate the flip angle range based off epg_dict that is being loaded into the model

    if epgdict.shape[1] == 1000: 
        # assume that epg dict was generated with 1000 points between 100-300 degrees 

        flip_angle_samples = 1000
        flip_angle_range = np.linspace(100,300,flip_angle_samples)
        
    elif epgdict.shape[1] == 320: 
        
        flip_angle_samples = 320
        flip_angle_range = np.linspace(120,280,flip_angle_samples)
    
    # expand dims of the flip angles used in EPG dict simulation 
    flip_angle_range_exp = np.expand_dims(flip_angle_range,axis=0)  # (N,) -> (N,1)
    flip_angle_range_exp = torch.FloatTensor(flip_angle_range_exp)
    
    return flip_angle_range_exp
        
def get_EPG_path():
    
    # CORRECTED EPG generating function 
    sample7 = "/home/ch215616/w/mwf_data/synthetic_data/training_data/flip_angle_dictionary/s20210805_v2/20210805-simulated_dictionary2000of2000.npy"    

    epg_dir = sample7
    print(f"PATH TO EPG_DICT -> {epg_dir}")
    
    EPG_dict = np.load(epg_dir)
    
    EPG_dict = torch.FloatTensor(EPG_dict)
    
    
    return EPG_dict



def find_matching_flip_angles_torch(fa,flip_angle_range):
    
    # for every value in the `flip_angles` image (aka B1 field), we find a matching closest flip angle in the simulated EPG dictionary 
    # this function returns the indices of where to look in the vector that defines the simulated flip angles for EPG dictionary 
    # the indices have the shape of the `flip_angles` image. 

    # create N copies of the flip angle vector, where N is the range of flip angles used in EPG dict simulation     
    flip_angles_1D_N = torch_tile_temp(fa,-1,flip_angle_range.shape[-1])
    
    # find absolute difference between the `measured` flip angles and simulated flip angles
    diff = torch.abs(torch.sub(flip_angles_1D_N,flip_angle_range))

    # get indices of the min values of the diff vector -> these are the indices of the flip_angle_range that we should use     
    ind = torch.argmin(diff, axis=-1)# shape = (L,)


    return ind


def torch_tile_temp(a, dim, n_tile):
    
    # This function is taken from StackOverflow as temporary replacement of torch.tile function which isn't available in torch v1.4 (but is in v1.9)
    
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)


def get_batch(iterable, n=1):
    # allows us to batch through data even if the last chunk is not equal to others 
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]



def concat_slices_into_volume2(savedir, basename,im_obj,im_shape,save_volume):

    if save_volume:

        # find slices that were generated and sort them in correct order 
        saved_slices_unsorted = glob.glob(savedir+basename+'s'+'[0-9]*.nii.gz')
        saved_slices_unsorted = [s.replace(savedir+basename+'s', '') for s in saved_slices_unsorted]
        saved_slices_unsorted = [s.replace('.nii.gz', '') for s in saved_slices_unsorted]
        indices = sorted([int(x) for x in saved_slices_unsorted])

        savevol = np.zeros((im_shape[0],im_shape[1],len(indices),32))

        for i,sl in enumerate(indices):
            f = savedir+basename+'s'+str(sl)+'.nii.gz'
            savevol[:,:,i,:] = nb.load(f).get_fdata()
        savename = savedir+basename+'.nii.gz'
        if not os.path.exists(savename):
            save_nii(savevol, savename, im_obj)

def save_nii(array,savename,im_obj,nii_dtype=None):
    # save images with base orientation

    # save all in default
    header = im_obj.header.copy()
    header.set_data_shape(array.shape)
    if nii_dtype is not None:
        header.set_data_dtype(nii_dtype)
    img = nb.Nifti1Image(array,affine=im_obj.affine, header=header)
    nb.save(img,savename)
    
    

def load_sigmas(im_shape, model='default'):
    
    if model=='default':
        sigma_myelin_ = np.array(5)
        sigma_ies_ = np.array(5)
        sigma_csf_ = np.array(5)
    elif model == 'anima' or model == 'anima_tissues' or model == 'anima_sample_mu_ies': 
        # based on Chatterjee 2018 - Multi-compartment model of brain tissues from T2 relaxometry MRI using gamma distribution
        # https://hal.archives-ouvertes.fr/hal-01744852/document

        # load sigmas 
        sigma_myelin_ = np.sqrt(50)
        sigma_ies_ = np.sqrt(100)
        sigma_csf_ = np.sqrt(6400)    
    else:
        sys.exit('not implemented')

    sigma_myelin_v,sigma_ies_v, sigma_csf_v = match_to_im_shape(im_shape,sigma_myelin_,sigma_ies_, sigma_csf_)
    
    return sigma_myelin_v,sigma_ies_v, sigma_csf_v


def load_sigmas_hybrid(im_shape, model='default', sigma_ies=None,sigma_ies_range=None):
    
    # based on Chatterjee 2018 - Multi-compartment model of brain tissues from T2 relaxometry MRI using gamma distribution
    # https://hal.archives-ouvertes.fr/hal-01744852/document

    sigma_myelin_ = np.sqrt(50)
    if sigma_ies is None and sigma_ies_range is None: 
        sigma_ies_ = np.sqrt(100)
    else: 
        assert (sigma_ies is None and sigma_ies_range is not None) or (sigma_ies is not None and sigma_ies_range is None), f"Either sigma_ies or sigma_ies_range have to be defined, but not both"
        if sigma_ies is not None:
            sigma_ies_ = np.float64(sigma_ies)
        elif sigma_ies_range is not None:
            value = sample_uniform(sigma_ies_range[0], sigma_ies_range[1]) 
            sigma_ies_ = np.float64(value)
    sigma_csf_ = np.sqrt(6400)  
        
    sigma_myelin_v,sigma_ies_v, sigma_csf_v = match_to_im_shape(im_shape,sigma_myelin_,sigma_ies_, sigma_csf_)
    
    return sigma_myelin_v,sigma_ies_v, sigma_csf_v



def load_means(im_shape, model='anima'):

    # based on Chatterjee 2018 - Multi-compartment model of brain tissues from T2 relaxometry MRI using gamma distribution
    # https://hal.archives-ouvertes.fr/hal-01744852/document
    
    if model == 'anima' or model == 'anima_sample_mu_ies':
        mu_myelin_ = np.float64(30)
        mu_ies_    = np.float64(100)
        mu_csf_    = np.float64(1700)
    else: 
        sys.exit('not implemented')
        
    mu_myelin_v,mu_ies_v, mu_csf_v = match_to_im_shape(im_shape, mu_myelin_,mu_ies_, mu_csf_)
    
    return mu_myelin_v,mu_ies_v, mu_csf_v


def load_means_hybrid(im_shape, model='anima', mu_ies=None, mu_ies_range=None):

    # based on Chatterjee 2018 - Multi-compartment model of brain tissues from T2 relaxometry MRI using gamma distribution
    # https://hal.archives-ouvertes.fr/hal-01744852/document
    
    mu_myelin_ = np.float64(30)
    if mu_ies is None and mu_ies_range is None:
        mu_ies_    = np.float64(100)
    else: 
        assert (mu_ies is None and mu_ies_range is not None) or (mu_ies is not None and mu_ies_range is None), f"Either mu_ies or mu_ies_range have to be defined, but not both"
        if mu_ies is not None:
            mu_ies_ = np.float64(mu_ies)
        elif mu_ies_range is not None:
            value = sample_uniform(mu_ies_range[0], mu_ies_range[1]) 
            mu_ies_ = np.float64(value)

    mu_csf_    = np.float64(1700)
        
    mu_myelin_v,mu_ies_v, mu_csf_v = match_to_im_shape(im_shape, mu_myelin_,mu_ies_, mu_csf_)
    
    return mu_myelin_v,mu_ies_v, mu_csf_v

def load_means_anima_tissues(im_shape, seg):

    # based on Chatterjee 2018 - Multi-compartment model of brain tissues from T2 relaxometry MRI using gamma distribution
    # https://hal.archives-ouvertes.fr/hal-01744852/document
    
    mu_myelin_ = np.float64(30)
    mu_ies_    = np.float64(100)
    mu_csf_    = np.float64(1700)

    mu_myelin_v,mu_ies_v, mu_csf_v = match_to_im_shape(im_shape, mu_myelin_,mu_ies_, mu_csf_)

    # add GM component
    mu_ies_gm    = np.float64(80)        
    gm_tissue_index = [200,300]
    indx = np.where(np.logical_and(seg>=gm_tissue_index[0], seg<gm_tissue_index[1])) #seg>=20, seg<30)), 
    mu_ies_v[indx] = mu_ies_gm
    
    return mu_myelin_v,mu_ies_v, mu_csf_v

def update_volume_fractions_low_mwf_in_gm(vf_myelin, vf_ies, vf_csf, seg, low_mwf='gm_only'):

    # set mwf to be very small in GM regions (and rest of contribution should come from IEWF and CSF tissues)

    # get indices of values for GM component       
    if low_mwf == 'gm_only':
        gm_tissue_index = [200,300, 20, 30]
        indx = np.where(np.logical_or(np.logical_and(seg>=gm_tissue_index[0], seg<gm_tissue_index[1]), np.logical_and(seg>=gm_tissue_index[2], seg<gm_tissue_index[3]))) 
    elif low_mwf == 'gm_and_csf':
        gm_tissue_index = [40,50,400,500] # non white matter tissues lie between [0-40], [50-400], [500+]
        indx = np.where(np.logical_or(np.logical_or(np.logical_and(seg>0,seg<gm_tissue_index[0]), np.logical_and(seg>=gm_tissue_index[1], seg<gm_tissue_index[2])), seg>=gm_tissue_index[3])) 

    # update IEWF by adding half of 90% of mwf values in the GM 
    vf_ies[indx] = np.add(vf_ies[indx], 0.5*np.multiply(vf_myelin[indx], 0.9))
    
    # update CSF by adding half of 90% of mwf values in the GM 
    vf_csf[indx] = np.add(vf_csf[indx], 0.5*np.multiply(vf_myelin[indx], 0.9))
    
    # update MWF in GM by reducing it to 10% of previous values 
    vf_myelin[indx] = np.multiply(vf_myelin[indx], 0.1)

    
    return vf_myelin, vf_ies, vf_csf

def match_to_im_shape(im_shape,myelin,ies, csf):
    
    myelin_v = np.full(shape=im_shape, fill_value = myelin)
    ies_v = np.full(shape=im_shape, fill_value = ies)
    csf_v = np.full(shape=im_shape, fill_value = csf)
    
    return myelin_v, ies_v, csf_v


def get_param_files(datadir, volume_range):

    if volume_range is not None:
        volume_range = [int(volume_range[0]), int(volume_range[1])]
        assert 0<=volume_range[0]<volume_range[1]

        files = [datadir+'/params_'+str(i)+'.nii.gz' for i in range(volume_range[0],volume_range[1])]
        #files = glob.glob(datadir+'/params_[0-9]*.nii.gz') # includes labels and images 

    else:
        files = glob.glob(datadir+'/params_[0-9]*.nii.gz') # includes labels and images 
        if not files: # if empty, check for old file names 
            files = glob.glob(datadir+'/sim_[0-9]*.nii.gz') # includes labels and images     
    ims = [f for f in files if 'labels' not in f] # fetch only images, not labels     
    ims = [f for f in ims if '_split_' not in f] # remove fslsplits...    
    
    return sorted(ims)



def estimate_volume_fractions(MWF, IEWF): 

    """ Estimates volume fractions from the simulated estiamtes of MWF, and initial estimate of IEWF (sampled from uniform distributions within a given range)

        Args: 
            MWF (numpy.ndarray) - 3D array with MWF water fraction estimates in the range of 0-1
            IEWF (numpy.ndarray) - 3D array with IES water fraction estimates in the range of 0-1
        Returns: 
            IEWF (numpy.ndarray) - corrected estimate of IEWF 
            CSF (numpy.ndarray) - estimated water fraction of CSF in the range of 0-1 

    """

    def debugging_fractions(F):
        if not np.logical_and(F > 0, F < 1).all(): 
            
            # set all negative values to zero 
            F[F<0] = 0 

            # set all values larger than 1 to 1 
            F[F>1] = 1


            # set nans to zero
            #F[np.isnan(F)] = 0 

        return F

    def debugging_fractions2(F,F2):

        # remove negatives 
        F2[np.where(F<0)] = 0 
        F[np.where(F<0)] = 0 

        F[np.where(F2<0)] = 0 
        F2[np.where(F2<0)] = 0 

        # remove numbers where fractions are bigger than 1 
        F2[np.where(F>1)] = 0 
        F[np.where(F>1)] = 0 

        F[np.where(F2>1)] = 0 
        F2[np.where(F2>1)] = 0

        # remove nans
        F2[np.isnan(F)] = 0 
        F[np.isnan(F)] = 0 

        F[np.isnan(F2)] = 0 
        F2[np.isnan(F2)] = 0


        return F, F2

    
    MWF = debugging_fractions(MWF)
    IEWF = debugging_fractions(IEWF)
    #MWF, IEWF = debugging_fractions2(MWF, IEWF)
    
    
    # check inputs     
    assert np.logical_and(MWF >= 0, MWF <= 1).all(), embed(header=sv.msg(f"MWF is not in the 0-1 range"))
    assert np.logical_and(IEWF >= 0, IEWF <= 1).all(), embed(header=sv.msg(f"IEWF is not in the 0-1 range"))
    
    # Correct IEWF: if IEWF+MWF > 1, IEWF_correction = 1-MWF 
    IEWF_MWF = np.add(IEWF, MWF) 
    ind_over_one = np.where(IEWF_MWF>1) #indices where the sum of the fractions > 1 
    ones = np.ones_like(IEWF)
    IEWF[ind_over_one] = np.subtract(ones, MWF)[ind_over_one] #estimate alternative values of IEWF by 1-MWF for instances where IEWF+MWF > 1 
    assert np.all(np.add(IEWF,MWF)<=1), f"np.add(IEWF and MWF)>1 AFTER correction. Please debug."

    # CSf = 1 - IEWF_corrected - MWF 
    CSF = np.subtract(ones,np.add(IEWF,MWF))
    assert np.all(np.add(CSF, np.add(IEWF,MWF))==1), f"Total volume fraction np.add(CSF, IEWF and MWF)>1. Please debug."

    return IEWF, CSF    
    



def estimate_volume_fractions_CSF(MWF, CSFF): 

    """ Estimates volume fractions from the simulated estiamtes of MWF, and initial estimate of IEWF (sampled from uniform distributions within a given range)

        Args: 
            MWF (numpy.ndarray) - 3D array with MWF water fraction estimates in the range of 0-1
            CSFF (numpy.ndarray) - 3D array with CSF water fraction estimates in the range of 0-1
        Returns: 
            IEWF (numpy.ndarray) - corrected estimate of IEWF 
            CSF (numpy.ndarray) - estimated water fraction of CSF in the range of 0-1 

    """

    def debugging_fractions(F):
        if not np.logical_and(F > 0, F < 1).all(): 
            
            # set all negative values to zero 
            F[F<0] = 0 

            # set all values larger than 1 to 1 
            F[F>1] = 1


            # set nans to zero
            #F[np.isnan(F)] = 0 

        return F

    def debugging_fractions2(F,F2):

        # remove negatives 
        F2[np.where(F<0)] = 0 
        F[np.where(F<0)] = 0 

        F[np.where(F2<0)] = 0 
        F2[np.where(F2<0)] = 0 

        # remove numbers where fractions are bigger than 1 
        F2[np.where(F>1)] = 0 
        F[np.where(F>1)] = 0 

        F[np.where(F2>1)] = 0 
        F2[np.where(F2>1)] = 0

        # remove nans
        F2[np.isnan(F)] = 0 
        F[np.isnan(F)] = 0 

        F[np.isnan(F2)] = 0 
        F2[np.isnan(F2)] = 0


        return F, F2

    
    MWF = debugging_fractions(MWF)
    CSFF = debugging_fractions(CSFF)
    #MWF, IEWF = debugging_fractions2(MWF, IEWF)
    
    
    # check inputs     
    assert np.logical_and(MWF >= 0, MWF <= 1).all(), embed(header=sv.msg(f"MWF is not in the 0-1 range"))
    assert np.logical_and(CSFF >= 0, CSFF <= 1).all(), embed(header=sv.msg(f"IEWF is not in the 0-1 range"))
    
    # Correct CSFF: if CSFF+MWF > 1, CSFF_correction = 1-MWF 
    CSFF_MWF = np.add(CSFF, MWF) 
    ind_over_one = np.where(CSFF_MWF>1) #indices where the sum of the fractions > 1 
    ones = np.ones_like(CSFF)
    CSFF[ind_over_one] = np.subtract(ones, MWF)[ind_over_one] #estimate alternative values of IEWF by 1-MWF for instances where IEWF+MWF > 1 
    assert np.all(np.add(CSFF,MWF)<=1), f"np.add(IEWF and MWF)>1 AFTER correction. Please debug."

    # # Correct CSFF: if CSFF< 0, CSFF_correction = +CSFF
    # CSFF = np.abs(CSFF)
    # assert np.all(CSFF>=0), f" Some CSF values are negative"

    # # Correct MWF: if MWF< 0, MWF_correction = +MWF
    # MWF = np.abs(MWF)
    # assert np.all(MWF>=0), f" Some MWF values are negative"

    # IEWF = 1 - CSFF_corrected - MWF 
    IEWF = np.subtract(ones,np.add(CSFF,MWF))

    ALL = np.add(CSFF, np.add(IEWF,MWF))
    assert np.all(np.logical_and(ALL>0.9999999, ALL<1.00000001)), f"Total volume fraction np.add(CSFF, IEWF and MWF)>1. Please debug."

    return IEWF, CSFF    


def update_volume_fractions(im):
    
    # add correct volume fractions to array of parameters 
    vf_myelin = im[:,:,3] 
    vf_ies_est = im[:,:,4]

    # check if volume fractions have been multiplied by a large number (in order to store them properly in nifti - as there was a datatype mismatch)
    if np.mean(vf_myelin[vf_myelin>0]) > 1: # mean value of volume fraction is larger than 1 
        vf_myelin = np.divide(vf_myelin,100)
        vf_ies_est = np.divide(vf_ies_est,100)


    vf_ies, vf_csf = estimate_volume_fractions(vf_myelin, vf_ies_est)
    
    vf_myelin = vf_myelin
    vf_ies = vf_ies
    vf_csf = vf_csf

    return vf_myelin, vf_ies, vf_csf



def update_volume_fractions_CSF(im):

    # assume that we had generated CSF instead of VF_IES
    
    # add correct volume fractions to array of parameters 
    vf_myelin = im[:,:,3] 
    vf_csf_est = im[:,:,4]

    # check if volume fractions have been multiplied by a large number (in order to store them properly in nifti - as there was a datatype mismatch)
    if np.mean(vf_myelin[vf_myelin>0]) > 1: # mean value of volume fraction is larger than 1 
        vf_myelin = np.divide(vf_myelin,100)
        vf_csf_est = np.divide(vf_csf_est,100)


    vf_ies, vf_csf = estimate_volume_fractions_CSF(vf_myelin, vf_csf_est)
    
    vf_myelin = vf_myelin
    vf_ies = vf_ies
    vf_csf = vf_csf

    return vf_myelin, vf_ies, vf_csf

def get_batch2(iterable, n=1):
    # allows us to COUNT through batches of data 
    l = len(iterable)
    for ndx in range(0, l, n):
        yield ndx, min(ndx + n, l)

def estimate_echoes_torch(pT2,EPG_batch):

    # We need to multiply a matrix of (N,2000) by (2000,N,32) 
    # to obtain a matrix of (N,1,32)
    
    # here is a toy example by 3D multiplication works 
    
    #     https://www.geeksforgeeks.org/numpy-3d-matrix-multiplication/
    #     (3,3,2) * (3,2,4) >> we are doing multiplication between 2D matrices of dims (3,2), (2,4) three times, which means that columns of first matrix (3,2) must match rows of second matrix (2,4)

    
    #     Therefore - we need to expand pT2 from (N,2000) to (N,1,2000)
    #     and switch EPG_batch from (2000,N,32) to (N,2000,32)
    #     where N is the number of voxels (samples)
    #     >>> 
    #     FINAL MULTIPLICATION IS: 
    #     (N,1,2000) * (N,2000,32)
    #     >>>      we multiply (1,2000) by (2000,32) N times => resuling in N results of (1,32). i.e. (N,1,32)
    
    
    #     multiplication is according to this formula: 
    #     https://cibm.ch/wp-content/uploads/2007.10225.pdf
    
    
    pT2_tr = pT2.unsqueeze(1) # (N,1,2000)    
    EPG_batch_tr = torch.transpose(EPG_batch,1,0) # (N, 2000, 32)
    mult = torch.matmul(pT2_tr,EPG_batch_tr) # (N,1,32)
    signal = torch.squeeze(mult) # (N,32)
    
    return signal

                  
if __name__ == "__main__":
    
    args = load_args()
    
    # run main 
    main(args)

