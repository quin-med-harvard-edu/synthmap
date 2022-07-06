"""Specify inputs for MWF related parameter image generation. 

The following variables can be specified: 
- A range of parameters specifying the bounds of the sampled prior values for EACH segment in the brain segmentation: 
    p['vf_myelin'] 
    p['vf_ies'] 
    p['mu_myelin'] 
    p['mu_ies'] 
    p['mu_csf'] 
    Each variable above must be made up of TWO numbers. 
    Depending on distribution, uniform or normal, the two numbers correspond to [low,high] or [mean,std] respectively. 

- Boolean parameters such as: 

    modified_opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True, 
                    'path_label_map':label_map,
                    'prior_distribution':'normal',
                    'n_examples':100,
                    'output_shape':None,
                    'vf_pop_var':0.2, # vary within 10% of what is set 
                    'mu_pop_var':0.2, # vary within 5% of what is set 
                    'save_prior_values':True}   

- Additional parameters such as: 
    opt['path_label_map'] = datadir+'example.nii.gz'
    opt['prior_distribution'] = 'normal'
    opt['generation_labels'] = np.array([1,2,3,4,5])
    opt['n_examples'] = 3
    opt['output_shape'] = None 
    opt['subject_count'] = 0 
    opt['load_from_file'] = 0 



This file is also a collection of experiments conducted in generation of MWF related parameter maps. 


`example_20` is the experiment that was used to generate parameter maps for ISMRM submission. 


To generate a new set of variables create a new function with `def example_<number>(opt)` and then run it inside `example_latest(). 

"""

import os

import numpy as np
import pandas as pd 

import svtools as sv

def example_latest(opt):

    #old #opt["experiment_name"] = 'experiment_'
    opt["load_from_file"] = False

    # default - generates 3 priors that differ by an integer multiple - e.g. 60/120/240 - no fractions 
    #opt = example_0(opt)

    # MIML priors, population var is half of intra-subject var;
    #opt = example_5(opt)

    # reduce intra-subject variance to half, turn off all spatial transforms
    #opt = example_6(opt)

    # reduce intra-ROI variance and inter-subject variance to one third of the default (MIML based values)
    #opt = example_7(opt)

    # zero variance in all parameters, all transforms are off (inc gmm sample)
    # DO NOT TOUCH
    #opt = example_8(opt)

    # generate examples with Onur's parameter suggestions. Pick 'uniform' distribution instead of 'normal'. No field map generation just yet. 
    #opt = example_10(opt)

    # generate same as example 10 with normal distribution, not uniform
    #opt = example_11(opt)

    # repeat experiment 11 but introduce blurring 
    #opt = example_12(opt)

    # repeat experiment 11 but introduce blurring, spatial transformations and generate 10 examples
    #opt = example_13(opt)

    # run example 11 on multiple segmentations 
    #opt = example_14(opt)

    # run example 13 on multiple segmentations (correct params, spatial transforms on, & multiseg with 20 examples)
    #opt = example_15(opt)

    # run example 11 but generate bias also 
    # opt = example_16(opt)

    # run example 11 (no spatial deformation) bu reduce range of variance in MWF to 0.1 and fix GM/WM values to be different, and add multi-seg
    #opt = example_17(opt)

    # run example 17 again, to check how to fix morphological closing operation properly 
    #opt = example_18(opt)

    # run example 17, with reduced variance in the paramter range, by setting the confidence interval to be 99.7 percent for specified parameter ranges 
    #opt = example_19(opt)
    #opt = example_19b(opt)
    # > decision - use 0.95 confidence, not 0.997 -> in other words -> example_17
    #opt = example_19c(opt)

    # run example 17 with all parameters switched ON, inc spatial transforms, for 100 volumes 
    #opt = example_20(opt)

    # repeat example20 for 100 volumes but introduce MORE variance between regions 
    #opt = example_21(opt)

    # repeat example21 for 100 volumes but introduce MORE variance between subjects 
    #opt = example_22(opt)

    # repeat example22 for 2 volumes as dummy test
    #opt = example_23(opt)

    
    # generate brain with crazy dimensions
    #opt = example_27(opt)    

    # generate brains from priors defined from .pkl file - 3 rois 
    #opt = example_28(opt)    
     
    # generate brains from priors defined from .pkl file - 95 rois 
    #opt = example_29(opt)   
    
    # repeat example 29 with severely reduced variances (prior_stds) - set to 10% of prior_means
    #opt = example_30(opt)    

    # repeat example 30 with no deformations
    #opt = example_31(opt)    

    # repeat example 30 (vars set to 10% of means) but with tissue n parcel segmentations
    #opt = example_31b(opt)    

    # repeat example 31b but with deformations
    #opt = example_32(opt)    
    #opt = example_32b(opt)    
    
    # repeat 32b 1000 times
    #opt = example_32c(opt)

    # vary individual parameters to learn how it changes the output - see examlpe_32z_... 
    #opt = example_32z_none(opt)

    # no spatial deformations of any kind - no: afine, nonlin, flip, blur, spatial bias 
    #opt = example_34(opt)

    # same as example 34, but extract vf_csf, instead of vf_ies
    opt = example_35(opt)    



    print("WARNING: interp set to nearest, see documentation")
    # def transform(vol, loc_shift, interp_method='nearest', indexing='ij'):


    return opt


########################################
##### PLEASE COMMENT / CLEAN THESE FUNCTIONS


# def load_priors_from_pickle(path_mean, path_std):
#     dfs_mean = pd.read_pickle(path_mean)
#     dfs_std = pd.read_pickle(path_std)

#     # extract parameters in the correct name format 
#     return dfs_mean, dfs_std 




# multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
def mult_vol_fraction(p,mult=100):
    p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
    p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
    return p
########################################

def example_35(opt): 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32z(opt)
    opt["experiment_name"] = 'test35_generate_vf_csf_instead_of_vf_ies'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'vary_prior_means_and_stds':True,
                    'blurring': False,
                    'apply_linear_trans':False,                    
                    'apply_nonlin_trans':False,
                    'blur_background':False,
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0} # vary within 5% of what is set

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    # load priors from pickle 
    rootdir = '/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/libs/'
    path_mean = rootdir + 'test_mean_v2_inc_tissue_n_parcel.pkl'
    path_std = rootdir + 'test_std_v2_inc_tissue_n_parcel.pkl'   
    dfs_mean = pd.read_pickle(path_mean)
    dfs_std = pd.read_pickle(path_std)
    

    """
    Currently statistics return the following: 

    mu_myelin              0.024113
    mu_ies                 0.081929
    mu_csf                 1.029994
    vf_myelin              0.035285
    vf_ies                 0.548513
    vf_csf                 0.416202
    sigma_myelin           0.000961
    sigma_ies              0.016912
    sigma_csf              0.144797
    Name: 44, dtype: object       


    REQUIREMENTS: 

    Parameters must be in this order: 
    - mu - in milliseconds - e.g. 23ms for short component at up to 1500 ms for long components 
    - sigma - in milliseconds - e.g. originally fixed to 5ms 
    - vf - originally set as a fraction - e.g. 0.25 for myelin - but it is multiplied by 100 - > therefore if should be in the range that generally varies from 1-100
    
    ACTIONS:

    mu_myelin              0.024113 >> mult by 1000
    mu_ies                 0.081929 >> mult by 1000
    mu_csf                 1.029994 >> mult by 1000
    vf_myelin              0.035285 >> mult by 100
    vf_ies                 0.548513 >> mult by 100
    vf_csf                 0.416202 >> mult by 100
    sigma_myelin           0.000961 >> mult by 1000
    sigma_ies              0.016912 >> mult by 1000
    sigma_csf              0.144797 >> mult by 1000    

    """

    # mult fractions by 100 (since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values)
    for p in ['vf_myelin', 'vf_ies', 'vf_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(100)
        dfs_std[p] = dfs_mean[p].copy().divide(10)
    
    for p in ['mu_myelin', 'mu_ies', 'mu_csf','sigma_myelin', 'sigma_ies', 'sigma_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(1000)
        dfs_std[p] = dfs_mean[p].copy().divide(10)

    # export parameters from pandas dataframe into a dictionary (historical compatibility reasons)
    params_all_rois = {}
    for roi in dfs_mean.index:
        
        # select dataframe series for given roi
        dfs_mean_roi = dfs_mean.loc[roi].to_dict()
        dfs_std_roi = dfs_std.loc[roi].to_dict()

        # fuse mean and std measurements together 
        dfs_roi = {}
        for k in dfs_mean_roi.keys():
            if k != 'roi_name': # avoid double copying the roi_name
                dfs_roi[k] = [dfs_mean_roi[k], dfs_std_roi[k]]
            else:
                dfs_roi[k] = dfs_mean_roi[k]
        
        params_all_rois[roi] = dfs_roi

    # add background priors 
    params_all_rois = add_background_priors(params_all_rois, background_label=0)

    opt['prior_means'], opt['prior_stds'],roi_map = convert_priors_to_synthseg_format_gauss_VF_CSF(params_all_rois, opt)


    # set generation labels 
    # make sure that the order of labels is the same as are the columns of prior_means and prior_sts with their corresponding rois
    #opt['generation_labels'] = np.array([1,2,3,4,5])
    #opt['generation_labels'] = np.array([int(i) for i in list(roi_map.keys())]) # the keys of roi_map dict contains the order, which we must present as a np.array of ints to the synthseg algorithm
    # import generation labels from file (pre computed - much better)
    # note that in future sections - we need to make sure that all regions are reflected     
    opt['generation_labels'] = np.array([int(i) for i in dfs_mean.index.tolist()] + [0])
    tissue_names = {'2':'GM', '3':'CSF', '4':'WM', '5':'VEN'}
    opt['generation_label_names'] = [tissue_names[i] + ' ' + j for i,j in zip(dfs_mean['tissue_id'].tolist(), dfs_mean['roi_name'].tolist())] + ['background']
    
    # a couple of checks to make sure the mapping is correct after conversion
    assert opt['prior_means'][:,0][0] == params_all_rois[str(opt['generation_labels'][0])]['mu_myelin'][0]
    assert opt['prior_means'][:,-1][0] == params_all_rois[str(opt['generation_labels'][-1])]['mu_myelin'][0]    

    # if priors need to be verified
    if 'check_priors' in opt and opt['check_priors']:
        assert 'roi_name_ids' in opt
        check_priors(params_all_rois, opt['roi_name_ids'])

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt    




def example_34(opt): # NOT DEFINED - not used 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32z(opt)
    opt["experiment_name"] = 'test34_no_spatial_deform'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'vary_prior_means_and_stds':True,
                    'blurring': False,
                    'apply_linear_trans':False,                    
                    'apply_nonlin_trans':False,
                    'blur_background':False,
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0} # vary within 5% of what is set

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt   

def example_32_z_affine(opt): # repeat example 32b with MINIMUM scaling / shearing / rotation bounds, but with controlled deformation field. 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32z(opt)
    opt["experiment_name"] = 'test32_z_affine'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':False,
                    'blurring': False,
                    'vary_prior_means_and_stds':False,
                    'scaling_bounds':0.15, #0.15
                    'shearing_bounds':0.01, #0.01  
                    'rotation_bounds':15, #15
                    'apply_nonlin_trans':False,
                    'nonlin_std':5., #1., #nonlin_std:3.,
                    'nonlin_shape_factor':0.0625,#nonlin_shape_factor:0.0625,    
                    'blur_background':False,
                    'blur_range': 1.0, #1.15,
                    'bias_field_std': 0.000000003,       
                    'bias_shape_factor':0.025,                                                                                  
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0} # vary within 5% of what is set

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt    

def example_32_z_vary_prior_means(opt): # repeat example 32b with MINIMUM scaling / shearing / rotation bounds, but with controlled deformation field. 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32z(opt)
    opt["experiment_name"] = 'test32_z_vary_prior_means'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':False,
                    'blurring': False,
                    'vary_prior_means_and_stds':True,
                    'scaling_bounds':0.00000000000000001, #0.15
                    'shearing_bounds':0.00000000000000001, #0.01  
                    'rotation_bounds':0.00000000000000001, #15
                    'apply_nonlin_trans':False,
                    'nonlin_std':1., #1., #nonlin_std:3.,
                    'nonlin_shape_factor':0.625,#nonlin_shape_factor:0.0625,    
                    'blur_background':False,
                    'blur_range': 1.0, #1.15,
                    'bias_field_std': 0.000000003,       
                    'bias_shape_factor':0.025,                                                                                  
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0} # vary within 5% of what is set

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt   

def example_32_z_sample_gmm(opt): # repeat example 32b with MINIMUM scaling / shearing / rotation bounds, but with controlled deformation field. 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32z(opt)
    opt["experiment_name"] = 'test32_z_sample_gmm'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring': False,
                    'vary_prior_means_and_stds':False,
                    'scaling_bounds':0.00000000000000001, #0.15
                    'shearing_bounds':0.00000000000000001, #0.01  
                    'rotation_bounds':0.00000000000000001, #15
                    'apply_nonlin_trans':False,
                    'nonlin_std':1., #1., #nonlin_std:3.,
                    'nonlin_shape_factor':0.625,#nonlin_shape_factor:0.0625,    
                    'blur_background':False,
                    'blur_range': 1.0, #1.15,
                    'bias_field_std': 0.000000003,       
                    'bias_shape_factor':0.025,                                                                                  
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0} # vary within 5% of what is set

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt   

def example_32_z_blurring(opt): # repeat example 32b with MINIMUM scaling / shearing / rotation bounds, but with controlled deformation field. 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32z(opt)
    opt["experiment_name"] = 'test32_z_blurring'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':False,
                    'blurring': True,
                    'vary_prior_means_and_stds':False,
                    'scaling_bounds':0.00000000000000001, #0.15
                    'shearing_bounds':0.00000000000000001, #0.01  
                    'rotation_bounds':0.00000000000000001, #15
                    'apply_nonlin_trans':False,
                    'nonlin_std':1., #1., #nonlin_std:3.,
                    'nonlin_shape_factor':0.625,#nonlin_shape_factor:0.0625,    
                    'blur_background':False,
                    'blur_range': 1.0, #1.15,
                    'bias_field_std': 0.000000003,       
                    'bias_shape_factor':0.025,                                                                                  
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0} # vary within 5% of what is set

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt    

def example_32_z_vary_population(opt): # repeat example 32b with MINIMUM scaling / shearing / rotation bounds, but with controlled deformation field. 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32z(opt)
    opt["experiment_name"] = 'test32_z_vary_population'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring': False,
                    'vary_prior_means_and_stds':True,
                    'scaling_bounds':0.00000000000000001, #0.15
                    'shearing_bounds':0.00000000000000001, #0.01  
                    'rotation_bounds':0.00000000000000001, #15
                    'apply_nonlin_trans':False,
                    'nonlin_std':1., #1., #nonlin_std:3.,
                    'nonlin_shape_factor':0.625,#nonlin_shape_factor:0.0625,    
                    'blur_background':False,
                    'blur_range': 1.0, #1.15,
                    'bias_field_std': 0.000000003,       
                    'bias_shape_factor':0.025,                                                                                  
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.2, # vary within 10% of what is set 
                    'mu_pop_var':0.2} # vary within 5% of what is set

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt    

def example_32_z_deformation_3_shape_factor_10x(opt): # repeat example 32b with MINIMUM scaling / shearing / rotation bounds, but with controlled deformation field. 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32z(opt)
    opt["experiment_name"] = 'test32_z_deformation3'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':False,
                    'blurring': False,
                    'vary_prior_means_and_stds':False,
                    'scaling_bounds':0.00000000000000001, #0.15
                    'shearing_bounds':0.00000000000000001, #0.01  
                    'rotation_bounds':0.00000000000000001, #15
                    'apply_nonlin_trans':True,
                    'nonlin_std':1., #1., #nonlin_std:3.,
                    'nonlin_shape_factor':0.625,#nonlin_shape_factor:0.0625,    
                    'blur_background':False,
                    'blur_range': 1.0, #1.15,
                    'bias_field_std': 0.000000003,       
                    'bias_shape_factor':0.025,                                                                                  
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0} # vary within 5% of what is set

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt    

def example_32_z_deformation_2(opt): # repeat example 32b with MINIMUM scaling / shearing / rotation bounds, but with controlled deformation field. 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32z(opt)
    opt["experiment_name"] = 'test32_z_deformation2'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':False,
                    'blurring': False,
                    'vary_prior_means_and_stds':False,
                    'scaling_bounds':0.00000000000000001, #0.15
                    'shearing_bounds':0.00000000000000001, #0.01  
                    'rotation_bounds':0.00000000000000001, #15
                    'apply_nonlin_trans':True,
                    'nonlin_std':2., #1., #nonlin_std:3.,
                    'nonlin_shape_factor':0.0625,#nonlin_shape_factor:0.0625,    
                    'blur_background':False,
                    'blur_range': 1.0, #1.15,
                    'bias_field_std': 0.000000003,       
                    'bias_shape_factor':0.025,                                                                                  
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0} # vary within 5% of what is set

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt    

def example_32_z_deformation_1(opt): # repeat example 32b with MINIMUM scaling / shearing / rotation bounds, but with controlled deformation field. 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32z(opt)
    opt["experiment_name"] = 'test32_z_deformation'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':False,
                    'blurring': False,
                    'vary_prior_means_and_stds':False,
                    'scaling_bounds':0.00000000000000001, #0.15
                    'shearing_bounds':0.00000000000000001, #0.01  
                    'rotation_bounds':0.00000000000000001, #15
                    'apply_nonlin_trans':True,
                    'nonlin_std':5., #1., #nonlin_std:3.,
                    'nonlin_shape_factor':0.0625,#nonlin_shape_factor:0.0625,    
                    'blur_background':False,
                    'blur_range': 1.0, #1.15,
                    'bias_field_std': 0.000000003,       
                    'bias_shape_factor':0.025,                                                                                  
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0} # vary within 5% of what is set

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt    

def example_32z_none(opt): # repeat example 32b with MINIMUM scaling / shearing / rotation bounds, but with controlled deformation field. 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32b(opt)
    opt["experiment_name"] = 'test32z_none'


    label_map = "/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/output_4/label_dir_2_do_not_delete/" # provide directory to multiple segmentations 

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':False,
                    'blurring': False,
                    'vary_prior_means_and_stds':False,
                    'scaling_bounds':0.00000000000000001, #0.15
                    'shearing_bounds':0.00000000000000001, #0.01  
                    'rotation_bounds':0.00000000000000001, #15
                    'apply_nonlin_trans':False,
                    'nonlin_std':5., #1., #nonlin_std:3.,
                    'nonlin_shape_factor':0.0625,#nonlin_shape_factor:0.0625,    
                    'blur_background':False,
                    'blur_range': 1.0, #1.15,
                    'bias_field_std': 0.000000003,       
                    'bias_shape_factor':0.025,                                                                                  
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0, # vary within 5% of what is set                      
                    'path_label_map':label_map}
    
    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt      

def example_32z(opt): # repeat example 32b with MINIMUM scaling / shearing / rotation bounds, but with controlled deformation field. 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32b(opt)
    opt["experiment_name"] = 'test32z_just_nonlin_deformation'


    label_map = "/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/output_4/label_dir_2_do_not_delete/" # provide directory to multiple segmentations 

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':False,
                    'blurring': False,
                    'vary_prior_means_and_stds':False,
                    'scaling_bounds':0.00000000000000001, #0.15
                    'shearing_bounds':0.00000000000000001, #0.01  
                    'rotation_bounds':0.00000000000000001, #15
                    'apply_nonlin_trans':True,
                    'nonlin_std':5., #1., #nonlin_std:3.,
                    'nonlin_shape_factor':0.0625,#nonlin_shape_factor:0.0625,    
                    'blur_background':False,
                    'blur_range': 1.0, #1.15,
                    'bias_field_std': 0.000000003,       
                    'bias_shape_factor':0.025,                                                                                  
                    'n_examples':3,
                    'flipping':False, 
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0, # vary within 5% of what is set                      
                    'path_label_map':label_map}
    
    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt      


def example_32d(opt): # repeat example 32b 100 times 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32b(opt)
    opt["experiment_name"] = 'test32c_generate_from_pkl_priors_285_rois_inc_tissues_10percent_variances_w_deformations'
    
    # over ride default extra-options with the following 
    modified_opt = {'n_examples':1000}
    
    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt       


def example_32c(opt): # repeat example 32b 100 times 

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_32b(opt)
    opt["experiment_name"] = 'test32c_generate_from_pkl_priors_285_rois_inc_tissues_10percent_variances_w_deformations'
    
    # over ride default extra-options with the following 
    modified_opt = {'n_examples':1000}
    
    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt       
    
    

def example_32b(opt): # repeat example 31b but with deformations

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_31b(opt)
    opt["experiment_name"] = 'test32b_generate_from_pkl_priors_285_rois_inc_tissues_10percent_variances_w_deformations'

    label_map = "/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/output_4/label_dir_2_do_not_delete/" # provide directory to multiple segmentations 
    
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'scaling_bounds':None,
                    'rotation_bounds':None,
                    'shearing_bounds':None,       
                    'nonlin_std':2., #1., #nonlin_std:3.,
                    'nonlin_shape_factor':0.0625,#nonlin_shape_factor:0.0625,    
                    'blur_background':True,
                    'blur_range':1.15,
                    'bias_field_std':0.3,       
                    'bias_shape_factor':0.025,                                                              
                    'flipping':True, 
                    'prior_distribution':'normal',  # if set to normal - the variance in parameters over population will be normal (this should be default practically always - do not set this to uniform - this is where the mistake was for example_24-26 - possibly the reason why generation wasn't great)
                    'n_examples':5,
                    'output_shape':None,
                    'vf_pop_var':0.1, # vary within 10% of what is set 
                    'mu_pop_var':0.1, # vary within 5% of what is set                               
                    'save_prior_values':True, 
                    'check_priors':False,
                    'roi_name_ids':[44],
                    'path_label_map': label_map}    
                 
    # # over ride default extra-options with the following 
    # modified_opt = {'sample_gmm':False,
    #                 'blurring':False, 
    #                 'vary_prior_means_and_stds':False,
    #                 'debug_prior_values':False,
    #                 'clipping':False,
    #                 'min_max_norm':False,
    #                 'gamma_augmentation':False,
    #                 'apply_bias_field':False, 
    #                 'apply_nonlin_trans':False,
    #                 'apply_linear_trans':False,
    #                 'scaling_bounds':None,
    #                 'rotation_bounds':None,
    #                 'shearing_bounds':None,       
    #                 'nonlin_std':2., #1., #nonlin_std:3.,
    #                 'nonlin_shape_factor':0.0625,#nonlin_shape_factor:0.0625,    
    #                 'blur_background':False,
    #                 'blur_range':1.15,
    #                 'bias_field_std':0.3,       
    #                 'bias_shape_factor':0.025,                                                              
    #                 'flipping':False, 
    #                 'prior_distribution':'normal',  # if set to normal - the variance in parameters over population will be normal (this should be default practically always - do not set this to uniform - this is where the mistake was for example_24-26 - possibly the reason why generation wasn't great)
    #                 'n_examples':5,
    #                 'output_shape':None,
    #                 'vf_pop_var':0.0, # vary within 10% of what is set 
    #                 'mu_pop_var':0.0, # vary within 5% of what is set                               
    #                 'save_prior_values':True, 
    #                 'check_priors':False,
    #                 'roi_name_ids':[44]}                   

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt   
    
def example_32(opt): # repeat example 31b but with deformations

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_31b(opt)
    opt["experiment_name"] = 'test32_generate_from_pkl_priors_285_rois_inc_tissues_10percent_variances_w_deformations'

    
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True, 
                    'prior_distribution':'normal',  # if set to normal - the variance in parameters over population will be normal (this should be default practically always - do not set this to uniform - this is where the mistake was for example_24-26 - possibly the reason why generation wasn't great)
                    'n_examples':5,
                    'output_shape':None,
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0, # vary within 5% of what is set                               
                    'save_prior_values':True, 
                    'check_priors':False,
                    'roi_name_ids':[44]}                 

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt    

def example_31b(opt): # repeat example 30 (vars set to 10% of means) but with tissue n parcel segmentations

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_29(opt)
    opt["experiment_name"] = 'test31_generate_from_pkl_priors_285_rois_inc_tissues_10percent_variances'



    # over ride default extra-options with the following 
    #segdir = '/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/output_4/'
    #segfile = "Hammers_mith-n30r95-MaxProbMap-full-MNI152-SPM12_resamp_reg_irtk_tr_reg_npeye.nii.gz"
    label_map = "/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/output_4/label_dir_2_do_not_delete/" # provide directory to multiple segmentations 
    
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':False, 
                    'vary_prior_means_and_stds':False,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':False,
                    'flipping':False, 
                    'path_label_map':label_map,
                    'prior_distribution':'normal',  # if set to normal - the variance in parameters over population will be normal (this should be default practically always - do not set this to uniform - this is where the mistake was for example_24-26 - possibly the reason why generation wasn't great)
                    'n_examples':5,
                    'output_shape':None,
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0, # vary within 5% of what is set                               
                    'save_prior_values':True, 
                    'check_priors':False,
                    'roi_name_ids':[44]}                 

    """
        :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
        sampled from a uniform distribution of predefined bounds. Can either be:
        1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
        (1-scaling_bounds, 1+scaling_bounds) for each dimension.
        2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
        (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
        3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
         of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
        4) the path to such a numpy array.
        If None (default), scaling_range = 0.15
        
    
        :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
        and 2, the bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
        If None (default), rotation_bounds = 15.
        
        :param shearing_bounds: (optional) same as scaling bounds. If None (default), shearing_bounds = 0.01.
        
        :param nonlin_std: (optional) If apply_nonlin_trans is True, maximum value for the standard deviation of the
        normal distribution from which we sample the first tensor for synthesising the deformation field.
        
    
        :param nonlin_shape_factor: (optional) If apply_nonlin_trans is True, ratio between the size of the input label
        maps and the size of the sampled tensor for synthesising the deformation field.
    """
    
    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    # load priors from pickle 
    rootdir = '/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/libs/'
    path_mean = rootdir + 'test_mean_v2_inc_tissue_n_parcel.pkl'
    path_std = rootdir + 'test_std_v2_inc_tissue_n_parcel.pkl'   
    dfs_mean = pd.read_pickle(path_mean)
    dfs_std = pd.read_pickle(path_std)
    

    """
    Currently statistics return the following: 

    mu_myelin              0.024113
    mu_ies                 0.081929
    mu_csf                 1.029994
    vf_myelin              0.035285
    vf_ies                 0.548513
    vf_csf                 0.416202
    sigma_myelin           0.000961
    sigma_ies              0.016912
    sigma_csf              0.144797
    Name: 44, dtype: object       


    REQUIREMENTS: 

    Parameters must be in this order: 
    - mu - in milliseconds - e.g. 23ms for short component at up to 1500 ms for long components 
    - sigma - in milliseconds - e.g. originally fixed to 5ms 
    - vf - originally set as a fraction - e.g. 0.25 for myelin - but it is multiplied by 100 - > therefore if should be in the range that generally varies from 1-100
    
    ACTIONS:

    mu_myelin              0.024113 >> mult by 1000
    mu_ies                 0.081929 >> mult by 1000
    mu_csf                 1.029994 >> mult by 1000
    vf_myelin              0.035285 >> mult by 100
    vf_ies                 0.548513 >> mult by 100
    vf_csf                 0.416202 >> mult by 100
    sigma_myelin           0.000961 >> mult by 1000
    sigma_ies              0.016912 >> mult by 1000
    sigma_csf              0.144797 >> mult by 1000    

    """

    # mult fractions by 100 (since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values)
    for p in ['vf_myelin', 'vf_ies', 'vf_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(100)
        dfs_std[p] = dfs_mean[p].copy().divide(10)
    
    for p in ['mu_myelin', 'mu_ies', 'mu_csf','sigma_myelin', 'sigma_ies', 'sigma_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(1000)
        dfs_std[p] = dfs_mean[p].copy().divide(10)

    # export parameters from pandas dataframe into a dictionary (historical compatibility reasons)
    params_all_rois = {}
    for roi in dfs_mean.index:
        
        # select dataframe series for given roi
        dfs_mean_roi = dfs_mean.loc[roi].to_dict()
        dfs_std_roi = dfs_std.loc[roi].to_dict()

        # fuse mean and std measurements together 
        dfs_roi = {}
        for k in dfs_mean_roi.keys():
            if k != 'roi_name': # avoid double copying the roi_name
                dfs_roi[k] = [dfs_mean_roi[k], dfs_std_roi[k]]
            else:
                dfs_roi[k] = dfs_mean_roi[k]
        
        params_all_rois[roi] = dfs_roi

    # add background priors 
    params_all_rois = add_background_priors(params_all_rois, background_label=0)

    opt['prior_means'], opt['prior_stds'],roi_map = convert_priors_to_synthseg_format_gauss(params_all_rois, opt)


    # set generation labels 
    # make sure that the order of labels is the same as are the columns of prior_means and prior_sts with their corresponding rois
    #opt['generation_labels'] = np.array([1,2,3,4,5])
    #opt['generation_labels'] = np.array([int(i) for i in list(roi_map.keys())]) # the keys of roi_map dict contains the order, which we must present as a np.array of ints to the synthseg algorithm
    # import generation labels from file (pre computed - much better)
    # note that in future sections - we need to make sure that all regions are reflected     
    opt['generation_labels'] = np.array([int(i) for i in dfs_mean.index.tolist()] + [0])
    tissue_names = {'2':'GM', '3':'CSF', '4':'WM', '5':'VEN'}
    opt['generation_label_names'] = [tissue_names[i] + ' ' + j for i,j in zip(dfs_mean['tissue_id'].tolist(), dfs_mean['roi_name'].tolist())] + ['background']
    
    # a couple of checks to make sure the mapping is correct after conversion
    assert opt['prior_means'][:,0][0] == params_all_rois[str(opt['generation_labels'][0])]['mu_myelin'][0]
    assert opt['prior_means'][:,-1][0] == params_all_rois[str(opt['generation_labels'][-1])]['mu_myelin'][0]    

    # if priors need to be verified
    if 'check_priors' in opt and opt['check_priors']:
        assert 'roi_name_ids' in opt
        check_priors(params_all_rois, opt['roi_name_ids'])

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt    



def example_31(opt): # repeat example 30 with no deformations

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_30(opt)
    opt["experiment_name"] = 'test31_repeat_example_30_without_deformations'


    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':False, 
                    'vary_prior_means_and_stds':True,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':False,
                    'flipping':False, 
                    'n_examples':5,                             
                    'save_prior_values':True}   

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]


    return opt 

def example_30(opt): # repeat example 29 with severely reduced variances (prior_stds) - set to 10% of prior_means

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_29(opt)
    opt["experiment_name"] = 'test30_generate_from_pkl_priors_95_rois_10percent_variances'



    # over ride default extra-options with the following 
    #segdir = '/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/output_4/'
    #segfile = "Hammers_mith-n30r95-MaxProbMap-full-MNI152-SPM12_resamp_reg_irtk_tr_reg_npeye.nii.gz"
    label_map = "/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/output_4/label_dir_do_not_delete/" # provide directory to multiple segmentations 
    
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':False, 
                    'vary_prior_means_and_stds':False,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':False,
                    'flipping':False, 
                    'path_label_map':label_map,
                    'prior_distribution':'normal',  # if set to normal - the variance in parameters over population will be normal (this should be default practically always - do not set this to uniform - this is where the mistake was for example_24-26 - possibly the reason why generation wasn't great)
                    'n_examples':5,
                    'output_shape':None,
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0, # vary within 5% of what is set                               
                    'save_prior_values':True, 
                    'check_priors':False,
                    'roi_name_ids':[44]}                 

    """
        :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
        sampled from a uniform distribution of predefined bounds. Can either be:
        1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
        (1-scaling_bounds, 1+scaling_bounds) for each dimension.
        2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
        (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
        3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
         of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
        4) the path to such a numpy array.
        If None (default), scaling_range = 0.15
        
    
        :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
        and 2, the bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
        If None (default), rotation_bounds = 15.
        
        :param shearing_bounds: (optional) same as scaling bounds. If None (default), shearing_bounds = 0.01.
        
        :param nonlin_std: (optional) If apply_nonlin_trans is True, maximum value for the standard deviation of the
        normal distribution from which we sample the first tensor for synthesising the deformation field.
        
    
        :param nonlin_shape_factor: (optional) If apply_nonlin_trans is True, ratio between the size of the input label
        maps and the size of the sampled tensor for synthesising the deformation field.
    """
    
    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    # load priors from pickle 
    rootdir = '/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/libs/'
    path_mean = rootdir + 'test_mean_v2.pkl'
    path_std = rootdir + 'test_std_v2.pkl'   
    dfs_mean = pd.read_pickle(path_mean)
    dfs_std = pd.read_pickle(path_std)
    

    """
    Currently statistics return the following: 

    mu_myelin              0.024113
    mu_ies                 0.081929
    mu_csf                 1.029994
    vf_myelin              0.035285
    vf_ies                 0.548513
    vf_csf                 0.416202
    sigma_myelin           0.000961
    sigma_ies              0.016912
    sigma_csf              0.144797
    Name: 44, dtype: object       


    REQUIREMENTS: 

    Parameters must be in this order: 
    - mu - in milliseconds - e.g. 23ms for short component at up to 1500 ms for long components 
    - sigma - in milliseconds - e.g. originally fixed to 5ms 
    - vf - originally set as a fraction - e.g. 0.25 for myelin - but it is multiplied by 100 - > therefore if should be in the range that generally varies from 1-100
    
    ACTIONS:

    mu_myelin              0.024113 >> mult by 1000
    mu_ies                 0.081929 >> mult by 1000
    mu_csf                 1.029994 >> mult by 1000
    vf_myelin              0.035285 >> mult by 100
    vf_ies                 0.548513 >> mult by 100
    vf_csf                 0.416202 >> mult by 100
    sigma_myelin           0.000961 >> mult by 1000
    sigma_ies              0.016912 >> mult by 1000
    sigma_csf              0.144797 >> mult by 1000    

    """

    # mult fractions by 100 (since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values)
    for p in ['vf_myelin', 'vf_ies', 'vf_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(100)
        dfs_std[p] = dfs_mean[p].copy().divide(10)
    
    for p in ['mu_myelin', 'mu_ies', 'mu_csf','sigma_myelin', 'sigma_ies', 'sigma_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(1000)
        dfs_std[p] = dfs_mean[p].copy().divide(10)

    # export parameters from pandas dataframe into a dictionary (historical compatibility reasons)
    params_all_rois = {}
    for roi in dfs_mean.index:
        
        # select dataframe series for given roi
        dfs_mean_roi = dfs_mean.loc[roi].to_dict()
        dfs_std_roi = dfs_std.loc[roi].to_dict()

        # fuse mean and std measurements together 
        dfs_roi = {}
        for k in dfs_mean_roi.keys():
            if k != 'roi_name': # avoid double copying the roi_name
                dfs_roi[k] = [dfs_mean_roi[k], dfs_std_roi[k]]
            else:
                dfs_roi[k] = dfs_mean_roi[k]
        
        params_all_rois[roi] = dfs_roi

    # add background priors 
    params_all_rois = add_background_priors(params_all_rois, background_label=0)

    opt['prior_means'], opt['prior_stds'],roi_map = convert_priors_to_synthseg_format_gauss(params_all_rois, opt)


    # set generation labels 
    # make sure that the order of labels is the same as are the columns of prior_means and prior_sts with their corresponding rois
    #opt['generation_labels'] = np.array([1,2,3,4,5])
    opt['generation_labels'] = np.array([int(i) for i in list(roi_map.keys())]) # the keys of roi_map dict contains the order, which we must present as a np.array of ints to the synthseg algorithm
    
    # a couple of checks to make sure the mapping is correct after conversion
    assert opt['prior_means'][:,0][0] == params_all_rois[str(opt['generation_labels'][0])]['mu_myelin'][0]
    assert opt['prior_means'][:,-1][0] == params_all_rois[str(opt['generation_labels'][-1])]['mu_myelin'][0]    

    # if priors need to be verified
    if 'check_priors' in opt and opt['check_priors']:
        assert 'roi_name_ids' in opt
        check_priors(params_all_rois, opt['roi_name_ids'])

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt    

def example_29(opt): # generate brains from priors defined from inside the .pkl file - 95 ROIs

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    opt = example_20(opt)
    opt["experiment_name"] = 'test29_generate_from_pkl_priors_95_rois'



    # over ride default extra-options with the following 
    #segdir = '/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/output_4/'
    #segfile = "Hammers_mith-n30r95-MaxProbMap-full-MNI152-SPM12_resamp_reg_irtk_tr_reg_npeye.nii.gz"
    label_map = "/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/output_4/label_dir_do_not_delete/" # provide directory to multiple segmentations 
    
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':False, 
                    'vary_prior_means_and_stds':False,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':False,
                    'flipping':False, 
                    'path_label_map':label_map,
                    'prior_distribution':'normal',  # if set to normal - the variance in parameters over population will be normal (this should be default practically always - do not set this to uniform - this is where the mistake was for example_24-26 - possibly the reason why generation wasn't great)
                    'n_examples':5,
                    'output_shape':None,
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0, # vary within 5% of what is set                               
                    'save_prior_values':True, 
                    'check_priors':False,
                    'roi_name_ids':[44]}         

    """
        :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
        sampled from a uniform distribution of predefined bounds. Can either be:
        1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
        (1-scaling_bounds, 1+scaling_bounds) for each dimension.
        2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
        (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
        3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
         of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
        4) the path to such a numpy array.
        If None (default), scaling_range = 0.15
        
    
        :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
        and 2, the bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
        If None (default), rotation_bounds = 15.
        
        :param shearing_bounds: (optional) same as scaling bounds. If None (default), shearing_bounds = 0.01.
        
        :param nonlin_std: (optional) If apply_nonlin_trans is True, maximum value for the standard deviation of the
        normal distribution from which we sample the first tensor for synthesising the deformation field.
        
    
        :param nonlin_shape_factor: (optional) If apply_nonlin_trans is True, ratio between the size of the input label
        maps and the size of the sampled tensor for synthesising the deformation field.
    """
    
    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    # load priors from pickle 
    rootdir = '/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/libs/'
    path_mean = rootdir + 'test_mean_v2.pkl'
    path_std = rootdir + 'test_std_v2.pkl'   
    dfs_mean = pd.read_pickle(path_mean)
    dfs_std = pd.read_pickle(path_std)
    

    """
    Currently statistics return the following: 

    mu_myelin              0.024113
    mu_ies                 0.081929
    mu_csf                 1.029994
    vf_myelin              0.035285
    vf_ies                 0.548513
    vf_csf                 0.416202
    sigma_myelin           0.000961
    sigma_ies              0.016912
    sigma_csf              0.144797
    Name: 44, dtype: object       


    REQUIREMENTS: 

    Parameters must be in this order: 
    - mu - in milliseconds - e.g. 23ms for short component at up to 1500 ms for long components 
    - sigma - in milliseconds - e.g. originally fixed to 5ms 
    - vf - originally set as a fraction - e.g. 0.25 for myelin - but it is multiplied by 100 - > therefore if should be in the range that generally varies from 1-100
    
    ACTIONS:

    mu_myelin              0.024113 >> mult by 1000
    mu_ies                 0.081929 >> mult by 1000
    mu_csf                 1.029994 >> mult by 1000
    vf_myelin              0.035285 >> mult by 100
    vf_ies                 0.548513 >> mult by 100
    vf_csf                 0.416202 >> mult by 100
    sigma_myelin           0.000961 >> mult by 1000
    sigma_ies              0.016912 >> mult by 1000
    sigma_csf              0.144797 >> mult by 1000    

    """

    # mult fractions by 100 (since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values)
    for p in ['vf_myelin', 'vf_ies', 'vf_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(100)
        dfs_std[p] = dfs_std[p].multiply(100)        
    
    for p in ['mu_myelin', 'mu_ies', 'mu_csf','sigma_myelin', 'sigma_ies', 'sigma_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(1000)
        dfs_std[p] = dfs_std[p].multiply(1000)        

    # export parameters from pandas dataframe into a dictionary (historical compatibility reasons)
    params_all_rois = {}
    for roi in dfs_mean.index:
        
        # select dataframe series for given roi
        dfs_mean_roi = dfs_mean.loc[roi].to_dict()
        dfs_std_roi = dfs_std.loc[roi].to_dict()

        # fuse mean and std measurements together 
        dfs_roi = {}
        for k in dfs_mean_roi.keys():
            if k != 'roi_name': # avoid double copying the roi_name
                dfs_roi[k] = [dfs_mean_roi[k], dfs_std_roi[k]]
            else:
                dfs_roi[k] = dfs_mean_roi[k]
        
        params_all_rois[roi] = dfs_roi

    # add background priors 
    params_all_rois = add_background_priors(params_all_rois, background_label=0)

    opt['prior_means'], opt['prior_stds'],roi_map = convert_priors_to_synthseg_format_gauss(params_all_rois, opt)


    # set generation labels 
    # make sure that the order of labels is the same as are the columns of prior_means and prior_sts with their corresponding rois
    #opt['generation_labels'] = np.array([1,2,3,4,5])
    opt['generation_labels'] = np.array([int(i) for i in list(roi_map.keys())]) # the keys of roi_map dict contains the order, which we must present as a np.array of ints to the synthseg algorithm
    
    # a couple of checks to make sure the mapping is correct after conversion
    assert opt['prior_means'][:,0][0] == params_all_rois[str(opt['generation_labels'][0])]['mu_myelin'][0]
    assert opt['prior_means'][:,-1][0] == params_all_rois[str(opt['generation_labels'][-1])]['mu_myelin'][0]    

    # if priors need to be verified
    if 'check_priors' in opt and opt['check_priors']:
        assert 'roi_name_ids' in opt
        check_priors(params_all_rois, opt['roi_name_ids'])

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt    



def check_priors(params_all_rois, roi_name_ids):

    # the the final priors that are passed to generation algorithm 

    print(f"PRINTING FINAL PARAMETER PRIORS FOR CHOSEN ROI_NAME_IDS")

    for i in roi_name_ids:
        print(params_all_rois[str(i)])
        input('press any key to proceed')

    

def example_28(opt): # generate brains from priors defined from inside the .pkl file - 3 rois

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    

    opt = example_20(opt)
    opt["experiment_name"] = 'test28_generate_from_pkl_priors_3_rois'

    # over ride default extra-options with the following 
    label_map = '/home/ch215616/w/mwf_data/synthetic_data/training_data/segmentation_targets/single_subject/' # provide directory to multiple segmentations 
    
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':False, 
                    'vary_prior_means_and_stds':False,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':False,
                    'flipping':False, 
                    'path_label_map':label_map,
                    'prior_distribution':'normal',  # if set to normal - the variance in parameters over population will be normal (this should be default practically always - do not set this to uniform - this is where the mistake was for example_24-26 - possibly the reason why generation wasn't great)
                    'n_examples':5,
                    'output_shape':None,
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0, # vary within 5% of what is set                               
                    'save_prior_values':True}         

    """
        :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
        sampled from a uniform distribution of predefined bounds. Can either be:
        1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
        (1-scaling_bounds, 1+scaling_bounds) for each dimension.
        2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
        (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
        3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
         of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
        4) the path to such a numpy array.
        If None (default), scaling_range = 0.15
        
    
        :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
        and 2, the bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
        If None (default), rotation_bounds = 15.
        
        :param shearing_bounds: (optional) same as scaling bounds. If None (default), shearing_bounds = 0.01.
        
        :param nonlin_std: (optional) If apply_nonlin_trans is True, maximum value for the standard deviation of the
        normal distribution from which we sample the first tensor for synthesising the deformation field.
        
    
        :param nonlin_shape_factor: (optional) If apply_nonlin_trans is True, ratio between the size of the input label
        maps and the size of the sampled tensor for synthesising the deformation field.
    """
    
    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]


    # load priors from pickle 
    rootdir = '/home/ch215616/w/code/mwf/experiments/s20210624-generate-brains-from-pickle-priors/'
    path_mean = rootdir + 'test_mean_v3.pkl'
    path_std = rootdir + 'test_std_v3.pkl'    
    dfs_mean = pd.read_pickle(path_mean)
    dfs_std = pd.read_pickle(path_std)

    
    # mult fractions by 100 (since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values)
    for p in ['vf_myelin', 'vf_ies', 'vf_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(100)
        dfs_std[p] = dfs_std[p].multiply(100)        

    for p in ['mu_myelin', 'mu_ies', 'mu_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(1000)
        dfs_std[p] = dfs_std[p].multiply(1000)        


    # rename columns (historical dependency)
    dfs_mean.rename(columns={"mu_m": "mu_myelin", "vf_m":"vf_myelin", "sigma_m":"sigma_myelin"}, inplace=True)
    dfs_std.rename(columns={"mu_m": "mu_myelin", "vf_m":"vf_myelin", "sigma_m":"sigma_myelin"}, inplace=True)

    # export parameters from pandas dataframe into a dictionary (historical compatibility reasons)
    params_all_rois = {}
    for roi in dfs_mean.index:
        
        # select dataframe series for given roi
        dfs_mean_roi = dfs_mean.loc[roi].to_dict()
        dfs_std_roi = dfs_std.loc[roi].to_dict()

        # fuse mean and std measurements together 
        dfs_roi = {}
        for k in dfs_mean_roi.keys():
            if k != 'roi_name': # avoid double copying the roi_name
                dfs_roi[k] = [dfs_mean_roi[k], dfs_std_roi[k]]
            else:
                dfs_roi[k] = dfs_mean_roi[k]
        
        params_all_rois[roi] = dfs_roi

    opt['generation_labels'] = np.array([3,2,4,1,5])
    
    opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_3ROIS(params_all_rois, opt)

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt            

def add_background_priors(params_all_rois, background_label=0):

    # add priors for the background (zeros everywhere)
    
    # NB no conversion needed between gauss<>uniform as all entries are zero 
    p = {}
    p['roi_name'] = 'background'
    p['mu_myelin'] = [0.0,0.0]
    p['mu_ies'] = [0.0,0.0]
    p['mu_csf'] = [0.0,0.0]  
    p['vf_myelin'] = [0.0,0.0]
    p['vf_ies'] = [0.0,0.0]
    p['vf_csf'] = [0.0,0.0]
    p['sigma_myelin']= [0.0,0.0]
    p['sigma_ies']= [0.0,0.0]
    p['sigma_csf']= [0.0,0.0]
    #p['bias'] = [0.0,0.0]  

    params_all_rois[str(background_label)] = p

    return params_all_rois

# convert all priors into the synthnet format 
def convert_priors_to_synthseg_format_gauss_VF_CSF(params_all_rois, opt, n_features=5, add_background_priors=True):
    # USES VF_CSF instead of VF_IES for the priors

    # NB prior name of the function = from_gauss_to_gauss_abstract

    """This function takes the statistics for each MWF parameter for each ROI, 
    and converts them into an appropriate format that can be ingested by the generative algorithm (synthseg). 

    Args: 
    params_all_rois (dict): dictionary with keys equal to rois. Each item of each key is a subdictionary with myelin model parameters as keys (i.e. {'corpus_callosum':{'mu_m':None, 'mu_ies':None, ... 'sigma_csf':None}}). Each items in each subdictionary holds two values - mean and std for the given MWF model parameter for the given roi
                            e.g. roi id of '44' = corresponds to corpus callosum
                            e.g. params_all_rois['44'] = {'vf_myelin': [20.0, 4.999999999999999], 'vf_ies': [80.0, 4.999999999999999], 'mu_myelin': [25.0, 0.0], 'mu_ies': [80.0, 0.0], 'mu_csf': [1750.0, 125.0]}
    add_background_priors (bool): if set to true, we will add an additional keys to params_all_rois dictionary that would be equivalent to the background 
    n_features (int): number of MWF parameters that we are simulating (e.g. mu_m,mu_ies,mu_csf,vf_m,vf_ies = 5 features)
    
    """

    # population variance for volume fraction and meanT2
    volumefraction_population_variance = opt['vf_pop_var']
    mu_population_variance = opt['mu_pop_var']

    # extract the number of ROIs that must be generated 
    n_rois = len(params_all_rois)    

    # priors are sets of tuples given to the generation script for each feature (e.g. 'vf_myelin') and for each ROI (e.g. 'corpus callosum') - from which the gaussians are sampled. Tuples correspond to mean and std for each particular parameter.
    prior_dimensions = n_features*2

    # setup empty numpy tensors with correct dimensions. The columns correspond to rois. 
    prior_means = np.zeros((prior_dimensions,n_rois))  # 5 is the number of regions of interest
    prior_stds = np.zeros((prior_dimensions,n_rois))

    # create a mapping between ordering of keys in a params dictionary (which correspond to ROI) and an index that will correspond to a column in a numpy array (necessary for generating priors)
    roi_name_ids = params_all_rois.keys()
    roi_nparray_column_index = range(0,len(params_all_rois.keys())) 
    assert len(roi_name_ids) == len(roi_nparray_column_index)
    roi_map = {k:v for k,v in zip(roi_name_ids, roi_nparray_column_index)}

    # iterate over each roi to fill in output prior_means and prior_stds (numpy format is required by the generative algorithm)
    for roi_name_id, column in roi_map.items():

        # get params for this roi
        params = params_all_rois[roi_name_id]  
        
        prior_means_column = prior_means[:,column]
        prior_stds_column = prior_stds[:,column]

        # assuming no subject to subject variation -> prior_means_odd = prior_means_even, prior_stds_odd = prior_stds_even
        prior_means_column[0] = params['mu_myelin'][0]   # accessing mean measure of mu_myelin
        prior_means_column[1] = params['mu_myelin'][0]*mu_population_variance   # accessing variance of the mean measure of mu_myelin

        prior_means_column[2] = params['mu_ies'][0] 
        prior_means_column[3] = params['mu_ies'][0]*mu_population_variance

        prior_means_column[4] = params['mu_csf'][0]
        prior_means_column[5] = params['mu_csf'][0]*mu_population_variance

        prior_means_column[6] = params['vf_myelin'][0]
        prior_means_column[7] = params['vf_myelin'][0]*volumefraction_population_variance

        prior_means_column[8] = params['vf_csf'][0]
        prior_means_column[9] = params['vf_csf'][0]*volumefraction_population_variance

        prior_stds_column[0] = params['mu_myelin'][1] # setting the intra-subject intra-roi variance of mu_myelin for this specific roi 
        prior_stds_column[1] = 0 # setting population variance for the variance measure of mu_myelin to zero 
        prior_stds_column[2] = params['mu_ies'][1]
        prior_stds_column[3] = 0
        prior_stds_column[4] = params['mu_csf'][1]
        prior_stds_column[5] = 0
        prior_stds_column[6] = params['vf_myelin'][1]
        prior_stds_column[7] = 0
        prior_stds_column[8] = params['vf_csf'][1]
        prior_stds_column[9] = 0

        # fill (put back the column)
        prior_means[:,column] = prior_means_column
        prior_stds[:,column] = prior_stds_column


    return prior_means, prior_stds,roi_map
                 


# convert all priors into the synthnet format 
def convert_priors_to_synthseg_format_gauss(params_all_rois, opt, n_features=5, add_background_priors=True):
    # NB prior name of the function = from_gauss_to_gauss_abstract

    """This function takes the statistics for each MWF parameter for each ROI, 
    and converts them into an appropriate format that can be ingested by the generative algorithm (synthseg). 

    Args: 
    params_all_rois (dict): dictionary with keys equal to rois. Each item of each key is a subdictionary with myelin model parameters as keys (i.e. {'corpus_callosum':{'mu_m':None, 'mu_ies':None, ... 'sigma_csf':None}}). Each items in each subdictionary holds two values - mean and std for the given MWF model parameter for the given roi
                            e.g. roi id of '44' = corresponds to corpus callosum
                            e.g. params_all_rois['44'] = {'vf_myelin': [20.0, 4.999999999999999], 'vf_ies': [80.0, 4.999999999999999], 'mu_myelin': [25.0, 0.0], 'mu_ies': [80.0, 0.0], 'mu_csf': [1750.0, 125.0]}
    add_background_priors (bool): if set to true, we will add an additional keys to params_all_rois dictionary that would be equivalent to the background 
    n_features (int): number of MWF parameters that we are simulating (e.g. mu_m,mu_ies,mu_csf,vf_m,vf_ies = 5 features)
    
    """

    # population variance for volume fraction and meanT2
    volumefraction_population_variance = opt['vf_pop_var']
    mu_population_variance = opt['mu_pop_var']

    # extract the number of ROIs that must be generated 
    n_rois = len(params_all_rois)    

    # priors are sets of tuples given to the generation script for each feature (e.g. 'vf_myelin') and for each ROI (e.g. 'corpus callosum') - from which the gaussians are sampled. Tuples correspond to mean and std for each particular parameter.
    prior_dimensions = n_features*2

    # setup empty numpy tensors with correct dimensions. The columns correspond to rois. 
    prior_means = np.zeros((prior_dimensions,n_rois))  # 5 is the number of regions of interest
    prior_stds = np.zeros((prior_dimensions,n_rois))

    # create a mapping between ordering of keys in a params dictionary (which correspond to ROI) and an index that will correspond to a column in a numpy array (necessary for generating priors)
    roi_name_ids = params_all_rois.keys()
    roi_nparray_column_index = range(0,len(params_all_rois.keys())) 
    assert len(roi_name_ids) == len(roi_nparray_column_index)
    roi_map = {k:v for k,v in zip(roi_name_ids, roi_nparray_column_index)}

    # iterate over each roi to fill in output prior_means and prior_stds (numpy format is required by the generative algorithm)
    for roi_name_id, column in roi_map.items():

        # get params for this roi
        params = params_all_rois[roi_name_id]  
        
        prior_means_column = prior_means[:,column]
        prior_stds_column = prior_stds[:,column]

        # assuming no subject to subject variation -> prior_means_odd = prior_means_even, prior_stds_odd = prior_stds_even
        prior_means_column[0] = params['mu_myelin'][0]   # accessing mean measure of mu_myelin
        prior_means_column[1] = params['mu_myelin'][0]*mu_population_variance   # accessing variance of the mean measure of mu_myelin

        prior_means_column[2] = params['mu_ies'][0] 
        prior_means_column[3] = params['mu_ies'][0]*mu_population_variance

        prior_means_column[4] = params['mu_csf'][0]
        prior_means_column[5] = params['mu_csf'][0]*mu_population_variance

        prior_means_column[6] = params['vf_myelin'][0]
        prior_means_column[7] = params['vf_myelin'][0]*volumefraction_population_variance

        prior_means_column[8] = params['vf_ies'][0]
        prior_means_column[9] = params['vf_ies'][0]*volumefraction_population_variance

        prior_stds_column[0] = params['mu_myelin'][1] # setting the intra-subject intra-roi variance of mu_myelin for this specific roi 
        prior_stds_column[1] = 0 # setting population variance for the variance measure of mu_myelin to zero 
        prior_stds_column[2] = params['mu_ies'][1]
        prior_stds_column[3] = 0
        prior_stds_column[4] = params['mu_csf'][1]
        prior_stds_column[5] = 0
        prior_stds_column[6] = params['vf_myelin'][1]
        prior_stds_column[7] = 0
        prior_stds_column[8] = params['vf_ies'][1]
        prior_stds_column[9] = 0

        # fill (put back the column)
        prior_means[:,column] = prior_means_column
        prior_stds[:,column] = prior_stds_column


    return prior_means, prior_stds,roi_map
                 



def convert_priors_to_synthseg_format_gauss_3ROIS(params_all_rois, opt, n_features=5, add_background_priors=True):
    # function for historical backcompatibility when fitting with 3 rois only - background is set to 1 (instead of 0) label, and ventricles are assigned copy of the same parameters as csf

    # NB prior name of the function = from_gauss_to_gauss_abstract

    """This function takes the statistics for each MWF parameter for each ROI, 
    and converts them into an appropriate format that can be ingested by the generative algorithm (synthseg). 

    Args: 
    params_all_rois (dict): dictionary with keys equal to rois. Each item of each key is a subdictionary with myelin model parameters as keys (i.e. {'corpus_callosum':{'mu_m':None, 'mu_ies':None, ... 'sigma_csf':None}}). Each items in each subdictionary holds two values - mean and std for the given MWF model parameter for the given roi
                            e.g. roi id of '44' = corresponds to corpus callosum
                            e.g. params_all_rois['44'] = {'vf_myelin': [20.0, 4.999999999999999], 'vf_ies': [80.0, 4.999999999999999], 'mu_myelin': [25.0, 0.0], 'mu_ies': [80.0, 0.0], 'mu_csf': [1750.0, 125.0]}
    add_background_priors (bool): if set to true, we will add an additional keys to params_all_rois dictionary that would be equivalent to the background 
    n_features (int): number of MWF parameters that we are simulating (e.g. mu_m,mu_ies,mu_csf,vf_m,vf_ies = 5 features)
    
    """

    # check if background labels need to be added
    if add_background_priors:
        
        # NB no conversion needed between gauss<>uniform as all entries are zero 
        p = {}
        p['roi_name'] = 'background'
        p['mu_myelin'] = [0.0,0.0]
        p['mu_ies'] = [0.0,0.0]
        p['mu_csf'] = [0.0,0.0]  
        p['vf_myelin'] = [0.0,0.0]
        p['vf_ies'] = [0.0,0.0]
        p['sigma_myelin']= [0.0,0.0]
        p['sigma_ies']= [0.0,0.0]
        p['sigma_csf']= [0.0,0.0]
        
        #p['bias'] = [0.0,0.0]  

        params_all_rois['1'] = p

    # add ventricle priors also - copy values from csf 
    params_all_rois['5'] = params_all_rois['3'].copy()
    params_all_rois['5']['roi_name'] = 'ventricles'

    # population variance for volume fraction and meanT2
    volumefraction_population_variance = opt['vf_pop_var']
    mu_population_variance = opt['mu_pop_var']

    # extract the number of ROIs that must be generated 
    n_rois = len(params_all_rois)    

    # priors are sets of tuples given to the generation script for each feature (e.g. 'vf_myelin') and for each ROI (e.g. 'corpus callosum') - from which the gaussians are sampled. Tuples correspond to mean and std for each particular parameter.
    prior_dimensions = n_features*2

    # setup empty numpy tensors with correct dimensions. The columns correspond to rois. 
    prior_means = np.zeros((prior_dimensions,n_rois))  # 5 is the number of regions of interest
    prior_stds = np.zeros((prior_dimensions,n_rois))

    # create a mapping between ordering of keys in a params dictionary (which correspond to ROI) and an index that will correspond to a column in a numpy array (necessary for generating priors)
    roi_name_ids = params_all_rois.keys()
    roi_nparray_column_index = range(0,len(params_all_rois.keys())) 
    assert len(roi_name_ids) == len(roi_nparray_column_index)
    roi_map = {k:v for k,v in zip(roi_name_ids, roi_nparray_column_index)}

    # iterate over each roi to fill in output prior_means and prior_stds (numpy format is required by the generative algorithm)
    for roi_name_id, column in roi_map.items():

        # get params for this roi
        params = params_all_rois[roi_name_id]  
        
        prior_means_column = prior_means[:,column]
        prior_stds_column = prior_stds[:,column]

        # assuming no subject to subject variation -> prior_means_odd = prior_means_even, prior_stds_odd = prior_stds_even
        prior_means_column[0] = params['mu_myelin'][0]   # accessing mean measure of mu_myelin
        prior_means_column[1] = params['mu_myelin'][0]*mu_population_variance   # accessing variance of the mean measure of mu_myelin

        prior_means_column[2] = params['mu_ies'][0] 
        prior_means_column[3] = params['mu_ies'][0]*mu_population_variance

        prior_means_column[4] = params['mu_csf'][0]
        prior_means_column[5] = params['mu_csf'][0]*mu_population_variance

        prior_means_column[6] = params['vf_myelin'][0]
        prior_means_column[7] = params['vf_myelin'][0]*volumefraction_population_variance

        prior_means_column[8] = params['vf_ies'][0]
        prior_means_column[9] = params['vf_ies'][0]*volumefraction_population_variance

        prior_stds_column[0] = params['mu_myelin'][1] # setting the intra-subject intra-roi variance of mu_myelin for this specific roi 
        prior_stds_column[1] = 0 # setting population variance for the variance measure of mu_myelin to zero 
        prior_stds_column[2] = params['mu_ies'][1]
        prior_stds_column[3] = 0
        prior_stds_column[4] = params['mu_csf'][1]
        prior_stds_column[5] = 0
        prior_stds_column[6] = params['vf_myelin'][1]
        prior_stds_column[7] = 0
        prior_stds_column[8] = params['vf_ies'][1]
        prior_stds_column[9] = 0

        # fill (put back the column)
        prior_means[:,column] = prior_means_column
        prior_stds[:,column] = prior_stds_column


    return prior_means, prior_stds

def example_27(opt): # no changes to nonlin deformations - just the params

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    

    opt = example_22(opt)
    opt["experiment_name"] = 'test27_totally_unrealistic_parameters_n_spatial_deformations'


    # over ride default extra-options with the following 
    label_map = '/home/ch215616/w/mwf_data/synthetic_data/training_data/segmentation_targets/single_subject/' # provide directory to multiple segmentations 

    ### OLD _ example 20 
    
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True, 
                    'path_label_map':label_map,
                    'prior_distribution':'normal',
                    'n_examples':100,
                    'output_shape':None,
                    'vf_pop_var':0.0,
                    'save_prior_values':True}   
    
    
    
    ### OLD _ example 22 
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True, 
                    'path_label_map':label_map,
                    'prior_distribution':'normal',
                    'n_examples':100,
                    'output_shape':None,
                    'vf_pop_var':0.2, # vary within 10% of what is set 
                    'mu_pop_var':0.2, # vary within 5% of what is set 
                    'save_prior_values':True}   
    
    
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':False,
                    'blurring':False, 
                    'vary_prior_means_and_stds':False,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':True,
                    'flipping':False, 
                    'path_label_map':label_map,
                    'prior_distribution':'uniform',
                    'n_examples':4,
                    'output_shape':None,
                    'vf_pop_var':0.0, # vary within 10% of what is set 
                    'mu_pop_var':0.0, # vary within 5% of what is set 
                    'scaling_bounds':0.0, #3.,
                    'rotation_bounds':0.0, #3.,
                    'shearing_bounds':0.0, #3.,                              
                    'nonlin_std':0.0, #3.,
                    'nonlin_shape_factor':0.0625,                                 
                    'save_prior_values':True}       

    """
        :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
        sampled from a uniform distribution of predefined bounds. Can either be:
        1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
        (1-scaling_bounds, 1+scaling_bounds) for each dimension.
        2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
        (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
        3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
         of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
        4) the path to such a numpy array.
        If None (default), scaling_range = 0.15
        
    
        :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
        and 2, the bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
        If None (default), rotation_bounds = 15.
        
        :param shearing_bounds: (optional) same as scaling bounds. If None (default), shearing_bounds = 0.01.
        
        :param nonlin_std: (optional) If apply_nonlin_trans is True, maximum value for the standard deviation of the
        normal distribution from which we sample the first tensor for synthesising the deformation field.
        
    
        :param nonlin_shape_factor: (optional) If apply_nonlin_trans is True, ratio between the size of the input label
        maps and the size of the sampled tensor for synthesising the deformation field.
    """
    
    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]


        
    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.05,0.5]  # reduced from 0.1 - 0.3  # increated back to previous amount
    p['vf_ies'] = [0.6,0.95]  # reduced from 0.7 - 0.9  # increated back to previous amount
    p['mu_myelin'] = [10,50]  # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [60,100]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.05]
    p['vf_ies'] = [0.8,1]
    p['mu_myelin'] = [10,50] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [80,140] # set 4ms variance. And changed from previous value of 80-80 (same as Wm) # increased to 20ms variance 
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.05]
    p['vf_ies'] = [0,0.05]
    p['mu_myelin'] = [10,50] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [60,100]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p)#, 'WM')


    # multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
    def mult_vol_fraction(p,mult=100):
        p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
        p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
        return p

    params_wm = mult_vol_fraction(params_wm)
    params_gm = mult_vol_fraction(params_gm)
    params_csf = mult_vol_fraction(params_csf)
        
    if opt['prior_distribution'] == 'uniform':
        # create prior_means and prior_stds vectors
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)
    else:
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,opt['vf_pop_var'],opt['mu_pop_var'])
        
        
    return opt


def example_26(opt):

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    

    opt = example_22(opt)
    opt["experiment_name"] = 'test26_totally_unrealistic_parameters_n_spatial_deformations'


    # over ride default extra-options with the following 
    label_map = '/home/ch215616/w/mwf_data/synthetic_data/training_data/segmentation_targets/' # provide directory to multiple segmentations 
    
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True, 
                    'path_label_map':label_map,
                    'prior_distribution':'uniform',
                    'n_examples':10,
                    'output_shape':None,
                    'vf_pop_var':0.8, # vary within 10% of what is set 
                    'mu_pop_var':0.8, # vary within 5% of what is set 
                    'nonlin_std':1000., #3.,
                    'nonlin_shape_factor':0.0625,                                 
                    'save_prior_values':True}       

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]


        
    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.05,0.5]  # reduced from 0.1 - 0.3  # increated back to previous amount
    p['vf_ies'] = [0.6,0.95]  # reduced from 0.7 - 0.9  # increated back to previous amount
    p['mu_myelin'] = [10,50]  # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [60,100]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.05]
    p['vf_ies'] = [0.8,1]
    p['mu_myelin'] = [10,50] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [80,140] # set 4ms variance. And changed from previous value of 80-80 (same as Wm) # increased to 20ms variance 
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.05]
    p['vf_ies'] = [0,0.05]
    p['mu_myelin'] = [10,50] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [60,100]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p)#, 'WM')


    # multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
    def mult_vol_fraction(p,mult=100):
        p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
        p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
        return p

    params_wm = mult_vol_fraction(params_wm)
    params_gm = mult_vol_fraction(params_gm)
    params_csf = mult_vol_fraction(params_csf)
        
    if opt['prior_distribution'] == 'uniform':
        # create prior_means and prior_stds vectors
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)
    else:
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,opt['vf_pop_var'],opt['mu_pop_var'])
        
        
    return opt



def example_25(opt): # VERY LITTLE VARIANCE - due to low nonlin_shape_factor

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    

    opt = example_22(opt)
    opt["experiment_name"] = 'test25_totally_unrealistic_parameters_n_spatial_deformations'


    # over ride default extra-options with the following 
    label_map = '/home/ch215616/w/mwf_data/synthetic_data/training_data/segmentation_targets/' # provide directory to multiple segmentations 
    
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True, 
                    'path_label_map':label_map,
                    'prior_distribution':'uniform',
                    'n_examples':2,
                    'output_shape':None,
                    'vf_pop_var':0.3, # vary within 10% of what is set 
                    'mu_pop_var':0.5, # vary within 5% of what is set 
                    'nonlin_std':15., #3.,
                    'nonlin_shape_factor':0.0000625,    #.0625,                                 
                    'save_prior_values':True}       

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]


        
    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.05,0.5]  # reduced from 0.1 - 0.3  # increated back to previous amount
    p['vf_ies'] = [0.6,0.95]  # reduced from 0.7 - 0.9  # increated back to previous amount
    p['mu_myelin'] = [10,50]  # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [60,100]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.05]
    p['vf_ies'] = [0.8,1]
    p['mu_myelin'] = [10,50] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [80,140] # set 4ms variance. And changed from previous value of 80-80 (same as Wm) # increased to 20ms variance 
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.05]
    p['vf_ies'] = [0,0.05]
    p['mu_myelin'] = [10,50] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [60,100]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p)#, 'WM')


    # multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
    def mult_vol_fraction(p,mult=100):
        p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
        p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
        return p

    params_wm = mult_vol_fraction(params_wm)
    params_gm = mult_vol_fraction(params_gm)
    params_csf = mult_vol_fraction(params_csf)
        
    if opt['prior_distribution'] == 'uniform':
        # create prior_means and prior_stds vectors
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)
    else:
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,opt['vf_pop_var'],opt['mu_pop_var'])
        
        
    return opt


def example_24(opt):

    # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
    

    opt = example_22(opt)
    opt["experiment_name"] = 'test24_totally_unrealistic_parameters_n_spatial_deformations'


    # over ride default extra-options with the following 
    label_map = '/home/ch215616/w/mwf_data/synthetic_data/training_data/segmentation_targets/' # provide directory to multiple segmentations 
    
    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True, 
                    'path_label_map':label_map,
                    'prior_distribution':'uniform',
                    'n_examples':2,
                    'output_shape':None,
                    'vf_pop_var':0.3, # vary within 10% of what is set 
                    'mu_pop_var':0.5, # vary within 5% of what is set 
                    'nonlin_std':15., #3.,
                    'nonlin_shape_factor':2.0625,    #.0625,                                 
                    'save_prior_values':True}       

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]


        
    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.05,0.5]  # reduced from 0.1 - 0.3  # increated back to previous amount
    p['vf_ies'] = [0.6,0.95]  # reduced from 0.7 - 0.9  # increated back to previous amount
    p['mu_myelin'] = [10,50]  # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [60,100]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.05]
    p['vf_ies'] = [0.8,1]
    p['mu_myelin'] = [10,50] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [80,140] # set 4ms variance. And changed from previous value of 80-80 (same as Wm) # increased to 20ms variance 
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.05]
    p['vf_ies'] = [0,0.05]
    p['mu_myelin'] = [10,50] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [60,100]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p)#, 'WM')


    # multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
    def mult_vol_fraction(p,mult=100):
        p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
        p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
        return p

    params_wm = mult_vol_fraction(params_wm)
    params_gm = mult_vol_fraction(params_gm)
    params_csf = mult_vol_fraction(params_csf)
        
    if opt['prior_distribution'] == 'uniform':
        # create prior_means and prior_stds vectors
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)
    else:
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,opt['vf_pop_var'],opt['mu_pop_var'])
        
        
    return opt


def example_23(opt):

    # repeat example22 for 2 volumes as dummy test
    

    opt = example_22(opt)
    opt["experiment_name"] = 'test23_repeat_test22_w_2_volumes_only'


    # over ride default extra-options with the following 
    label_map = '/home/ch215616/abd/mwf_data/synthetic_data/training_data/segmentation_targets/' # provide directory to multiple segmentations 

    # over ride default extra-options with the following 
    modified_opt = {'path_label_map':label_map, 
        'n_examples':2}   

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    return opt




def example_22(opt):

    # the purpose of this test is to generate the final set of parameter maps to test over but with much more variance 
    # BOTH for intra-ROI parameter variation AND between subject variance (which was set to no vary until now)

    opt = example_20(opt)
    opt["experiment_name"] = 'test22_generate_100volumes_more_intraROI_variance_and_pop_variance'


    # over ride default extra-options with the following 
    label_map = '/home/ch215616/code/SynthSeg/sv407_ismrm/segmentation_targets/' # provide directory to multiple segmentations 

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True, 
                    'path_label_map':label_map,
                    'prior_distribution':'normal',
                    'n_examples':100,
                    'output_shape':None,
                    'vf_pop_var':0.2, # vary within 10% of what is set 
                    'mu_pop_var':0.2, # vary within 5% of what is set 
                    'save_prior_values':True}   

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]

    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.1,0.3]  # reduced from 0.1 - 0.3  # increated back to previous amount
    p['vf_ies'] = [0.7,0.9]  # reduced from 0.7 - 0.9  # increated back to previous amount
    p['mu_myelin'] = [20,30]  # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [75,85]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0.9,1]
    p['mu_myelin'] = [20,30] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [100,120] # set 4ms variance. And changed from previous value of 80-80 (same as Wm) # increased to 20ms variance 
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0,0.1]
    p['mu_myelin'] = [20,30] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [75,85]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p)#, 'WM')


    # multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
    def mult_vol_fraction(p,mult=100):
        p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
        p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
        return p

    params_wm = mult_vol_fraction(params_wm)
    params_gm = mult_vol_fraction(params_gm)
    params_csf = mult_vol_fraction(params_csf)



    if opt['prior_distribution'] == 'uniform':
        # create prior_means and prior_stds vectors
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)
    else:
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,opt['vf_pop_var'],opt['mu_pop_var'])


    return opt


def example_21(opt):

    # the purpose of this test is to generate the final set of parameter maps to test over but with much more variance then example20

    opt = example_20(opt)
    opt["experiment_name"] = 'test21_generate_100volumes_more_intraROI_variance'


    # over ride default extra-options with the following 
    label_map = '/home/ch215616/code/SynthSeg/sv407_ismrm/segmentation_targets/' # provide directory to multiple segmentations 

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True, 
                    'path_label_map':label_map,
                    'prior_distribution':'normal',
                    'n_examples':100,
                    'output_shape':None,
                    'vf_pop_var':0.0,
                    'save_prior_values':True}   


    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.1,0.3]  # reduced from 0.1 - 0.3  # increated back to previous amount
    p['vf_ies'] = [0.7,0.9]  # reduced from 0.7 - 0.9  # increated back to previous amount
    p['mu_myelin'] = [20,30]  # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [75,85]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0.9,1]
    p['mu_myelin'] = [20,30] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [100,120] # set 4ms variance. And changed from previous value of 80-80 (same as Wm) # increased to 20ms variance 
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0,0.1]
    p['mu_myelin'] = [20,30] # set 4ms variance. Before it was 25-25 # increased to 10ms variance 
    p['mu_ies'] = [75,85]  # set 4ms variance. Before it was 80,80 # increased to 10ms variance 
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p)#, 'WM')


    # multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
    def mult_vol_fraction(p,mult=100):
        p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
        p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
        return p

    params_wm = mult_vol_fraction(params_wm)
    params_gm = mult_vol_fraction(params_gm)
    params_csf = mult_vol_fraction(params_csf)


    if opt['prior_distribution'] == 'uniform':
        # create prior_means and prior_stds vectors
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)
    else:
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,opt['vf_pop_var'])



    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt


def example_20(opt):

    # the purpose of this test is to generate the final set of parameter maps to test over 

    opt = example_17(opt)
    opt["experiment_name"] = 'test20_generate_100volumes'


    # over ride default extra-options with the following 
    label_map = '/home/ch215616/code/SynthSeg/sv407_ismrm/segmentation_targets/' # provide directory to multiple segmentations 

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':False,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True, 
                    'path_label_map':label_map,
                    'prior_distribution':'normal',
                    'n_examples':100,
                    'output_shape':None,
                    'vf_pop_var':0.0,
                    'save_prior_values':True}   


    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt



def generate(datadir):

    #############################################################
    # default generation controls (overridden by `example_latest` function if necessary )
    opt = {'sample_gmm':True,
                    'blurring':True, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':True,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True}

    opt['path_label_map'] = datadir+'example.nii.gz'
    opt['prior_distribution'] = 'normal'
    opt['generation_labels'] = np.array([1,2,3,4,5])
    opt['n_examples'] = 3
    opt['output_shape'] = None 
    opt['subject_count'] = 0 

    opt = example_latest(opt)
    #############################################################


    # update datadir if experiment name is specified 
    opt['datadir'] = datadir+opt["experiment_name"]+'/' if opt["experiment_name"] else datadir
    
    # load options from file if set to True
    opt = load_from_file(opt)

    # convert priors to numpy arrays
    if not isinstance(opt['prior_means'], np.ndarray): 
        opt['prior_means'] = np.array(opt['prior_means'])
    if not isinstance(opt['prior_stds'], np.ndarray): 
        opt['prior_stds'] = np.array(opt['prior_stds'])

    # print warning messages
    for k,v in opt.items():
        if isinstance(v,bool):# only select binary options 
            if not v: 
                print(f"WARNING: {k} is turned off.")
    

    # create dir if not present  
    os.makedirs(opt['datadir'],exist_ok=True)
    print(f"Saving result to: {opt['datadir']}")

    # save to file 
    save_to_file(opt)

    return opt, opt['datadir']

def load_from_file(opt):

    if opt["load_from_file"]:
        opt_ = sv.read_from_json(opt['datadir'] + "opt.json")

        # turn lists into np arrays 
        for k in opt:
            if isinstance(opt[k],list):
                # turn into list
                opt[k] = np.array(opt[k])
    
    return opt


def save_to_file(opt):

    opt_copy = opt.copy() # create a copy so that we don't change values to 'list' in actual array
    for k in opt_copy:
        if isinstance(opt_copy[k],np.ndarray):
            # turn numpy arrays into list (cannot save numpy in dictionary)
            opt_copy[k] = opt_copy[k].tolist()

    filename = opt['datadir'] + "opt.json"
    sv.write_to_json(opt_copy,filename)



def uniform_to_gauss(min_max,confidence=0.95):
    # take a uniform range's min_max and convert to mean and std of the gaussian 
    mean = np.mean(min_max).tolist()

    if confidence == 0.95: 
        std = (min_max[1]-mean)/2 
    elif confidence == 0.997:
        print('Setting the confidence in the range of  the provided parameter values to be within 99.7 percent of the time ')
        std = (min_max[1]-mean)/3 


    return [mean, std] 
    
def convert_params(params,confidence=0.95):#, ROI):
    # convert parameters from uniform to gauss 
    for k in params: 
        params[k] = uniform_to_gauss(params[k],confidence=0.95)

    # params['ROI'] = ROI
    return params 


def fill_prior_column(params, column,prior_means,prior_stds):
    
    prior_means_column = prior_means[:,column]
    prior_stds_column = prior_stds[:,column]


    # assuming no subject to subject variation -> prior_means_odd = prior_means_even, prior_stds_odd = prior_stds_even

    prior_means_column[0] = params['mu_myelin'][0]
    prior_means_column[1] = params['mu_myelin'][0]
    prior_means_column[2] = params['mu_ies'][0]
    prior_means_column[3] = params['mu_ies'][0]
    prior_means_column[4] = params['mu_csf'][0]
    prior_means_column[5] = params['mu_csf'][0]        
    prior_means_column[6] = params['vf_myelin'][0]
    prior_means_column[7] = params['vf_myelin'][0]
    prior_means_column[8] = params['vf_ies'][0]
    prior_means_column[9] = params['vf_ies'][0]                

    prior_stds_column[0] = params['mu_myelin'][1]
    prior_stds_column[1] = params['mu_myelin'][1]
    prior_stds_column[2] = params['mu_ies'][1]
    prior_stds_column[3] = params['mu_ies'][1]
    prior_stds_column[4] = params['mu_csf'][1]
    prior_stds_column[5] = params['mu_csf'][1]        
    prior_stds_column[6] = params['vf_myelin'][1]
    prior_stds_column[7] = params['vf_myelin'][1]
    prior_stds_column[8] = params['vf_ies'][1]
    prior_stds_column[9] = params['vf_ies'][1]                        
    
    # fill put back the column 
    prior_means[:,column] = prior_means_column
    prior_stds[:,column] = prior_stds_column

    return prior_means,prior_stds




def fill_prior_column_gauss(params, column,prior_means,prior_stds,volumefraction_population_variance=0.0,mu_population_variance=0.0):
    
    prior_means_column = prior_means[:,column]
    prior_stds_column = prior_stds[:,column]


    # assuming no subject to subject variation -> prior_means_odd = prior_means_even, prior_stds_odd = prior_stds_even

    prior_means_column[0] = params['mu_myelin'][0]
    prior_means_column[1] = params['mu_myelin'][0]*mu_population_variance

    prior_means_column[2] = params['mu_ies'][0]
    prior_means_column[3] = params['mu_ies'][0]*mu_population_variance

    prior_means_column[4] = params['mu_csf'][0]
    prior_means_column[5] = params['mu_csf'][0]*mu_population_variance

    prior_means_column[6] = params['vf_myelin'][0]
    prior_means_column[7] = params['vf_myelin'][0]*volumefraction_population_variance

    prior_means_column[8] = params['vf_ies'][0]
    prior_means_column[9] = params['vf_ies'][0]*volumefraction_population_variance
    
    if prior_means_column.shape[0]==12:
        # fill 10n bias 1
        prior_means_column[10] = params['bias'][0]
        prior_means_column[11] = 0



    prior_stds_column[0] = params['mu_myelin'][1]
    prior_stds_column[1] = 0
    prior_stds_column[2] = params['mu_ies'][1]
    prior_stds_column[3] = 0
    prior_stds_column[4] = params['mu_csf'][1]
    prior_stds_column[5] = 0
    prior_stds_column[6] = params['vf_myelin'][1]
    prior_stds_column[7] = 0
    prior_stds_column[8] = params['vf_ies'][1]
    prior_stds_column[9] = 0
    if prior_stds_column.shape[0]==12:
        # fill 10n bias 1
        prior_stds_column[10] = params['bias'][1]
        prior_stds_column[11] = 0    
    
    # fill put back the column 
    prior_means[:,column] = prior_means_column
    prior_stds[:,column] = prior_stds_column

    return prior_means,prior_stds

def convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,volumefraction_population_variance=0,mu_population_variance=0, B1_prior=False):
    
    """This function takes range inputs for each parameter 
    and spits out the mean and standard deviation that each measurements should take 

    It assumes no variation between subjects... 

    If variation between subjects is necessary - change the vectors that it spits out a little bit...
    (but do it in a separate function )

    """

    ###################################################### SPATIALLY varying parameter range ######################################################

    np.set_printoptions(suppress=True)

    
    ################## VEN ##################
    params_ven = params_csf.copy()

    ################## BACKGROUND ##################
    p = {}
    p['vf_myelin'] = [0.0,0.0]
    p['vf_ies'] = [0.0,0.0]
    p['mu_myelin'] = [0.0,0.0]
    p['mu_ies'] = [0.0,0.0]
    p['mu_csf'] = [0.0,0.0]  
    p['bias'] = [0.0,0.0]  

    params_bg = convert_params(p)#, 'WM')

        
    # init priors 
    prior_dims = 10 if not B1_prior else 12  # prior_dims should be equal to the number of mwf parameters that we must simulate - e.g. mu_myelin, mu_ies, mu_csf, vf_myelin, vf_ies -> 5 x 2  - 2 is required as it refers to mean and std of each parameter in this range
                                             # note that we add additional prior for B1 variation - should be removed as it is mostly not used 
    prior_means = np.zeros((prior_dims,5))  # 5 is the number of regions of interest
    prior_stds = np.zeros((prior_dims,5))

    
    # fill in the values - should be repeated over each ROI
    prior_means,prior_stds = fill_prior_column_gauss(params_bg, 0,prior_means,prior_stds,volumefraction_population_variance,mu_population_variance)
    prior_means,prior_stds = fill_prior_column_gauss(params_gm, 1,prior_means,prior_stds,volumefraction_population_variance,mu_population_variance)
    prior_means,prior_stds = fill_prior_column_gauss(params_csf, 2,prior_means,prior_stds,volumefraction_population_variance,mu_population_variance)
    prior_means,prior_stds = fill_prior_column_gauss(params_wm, 3,prior_means,prior_stds,volumefraction_population_variance,mu_population_variance)
    prior_means,prior_stds = fill_prior_column_gauss(params_ven, 4,prior_means,prior_stds,volumefraction_population_variance,mu_population_variance)


    return prior_means, prior_stds




def convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf):
    
    """This function takes range inputs for each parameter 
    and spits out the mean and standard deviation that each measurements should take 

    It assumes no variation between subjects... 

    If variation between subjects is necessary - change the vectors that it spits out a little bit...
    (but do it in a separate function )

    """

    ###################################################### SPATIALLY varying parameter range ######################################################

    np.set_printoptions(suppress=True)

    
    ################## VEN ##################
    params_ven = params_csf.copy()

    ################## BACKGROUND ##################
    p = {}
    p['vf_myelin'] = [1.0,1.0]
    p['vf_ies'] = [0.0,0.0]
    p['mu_myelin'] = [0.0,0.0]
    p['mu_ies'] = [0.0,0.0]
    p['mu_csf'] = [0.0,0.0]  

    params_bg = convert_params(p)#, 'WM')

    # init priors 
    prior_means = np.zeros((10,5))
    prior_stds = np.zeros((10,5))

    
    # fill in the values 
    prior_means,prior_stds = fill_prior_column(params_bg, 0,prior_means,prior_stds)
    prior_means,prior_stds = fill_prior_column(params_gm, 1,prior_means,prior_stds)
    prior_means,prior_stds = fill_prior_column(params_csf, 2,prior_means,prior_stds)
    prior_means,prior_stds = fill_prior_column(params_wm, 3,prior_means,prior_stds)
    prior_means,prior_stds = fill_prior_column(params_ven, 4,prior_means,prior_stds)

    return prior_means, prior_stds





def from_range_to_gauss_v1():
    
    """This function takes range inputs for each parameter 
    and spits out the mean and standard deviation that each measurements should take 

    It assumes no variation between subjects... 

    If variation between subjects is necessary - change the vectors that it spits out a little bit...
    (but do it in a separate function )

    """

    ###################################################### SPATIALLY varying parameter range ######################################################

    def uniform_to_gauss(min_max):
        # take a uniform range's min_max and convert to mean and std of the gaussian 
        mean = np.mean(min_max).tolist()

        std = (min_max[1]-mean)/2 

        return [mean, std] 
        
    def convert_params(params):#, ROI):
        # convert parameters from uniform to gauss 
        for k in params: 
            params[k] = uniform_to_gauss(params[k])

        # params['ROI'] = ROI
        return params 

    def fill_prior_column(params, column,prior_means,prior_stds):
        
        prior_means_column = prior_means[:,column]
        prior_stds_column = prior_stds[:,column]


        # assuming no subject to subject variation -> prior_means_odd = prior_means_even, prior_stds_odd = prior_stds_even

        prior_means_column[0] = params['mu_myelin'][0]
        prior_means_column[1] = params['mu_myelin'][0]
        prior_means_column[2] = params['mu_ies'][0]
        prior_means_column[3] = params['mu_ies'][0]
        prior_means_column[4] = params['mu_csf'][0]
        prior_means_column[5] = params['mu_csf'][0]        
        prior_means_column[6] = params['vf_myelin'][0]
        prior_means_column[7] = params['vf_myelin'][0]
        prior_means_column[8] = params['vf_ies'][0]
        prior_means_column[9] = params['vf_ies'][0]                

        prior_stds_column[0] = params['mu_myelin'][1]
        prior_stds_column[1] = params['mu_myelin'][1]
        prior_stds_column[2] = params['mu_ies'][1]
        prior_stds_column[3] = params['mu_ies'][1]
        prior_stds_column[4] = params['mu_csf'][1]
        prior_stds_column[5] = params['mu_csf'][1]        
        prior_stds_column[6] = params['vf_myelin'][1]
        prior_stds_column[7] = params['vf_myelin'][1]
        prior_stds_column[8] = params['vf_ies'][1]
        prior_stds_column[9] = params['vf_ies'][1]                        
        
        # fill put back the column 
        prior_means[:,column] = prior_means_column
        prior_stds[:,column] = prior_stds_column

        return prior_means,prior_stds

    np.set_printoptions(suppress=True)

    ################## WHITE MATTER ##################
    p = {}
    p['vf_myelin'] = [0.1,0.3]
    p['vf_ies'] = [0.7,0.9]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0.9,1]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0,0.1]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p)#, 'WM')
    
    ################## VEN ##################
    params_ven = params_csf.copy()

    ################## BACKGROUND ##################
    p = {}
    p['vf_myelin'] = [1.0,1.0]
    p['vf_ies'] = [0.0,0.0]
    p['mu_myelin'] = [0.0,0.0]
    p['mu_ies'] = [0.0,0.0]
    p['mu_csf'] = [0.0,0.0]  

    params_bg = convert_params(p)#, 'WM')

    # init priors 
    prior_means = np.zeros((10,5))
    prior_stds = np.zeros((10,5))

    
    # fill in the values 
    prior_means,prior_stds = fill_prior_column(params_bg, 0,prior_means,prior_stds)
    prior_means,prior_stds = fill_prior_column(params_gm, 1,prior_means,prior_stds)
    prior_means,prior_stds = fill_prior_column(params_csf, 2,prior_means,prior_stds)
    prior_means,prior_stds = fill_prior_column(params_wm, 3,prior_means,prior_stds)
    prior_means,prior_stds = fill_prior_column(params_ven, 4,prior_means,prior_stds)

    return prior_means, prior_stds




######################## PRE ISMRM SUBMISSION EXPERIMENTS

def example_19c(opt):

    # creating an extra copy of generated volume with negative values (instead of zeros) - to then create a function to show it 

    opt = example_17(opt)
    opt["experiment_name"] = 'test19c_removing_negatives'

    modified_opt = {"n_examples":1}    

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt



def example_19b(opt):

    # repeat 19 but with original confidence range of 0.95percent 


    opt = example_19(opt)
    opt["experiment_name"] = 'test19b_confidence_of_095/'

    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.15,0.25]  # reduced from 0.1 - 0.3
    p['vf_ies'] = [0.75,0.85]  # reduced from 0.7 - 0.9 
    p['mu_myelin'] = [23,27]  # set 4ms variance. Before it was 25-25
    p['mu_ies'] = [78,82]  # set 4ms variance. Before it was 80,80
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p,confidence=0.95)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0.9,1]
    p['mu_myelin'] = [23,27] # set 4ms variance. Before it was 25-25
    p['mu_ies'] = [108,112] # set 4ms variance. And changed from previous value of 80-80 (same as Wm)
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p,confidence=0.95)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0,0.1]
    p['mu_myelin'] = [23,27] # set 4ms variance. Before it was 25-25
    p['mu_ies'] = [78,82]  # set 4ms variance. Before it was 80,80
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p,confidence=0.95)#, 'WM')


    # multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
    def mult_vol_fraction(p,mult=100):
        p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
        p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
        return p

    params_wm = mult_vol_fraction(params_wm)
    params_gm = mult_vol_fraction(params_gm)
    params_csf = mult_vol_fraction(params_csf)


    if opt['prior_distribution'] == 'uniform':
        # create prior_means and prior_stds vectors
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)
    else:
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,opt['vf_pop_var'])

    modified_opt = {"n_examples":5}    

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]    

    return opt


def example_19(opt):

    # reduce the variance of parameters significantly by modifying the STD to fall within the provided uniform range 99.7% of the time (i.e. std = (max-mean)/3, instead of std = (max-mean)/2)
    # repeat example_17 but over-ride the prior values

    opt = example_17(opt)
    opt["experiment_name"] = 'test19_confidence_of_0995/'

    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.15,0.25]  # reduced from 0.1 - 0.3
    p['vf_ies'] = [0.75,0.85]  # reduced from 0.7 - 0.9 
    p['mu_myelin'] = [23,27]  # set 4ms variance. Before it was 25-25
    p['mu_ies'] = [78,82]  # set 4ms variance. Before it was 80,80
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p,confidence=0.997)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0.9,1]
    p['mu_myelin'] = [23,27] # set 4ms variance. Before it was 25-25
    p['mu_ies'] = [108,112] # set 4ms variance. And changed from previous value of 80-80 (same as Wm)
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p,confidence=0.997)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0,0.1]
    p['mu_myelin'] = [23,27] # set 4ms variance. Before it was 25-25
    p['mu_ies'] = [78,82]  # set 4ms variance. Before it was 80,80
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p,confidence=0.997)#, 'WM')


    # multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
    def mult_vol_fraction(p,mult=100):
        p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
        p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
        return p

    params_wm = mult_vol_fraction(params_wm)
    params_gm = mult_vol_fraction(params_gm)
    params_csf = mult_vol_fraction(params_csf)


    if opt['prior_distribution'] == 'uniform':
        # create prior_means and prior_stds vectors
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)
    else:
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,opt['vf_pop_var'])

    modified_opt = {"n_examples":5}    

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt



def example_18(opt):

    # repeating test 17 entirely. Generating new results only to check if `<0` values in MWF and IEWF can be resolved without `morphological closing`

    opt = example_17(opt)
    opt["experiment_name"] = 'test18_check_morphological_closing'

    modified_opt = {"n_examples":1}    

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt



def example_17(opt):
    # run example 11 (no spatial deformation) but reduce range of variance in MWF to 0.1 and fix GM/WM values to be different, and add multi-seg

    opt = example_11(opt)
    opt["experiment_name"] = 'test17_multiseg_fix_GM_WM_T2_reduce_MWF_variance'

    # over ride default extra-options with the following 
    label_map = '/home/ch215616/code/SynthSeg/sv407_ismrm/segmentation_targets/' # provide directory to multiple segmentations 

    modified_opt = {'path_label_map':label_map,
                            "n_examples":5}                                

    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.15,0.25]  # reduced from 0.1 - 0.3
    p['vf_ies'] = [0.75,0.85]  # reduced from 0.7 - 0.9 
    p['mu_myelin'] = [23,27]  # set 4ms variance. Before it was 25-25
    p['mu_ies'] = [78,82]  # set 4ms variance. Before it was 80,80
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0.9,1]
    p['mu_myelin'] = [23,27] # set 4ms variance. Before it was 25-25
    p['mu_ies'] = [108,112] # set 4ms variance. And changed from previous value of 80-80 (same as Wm)
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0,0.1]
    p['mu_myelin'] = [23,27] # set 4ms variance. Before it was 25-25
    p['mu_ies'] = [78,82]  # set 4ms variance. Before it was 80,80
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p)#, 'WM')


    # multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
    def mult_vol_fraction(p,mult=100):
        p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
        p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
        return p

    params_wm = mult_vol_fraction(params_wm)
    params_gm = mult_vol_fraction(params_gm)
    params_csf = mult_vol_fraction(params_csf)


    if opt['prior_distribution'] == 'uniform':
        # create prior_means and prior_stds vectors
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)
    else:
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,opt['vf_pop_var'])


    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt

def example_16(opt):
    
    # Example: 
    # - add bias field generation
    # - run example 11 - no spatial deformations, no blurring, no multiseg examples, but correct myelin params 

    opt = example_11(opt)

    opt["experiment_name"] = 'test16_generate_bias_field'

    modified_opt = {'B1_prior':True}    

    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.1,0.3]
    p['vf_ies'] = [0.7,0.9]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]
    p['bias'] = [1.,1.]

    params_wm = convert_params(p)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0.9,1]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]
    p['bias'] = [1.,1.]

    params_gm = convert_params(p)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0,0.1]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]  
    p['bias'] = [1.,1.]

    params_csf = convert_params(p)#, 'WM')




    # multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
    def mult_vol_fraction(p,mult=100):
        p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
        p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
        p['bias'] = [p['bias'][0]*mult,p['bias'][1]*mult]        
        return p

    params_wm = mult_vol_fraction(params_wm)
    params_gm = mult_vol_fraction(params_gm)
    params_csf = mult_vol_fraction(params_csf)






    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    

    if opt['prior_distribution'] == 'uniform':
        # create prior_means and prior_stds vectors
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)
    else:
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,opt['vf_pop_var'], opt['B1_prior'])

    return opt

def example_15(opt):

    # run multi-seg with example 13 (TURN ON spatial transformations, blurring and other parameters)    

    opt = example_13(opt)
    opt["experiment_name"] = 'test15_multiseg_with_spatial_transforms'

    # over ride default extra-options with the following 
    label_map = '/home/ch215616/code/SynthSeg/sv407_ismrm/segmentation_targets/' # provide directory to multiple segmentations 

    modified_opt = {'path_label_map':label_map,
                            "n_examples":30}                                

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt

def example_14(opt):

    # run example 11 for multi-segmentation case 

    # reminder: 
    # Example 11: 
    # - turn off spatial deformations (to make comparisons easy)
    # - change all the parameters as per onur's suggestions 
    # - normal distribution 
    # - turn off bias field 
    # - introduce some parameter variance 
    # - turn on blur 
    

    opt = example_11(opt)
    opt["experiment_name"] = 'test14_multiple_segmentations_targets'

    # over ride default extra-options with the following 
    label_map = '/home/ch215616/code/SynthSeg/sv407_ismrm/segmentation_targets/' # provide directory to multiple segmentations 

    modified_opt = {'path_label_map':label_map,
                            "n_examples":8}                                

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt



def example_13b(opt):
    # DOES NOT WORK - IT IS NOT POSSIBLE TO ON NONLINDEFORM AND OFF LINDEFORM


    # run example 13 but turn off flipping and linear transforms - to view how much nonlin deformations there is 
    opt = example_13(opt)
    opt["experiment_name"] = 'test13_Onurs_params_with_blurring_and_non_linear_transform_only'

    # over ride default extra-options with the following 
    modified_opt = {'blurring':True,
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':False,
                    'flipping':False,
                    'n_examples':10}                                

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt


def example_13(opt):

    # run example 11 but switch on spatial transformation - and increase number of subjects to 10 
    opt = example_11(opt)
    opt["experiment_name"] = 'test12_Onurs_params_with_blurring_and_spatial_transform'

    # over ride default extra-options with the following 
    modified_opt = {'blurring':True,
                    'apply_nonlin_trans':True,
                    'apply_linear_trans':True,
                    'flipping':True,
                    'n_examples':10}                                

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt



def example_12(opt):

    # run example 11 but switch on blurring
    opt = example_11(opt)
    opt["experiment_name"] = 'test11_Onurs_params_with_blurring'

    # over ride default extra-options with the following 
    modified_opt = {'blurring':True}                                

    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    
    return opt


def example_11(opt):
    
    # Example: 
    # - change to normal distribution

    # Example 10: 
    # - change all the parameters as per onur's suggestions 
    # - change distribution to uniform, not normal 
    # - turn off bias field 
    # - introduce some parameter variance 
    # - turn on blur 
    # - turn off spatial deformations (to make comparisons easy)


    opt["experiment_name"] = 'test10_Onurs_params'

    

    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.1,0.3]
    p['vf_ies'] = [0.7,0.9]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0.9,1]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0,0.1]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p)#, 'WM')


    # multiply volume fraction by a hundred since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values
    def mult_vol_fraction(p,mult=100):
        p['vf_myelin'] = [p['vf_myelin'][0]*mult,p['vf_myelin'][1]*mult]
        p['vf_ies'] = [p['vf_ies'][0]*mult,p['vf_ies'][1]*mult]        
        return p

    params_wm = mult_vol_fraction(params_wm)
    params_gm = mult_vol_fraction(params_gm)
    params_csf = mult_vol_fraction(params_csf)


    ################## Spatial Deformations ##################

    label_map = '/home/ch215616/code/SynthSeg/sv407_ismrm/example.nii.gz'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':False, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':True,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':False,
                    'flipping':False, 
                    'path_label_map':label_map,
                    'prior_distribution':'normal',
                    'n_examples':2,
                    'output_shape':None,
                    'vf_pop_var':0.0}                                

    # vf_pop_var - the amount of variation in the choice of mean parameter for volume fraction only 


    # # each column represents a label: background, GM, CSF, WM, ventricles respectively 
    # r = 0.3 # ratio of `across population variance` to `intra-region variance`

    # # ASSUME UNIFORM DISTRIBUTION 
    # # ODD ROW - min value inside ROI, EVEN ROW - if that min value should change between each generated brain
    # # i.e. if odd and even are equal - no variation between subjects
    #                         #        BG   GM   CSF  WM   VEN
    # opt['prior_means'] = [[0.0, 25., 25., 25., 25.], \
    #                                 [0.0, 25., 25., 25., 25.], \
    #                                 [0.0, 80., 80., 80., 80.], \
    #                                 [0.0, 80., 80., 80., 80.], \
    #                                 [0.0, 1500., 1500., 1500., 1500.], \
    #                                 [0.0, 1500., 1500., 1500., 1500.], \
    #                                 [1.0, 0.0, 0.0, 0.1, 0.0], \
    #                                 [1.0, 0.0, 0.0, 0.1, 0.0], \
    #                                 [0.0, 0.9, 0.0, 0.7, 0.0], \
    #                                 [0.0, 0.9, 0.0, 0.7, 0.0]]

    # # ODD ROW - max value inside ROI, EVEN ROW - if that max value inside ROI should change between each generated brain 
    # # i.e. [prior_means_ODD, prior_stds_ODD] - defines the min-max range of the uniform distribution inside each ROI 
    # #      [prior_means_EVEN] - defines if the min range of the uniform distribution is different for each generated subject 
    # #      [prior_stds_EVEN] - defines if the max range of the uniform distribution is different for each generated subject 
    # # i.e. if [prior_means_ODD == prior_means_EVEN]

    # # if odd and even are equal - no variation between subjects
    # # The first image should be all pretty random numbers (as the averages are exactly the same - look at prior_means)
    # opt['prior_stds'] = [[0.0, 25., 25., 25., 25.], \
    #                             [0.0, 25., 25., 25., 25.], \
    #                             [0.0, 80., 80., 80., 80.], \
    #                             [0.0, 80., 80., 80., 80.], \
    #                             [0.0, 2000., 2000., 2000., 2000.], \
    #                             [0.0, 2000., 2000., 2000., 2000.], \
    #                             [1.0, 0.1, 0.1, 0.3, 0.1], \
    #                             [1.0, 0.1, 0.1, 0.3, 0.1], \
    #                             [0.0, 1.0, 0.1, 0.9, 0.1], \
    #                             [0.0, 1.0, 0.1, 0.9, 0.1]]  


    # over-ride SynthSeg options 
    for k in modified_opt:
        opt[k] = modified_opt[k]
    

    if opt['prior_distribution'] == 'uniform':
        # create prior_means and prior_stds vectors
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)
    else:
        opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_gauss_manual(params_wm,params_gm,params_csf,opt['vf_pop_var'])

    return opt

def example_10(opt):
    
    # Example 10: 
    # - change all the parameters as per onur's suggestions 
    # - change distribution to uniform, not normal 
    # - turn off bias field 
    # - introduce some parameter variance 
    # - turn on blur 
    # - turn off spatial deformations (to make comparisons easy)


    opt["experiment_name"] = 'test10_Onurs_params'

    

    ################## WHITE MATTER ################## UNIFORM
    p = {}
    p['vf_myelin'] = [0.1,0.3]
    p['vf_ies'] = [0.7,0.9]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]

    params_wm = convert_params(p)#, 'WM')

    ################## GRAY MATTER ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0.9,1]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]

    params_gm = convert_params(p)#, 'WM')

    ################## CSF ##################
    p = {}
    p['vf_myelin'] = [0,0.1]
    p['vf_ies'] = [0,0.1]
    p['mu_myelin'] = [25,25]
    p['mu_ies'] = [80,80]
    p['mu_csf'] = [1500,2000]  

    params_csf = convert_params(p)#, 'WM')

    ################## Spatial Deformations ##################

    label_map = '/home/ch215616/code/SynthSeg/sv407_ismrm/example.nii.gz'

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':False, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':True,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':False,
                    'flipping':False, 
                    'path_label_map':label_map,
                    'prior_distribution':'uniform',
                    'n_examples':2,
                    'output_shape':None}                                




    # # each column represents a label: background, GM, CSF, WM, ventricles respectively 
    # r = 0.3 # ratio of `across population variance` to `intra-region variance`

    # # ASSUME UNIFORM DISTRIBUTION 
    # # ODD ROW - min value inside ROI, EVEN ROW - if that min value should change between each generated brain
    # # i.e. if odd and even are equal - no variation between subjects
    #                         #        BG   GM   CSF  WM   VEN
    # opt['prior_means'] = [[0.0, 25., 25., 25., 25.], \
    #                                 [0.0, 25., 25., 25., 25.], \
    #                                 [0.0, 80., 80., 80., 80.], \
    #                                 [0.0, 80., 80., 80., 80.], \
    #                                 [0.0, 1500., 1500., 1500., 1500.], \
    #                                 [0.0, 1500., 1500., 1500., 1500.], \
    #                                 [1.0, 0.0, 0.0, 0.1, 0.0], \
    #                                 [1.0, 0.0, 0.0, 0.1, 0.0], \
    #                                 [0.0, 0.9, 0.0, 0.7, 0.0], \
    #                                 [0.0, 0.9, 0.0, 0.7, 0.0]]

    # # ODD ROW - max value inside ROI, EVEN ROW - if that max value inside ROI should change between each generated brain 
    # # i.e. [prior_means_ODD, prior_stds_ODD] - defines the min-max range of the uniform distribution inside each ROI 
    # #      [prior_means_EVEN] - defines if the min range of the uniform distribution is different for each generated subject 
    # #      [prior_stds_EVEN] - defines if the max range of the uniform distribution is different for each generated subject 
    # # i.e. if [prior_means_ODD == prior_means_EVEN]

    # # if odd and even are equal - no variation between subjects
    # # The first image should be all pretty random numbers (as the averages are exactly the same - look at prior_means)
    # opt['prior_stds'] = [[0.0, 25., 25., 25., 25.], \
    #                             [0.0, 25., 25., 25., 25.], \
    #                             [0.0, 80., 80., 80., 80.], \
    #                             [0.0, 80., 80., 80., 80.], \
    #                             [0.0, 2000., 2000., 2000., 2000.], \
    #                             [0.0, 2000., 2000., 2000., 2000.], \
    #                             [1.0, 0.1, 0.1, 0.3, 0.1], \
    #                             [1.0, 0.1, 0.1, 0.3, 0.1], \
    #                             [0.0, 1.0, 0.1, 0.9, 0.1], \
    #                             [0.0, 1.0, 0.1, 0.9, 0.1]]  


    # over-ride SynthSeg options 
    for k in opt:
        if k in modified_opt:
            opt[k] = modified_opt[k]


    # create prior_means and prior_stds vectors
    opt['prior_means'], opt['prior_stds'] = convert_priors_to_synthseg_format_uniform_manual(params_wm,params_gm,params_csf)

    return opt











def example_11_old(opt):




    def set_params(ROI, prior_type, p, vf_myelin, vf_ies, mu_myelin, mu_ies, mu_csf):
        """ Sets the parmaeters according to the variables passed to it
        
        Creates a dict with the values... 
        
        """

        if prior_type == 'spatial': 

            pass
        else: 
            pass 
        p=0

        return p 

    def generate_priors(p):
        """
        Args: 
            p (dict): dictionary with values of where each parameter belongs
        
        Returns: 
            prior_means, prior_stds (numpy.ndarray): default expected parameters of synthseg for parameter generation

        """
        prior_means = 0
        prior_stds = 0 


        # sets the Ventricles to be like CSF 
        # sets background to be around zero everywhere

        return prior_means, prior_stds


    # Example 11 
    # - same as example 10 but with the improved visualization of priors 

    # To ANALYZE closely -> run get_MWF_parameters_MIML.py file -> function reformat_priors_to_pandas and synthseg_priors(requires copy paste of manual values below). 

    # Take std distirbution of each parameter in each region from MIML paper and set it as intra-region variation. Set very little intra-subject variation 
    # see get_MWF_parameters_MIML.py -> output of function `reformat_to_synthseg()`


    opt["experiment_name"] = 'test11_Onurs_params_new_load_framework'

    ###################################################### SPATIALLY varying parameter range ######################################################

    ################## WHITE MATTER ##################
    vf_myelin = [0.1,0.3]
    vf_ies = [0.7,0.9]
    mu_myelin = [25,25]
    mu_ies = [80,80]
    mu_csf = [1500,2000]

    p = set_params('WM','spatial', p, vf_myelin,vf_ies,mu_myelin,mu_ies,mu_csf)

    ################## GRAY MATTER ##################
    vf_myelin = [0,0.1]
    vf_ies = [0.9,1]
    mu_myelin = [25,25]
    mu_ies = [80,80]
    mu_csf = [1500,2000]

    p = set_params('GM','spatial', p, vf_myelin,vf_ies,mu_myelin,mu_ies,mu_csf)

    ################## CSF ##################
    vf_myelin = [0,0.1]
    vf_ies = [0,0.1]
    mu_myelin = [25,25]
    mu_ies = [80,80]
    mu_csf = [1500,2000]  
    
    p = set_params('CSF','spatial', p, vf_myelin,vf_ies,mu_myelin,mu_ies,mu_csf)


    ###################################################### POPULATION varying parameter range ######################################################

    ################## WHITE MATTER ##################
    population_vf_myelin = []
    population_vf_ies = []
    population_mu_myelin = [25,25]
    population_mu_ies = [80,80]
    population_mu_csf = [1500,2000]  

    p = set_params('WM','population', p, population_vf_myelin,population_vf_ies,population_mu_myelin,population_mu_ies,population_mu_csf)

    ################## GRAY MATTER ##################
    population_vf_myelin = []
    population_vf_ies = []
    population_mu_myelin = [25,25]
    population_mu_ies = [80,80]
    population_mu_csf = [1500,2000]  

    p = set_params('GM','population', p, population_vf_myelin,population_vf_ies,population_mu_myelin,population_mu_ies,population_mu_csf)

    ################## CSF ##################
    population_vf_myelin = [0,0.1]
    population_vf_ies = [0,0.1]
    population_mu_myelin = [25,25]
    population_mu_ies = [80,80]
    population_mu_csf = [1500,2000]  

    p = set_params('CSF','population', p, population_vf_myelin,population_vf_ies,population_mu_myelin,population_mu_ies,population_mu_csf)

    # IMPORTANT NOTES: 
    # - each parameter has TWO elements - if distribution==uniform -> [min,max], elseif distribution==normal -> [mean,std]    
    # - [sigma_myelin, sigma_ies, sigma_csf] are FIXED and not used for generating spatially varying samples 
    # - [vf_csf] is calculated through a relationship with vf_myelin and vf_ies (i.e. all fractions must sum to 1), hence no need to generate it 
    # - bias field would be set in follow up experiments 

    ###################################################### SYNTHSEG spatial transformations ######################################################

    # over ride default SynthSeg options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':False, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':True,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':True, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':False,
                    'flipping':False}                                









    # update SynthSeg default options with the modified_options
    for k in opt:
        if k in modified_opt:
            opt[k] = modified_opt[k]

    # update opt with the priors 
    opt['prior_means'],opt['prior_stds']  = generate_priors(p)

    # opt['prior_means'] = [[0.0, 22., 22., 22., 22.], \
    #                                 [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                                 [0.0, 80., 80., 80., 80.], \
    #                                 [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                                 [0.0, 1500., 1500., 1500., 1500.], \
    #                                 [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                                 [1.0, 0.0, 0.0, 0.3, 0.0], \
    #                                 [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                                 [0.0, 1.0, 0.0, 0.3, 0.0], \
    #                                 [0.0, 0.0, 0.0, 0.0, 0.0]]


    # ventricles and background - were set to be 1., with no variation of the mean from subject to subject 
    # may need to re-calculate the actual value of T2 in ventricles (e.g. it may be filled with CSF like fluid)


    # # The first image should be all pretty random numbers (as the averages are exactly the same - look at prior_means)
    # opt['prior_stds'] = [[0.0, 0.0, 0.0, 0.0, 0.0], \
    #                             [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                             [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                             [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                             [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                             [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                             [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                             [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                             [0.0, 0.0, 0.0, 0.0, 0.0], \
    #                             [0.0, 0.0, 0.0, 0.0, 0.0]]  



    return opt

















def example_6(opt):
    # Example 6 

    # Diff to example 5: 
    # - reduce intra-subject variance to half 
    # - turn off all spatial transforms, inc blurring (just sampling from priors)

    # Set ventricles same as CSF.
    # Set `intra-region variance` to `across population variance`
    # Then set `across population variance` to half of the `intra-region variance`
    # Set `intra-region variance across population` to zero (this is not important metric and should be similar). i.e. prior_stds[1]
    # Set `intra-region variance` to zero for background
    


    # To ANALYZE closely -> run get_MWF_parameters_MIML.py file -> function reformat_priors_to_pandas and synthseg_priors(requires copy paste of manual values below). 

    # Take std distirbution of each parameter in each region from MIML paper and set it as intra-region variation. Set very little intra-subject variation 
    # see get_MWF_parameters_MIML.py -> output of function `reformat_to_synthseg()`

    # each column represents a label: background, GM, CSF, WM, ventricles respectively 
    r = 0.5 # ratio of `across population variance` to `intra-region variance`
    opt['prior_means'] = [[0.0, 22.5, 22.5, 22.5, 22.5], \
                                    [0.0, 4.3*r, 4.3*r, 4.3*r, 4.3*r], \
                                    [0.0, 180.1, 80.0, 84.9, 80.0], \
                                    [0.0, 69.2*r, 11.6*r, 20.2*r, 11.6*r], \
                                    [0.0, 1500.8, 1499.9, 1499.5, 1499.9], \
                                    [0.0, 288.9*r, 289.2*r, 289.2*r, 289.2*r], \
                                    [1.0, 0.0, 0.0, 0.3, 0.0], \
                                    [0.0, 0.0, 0.0, 0.2*r, 0.0], \
                                    [0.0, 1.0, 0.0, 0.3, 0.0], \
                                    [0.0, 0.0, 0.0, 0.2*r, 0.0]]


    # ventricles and background - were set to be 1., with no variation of the mean from subject to subject 
    # may need to re-calculate the actual value of T2 in ventricles (e.g. it may be filled with CSF like fluid)


    # The first image should be all pretty random numbers (as the averages are exactly the same - look at prior_means)
    opt['prior_stds'] = [[0.0, 4.3, 4.3, 4.3, 4.3], \
                                [0.0, 4.3*r, 4.3*r, 4.3*r, 4.3*r], \
                                [0.0, 69.2*r, 11.6*r, 20.2*r, 11.6*r], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 288.9*r, 289.2*r, 289.2*r, 289.2*r], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.2*r, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.2*r, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0]]  

    return opt

def example_9(opt):
    # Example 9
    # Diff to example 8
    # - turned on bias field and did this for 3 samples - to show onur how bias is generated. 

    # Diff to example 7 
    # - reduce all variance to zero, reduce all other params to zero 

    # Diff to example 6 
    # - reduce intra-subject variance to one third  
    # - over ride spatial transform options directly here through modified_opt

    # Diff to example 5: 
    # Set ventricles same as CSF.
    # Set `intra-region variance` to `across population variance`
    # Then set `across population variance` to half of the `intra-region variance`
    # Set `intra-region variance across population` to zero (this is not important metric and should be similar). i.e. prior_stds[1]
    # Set `intra-region variance` to zero for background
    


    # To ANALYZE closely -> run get_MWF_parameters_MIML.py file -> function reformat_priors_to_pandas and synthseg_priors(requires copy paste of manual values below). 

    # Take std distirbution of each parameter in each region from MIML paper and set it as intra-region variation. Set very little intra-subject variation 
    # see get_MWF_parameters_MIML.py -> output of function `reformat_to_synthseg()`


    opt["experiment_name"] = 'test_9_zero_variance_all_params_off_bias_on'

    # each column represents a label: background, GM, CSF, WM, ventricles respectively 
    r = 0.3 # ratio of `across population variance` to `intra-region variance`
    opt['prior_means'] = [[0.0, 22., 22., 22., 22.], \
                                    [0.0, 0.0, 0.0, 0.0, 0.0], \
                                    [0.0, 80., 80., 80., 80.], \
                                    [0.0, 0.0, 0.0, 0.0, 0.0], \
                                    [0.0, 1500., 1500., 1500., 1500.], \
                                    [0.0, 0.0, 0.0, 0.0, 0.0], \
                                    [1.0, 0.0, 0.0, 0.3, 0.0], \
                                    [0.0, 0.0, 0.0, 0.0, 0.0], \
                                    [0.0, 1.0, 0.0, 0.3, 0.0], \
                                    [0.0, 0.0, 0.0, 0.0, 0.0]]


    # ventricles and background - were set to be 1., with no variation of the mean from subject to subject 
    # may need to re-calculate the actual value of T2 in ventricles (e.g. it may be filled with CSF like fluid)


    # The first image should be all pretty random numbers (as the averages are exactly the same - look at prior_means)
    opt['prior_stds'] = [[0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0]]  

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':False, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':True,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':True, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':False,
                    'flipping':False}                                


    # over-ride SynthSeg options 
    for k in opt:
        if k in modified_opt:
            opt[k] = modified_opt[k]


    return opt

def example_8(opt):
    # Example 8
    # Diff to example 7 
    # - reduce all variance to zero, reduce all other params to zero 

    # Diff to example 6 
    # - reduce intra-subject variance to one third  
    # - over ride spatial transform options directly here through modified_opt

    # Diff to example 5: 
    # Set ventricles same as CSF.
    # Set `intra-region variance` to `across population variance`
    # Then set `across population variance` to half of the `intra-region variance`
    # Set `intra-region variance across population` to zero (this is not important metric and should be similar). i.e. prior_stds[1]
    # Set `intra-region variance` to zero for background
    


    # To ANALYZE closely -> run get_MWF_parameters_MIML.py file -> function reformat_priors_to_pandas and synthseg_priors(requires copy paste of manual values below). 

    # Take std distirbution of each parameter in each region from MIML paper and set it as intra-region variation. Set very little intra-subject variation 
    # see get_MWF_parameters_MIML.py -> output of function `reformat_to_synthseg()`


    opt["experiment_name"] = 'test_8_zero_variance_all_params_off'

    # each column represents a label: background, GM, CSF, WM, ventricles respectively 
    r = 0.3 # ratio of `across population variance` to `intra-region variance`
    opt['prior_means'] = [[0.0, 22., 22., 22., 22.], \
                                    [0.0, 0.0, 0.0, 0.0, 0.0], \
                                    [0.0, 80., 80., 80., 80.], \
                                    [0.0, 0.0, 0.0, 0.0, 0.0], \
                                    [0.0, 1500., 1500., 1500., 1500.], \
                                    [0.0, 0.0, 0.0, 0.0, 0.0], \
                                    [1.0, 0.0, 0.0, 0.3, 0.0], \
                                    [0.0, 0.0, 0.0, 0.0, 0.0], \
                                    [0.0, 1.0, 0.0, 0.3, 0.0], \
                                    [0.0, 0.0, 0.0, 0.0, 0.0]]


    # ventricles and background - were set to be 1., with no variation of the mean from subject to subject 
    # may need to re-calculate the actual value of T2 in ventricles (e.g. it may be filled with CSF like fluid)


    # The first image should be all pretty random numbers (as the averages are exactly the same - look at prior_means)
    opt['prior_stds'] = [[0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0]]  

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':False, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':True,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':False,
                    'flipping':False}                                


    # over-ride SynthSeg options 
    for k in opt:
        if k in modified_opt:
            opt[k] = modified_opt[k]


    return opt

def example_7(opt):
    # Example 7 
    # Diff to example 6 
    # - reduce intra-subject variance to one third  
    # - over ride spatial transform options directly here through modified_opt

    # Diff to example 5: 
    # Set ventricles same as CSF.
    # Set `intra-region variance` to `across population variance`
    # Then set `across population variance` to half of the `intra-region variance`
    # Set `intra-region variance across population` to zero (this is not important metric and should be similar). i.e. prior_stds[1]
    # Set `intra-region variance` to zero for background
    


    # To ANALYZE closely -> run get_MWF_parameters_MIML.py file -> function reformat_priors_to_pandas and synthseg_priors(requires copy paste of manual values below). 

    # Take std distirbution of each parameter in each region from MIML paper and set it as intra-region variation. Set very little intra-subject variation 
    # see get_MWF_parameters_MIML.py -> output of function `reformat_to_synthseg()`


    opt["experiment_name"] = 'test_7_reduce_intra_ROI_and_subj_variance_to_third_no_spatial_trans'

    # each column represents a label: background, GM, CSF, WM, ventricles respectively 
    r = 0.3 # ratio of `across population variance` to `intra-region variance`
    opt['prior_means'] = [[0.0, 22.5, 22.5, 22.5, 22.5], \
                                    [0.0, 4.3*r, 4.3*r, 4.3*r, 4.3*r], \
                                    [0.0, 180.1, 80.0, 84.9, 80.0], \
                                    [0.0, 69.2*r, 11.6*r, 20.2*r, 11.6*r], \
                                    [0.0, 1500.8, 1499.9, 1499.5, 1499.9], \
                                    [0.0, 288.9*r, 289.2*r, 289.2*r, 289.2*r], \
                                    [1.0, 0.0, 0.0, 0.3, 0.0], \
                                    [0.0, 0.0, 0.0, 0.2*r, 0.0], \
                                    [0.0, 1.0, 0.0, 0.3, 0.0], \
                                    [0.0, 0.0, 0.0, 0.2*r, 0.0]]


    # ventricles and background - were set to be 1., with no variation of the mean from subject to subject 
    # may need to re-calculate the actual value of T2 in ventricles (e.g. it may be filled with CSF like fluid)


    # The first image should be all pretty random numbers (as the averages are exactly the same - look at prior_means)
    opt['prior_stds'] = [[0.0, 4.3, 4.3, 4.3, 4.3], \
                                [0.0, 4.3*r, 4.3*r, 4.3*r, 4.3*r], \
                                [0.0, 69.2*r, 11.6*r, 20.2*r, 11.6*r], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 288.9*r, 289.2*r, 289.2*r, 289.2*r], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.2*r, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.2*r, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0]]  

    # over ride default extra-options with the following 
    modified_opt = {'sample_gmm':True,
                    'blurring':False, 
                    'vary_prior_means_and_stds':True,
                    'debug_prior_values':True,
                    'clipping':False,
                    'min_max_norm':False,
                    'gamma_augmentation':False,
                    'apply_bias_field':False, 
                    'apply_nonlin_trans':False,
                    'apply_linear_trans':False,
                    'flipping':False}                                


    # over-ride SynthSeg options 
    for k in opt:
        if k in modified_opt:
            opt[k] = modified_opt[k]


    return opt


    




    # Assignment of dimensions: 
    # - 0 = mu of myelin
    # - 1 = mu of IES 
    # - 2 = mu of CSF
    # - 3 = vol_fraction of myelin 
    # - 4 = vol_fraction of IES 
    # - 5 = bias_field? (maybe bias field should be computed PRIOR to deformation... or separately altogether)

    # single channel examples 

    # EXAMPLE 0 
    # (small) variation between subjects and intra-region; labels are 1,100,80,160,60 for background, GM, CSF, WM and ventricles respectively
    # opt['prior_means'] = [[1.,100.,80.,160.,60.],[1.,1.,1.,1.,1.]]                                    
    # opt['prior_stds'] = [[1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.]]

    # EXAMPLE 1
    # zero variation between subjects and intra-region
    # opt['prior_means'] = [[1.,100.,80.,160.,60.],[0.,0.,0.,0.,0.]] 
    # opt['prior_stds'] = [[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]  

    # EXAMPLE 2
    # zero variation intra-region
    # opt['prior_means'] = [[1.,100.,80.,160.,60.],[1.,1.,1.,1.,1.]]  
    # opt['prior_stds'] = [[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]  

    # EXAMPLE 3
    # zero variation between subjects 
    #        [& the amount of variation intra-region is the same between subjects - i.e. the level of noise for each example is THE SAME!]
    #        [this is how we can vary SNR in parameter estimates [or indeed in signal estiamtes images]
    #        ... although remember that noise is added separately to EACH segment - 
    #        ... so we can vary the amount of noise added to EACH segment differently! how cool is that]]
    #        ... to add the same SNR everywhere - just make the values of each prior_stds[0] the same (like shown above)
    #        ... to make SNR different for each region - make the values of prior_stds[0] vary from segment to segment
    # opt['prior_means'] = [[1.,100.,80.,160.,60.],[0.,0.,0.,0.,0.]]  
    # opt['prior_stds'] = [[10.,10.,10.,10.,10.],[0.,0.,0.,0.,0.]]  

    # EXAMPLE 4
    # zero variation between subjects [in mean values of each parameter] 
    #        [& some variation intra-region between subjects - i.e. the amount of noise added to each region between each example varies!]
    # opt['prior_means'] = [[1.,100.,80.,160.,60.],[0.,0.,0.,0.,0.]] 
    # opt['prior_stds'] = [[0.,0.,0.,0.,0.],[10.,10.,10.,10.,10.]]

def example_5(opt):
    # Example 5
    # Set ventricles same as CSF.
    # Set `intra-region variance` to `across population variance`
    # Then set `across population variance` to half of the `intra-region variance`
    # Set `intra-region variance across population` to zero (this is not important metric and should be similar). i.e. prior_stds[1]
    # Set `intra-region variance` to zero for background
    


    # To ANALYZE closely -> run get_MWF_parameters_MIML.py file -> function reformat_priors_to_pandas and synthseg_priors(requires copy paste of manual values below). 

    # Take std distirbution of each parameter in each region from MIML paper and set it as intra-region variation. Set very little intra-subject variation 
    # see get_MWF_parameters_MIML.py -> output of function `reformat_to_synthseg()`

    # each column represents a label: background, GM, CSF, WM, ventricles respectively 
    r = 0.5 # ratio of `across population variance` to `intra-region variance`
    opt['prior_means'] = [[0.0, 22.5, 22.5, 22.5, 22.5], \
                                    [0.0, 4.3*r, 4.3*r, 4.3*r, 4.3*r], \
                                    [0.0, 180.1, 80.0, 84.9, 80.0], \
                                    [0.0, 69.2*r, 11.6*r, 20.2*r, 11.6*r], \
                                    [0.0, 1500.8, 1499.9, 1499.5, 1499.9], \
                                    [0.0, 288.9*r, 289.2*r, 289.2*r, 289.2*r], \
                                    [1.0, 0.0, 0.0, 0.3, 0.0], \
                                    [0.0, 0.0, 0.0, 0.2*r, 0.0], \
                                    [0.0, 1.0, 0.0, 0.3, 0.0], \
                                    [0.0, 0.0, 0.0, 0.2*r, 0.0]]


    # ventricles and background - were set to be 1., with no variation of the mean from subject to subject 
    # may need to re-calculate the actual value of T2 in ventricles (e.g. it may be filled with CSF like fluid)


    # The first image should be all pretty random numbers (as the averages are exactly the same - look at prior_means)
    opt['prior_stds'] = [[0.0, 4.3, 4.3, 4.3, 4.3], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 69.2, 11.6, 20.2, 11.6], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 288.9, 289.2, 289.2, 289.2], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.2, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.2, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0]]  

    return opt


def example_0(opt):
    # Example 0 
    # For prior_means[0] only (i.e. mean of each parameter] -> Channel1 == Channel2 / 2 == Channel3 * 2 
    # All other parameters are set to zero. I.e. no variation between example, no variation intra-region
    opt['prior_means'] = [[1.,100.,80.,160.,60.],[0.,0.,0.,0.,0.],\
                                    [1.,100./2,80./2,160./2,60./2],[0.,0.,0.,0.,0.], \
                                    [1.,100.*2,80.*2,160.*2,60.*2],[0.,0.,0.,0.,0.]]   
    
    opt['prior_stds'] = [[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.], \
                                    [0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.], \
                                    [0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.]]  
    return opt

def example_1(opt):
    # Example 1 
    # vary all parameters smoothly between subjects and intra-region 
    opt['prior_means'] = [[1.,100.,80.,160.,60.],[0.,0.,0.,0.,0.],\
                                    [1.,100./2,80./2,160./2,60./2],[0.,0.,0.,0.,0.], \
                                    [1.,100.*2,80.*2,160.*2,60.*2],[0.,0.,0.,0.,0.]]   
    
    opt['prior_stds'] = [[1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.], \
                                    [1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.], \
                                    [1.,1.,1.,1.,1.],[1.,1.,1.,1.,1.]]  

    return opt


def example_3(opt):
    # Example 3 
    # Add volume fraction:
    opt['prior_means'] = [[1.,100.,80.,160.,60.],[0.,0.,0.,0.,0.],\
                                    [1.,100./2,80./2,160./2,60./2],[0.,0.,0.,0.,0.], \
                                    [1.,100.*2,80.*2,160.*2,60.*2],[0.,0.,0.,0.,0.], \
                                    [0.3,0.4,0.3,0.7,0.1],[0.,0.,0.,0.,0.], \
                                    [0.6,0.6,0.3,0.3,0.9],[0.,0.,0.,0.,0.]]   
    
    opt['prior_stds'] = [[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.], \
                                    [0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.], \
                                    [0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],\
                                    [0.2,0.2,0.2,0.2,0.2],[0.,0.,0.,0.,0.],\
                                    [0.2,0.2,0.2,0.2,0.2],[0.,0.,0.,0.,0.]]      

    return opt    

def example_4a(opt):
    # Example 4A - abandoned 
    # Take a min-max range from MIML paper. Set min-max as population wide variation, while intra-region variation is small (in the range of 0)

    # each column represents a label: background, GM, CSF, WM, ventricles respectively 
    opt['prior_means'] = [[1.,13.8,13.8,13.8,1.],[0.,31.2,31.2,31.2,0.],\
                                    [1.,41.6,56.8,44.5,1.],[0.,318.6,103.1,125.4,0.], \
                                    [1.,923.0,923.0,923.0,1.],[0.,2078.5,2078.5,2078.5,0.], \
                                    [1.,0.,0.,0.,1.],[1.0,0.,0.,0.8,1.], \
                                    [0.,1.0,0.,0.,0.],[0.,1.0,0.,0.8,0.]]   
    # ventricles and background - were set to be 1., with no variation of the mean from subject to subject 
    # may need to re-calculate the actual value of T2 in ventricles (e.g. it may be filled with CSF like fluid)

    opt['prior_stds'] = [[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.], \
                                    [0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.], \
                                    [0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],\
                                    [0.2,0.2,0.2,0.2,0.2],[0.,0.,0.,0.,0.],\
                                    [0.2,0.2,0.2,0.2,0.2],[0.,0.,0.,0.,0.]]  
    opt['prior_distribution'] = 'uniform'
    # What is the variation of values inside each region? It is defined by prior_stds[0] values... 
    # Let's make the values range 
    # we set the variation from subject to subject to zero for now. We will keep the amount of random 'variation' the same for each subject


    return opt    
def example_4b(opt):
    # Example 4B
    # Take a normal distirbution from MIML paper and set it as intra-region variation. Set between subject variation to be the same as intra-subject variation. 
    # see get_MWF_parameters_MIML.py -> output of function `reformat_to_synthseg()`

    # each column represents a label: background, GM, CSF, WM, ventricles respectively 
    opt['prior_means'] = [[0.0, 22.5, 22.5, 22.5, 0.0], \
                                    [0.0, 4.3, 4.3, 4.3, 0.0], \
                                    [0.0, 180.1, 80.0, 84.9, 0.0], \
                                    [0.0, 69.2, 11.6, 20.2, 0.0], \
                                    [0.0, 1500.8, 1499.9, 1499.5, 0.0], \
                                    [0.0, 288.9, 289.2, 289.2, 0.0], \
                                    [1.0, 0.0, 0.0, 0.3, 1.0], \
                                    [0.0, 0.0, 0.0, 0.2, 0.0], \
                                    [0.0, 1.0, 0.0, 0.3, 0.0], \
                                    [0.0, 0.0, 0.0, 0.2, 0.0]]


    # ventricles and background - were set to be 1., with no variation of the mean from subject to subject 
    # may need to re-calculate the actual value of T2 in ventricles (e.g. it may be filled with CSF like fluid)


    # The first image should be all pretty random numbers (as the averages are exactly the same - look at prior_means)
    opt['prior_stds'] = [[1.0, 4.3, 4.3, 4.3, 1.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [1.0, 69.2, 11.6, 20.2, 1.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [1.0, 288.9, 289.2, 289.2, 1.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.2, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.2, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0]]  

    # THIS IS LESS correct (maybe?)
    opt['prior_stds'] = [[1.0, 1.0, 1.0, 1.0, 1.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [1.0, 1.0, 1.0, 1.0, 1.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [1.0, 1.0, 1.0, 1.0, 1.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0], \
                                [0.0, 0.0, 0.0, 0.0, 0.0]]                   
    return opt            
def example_X(opt):
    #return opt            
    pass




def help():
    """



        # Example 4
        # With realistic MWF parameters: mu_myelin, mu_IES, mu_CSF, MWF, IEWF.
        # We extract the range (min-max) of all 9 parameters from MIML data for CSF, WM, GM - 9 parameters are (mu,sigma,vol_fraction) * (myelin, IES,CSF)
        # The parameters are extracted with a file called `get_MWF_parameters_MIML_study.py` in `~/code/mwf/source/MIML/dataset_for_serge/`
        # labels = {'background':1,'GM':2,'CSF':3,'WM':4, 'ventricles':5} 
        # Each label value corresponds to the index of prior_means and prior_stds. I.e. prior_means[0][0] -> background, prior_means[0][3] -> WM 
        
        # Population statistics - reflected in prior_means 
        # Variation in each region - reflected in prior_stds 

        # In this instance we sample from a uniform distribution (since it is easier to define over population statistics)
        
        # each row below represents a certain parameter of the MWF model: 
        # Row 1: mu_myelin
        # Row 2: mu_ies 
        # Row 3: mu_csf 
        # Row 4: vol_frac_myelin
        # Row 5: vol_frac_IES




               
        
    """    

if __name__ == '__main__':

    opt = {}
    opt = example_latest(opt)
