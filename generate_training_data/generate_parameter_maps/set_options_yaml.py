"""Specify inputs for MWF related parameter image generation from yaml file 

To generate a new set of variables create a new function with `def example_<number>(opt)` and then run it inside `example_latest(). 

"""

import os

import numpy as np
import pandas as pd 

import svtools as sv


def read_priors(opt):

    # read from pickle
    dfs_mean = pd.read_pickle(opt['prior_means_path'])
    dfs_std = pd.read_pickle(opt['prior_stds_path'])

    # identify how much to vary std w.r.t. to mean 
    

    # mult fractions by 100 (since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values)
    for p in ['vf_myelin', 'vf_ies', 'vf_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(100)
        if opt['std_to_mean_ratio'] is not None: 
            if p == 'vf_myelin': 
                i = 3
            elif p == 'vf_ies':
                i = 4 
            else: 
                i = 5
            dfs_std[p] = dfs_mean[p].copy().divide(100*opt['std_to_mean_ratio'][i]) # 10% 
        else: 
            dfs_std[p] = dfs_std[p].multiply(100)

    for p in ['mu_myelin', 'mu_ies', 'mu_csf']:
        dfs_mean[p] = dfs_mean[p].multiply(1000)
        if opt['std_to_mean_ratio'] is not None: 
            if p == 'mu_myelin': 
                i = 0
            elif p == 'mu_ies':
                i = 1
            else: 
                i = 2             
            dfs_std[p] = dfs_mean[p].copy().multiply(opt['std_to_mean_ratio'][i])
        else:
            dfs_std[p] = dfs_std[p].multiply(1000)

    for p in ['sigma_myelin', 'sigma_ies', 'sigma_csf']:
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

    # convert to synthseg format 
    opt['prior_means'], opt['prior_stds'],roi_map = convert_priors_to_synthseg_format_gauss(params_all_rois, opt)      


    # get generation labels and their names     
    opt['generation_labels'] = np.array([int(i) for i in dfs_mean.index.tolist()] + [0]) #opt['generation_labels'] = np.array([int(i) for i in list(roi_map.keys())]) # the keys of roi_map dict contains the order, which we must present as a np.array of ints to the synthseg algorithm
    
    #tissue_names = {'2':'GM', '3':'CSF', '4':'WM', '5':'VEN'}
    #opt['generation_label_names'] = [tissue_names[i] + ' ' + j for i,j in zip(dfs_mean['tissue_id'].tolist(), dfs_mean['roi_name'].tolist())] + ['background']
    
    # a couple of checks to make sure the mapping is correct after conversion
    assert opt['prior_means'][:,0][0] == params_all_rois[str(opt['generation_labels'][0])]['mu_myelin'][0]
    assert opt['prior_means'][:,-1][0] == params_all_rois[str(opt['generation_labels'][-1])]['mu_myelin'][0]    

    return opt 


def check_priors(params_all_rois, roi_name_ids):

    # the the final priors that are passed to generation algorithm 

    print(f"PRINTING FINAL PARAMETER PRIORS FOR CHOSEN ROI_NAME_IDS")

    for i in roi_name_ids:
        print(params_all_rois[str(i)])
        input('press any key to proceed')

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




def convert_priors_to_synthseg_format_gauss_CSF(params_all_rois, opt, n_features=5, add_background_priors=True):
    # convert all priors into the synthnet format 
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
        prior_means_column[1] = params['mu_myelin'][0]*mu_population_variance[0]   # accessing variance of the mean measure of mu_myelin

        prior_means_column[2] = params['mu_ies'][0] 
        prior_means_column[3] = params['mu_ies'][0]*mu_population_variance[1]

        prior_means_column[4] = params['mu_csf'][0]
        prior_means_column[5] = params['mu_csf'][0]*mu_population_variance[2]
        #prior_means_column[5] = params['mu_csf'][0]

        prior_means_column[6] = params['vf_myelin'][0]
        prior_means_column[7] = params['vf_myelin'][0]*volumefraction_population_variance[0]

        prior_means_column[8] = params['vf_csf'][0]
        prior_means_column[9] = params['vf_csf'][0]*volumefraction_population_variance[1]
        #prior_means_column[9] = params['vf_ies'][0]

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

def convert_priors_to_synthseg_format_gauss(params_all_rois, opt, n_features=5, add_background_priors=True):
    # convert all priors into the synthnet format 
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
        prior_means_column[1] = params['mu_myelin'][0]*mu_population_variance[0]   # accessing variance of the mean measure of mu_myelin

        prior_means_column[2] = params['mu_ies'][0] 
        prior_means_column[3] = params['mu_ies'][0]*mu_population_variance[1]

        prior_means_column[4] = params['mu_csf'][0]
        prior_means_column[5] = params['mu_csf'][0]*mu_population_variance[2]
        #prior_means_column[5] = params['mu_csf'][0]

        prior_means_column[6] = params['vf_myelin'][0]
        prior_means_column[7] = params['vf_myelin'][0]*volumefraction_population_variance[0]

        prior_means_column[8] = params['vf_ies'][0]
        prior_means_column[9] = params['vf_ies'][0]*volumefraction_population_variance[1]
        #prior_means_column[9] = params['vf_ies'][0]

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

def prepare_opt(opt):

    # Create new directory 
    opt['datadir'] = opt['datadir']+opt["experiment_name"]+'/'
    os.makedirs(opt['datadir'],exist_ok=True)
    print(f"Saving result to: {opt['datadir']}")

    # read priors from pandas dataframe 
    opt = read_priors(opt)

    # if priors need to be verified
    if 'check_priors' in opt and opt['check_priors']:
        assert 'roi_name_ids' in opt
        check_priors(params_all_rois, opt['roi_name_ids'])


    # save all variables to json file (necessary as we also want to save priors)
    save_to_file(opt)

    return opt, opt['datadir']

def save_to_file(opt):

    opt_copy = opt.copy() # create a copy so that we don't change values to 'list' in actual array
    for k in opt_copy:
        if isinstance(opt_copy[k],np.ndarray):
            # turn numpy arrays into list (cannot save numpy in dictionary)
            opt_copy[k] = opt_copy[k].tolist()

    # save to json 
    sv.write_to_json(opt_copy,opt['datadir'] + "opt.json")
    # save to pkl
    sv.pickle_dump(opt['datadir'] + "opt.pkl", opt)






# def example_31b(opt): # repeat example 30 (vars set to 10% of means) but with tissue n parcel segmentations

#     # generate completely unrealistic signals - both in parameter dimension and in spatial dimension 
#     opt = example_29(opt)
#     opt["experiment_name"] = 'test31_generate_from_pkl_priors_285_rois_inc_tissues_10percent_variances'



#     # over ride default extra-options with the following 
#     #segdir = '/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/output_4/'
#     #segfile = "Hammers_mith-n30r95-MaxProbMap-full-MNI152-SPM12_resamp_reg_irtk_tr_reg_npeye.nii.gz"
#     label_map = "/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/output_4/label_dir_2_do_not_delete/" # provide directory to multiple segmentations 
    
#     # over ride default extra-options with the following 
#     modified_opt = {'sample_gmm':True,
#                     'blurring':False, 
#                     'vary_prior_means_and_stds':False,
#                     'debug_prior_values':False,
#                     'clipping':False,
#                     'min_max_norm':False,
#                     'gamma_augmentation':False,
#                     'apply_bias_field':False, 
#                     'apply_nonlin_trans':False,
#                     'apply_linear_trans':False,
#                     'flipping':False, 
#                     'path_label_map':label_map,
#                     'prior_distribution':'normal',  # if set to normal - the variance in parameters over population will be normal (this should be default practically always - do not set this to uniform - this is where the mistake was for example_24-26 - possibly the reason why generation wasn't great)
#                     'n_examples':5,
#                     'output_shape':None,
#                     'vf_pop_var':0.0, # vary within 10% of what is set 
#                     'mu_pop_var':0.0, # vary within 5% of what is set                               
#                     'save_prior_values':True, 
#                     'check_priors':False,
#                     'roi_name_ids':[44]}                 

#     """
#         :param scaling_bounds: (optional) if apply_linear_trans is True, the scaling factor for each dimension is
#         sampled from a uniform distribution of predefined bounds. Can either be:
#         1) a number, in which case the scaling factor is independently sampled from the uniform distribution of bounds
#         (1-scaling_bounds, 1+scaling_bounds) for each dimension.
#         2) a sequence, in which case the scaling factor is sampled from the uniform distribution of bounds
#         (1-scaling_bounds[i], 1+scaling_bounds[i]) for the i-th dimension.
#         3) a numpy array of shape (2, n_dims), in which case the scaling factor is sampled from the uniform distribution
#          of bounds (scaling_bounds[0, i], scaling_bounds[1, i]) for the i-th dimension.
#         4) the path to such a numpy array.
#         If None (default), scaling_range = 0.15
        
    
#         :param rotation_bounds: (optional) same as scaling bounds but for the rotation angle, except that for cases 1
#         and 2, the bounds are centred on 0 rather than 1, i.e. (0+rotation_bounds[i], 0-rotation_bounds[i]).
#         If None (default), rotation_bounds = 15.
        
#         :param shearing_bounds: (optional) same as scaling bounds. If None (default), shearing_bounds = 0.01.
        
#         :param nonlin_std: (optional) If apply_nonlin_trans is True, maximum value for the standard deviation of the
#         normal distribution from which we sample the first tensor for synthesising the deformation field.
        
    
#         :param nonlin_shape_factor: (optional) If apply_nonlin_trans is True, ratio between the size of the input label
#         maps and the size of the sampled tensor for synthesising the deformation field.
#     """
    
#     # over-ride SynthSeg options 
#     for k in modified_opt:
#         opt[k] = modified_opt[k]
    
#     # load priors from pickle 
#     rootdir = '/home/ch215616/w/code/mwf/experiments/s20210630-FULL-pipeline-hammers-to-mwf-prior-stats/single_image_example/libs/'
#     path_mean = rootdir + 'test_mean_v2_inc_tissue_n_parcel.pkl'
#     path_std = rootdir + 'test_std_v2_inc_tissue_n_parcel.pkl'   
#     dfs_mean = pd.read_pickle(path_mean)
#     dfs_std = pd.read_pickle(path_std)
    

#     """
#     Currently statistics return the following: 

#     mu_myelin              0.024113
#     mu_ies                 0.081929
#     mu_csf                 1.029994
#     vf_myelin              0.035285
#     vf_ies                 0.548513
#     vf_csf                 0.416202
#     sigma_myelin           0.000961
#     sigma_ies              0.016912
#     sigma_csf              0.144797
#     Name: 44, dtype: object       


#     REQUIREMENTS: 

#     Parameters must be in this order: 
#     - mu - in milliseconds - e.g. 23ms for short component at up to 1500 ms for long components 
#     - sigma - in milliseconds - e.g. originally fixed to 5ms 
#     - vf - originally set as a fraction - e.g. 0.25 for myelin - but it is multiplied by 100 - > therefore if should be in the range that generally varies from 1-100
    
#     ACTIONS:

#     mu_myelin              0.024113 >> mult by 1000
#     mu_ies                 0.081929 >> mult by 1000
#     mu_csf                 1.029994 >> mult by 1000
#     vf_myelin              0.035285 >> mult by 100
#     vf_ies                 0.548513 >> mult by 100
#     vf_csf                 0.416202 >> mult by 100
#     sigma_myelin           0.000961 >> mult by 1000
#     sigma_ies              0.016912 >> mult by 1000
#     sigma_csf              0.144797 >> mult by 1000    

#     """

#     # mult fractions by 100 (since .nii.gz cannot store such small 0.0123 like numbers properly - just defaults to single values)
#     for p in ['vf_myelin', 'vf_ies', 'vf_csf']:
#         dfs_mean[p] = dfs_mean[p].multiply(100)
#         dfs_std[p] = dfs_mean[p].copy().divide(10)
    
#     for p in ['mu_myelin', 'mu_ies', 'mu_csf','sigma_myelin', 'sigma_ies', 'sigma_csf']:
#         dfs_mean[p] = dfs_mean[p].multiply(1000)
#         dfs_std[p] = dfs_mean[p].copy().divide(10)

#     # export parameters from pandas dataframe into a dictionary (historical compatibility reasons)
#     params_all_rois = {}
#     for roi in dfs_mean.index:
        
#         # select dataframe series for given roi
#         dfs_mean_roi = dfs_mean.loc[roi].to_dict()
#         dfs_std_roi = dfs_std.loc[roi].to_dict()

#         # fuse mean and std measurements together 
#         dfs_roi = {}
#         for k in dfs_mean_roi.keys():
#             if k != 'roi_name': # avoid double copying the roi_name
#                 dfs_roi[k] = [dfs_mean_roi[k], dfs_std_roi[k]]
#             else:
#                 dfs_roi[k] = dfs_mean_roi[k]
        
#         params_all_rois[roi] = dfs_roi

#     # add background priors 
#     params_all_rois = add_background_priors(params_all_rois, background_label=0)

#     opt['prior_means'], opt['prior_stds'],roi_map = convert_priors_to_synthseg_format_gauss(params_all_rois, opt)


#     # set generation labels 
#     # make sure that the order of labels is the same as are the columns of prior_means and prior_sts with their corresponding rois
#     #opt['generation_labels'] = np.array([1,2,3,4,5])
#     #opt['generation_labels'] = np.array([int(i) for i in list(roi_map.keys())]) # the keys of roi_map dict contains the order, which we must present as a np.array of ints to the synthseg algorithm
#     # import generation labels from file (pre computed - much better)
#     # note that in future sections - we need to make sure that all regions are reflected     
#     opt['generation_labels'] = np.array([int(i) for i in dfs_mean.index.tolist()] + [0])
#     tissue_names = {'2':'GM', '3':'CSF', '4':'WM', '5':'VEN'}
#     opt['generation_label_names'] = [tissue_names[i] + ' ' + j for i,j in zip(dfs_mean['tissue_id'].tolist(), dfs_mean['roi_name'].tolist())] + ['background']
    
#     # a couple of checks to make sure the mapping is correct after conversion
#     assert opt['prior_means'][:,0][0] == params_all_rois[str(opt['generation_labels'][0])]['mu_myelin'][0]
#     assert opt['prior_means'][:,-1][0] == params_all_rois[str(opt['generation_labels'][-1])]['mu_myelin'][0]    


#     # over-ride SynthSeg options 
#     for k in modified_opt:
#         opt[k] = modified_opt[k]
    
#     return opt    


# def convert_priors_to_synthseg_format_gauss_3ROIS(params_all_rois, opt, n_features=5, add_background_priors=True):
    
#     # legacy function

#     # function for historical backcompatibility when fitting with 3 rois only - background is set to 1 (instead of 0) label, and ventricles are assigned copy of the same parameters as csf

#     # NB prior name of the function = from_gauss_to_gauss_abstract

#     """This function takes the statistics for each MWF parameter for each ROI, 
#     and converts them into an appropriate format that can be ingested by the generative algorithm (synthseg). 

#     Args: 
#     params_all_rois (dict): dictionary with keys equal to rois. Each item of each key is a subdictionary with myelin model parameters as keys (i.e. {'corpus_callosum':{'mu_m':None, 'mu_ies':None, ... 'sigma_csf':None}}). Each items in each subdictionary holds two values - mean and std for the given MWF model parameter for the given roi
#                             e.g. roi id of '44' = corresponds to corpus callosum
#                             e.g. params_all_rois['44'] = {'vf_myelin': [20.0, 4.999999999999999], 'vf_ies': [80.0, 4.999999999999999], 'mu_myelin': [25.0, 0.0], 'mu_ies': [80.0, 0.0], 'mu_csf': [1750.0, 125.0]}
#     add_background_priors (bool): if set to true, we will add an additional keys to params_all_rois dictionary that would be equivalent to the background 
#     n_features (int): number of MWF parameters that we are simulating (e.g. mu_m,mu_ies,mu_csf,vf_m,vf_ies = 5 features)
    
#     """

#     # check if background labels need to be added
#     if add_background_priors:
        
#         # NB no conversion needed between gauss<>uniform as all entries are zero 
#         p = {}
#         p['roi_name'] = 'background'
#         p['mu_myelin'] = [0.0,0.0]
#         p['mu_ies'] = [0.0,0.0]
#         p['mu_csf'] = [0.0,0.0]  
#         p['vf_myelin'] = [0.0,0.0]
#         p['vf_ies'] = [0.0,0.0]
#         p['sigma_myelin']= [0.0,0.0]
#         p['sigma_ies']= [0.0,0.0]
#         p['sigma_csf']= [0.0,0.0]
        
#         #p['bias'] = [0.0,0.0]  

#         params_all_rois['1'] = p

#     # add ventricle priors also - copy values from csf 
#     params_all_rois['5'] = params_all_rois['3'].copy()
#     params_all_rois['5']['roi_name'] = 'ventricles'

#     # population variance for volume fraction and meanT2
#     volumefraction_population_variance = opt['vf_pop_var']
#     mu_population_variance = opt['mu_pop_var']

#     # extract the number of ROIs that must be generated 
#     n_rois = len(params_all_rois)    

#     # priors are sets of tuples given to the generation script for each feature (e.g. 'vf_myelin') and for each ROI (e.g. 'corpus callosum') - from which the gaussians are sampled. Tuples correspond to mean and std for each particular parameter.
#     prior_dimensions = n_features*2

#     # setup empty numpy tensors with correct dimensions. The columns correspond to rois. 
#     prior_means = np.zeros((prior_dimensions,n_rois))  # 5 is the number of regions of interest
#     prior_stds = np.zeros((prior_dimensions,n_rois))

#     # create a mapping between ordering of keys in a params dictionary (which correspond to ROI) and an index that will correspond to a column in a numpy array (necessary for generating priors)
#     roi_name_ids = params_all_rois.keys()
#     roi_nparray_column_index = range(0,len(params_all_rois.keys())) 
#     assert len(roi_name_ids) == len(roi_nparray_column_index)
#     roi_map = {k:v for k,v in zip(roi_name_ids, roi_nparray_column_index)}

#     # iterate over each roi to fill in output prior_means and prior_stds (numpy format is required by the generative algorithm)
#     for roi_name_id, column in roi_map.items():

#         # get params for this roi
#         params = params_all_rois[roi_name_id]  
        
#         prior_means_column = prior_means[:,column]
#         prior_stds_column = prior_stds[:,column]

#         # assuming no subject to subject variation -> prior_means_odd = prior_means_even, prior_stds_odd = prior_stds_even
#         prior_means_column[0] = params['mu_myelin'][0]   # accessing mean measure of mu_myelin
#         prior_means_column[1] = params['mu_myelin'][0]*mu_population_variance   # accessing variance of the mean measure of mu_myelin

#         prior_means_column[2] = params['mu_ies'][0] 
#         prior_means_column[3] = params['mu_ies'][0]*mu_population_variance

#         prior_means_column[4] = params['mu_csf'][0]
#         prior_means_column[5] = params['mu_csf'][0]*mu_population_variance

#         prior_means_column[6] = params['vf_myelin'][0]
#         prior_means_column[7] = params['vf_myelin'][0]*volumefraction_population_variance

#         prior_means_column[8] = params['vf_ies'][0]
#         prior_means_column[9] = params['vf_ies'][0]*volumefraction_population_variance

#         prior_stds_column[0] = params['mu_myelin'][1] # setting the intra-subject intra-roi variance of mu_myelin for this specific roi 
#         prior_stds_column[1] = 0 # setting population variance for the variance measure of mu_myelin to zero 
#         prior_stds_column[2] = params['mu_ies'][1]
#         prior_stds_column[3] = 0
#         prior_stds_column[4] = params['mu_csf'][1]
#         prior_stds_column[5] = 0
#         prior_stds_column[6] = params['vf_myelin'][1]
#         prior_stds_column[7] = 0
#         prior_stds_column[8] = params['vf_ies'][1]
#         prior_stds_column[9] = 0

#         # fill (put back the column)
#         prior_means[:,column] = prior_means_column
#         prior_stds[:,column] = prior_stds_column


#     return prior_means, prior_stds

