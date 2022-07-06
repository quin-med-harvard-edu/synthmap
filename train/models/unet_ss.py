# THIS IS A LEGACY FILE - excludes hyperunet version of the model (since it cannot be used with tch1_yml environment - because it houses tf<2.0)

import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.math import exp,negative, add, multiply # not available in tensorflow 2.0 (on e2)
from tensorflow import math
import tensorflow as tf
import keras.backend as K


def unet(img_size,channel_size,loss,input_type,otherparams=None):

    # downsampling in the encoder
    def downsample(inputs,filters,kernel_size,dropout_level):
        
        x = layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="same")(inputs)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(rate=dropout_level)(x)

        return x

    # upsampling in the decoder
    def upsample(inputs,filters,kernel_size,dropout_level):
        
        x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,padding="same",strides=(2,2))(inputs)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(rate=dropout_level)(x)

        return x

    
    # ------------------ 
    # INIT
    # ------------------ 
    
    # init parameters
    p = {}
    p['kernel_size'] = 3 
    p['unet_resolution_levels'] = 4 
    p['conv_times'] = 3 
    p['filters_init'] = 64
    p['dropout_level'] = 0.1

    # update init parameters if specified 
    if otherparams is not None: 
        for k in p.keys():
            if k in otherparams:
                if otherparams[k] != p[k]:
                    print(f"ATTENTION: Default parameter {k.upper()} changed from {p[k]}: to: {otherparams[k]}")
                p[k] = otherparams[k]
                


    # define input layer
    inputs = keras.Input(shape=img_size + (channel_size,))
    
    # init skip connection list 
    skip_layers = []

    # ------------------ 
    # ENCODER
    # ------------------ 
    
    # downsampling the input only 
    x = downsample(inputs,p['filters_init'],p['kernel_size'],p['dropout_level'])

    # downsampling each subsequent layer with increasing number of filters (based on layer number)
    for layer in range(p['unet_resolution_levels']):
        
        # update number of filters in the block 
        filters = 2 ** layer * p['filters_init']

        # convolve the input with the same number of filters 3 times 
        for _ in range(0,p['conv_times']):
            x = downsample(x, filters, p['kernel_size'], p['dropout_level'])
            
        # add skip connection 
        skip_layers.append(x)
        
        # maxpool the results 
        x = layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
    
    # restart number of filters (8*64) 
    filters = 2 ** p['unet_resolution_levels'] * p['filters_init']
    
    # convole the output with convolution filter 3 times 
    for _ in range(0,p['conv_times']):
        x = downsample(x, filters, p['kernel_size'], p['dropout_level'])

    # ------------------ 
    # DECODER 
    # ------------------ 
    
    # upsampling the result 
    for layer in range(p['unet_resolution_levels']-1, -1, -1):
        
        # update number of filters in the block 
        filters = 2 ** layer * p['filters_init'] 
        
        # upsample the output 
        x = upsample(x,filters,p['kernel_size'],p['dropout_level'])
        
        # concatenate the skip connection to the output 
        x = layers.Concatenate(axis=-1)([x,skip_layers[layer]])
        
        # convole 3 times 
        for _ in range(0, p['conv_times']):
            x = downsample(x,filters,p['kernel_size'],p['dropout_level'])

    # ------------------ 
    # OUTPUT LAYER
    # ------------------             
            
    # output layer 
    #output_nc = 1 if input_type == 'mwf' else 4 
    output_nc = 2
    outputs = layers.Conv2D(filters=output_nc, kernel_size=p['kernel_size'],padding="same")(x)
    #outputs = keras.backend.abs(outputs) # only allow positive values to be predicted 


    # ------------------ 
    # Self supervised layer 
    # ------------------             

    # task 1: outputs must predict 2 parameters (vf_myelin and vf_csf)

    # source: w/code/mwf/experiments/s20210901_self_supervised/train.py

    # define fixed params 
    !!! image_size (not batchsize)
    fixed_params = get_fixed_params(image_size) # should reflect size of image (just make it into a tensor)
    

    # load EPG dictionary 
    EPG_dict = get_EPG() # 

    # load flip angle range (must match with simulated EPG dictionary )
    flip_angle_range = get_flip_angle_range()

    # get a list of T2 samples to simulate the T2 distribution 
    T2range = get_T2()   

    # add parameter bounds for learned mwf parameters 
    bounds = get_parameter_bounds()
    
    # add 2*pi squared in torch 
    sqrt_2pi = torch_2pi_squared()
    
    # store opt also 
    #self.opt = opt




!!!! fix this 
    # pass the input through encoder 
    out = encoder(X)    



    # placeholder for params
    est_params = {}
    
    # apply softmax to volume fractions to guarantee that the fractions add up to 1 
    # slice it: https://discuss.pytorch.org/t/how-to-apply-different-activation-fuctions-to-different-neurons/13475
    vfs = torch.softmax(out[:,1:4],1,torch.float32)
    est_params['vf_m'] = vfs[:,0].unsqueeze(1)
    est_params['vf_ies'] = vfs[:,1].unsqueeze(1)
    est_params['vf_csf'] = vfs[:,2].unsqueeze(1)
    
    # extract params into corresponding dict (easier to operate)
    est_params['mu_ies'] = out[:, 0].unsqueeze(1) 
    #est_params['vf_m'] = out[:, 1].unsqueeze(1) 
    #est_params['vf_ies'] = out[:, 2].unsqueeze(1) 
    #est_params['vf_csf'] = out[:, 3].unsqueeze(1) 
    est_params['fa'] = out[:, 4].unsqueeze(1) 
    
    
    # REMOVING ANY BOUNDS - letting the algorithm run free
    est_params['mu_ies'] = bounds['mu_ies'][0] + \
                                        torch.sigmoid(est_params['mu_ies']) * \
                                        bounds['mu_ies'][2]
#         est_params['vf_m'] = self.bounds['vf_m'][0] + \
#                                             torch.sigmoid(est_params['vf_m']) * \
#                                             self.bounds['vf_m'][2]
#         est_params['vf_ies'] = self.bounds['vf_ies'][0] + \
#                                             torch.sigmoid(est_params['vf_ies']) * \
#                                             self.bounds['vf_ies'][2]
#         est_params['vf_csf'] = self.bounds['vf_csf'][0] + \
#                                             torch.sigmoid(est_params['vf_csf']) * \
#                                             self.bounds['vf_csf'][2]
    est_params['fa'] = bounds['fa'][0] + \
                                        torch.sigmoid(est_params['fa']) * \
                                        bounds['fa'][2] 
    
    # estimate signal 
    X_est = mwf_forward_model(est_params)
    
    # normalize the signal - divide signal by first echo
    X_est = torch.div(X_est, X_est[:,0].unsqueeze(1))


    # load into the model
    model = keras.Model(inputs, outputs)

    # ------------------ 
    # COMPILE
    # ------------------             
    
    # Compile 
    model.compile(optimizer=keras.optimizers.Adam(lr=otherparams.lr), loss=loss)
    #model.compile(optimizer=keras.optimizers.Adam(lr=0.0008), loss=loss)
    
    

    return model


def get_EPG_path():
    
    # CORRECTED EPG generating function 
    file = "/home/ch215616/w/mwf_data/synthetic_data/training_data/flip_angle_dictionary/s20210805_v2/20210805-simulated_dictionary2000of2000.npy"    
    print(f"PATH TO EPG_DICT -> {file}")
    
    return file 

def get_EPG():

    # get epg dict 
    epg_dir = get_EPG_path()
    EPG_dict = np.load(epg_dir)
    
    #EPG_dict = torch.FloatTensor(EPG_dict)
    EPG_dict = K.constant(EPG_dict)
    
    return EPG_dict


def get_flip_angle_range():
    
    # define flip angle 
    fixed_FA = 180.     
    flip_angle_samples = 320
    input('how many input samples are there?')
    flip_angle_range = np.linspace(120,280,flip_angle_samples)
    
    # expand dims of the flip angles used in EPG dict simulation 
    flip_angle_range_exp = np.expand_dims(flip_angle_range,axis=0)  # (N,) -> (N,1)
    #flip_angle_range_exp = torch.FloatTensor(flip_angle_range_exp)
    flip_angle_range_exp = K.constant(flip_angle_range_exp)

    return flip_angle_range_exp

def get_fixed_params(batchsize):
    
   
    # load non varying MWF model parameters 
    # the only parameters to be learned are: 3 volume fractions, mu_ies which should vary between 100-125ms 
    # based on Chatterjee 2018 - Multi-compartment model of brain tissues from T2 relaxometry MRI using gamma distribution
    # https://hal.archives-ouvertes.fr/hal-01744852/document
    sigma_myelin_v, sigma_ies_v, sigma_csf_v = load_sigmas()
    mu_myelin_v, mu_csf_v = load_means()
    
    # tile parameter arrays according to size of batch
    mu_myelin_v = np.tile(mu_myelin_v, (batchsize,1))
    mu_csf_v = np.tile(mu_csf_v, (batchsize,1))
    sigma_myelin_v = np.tile(sigma_myelin_v, (batchsize,1))
    sigma_ies_v = np.tile(sigma_ies_v, (batchsize,1))
    sigma_csf_v = np.tile(sigma_csf_v, (batchsize,1))
    
    # turn parameters arrays into torch Tensors 
    # mu_myelin_v = torch.FloatTensor(mu_myelin_v)
    # mu_csf_v = torch.FloatTensor(mu_csf_v)
    # sigma_myelin_v = torch.FloatTensor(sigma_myelin_v)
    # sigma_ies_v = torch.FloatTensor(sigma_ies_v)
    # sigma_csf_v = torch.FloatTensor(sigma_csf_v)   
    mu_myelin_v = K.constant(mu_myelin_v)
    mu_csf_v = K.constant(mu_csf_v)
    sigma_myelin_v = K.constant(sigma_myelin_v)
    sigma_ies_v = K.constant(sigma_ies_v)
    sigma_csf_v = K.constant(sigma_csf_v)   

    
    #save into dictionary 
    params = {'mu_myelin_v':mu_csf_v,
              'mu_csf_v':mu_csf_v,
              'sigma_myelin_v':sigma_myelin_v,
              'sigma_ies_v':sigma_ies_v,
              'sigma_csf_v':sigma_csf_v}
    

    return params

def load_sigmas():
    # based on Chatterjee 2018 - Multi-compartment model of brain tissues from T2 relaxometry MRI using gamma distribution
    # https://hal.archives-ouvertes.fr/hal-01744852/document
    
    # load sigmas 
    sigma_myelin_ = np.sqrt(50)
    sigma_ies_ = np.sqrt(100)
    sigma_csf_ = np.sqrt(6400)    
    sigma_myelin_v = np.expand_dims(np.array(sigma_myelin_),0)
    sigma_ies_v = np.expand_dims(np.array(sigma_ies_),0)
    sigma_csf_v = np.expand_dims(np.array(sigma_csf_),0)
    
    return sigma_myelin_v, sigma_ies_v, sigma_csf_v

def load_means():
    # based on Chatterjee 2018 - Multi-compartment model of brain tissues from T2 relaxometry MRI using gamma distribution
    # https://hal.archives-ouvertes.fr/hal-01744852/document

    # load gaussian means 
    mu_m = 30
    mu_csf = 2000
    mu_m_v = np.expand_dims(np.array(mu_m),0)
    mu_csf_v = np.expand_dims(np.array(mu_csf),0)    
    
    return mu_m_v, mu_csf_v

def get_T2():
    
    # get T2 range 
    t2rangetype='linspace'
    T2_samples = 2000
    T2range = np.linspace(10, 2000, num=T2_samples)  # T2 must be defined in the same range as pre-generated EPG dictionary        
    batch_size = 1
    T2range = np.tile(T2range,(batch_size,1))  

    # convert to torch 
    # T2range_torch = torch.FloatTensor(T2range)    
    T2range_torch = K.constant(T2range)        
    
    return T2range_torch



def get_parameter_bounds():

    # we fix parameters bounds based on our estimates 
#     bounds={'mu_ies':[100,125],
#             'vf_m':[0,1],
#             'vf_ies':[0,1],
#             'vf_csf':[0,1],
#             'fa':[120,280]}

    input('warning - bounds are incorrrect . proceed?')
    bounds={'mu_ies':[10,500],
            'vf_m':[0,1],
            'vf_ies':[0,1],
            'vf_csf':[0,1],
            'fa':[50,500]}



    # precalculate 'range' of max - min 
    for k,v in bounds.items():
        bounds[k].append(bounds[k][1]-bounds[k][0])

    return bounds

def torch_2pi_squared():    
    
    # preallocated 2pi**2 as a torch tensor to save comp time
    #pi = K.constant([np.pi])
    sqrt_2pi = np.sqrt(2*np.pi)
    sqrt_2pi = K.constant([sqrt_2pi])

    return sqrt_2pi    

    def mwf_forward_model(self,est_params):

        """ the following lines estimate the MWF signal from given the estimate MWF parameters and the b1 value (flip angle ratio) """

        # perform basic scale checks for the params
        for k in ['vf_m', 'vf_ies','vf_csf']:        
            # all volume fractions must be in 0-1 range 
            assert torch.all(est_params[k]>=0) and torch.all(est_params[k]<=1), embed(header=sv.msg())

        # mu_ies has to be in the range of 100-125 
        ##boundchecks
        #assert torch.all(est_params['mu_ies']>=self.bounds['mu_ies'][0]) and torch.all(est_params['mu_ies']<=self.bounds['mu_ies'][1]), embed(header=sv.msg())

        # fa have to be in the min-max range of flip angles 
        #assert torch.all(est_params['fa']>=self.bounds['fa'][0]) and torch.all(est_params['fa']<=self.bounds['fa'][1]), embed(header=sv.msg())


        # get indices of flip angles that are matching to current signal
        fa = est_params['fa'] #fa * 180    
        indx_batch = find_matching_flip_angles_torch(fa,
                                                     self.flip_angle_range) 

        # match EPG dictionary - get pre-caluclated EPG signal evolution based on indices
        EPG_batch=self.EPG_dict[:,indx_batch,:]
        
        # estimate individual gaussians
        gauss_myelin = get_gaussian_torch(self.T2range,
                                          self.fixed_params['mu_myelin_v'],
                                          self.fixed_params['sigma_myelin_v'],
                                          est_params['vf_m'],self.sqrt_2pi)
        gauss_ies    = get_gaussian_torch(self.T2range,
                                          est_params['mu_ies'],
                                          self.fixed_params['sigma_ies_v'],
                                          est_params['vf_ies'],self.sqrt_2pi)
        gauss_csf    = get_gaussian_torch(self.T2range,
                                          self.fixed_params['mu_csf_v'],
                                          self.fixed_params['sigma_csf_v'],
                                          est_params['vf_csf'],self.sqrt_2pi)    

        # get T2 density function 
        pT2 = gauss_myelin+gauss_ies+gauss_csf 

        # get signals from pT2 distribution and EPG dict 
        signals = estimate_echoes_torch(pT2,EPG_batch)

        return signals    


def get_gaussian_torch(T2range,mu,sigma,vf,sqrt_2pi):     

    # subtract mu from T2 range n
    res = torch.sub(T2range,mu)

    
    ### BELOW lines cover the following equation
        # gauss = np.exp(-0.5*res**2/sigma**2)/(sigma*np.sqrt(2*np.pi))    
        
    # denominator
    denominator = torch.mul(sigma,sqrt_2pi)
    
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

if __name__ == '__main__':
    
    # debug tools 
    img_size = (160, 192)
    channel_size = 7
    loss = 'mean_squared_error'
    net = unet(img_size,channel_size,loss)
    
    net.summary()
    len(net.layers)

    
    
    net.layers[-1].output
    net.layers[-2].output
    net.layers[-3].output
    
