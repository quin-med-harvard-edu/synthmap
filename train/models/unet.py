# THIS IS A LEGACY FILE - excludes hyperunet version of the model (since it cannot be used with tch1_yml environment - because it houses tf<2.0)

import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers
#from tensorflow.math import exp,negative, add, multiply # not available in tensorflow 2.0 (on e2)
from tensorflow import math
import tensorflow as tf
import keras.backend as K
from kerastuner import HyperModel

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
    output_nc = 1 if input_type == 'mwf' else 4 
    outputs = layers.Conv2D(filters=output_nc, kernel_size=p['kernel_size'],padding="same")(x)
    #outputs = keras.backend.abs(outputs) # only allow positive values to be predicted 
    

    # load into the model
    model = keras.Model(inputs, outputs)

    # ------------------ 
    # COMPILE
    # ------------------             
    
    # Compile 
    model.compile(optimizer=keras.optimizers.Adam(lr=otherparams.lr), loss=loss)
    #model.compile(optimizer=keras.optimizers.Adam(lr=0.0008), loss=loss)
    
    

    return model

class Unet(HyperModel):
    def __init__(self, args, loss):
        self.args = args
        self.loss = loss       


    def build(self, hp):

        # downsampling in the encoder
        def downsample(inputs,filters,kernel_size,dropout_level, activation):
            
            x = layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="same")(inputs)
            x = layers.Activation(activation)(x)
            x = layers.Dropout(rate=dropout_level)(x)

            return x

        # upsampling in the decoder
        def upsample(inputs,filters,kernel_size,dropout_level,activation):
            
            x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,padding="same",strides=(2,2))(inputs)
            x = layers.Activation(activation)(x)
            x = layers.Dropout(rate=dropout_level)(x)

            return x

        def configure_tunable_parameters(self,hp):
            # Search for the configuration of hyperparameters in the config file. Set to default value if config for the given parameter is fixed (fix==True)
            d = {}
            args = self.args

            # ----
            # Check all public attributes of hyperparameter configuration object and select only those which are fixed 
            # ----

            # get names of attributes
            obj_attr = [name for name in dir(args.tunerconfig) if not name.startswith('_')]
            obj_attr.remove('to_dict') 

            # set all 'fixed' attributes (SEE YAML CONFIG FILE)
            for key in obj_attr:
                attr = getattr(args.tunerconfig, key)
                if attr.fix:
                    # set to fixed value 
                    d[key] = attr.default 

            # OLD unused
            # # if the tunable parameter is set to 'fix' inside the configuration file, we fetch it's default value instead. 
            # for key in d.keys():
            #     attr = getattr(args.tunerconfig, key)
            #     d[key] = attr.default if attr.fix else d[key]


            # now check if attribute had already been set in the dictionary, if not - proceed to configure it with kerastuner

            if not 'optimizer' in d:
                d['optimizer'] = hp.Choice(
                    "optimizer",
                    values=args.tunerconfig.optimizer.values,
                    default=args.tunerconfig.optimizer.default) 
            # if not d.has_key('adam_learning_rate'):                    
            #     d['adam_learning_rate'] = hp.Float(
            #         "adam_learning_rate",
            #         min_value=float(args.tunerconfig.adam_learning_rate.values[0]),
            #         max_value=float(args.tunerconfig.adam_learning_rate.values[1]),
            #         sampling=args.tunerconfig.adam_learning_rate.sampling,
            #         default=args.tunerconfig.adam_learning_rate.default) 
            #         # min_value=1e-3,
            #         # max_value=8e-6,
            #         # sampling="LOG",
            #         # default=8e-5) # 0.00008                
            if not 'learning_rate' in d:
                d['learning_rate'] = hp.Float(
                    "learning_rate",
                    min_value=float(args.tunerconfig.learning_rate.values[0]),
                    max_value=float(args.tunerconfig.learning_rate.values[1]),
                    sampling=args.tunerconfig.learning_rate.sampling,
                    default=args.tunerconfig.learning_rate.default) 
            if not 'batchsize' in d:
                d['batchsize'] = hp.Choice(
                    "batchsize",
                    values=args.tunerconfig.batchsize.values,
                    default=args.tunerconfig.batchsize.default) 
                # d['loss_function'] = hp.Choice(
                #     "loss_function",
                #     values=args.tunerconfig.loss_function.values,
                #     default=args.tunerconfig.loss_function.default) 
            if not 'limit_input_to' in d:
                d['limit_input_to'] = hp.Choice(
                    "limit_input_to",
                    values=args.tunerconfig.limit_input_to.values_percent,
                    default=args.tunerconfig.limit_input_to.default) 
            if not 'random_seed' in d:
                d['random_seed'] = hp.Choice(
                    "random_seed",
                    values=args.tunerconfig.random_seed.values,
                    default=args.tunerconfig.random_seed.default) 
            if not 'dropout_level' in d:
                d['dropout_level'] = hp.Float(
                    "dropout_level",
                    min_value=float(args.tunerconfig.dropout_level.values[0]),
                    max_value=float(args.tunerconfig.dropout_level.values[1]),
                    step=args.tunerconfig.dropout_level.step,
                    default=args.tunerconfig.dropout_level.default) 
            if not 'kernel_size' in d:
                d['kernel_size'] = hp.Choice(
                    "kernel_size",
                    values=args.tunerconfig.kernel_size.values,
                    default=args.tunerconfig.kernel_size.default) 
            if not 'unet_resolution_levels' in d:
                d['unet_resolution_levels'] = hp.Choice(
                    "unet_resolution_levels",
                    values=args.tunerconfig.unet_resolution_levels.values,
                    default=args.tunerconfig.unet_resolution_levels.default) 
            if not 'conv_times' in d:
                d['conv_times'] = hp.Choice(
                    "conv_times",
                    values=args.tunerconfig.conv_times.values,
                    default=args.tunerconfig.conv_times.default) 
            if not 'filters_init' in d:
                d['filters_init'] = hp.Choice(
                    "filters_init",
                    values=args.tunerconfig.filters_init.values,
                    default=args.tunerconfig.filters_init.default) 
            if not 'pooling' in d:
                d['pooling'] = hp.Choice(
                    "pooling",
                    values=args.tunerconfig.pooling.values,
                    default=args.tunerconfig.pooling.default)             
            if not 'activation' in d:
                d['activation'] = hp.Choice(
                    "activation",
                    values=args.tunerconfig.activation.values,
                    default=args.tunerconfig.activation.default)             

            self.hyperconfig = d

        # ------------------ 
        # INIT
        # ------------------ 

        # configure hyperparameter ranges
        configure_tunable_parameters(self, hp)
        
        # set hyperparameters 
        kernel_size = self.hyperconfig['kernel_size']
        unet_resolution_levels = self.hyperconfig['unet_resolution_levels']
        conv_times = self.hyperconfig['conv_times']
        filters_init = self.hyperconfig['filters_init']
        dropout_level = self.hyperconfig['dropout_level']
        pooling = self.hyperconfig['pooling']
        activation = self.hyperconfig['activation']
        if self.hyperconfig['optimizer'] == "adam":
            optimizer = keras.optimizers.Adam(lr=self.hyperconfig['learning_rate'])
        elif self.hyperconfig['optimizer'] == "sgd":
            optimizer = keras.optimizers.SGD(lr=self.hyperconfig['learning_rate'])


        # TBC at later date - hyperconfig for - lr_schedule, custom_combined_losses, random_seed, batchsize, limit_input_to
                        # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                        #     initial_learning_rate=1e-2,
                        #     decay_steps=10000,
                        #     decay_rate=0.9)
                        # optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)

        # define input layer
        inputs = keras.Input(shape=tuple(self.args.shape) + (self.args.input_nc,))
        
        # init skip connection list 
        skip_layers = []

        # ------------------ 
        # ENCODER
        # ------------------ 
        
        # downsampling the input only 
        x = downsample(inputs,filters_init,kernel_size,dropout_level,activation)

        # downsampling each subsequent layer with increasing number of filters (based on layer number)
        for layer in range(unet_resolution_levels):
            
            # update number of filters in the block 
            filters = 2 ** layer * filters_init

            # convolve the input with the same number of filters 3 times 
            for _ in range(0,conv_times):
                x = downsample(x, filters, kernel_size, dropout_level, activation)
                
            # add skip connection 
            skip_layers.append(x)
            
            # maxpool the results 
            if pooling == "max":
                x = layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
            elif pooling == "average":
                x = layers.AveragePooling2D(pool_size=(2,2),strides=(2,2))(x)
        
        # restart number of filters (8*64) 
        filters = 2 ** unet_resolution_levels * filters_init
        
        # convole the output with convolution filter 3 times 
        for _ in range(0,conv_times):
            x = downsample(x, filters, kernel_size, dropout_level,activation)

        # ------------------ 
        # DECODER 
        # ------------------ 
        
        # upsampling the result 
        for layer in range(unet_resolution_levels-1, -1, -1):
            
            # update number of filters in the block 
            filters = 2 ** layer * filters_init 
            
            # upsample the output 
            x = upsample(x,filters,kernel_size,dropout_level,activation)
            
            # concatenate the skip connection to the output 
            x = layers.Concatenate(axis=-1)([x,skip_layers[layer]])
            
            # convole 3 times 
            for _ in range(0, conv_times):
                x = downsample(x,filters,kernel_size,dropout_level,activation)

        # ------------------ 
        # OUTPUT LAYER
        # ------------------             
                
        # output layer 
        output_nc = 1 if self.args.input_type == 'mwf' else 4 
        outputs = layers.Conv2D(filters=output_nc, kernel_size=kernel_size,padding="same")(x)
        #outputs = keras.backend.abs(outputs) # only allow positive values to be predicted 
        

        # load into the model
        model = keras.Model(inputs, outputs)

        # ------------------ 
        # COMPILE
        # ------------------             
                
        # Compile 
        model.compile(optimizer=optimizer, loss=self.loss)
        
        return model




def ivim_loss(y_true,y_pred):

    # ------------------ 
    # PARAMETERS > MRI SIGNAL      S(b) = S0*(f*exp(-b*D_fast)+(1-f)*exp(-b*D_slow))
    # ------------------      
    
    # define parameters
    S0 = y_pred[:,:,:,0,None]
    D = y_pred[:,:,:,1,None]
    Dstar = y_pred[:,:,:,2,None]
    f = y_pred[:,:,:,3,None]
    
    # Set constants 
    ones = tf.constant(1, dtype=tf.float32)   
    bvals_in = [0,50,100,200,400,600,800]
    bvals = tf.convert_to_tensor(np.array(bvals_in) / 1e3, dtype="float32") 

    # exp(-b*D_fast)
    Dfast_coef = math.exp(math.multiply(math.negative(bvals), Dstar))
    # f * exp(-b*D_fast)
    Dfast_frac = math.multiply(f,Dfast_coef)
    # exp(-b*D_slow)
    Dslow_coef = math.exp(math.multiply(math.negative(bvals), D))
    # (1-f) * exp(-b*D_slow)
    Dslow_frac = math.multiply(math.add(ones,math.negative(f)),Dslow_coef)
    # S0 * (exp(-b*D_slow) + (1-f) * exp(-b*D_slow)) 
    signals = math.multiply(S0, math.add(Dslow_frac, Dfast_frac))

    #from tensorflow.python.ops import math_ops

    #y_true = math_ops.cast(y_true, y_pred.dtype)
    #y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    mse = K.mean(K.square(y_true - signals), axis=(-1))
    #mse = K.mean(math_ops.squared_difference(y_true - signals), axis=(-1))
    #mse = K.mean(K.square(y_true - signals), axis=(-1,-2,-3))

    return mse

# class IVIM(Layer):

#   def __init__(self, units=32):
#       super(IVIM, self).__init__()
#       self.units = units

#   def build(self, input_shape):  # Create the state of the layer (weights)
#     w_init = tf.random_normal_initializer()
#     self.w = tf.Variable(
#         initial_value=w_init(shape=(input_shape[-1], self.units),
#                              dtype='float32'),
#         trainable=True)
#     b_init = tf.zeros_initializer()
#     self.b = tf.Variable(
#         initial_value=b_init(shape=(self.units,), dtype='float32'),
#         trainable=True)

#   def call(self, inputs):  # Defines the computation from inputs to outputs
#       return tf.matmul(inputs, self.w) + self.b




# # DEPRECATED
# def unet_ivim(img_size,channel_size, loss):

#     # define input layer
#     inputs = keras.Input(shape=img_size + (channel_size,))

#     # build basic unet 
#     model = unet(img_size,channel_size, loss)
    
#     # remove previously added 2 output layers 
#     #x = model.layers[-3].output
#     x = model.layers[-2].output

#     # init parameters
#     kernel_size = 3 
    
    
#     # ------------------ 
#     # OUTPUT LAYER
#     # ------------------             
            
#     # output layer with FOUR convolutions
#     outputs = layers.Conv2D(filters=4, kernel_size=kernel_size,padding="same")(x)
#     #outputs = keras.backend.abs(outputs) # only allow positive values to be predicted 

    
#     # ------------------ 
#     # PARAMETERS > MRI SIGNAL      S(b) = S0*(f*exp(-b*D_fast)+(1-f)*exp(-b*D_slow))
#     # ------------------      
    
#     # define parameters
#     S0 = outputs[:,:,:,0,None]
#     D = outputs[:,:,:,1,None]
#     Dstar = outputs[:,:,:,2,None]
#     f = outputs[:,:,:,3,None]
    
    
#     # Set constants 
#     ones = tf.constant(1, dtype=tf.float32)   
#     bvals_in = [0,50,100,200,400,600,800]
#     bvals = tf.convert_to_tensor(np.array(bvals_in) / 1e3, dtype="float32") 
    
#     # exp(-b*D_fast)
#     Dfast_coef = exp(multiply(negative(bvals), Dstar))
#     # f * exp(-b*D_fast)
#     Dfast_frac = multiply(f,Dfast_coef)
#     # exp(-b*D_slow)
#     Dslow_coef = exp(multiply(negative(bvals), D))
#     # (1-f) * exp(-b*D_slow)
#     Dslow_frac = multiply(add(ones,negative(f)),Dslow_coef)
#     # S0 * (exp(-b*D_slow) + (1-f) * exp(-b*D_slow)) 
#     signals = multiply(S0, add(Dslow_frac, Dfast_frac))
    
#     # load into the model
#     model = keras.Model(inputs, signals)

#     # ------------------ 
#     # COMPILE
#     # ------------------             
    
#     # Compile 
#     model.compile(optimizer=keras.optimizers.Adam(lr=0.00008), loss=loss)
    
#     return model

# # DEPRECATED
# def unet_ivim2(img_size,channel_size, loss):

#     """TEMPORARY FUNCTION THAT RETURNS MODEL WITHOUT THE IVIM COMPUTATION"""

#     # define input layer
#     inputs = keras.Input(shape=img_size + (channel_size,))

#     # build basic unet 
#     model = unet(img_size,channel_size, loss)
    
#     # remove previously added 2 output layers 
#     #x = model.layers[-3].output
#     x = model.layers[-2].output

#     # init parameters
#     kernel_size = 3 
    
#     # ------------------ 
#     # OUTPUT LAYER
#     # ------------------             
            
#     # output layer with FOUR convolutions
#     outputs = layers.Conv2D(filters=4, kernel_size=kernel_size,padding="same")(x)
#     #outputs = keras.backend.abs(outputs) # only allow positive values to be predicted 
        
#     # load into the model
#     model = keras.Model(inputs, outputs)

#     # ------------------ 
#     # COMPILE
#     # ------------------             
    
#     # Compile 
#     model.compile(optimizer=keras.optimizers.Adam(lr=0.00008), loss=loss)
    
#     return model    


# DEPRECATED


# def ivim_loss2(y_true,y_pred):

#     # ------------------ 
#     # PARAMETERS > MRI SIGNAL      S(b) = S0*(f*exp(-b*D_fast)+(1-f)*exp(-b*D_slow))
#     # ------------------      
    
#     # define parameters
#     S0 = y_pred[:,:,:,0,None]
#     D = y_pred[:,:,:,1,None]
#     Dstar = y_pred[:,:,:,2,None]
#     f = y_pred[:,:,:,3,None]
    
#     # Set constants 
#     ones = tf.constant(1, dtype=tf.float32)   
#     bvals_in = [0,50,100,200,400,600,800]
#     bvals = tf.convert_to_tensor(np.array(bvals_in) / 1e3, dtype="float32") 
    
#     # exp(-b*D_fast)
#     Dfast_coef = exp(multiply(negative(bvals), Dstar))
#     # f * exp(-b*D_fast)
#     Dfast_frac = multiply(f,Dfast_coef)
#     # exp(-b*D_slow)
#     Dslow_coef = exp(multiply(negative(bvals), D))
#     # (1-f) * exp(-b*D_slow)
#     Dslow_frac = multiply(add(ones,negative(f)),Dslow_coef)
#     # S0 * (exp(-b*D_slow) + (1-f) * exp(-b*D_slow)) 
#     signals = multiply(S0, add(Dslow_frac, Dfast_frac))

#     squared_difference = tf.square(y_true - signals)

#     return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`


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
    
