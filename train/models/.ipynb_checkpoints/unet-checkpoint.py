# THIS IS A LEGACY FILE - excludes hyperunet version of the model (since it cannot be used with tch1_yml environment - because it houses tf<2.0)

import numpy as np 
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.math import exp,negative, add, multiply
import tensorflow as tf

def unet(img_size,channel_size,loss):

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
    kernel_size = 3 
    unet_resolution_levels = 4 
    conv_times = 3 
    filters_init = 64
    dropout_level = 0.1

    # define input layer
    inputs = keras.Input(shape=img_size + (channel_size,))
    
    # init skip connection list 
    skip_layers = []

    # ------------------ 
    # ENCODER
    # ------------------ 
    
    # downsampling the input only 
    x = downsample(inputs,filters_init,kernel_size,dropout_level)

    # downsampling each subsequent layer with increasing number of filters (based on layer number)
    for layer in range(unet_resolution_levels):
        
        # update number of filters in the block 
        filters = 2 ** layer * filters_init

        # convolve the input with the same number of filters 3 times 
        for _ in range(0,conv_times):
            x = downsample(x, filters, kernel_size, dropout_level)
            
        # add skip connection 
        skip_layers.append(x)
        
        # maxpool the results 
        x = layers.MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
    
    # restart number of filters (8*64) 
    filters = 2 ** unet_resolution_levels * filters_init
    
    # convole the output with convolution filter 3 times 
    for _ in range(0,conv_times):
        x = downsample(x, filters, kernel_size, dropout_level)

    # ------------------ 
    # DECODER 
    # ------------------ 
    
    # upsampling the result 
    for layer in range(unet_resolution_levels-1, -1, -1):
        
        # update number of filters in the block 
        filters = 2 ** layer * filters_init 
        
        # upsample the output 
        x = upsample(x,filters,kernel_size,dropout_level)
        
        # concatenate the skip connection to the output 
        x = layers.Concatenate(axis=-1)([x,skip_layers[layer]])
        
        # convole 3 times 
        for _ in range(0, conv_times):
            x = downsample(x,filters,kernel_size,dropout_level)

    # ------------------ 
    # OUTPUT LAYER
    # ------------------             
            
    # output layer 
    outputs = layers.Conv2D(filters=1, kernel_size=kernel_size,padding="same")(x)
    outputs = keras.backend.abs(outputs) # only allow positive values to be predicted 

    # load into the model
    model = keras.Model(inputs, outputs)

    # ------------------ 
    # COMPILE
    # ------------------             
    
    # Compile 
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00008), loss=loss)
    

    return model
    
def unet_ivim2(img_size,channel_size, loss)

    # build basic unet 
    model = unet(img_size,channel_size, loss)
    
    # remove previously added 2 output layers 
    x = model.layers[-3].output

    # init parameters
    kernel_size = 3 
    
    
    # ------------------ 
    # OUTPUT LAYER
    # ------------------             
            
    # output layer with FOUR convolutions
    outputs = layers.Conv2D(filters=4, kernel_size=kernel_size,padding="same")(x)
    outputs = keras.backend.abs(outputs) # only allow positive values to be predicted 

    outputs
    # ------------------ 
    # PARAMETERS > MRI SIGNAL      S(b) = S0*(f*exp(-b*D_fast)+(1-f)*exp(-b*D_slow))
    # ------------------      
    
    # define parameters
    S0 = outputs[:,:,:,0,None]
    D = outputs[:,:,:,1,None]
    Dstar = outputs[:,:,:,2,None]
    f = outputs[:,:,:,3,None]
    
    
    # Set constants 
    ones = tf.constant(1, dtype=tf.float32)   
    bvals_in = [0,50,100,200,400,600,800]
    bvals = tf.convert_to_tensor(np.array(bvals_in) / 1e3, dtype="float32") 
    
    # exp(-b*D_fast)
    Dfast_coef = exp(multiply(negative(bvals), Dstar))
    # f * exp(-b*D_fast)
    Dfast_frac = multiply(f,Dfast_coef)
    # exp(-b*D_slow)
    Dslow_coef = exp(multiply(negative(bvals), D))
    # (1-f) * exp(-b*D_slow)
    Dslow_frac = multiply(add(ones,negative(f)),Dslow_coef)
    # S0 * (exp(-b*D_slow) + (1-f) * exp(-b*D_slow)) 
    signals = multiply(S0, add(Dslow_frac, Dfast_frac))
    
    # load into the model
    model = keras.Model(inputs, signals)

    # ------------------ 
    # COMPILE
    # ------------------             
    
    # Compile 
    model.compile(optimizer=keras.optimizers.Adam(lr=0.00008), loss=loss)
    

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
    
