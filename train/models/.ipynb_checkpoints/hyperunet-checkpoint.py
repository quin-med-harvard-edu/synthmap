from tensorflow import keras
from tensorflow.keras import layers
from kerastuner import HyperModel

class HyperUnet(HyperModel):

    def __init__(self, img_size,channel_size):
        self.img_size = img_size
        self.channel_size = channel_size
    
    def build(self, hp):
        # downsampling in the encoder
        def downsample(inputs,filters,kernel_size,dropout_level):
            
            x = layers.Conv2D(filters=filters, kernel_size=kernel_size,padding="same")(inputs)
            #        x = layers.Activation("relu")(x)
            x = layers.Activation(
                activation=hp.Choice(
                    "activation_down",
                    values=["relu", "tanh", "sigmoid"],
                    default="relu"))(x)
            
            #        x = layers.Dropout(rate=dropout_level)(x)
            x = layers.Dropout(
                rate=hp.Float(
                    "dropout_down", 
                    min_value=0.0, 
                    max_value=0.2, 
                    default=dropout_level, step=0.05))(x)


            return x

        # upsampling in the decoder
        def upsample(inputs,filters,kernel_size,dropout_level):
            
            x = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size,padding="same",strides=(2,2))(inputs)
            #        x = layers.Activation("relu")(x)
            x = layers.Activation(activation=hp.Choice(
                        "activation_up",
                        values=["relu", "tanh", "sigmoid"],
                        default="relu"))(x)        
            x = layers.Dropout(
                rate=hp.Float(
                    "dropout_up", min_value=0.0, max_value=0.2, default=dropout_level, step=0.05))(x)

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
        inputs = keras.Input(shape=self.img_size + (self.channel_size,))
        
        # init skip connection list 
        skip_layers = []

        # ------------------ 
        # ENCODER
        # ------------------ 
        
        # downsampling the input only 
        #    x = downsample(inputs,filters_init,kernel_size,dropout_level)
        x = downsample(inputs,
                    hp.Choice("init_filters", values=[32, 64], default=filters_init),
                    kernel_size,dropout_level)

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

        # load into the model
        model = keras.Model(inputs, outputs)

        # ------------------ 
        # COMPILE
        # ------------------             

        # Compile 
        #    model.compile(optimizer=keras.optimizers.Adam(lr=0.00008), loss=args.loss)
        model.compile(
            optimizer=keras.optimizers.Adam(
                hp.Float(
                    "learning_rate",
                    min_value=1e-3,
                    max_value=8e-6,
                    sampling="LOG",
                    default=8e-5, # 0.00008
                )
            ),
            loss=hp.Choice(
                "loss", 
                values=["mean_squared_error",
                                "mean_squared_logarithmic_error",
                                "mean_absolute_error"],
                default="mean_squared_error"),       
#            metrics=["val_loss"],
        )


        return model

