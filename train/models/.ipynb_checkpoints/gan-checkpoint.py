
def discriminator():
    
    pass



# SOURCE - machine learning master https://machinelearningmastery.com/how-to-develop-cyclegan-models-from-scratch-with-keras/
# define the discriminator model
def define_discriminator(image_shape):
	# weight initialization
	init = RandomNormal(stddev=0.02)
	# source image input
	in_image = Input(shape=image_shape)
	# C64
	d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
	d = LeakyReLU(alpha=0.2)(d)
	# C128
	d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C256
	d = Conv2D(256, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# C512
	d = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# second last output layer
	d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
	d = InstanceNormalization(axis=-1)(d)
	d = LeakyReLU(alpha=0.2)(d)
	# patch output
	patch_out = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
	# define model
	model = Model(in_image, patch_out)
	# compile model
	model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
	return model
 


def addGAN(args,model):

    # add a GAN to the training routine 
    #parser.add_argument('--GAN', action='store_true', help='add a GAN to the transfer learning procedure - the model should be able to reproduce the true / fake images as close as possible')       
    #parser.add_argument('--GANlossweight', type=float, help='how much weighting does GAN loss get?')       


    # 3. Add three more layers
    # ll = model.layers[8].output
    # ll = Dense(32)(ll)
    # ll = Dense(64)(ll)
    # ll = Dense(num_classes,activation="softmax")(ll)
    # new_model = Model(inputs=model.input,outputs=ll)

    
    # tf.keras.applications.Xception(
    #     include_top=True,
    #     weights="imagenet",
    #     input_tensor=None,
    #     input_shape=None,
    #     pooling=None,
    #     classes=1000,
    #     classifier_activation="softmax",
    # )


    model.layers[-1]

    return model 
