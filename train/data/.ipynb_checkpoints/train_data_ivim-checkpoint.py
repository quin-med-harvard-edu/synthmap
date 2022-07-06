import os 
import glob 


from train_data import split_data, cut_training_data, check_data_paths
from generator import DataGenerator

def get_data(args):

    # Get lists of X and Y  paths
    if args.ivim:
        X_paths, Y_paths = ivim_data(args)

    # check data paths 
    check_data_paths(X_paths, Y_paths)

    # cut training data 
    if args.limit_input_to is not None:
        X_paths, Y_paths = cut_training_data(X_paths,Y_paths,args.limit_input_to)       

    return X_paths, Y_paths

def ivim_data(args):

    def x2y(i):

        #REPLACE 'average6' with 'DIPY6' or similar (for supervision) - for unsupervised -just define new loss function
        return args.y + os.path.basename(i)

    # get paths 
    X_paths = glob.glob(args.x+'*.nii.gz')
    Y_paths = [x2y(i) for i in X_paths]

    X_paths=glob.glob("/home/ch215616/abd/IVIM_data/crohn/*/average6/slices/*.nii.gz")


    return X_paths, Y_paths


class IvimDataGenerator(DataGenerator):

    # over write default method 

    def __init__(self, args, img_size, input_img_paths, target_img_paths):
        self.batchsize = args.batchsize
        self.img_size = img_size
        self.normalize = args.normalize 
        self.noisevariance = args.noisevariance
        self.input_img_paths = input_img_paths
        self.mode = args.mode

        if self.mode == 'train':
            self.target_img_paths = target_img_paths

        #self.learning_type == args.learning_type  !!!!! INSERT THIS INTO GENERATOR INSTANCE DEFINTION

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batchsize
        batch_input_img_paths = self.input_img_paths[i : i + self.batchsize]
        x = np.zeros((self.batchsize,) + self.img_size + (7,), dtype="float32")

        for j, path in enumerate(batch_input_img_paths):

            img = nb.load(path).get_fdata()

            # check for extra slice dimension e.g. (160,192,1,32) instead of (160,192,32)
            if img.ndim == 4: 
                assert img.shape[-2] == 1, f"Current shape is {img.shape}. Third dimension must be 1, not {img.shape[-2]}"
                img = img[:,:,0,:]


            if self.normalize:
                img = self.normalize_image(img)

            if self.noisevariance is not None:
                img = self.add_noise(img, self.noisevariance) 

            x[j] = img

        if self.mode == 'train':

            if self.learning_type == 'supervised':
                channels = 4 
            elif self.learning_type == 'self-supervised':
                channels = 7 

            batch_target_img_paths = self.target_img_paths[i : i + self.batchsize]
            y = np.zeros((self.batchsize,) + self.img_size + (channels,), dtype="float32")

            for j, path in enumerate(batch_target_img_paths):
                img = nb.load(path).get_fdata()

                assert img.ndim == 4, f"Supervised IVIM training requires input dim to be 4"
                assert img.shape[-2] == 1, f"Current shape is {img.shape}. Third dimension must be 1, not {img.shape[-2]}"

                y[j] = img

            return x,y

        else:
            return x    


def loss():


        import sys 

        #from tensorflow as tf
        from tensorflow.math import multiply, add, subtract, exp, negative

        

        if args.data == 'ivim':
            channels = 7
        elif args.data == 'mwf':
            channels = 32


        # define input layer
        inputs = keras.Input(shape=img_size + (channels,))  


        # ------------------ 
        # OUTPUT LAYER
        # ------------------             

        def ivim_forward_model(features):

            # output layer 
            parameters = layers.Conv2D(filters=4, kernel_size=kernel_size,padding="same")(features) 

            # Forward model 
            S0 = parameters[:,:,:,0:1] # S0 
            D = parameters[:,:,:,1:2] # D
            Dstar = parameters[:,:,:,2:3] # Dstar
            f = parameters[:,:,:,3:4] # f

            # IVIM equation: 
            # S(b) = S0*(f*exp(-b*Dstar)+(1-f)*exp(-b*D))
            ONE = tf.constant(1, dtype=tf.float32)
            perf_term = multiply(f,exp(multiply(negative(bvals),Dstar))) # f*exp(-b*Dstar)
            diff_term = multiply(subtract(ONE, f),exp(multiply(negative(bvals),D)))  # (1-f)*exp(-b*D)
            outputs = multiply(add(perf_term,diff_term), S0)  

            return outputs 


            def mwf_forward_model(features):


                std_myelin = std_ies = std_csf = 5 # ms 

                # output layer 
                parameters = layers.Conv2D(filters=5, kernel_size=kernel_size,padding="same")(features) 

                # Forward model 
                mu_myelin = parameters[:,:,:,0:1] # mu_myelin
                mu_ies = parameters[:,:,:,1:2] # mu_ies
                mu_csf = parameters[:,:,:,2:3] # mu_csf
                MWF = parameters[:,:,:,3:4] # MWF
                IEWF = parameters[:,:,:,4:5] # IEWF


                # MWF equation: 
                # S(TE) = MWF*N(mu_myelin,std_myelin) + IEWF*N(mu_ies,std_ies) + (1-MWF-IWEF)*N(mu_csf,std_csf)
                # QUESTION - how to implement normal distribution in tensorflow? 
                # QUESTION2 - how to sample 60 values from the distribution vector? 

                outputs = myelin_component + ies_component + csf_component

                return outputs2      

        if args.data == 'ivim' and args.learning_type == 'self-supervised':

            outputs = ivim_forward_model(x)

        elif args.data == 'ivim' and args.learning_type == 'supervised':
            # output layer 
            outputs = layers.Conv2D(filters=4, kernel_size=kernel_size,padding="same")(x)

        elif args.data == 'mwf' and args.learning_type == 'supervised':
            # output layer 
            outputs = layers.Conv2D(filters=1, kernel_size=kernel_size,padding="same")(x)

        elif args.data == 'mwf' and args.learning_type == 'self-supervised':
            # output layer 
            outputs1 = layers.Conv2D(filters=1, kernel_size=kernel_size,padding="same")(x)
            
            sys.exit('not implemented yet')
            
            # output layer 
            outputs2 = layers.Conv2D(filters=1, kernel_size=kernel_size,padding="same")(x)
            
            #> return two different values to loss function and weigh them accordingly... 

        # load into the model
        model = keras.Model(inputs, outputs)  

    

