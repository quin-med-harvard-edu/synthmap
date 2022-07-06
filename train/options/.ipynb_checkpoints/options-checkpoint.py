import argparse

import svtools as sv

class BaseArgs():
    
    """Defines Base args for test and train"""

    def __init__(self,fromnotebook=False):

        """ Define shared input arguments"""
        if fromnotebook:
            self.parser = sv.init_argparse()
        else:
            self.parser = argparse.ArgumentParser()

        # Training
        self.parser.add_argument('--batchsize',type=int,default=8)
        self.parser.add_argument('--gpu', type=int, required=True, help='select GPU to run the model on')
        self.parser.add_argument('--custom_checkpoint_dir', type=str, default=None, help='if specified, weights will be saved to this directory')

        # Input Modifications
        self.parser.add_argument('--normalize', action='store_true', help='Normalize input. Not necessary for synthetic data as generated signals (echoes) are already normalized')
        
        self.parser.add_argument('--noisevariance', '-noise', type=int, nargs='+', default=None, help='specify lower and upper bounds for random uniform noise to be added to each input image')
        self.parser.add_argument('--limit_input_to', type=int, default=None, help='limit input to specific number of images')
        self.parser.add_argument('--normalize_type', type=str,choices=['norm_by_first_echo','norm_by_voxel'],help='Type of normalization to use on input image. ')        
        self.parser.add_argument('--reject_outliers', action='store_true', help='reject outliers when normalizing the image. Only applies to norm_by_first_echo method')
        
        # choose input type
        self.parser.add_argument('--input_type', type=str, choices=['mwf','ivim'], default = 'mwf', help='choose which data to train with')       
        
        

    def get_args(self):
        
        pass # shell method 

    def parse(self):

        # get extra args
        self.get_args()

        # parse args 
        args = self.parser.parse_args()

        # Print args
        print("Executing with the following args:")
        sv.print_args(args)

        return args


class TrainArgs(BaseArgs):

    def get_args(self):
        
        # Name
        self.parser.add_argument('--name','-n', type=str, required=True, help='experiment name')

        # Data 
        self.parser.add_argument('--x','-x','--signaldir', type=str,required=True,help='specify location of training data - signals')
        self.parser.add_argument('--y','-y','--paramdir',  type=str,required=True,help='specify location of training data - parameter maps')

        # Training 
        self.parser.add_argument('--loss', type=str, default="mean_squared_error")
        self.parser.add_argument('--epochs', type=int,default=300, help='epochs to train for')
        self.parser.add_argument('--valsize', type=int,default=500, help='validation size')
        self.parser.add_argument('--resume_training', type=str, default=None, help='if specified, it must point to the latest epoch from which training should resume')
        self.parser.add_argument('--resume_epochs', type=int, default=None, help='specify the new epoch from which training will begin.')
        
        # Transfer learning options 
        self.parser.add_argument('--transfer_learning', action='store_true', help='freeze N layers in the models and train from scratch on new data')       
        self.parser.add_argument('--pretrained_weights', type=str, default=None, help='specify directory from which pre trained weights would be fetched')
        self.parser.add_argument('--trainable_layers', type=int, default=None, help='specify number N of last trainable layers that will not be frozen (from the end of the model)')    
        self.parser.add_argument('--labels_source', type=str, choices=['julia','julia_norm','miml'],default=None, help='define the source algorithm from which the labels are taken for transfer learning')    

        # Training data type 
        self.parser.add_argument('--training_data_type', type=str, choices=['synthetic', 'real'], default = 'synthetic', help='choose training data type')               

        # Evaluate hyperparameters with kerastuner
        self.parser.add_argument('--kerastuner', action='store_true', help='evaluate multiple hyperparameters via kerastuner')       
        self.parser.add_argument('--maxepochs', type=int, default=30, help='maximum number of epochs that any good model will train in tuner')       
        self.parser.add_argument('--executions_per_trial', type=int, default=2, help='maximum number of repetitions per trial of each parameter')       

        # self supervised mode
        self.parser.add_argument('--selfsupervised', action='store_true', help='trains the model in self supervised mode based on forward equation')       
        
        # add a GAN to the training routine 
        self.parser.add_argument('--GAN', action='store_true', help='add a GAN to the transfer learning procedure - the model should be able to reproduce the true / fake images as close as possible')       
        self.parser.add_argument('--GANlossweight', type=float, help='how much weighting does GAN loss get?')       

        # train in patches - should filter size be 32x32x32?
        self.parser.add_argument('--patchNet', action='store_true', help='read the input data in patches instead of whole image.')       

        # Legacy 
        self.parser.add_argument('--ismrm', action='store_true', help='matches the data routine built for ISMRM conference submission')        

        # mode 
        self.parser.add_argument('--mode', type=str, default='train',help='sets mode')       

class TestArgs(BaseArgs):

    def get_args(self):
    
        # Trained Network
        self.parser.add_argument('--trained_weights', '-t', type=str, required=True, help='specify directory from which weights would be fetched')

        # Save to 
        self.parser.add_argument('--savedir', type=str, required=True, help='path to directory where to save the results')

        # Data 
        self.parser.add_argument('--test_dir', type=str, default=None, help='folder with test data')
        self.parser.add_argument('--test_regexp', type=str, nargs='+',default=None, help='test on specific subject identifier inside a folder')
        self.parser.add_argument('--test_file', type=str, default=None, help='test on specific file')   

        # mode 
        self.parser.add_argument('--mode', type=str, default='test',help='sets mode')       

        

if __name__ == '__main__':

    """USED FOR TESTS ONLY """

    # 1. Test from CLI 
    args = TrainArgs().parse()
    sv.print_args(args)
    
    # 2. Load previously saved network and convert to obj
    import sys 
    sys.path.append('/home/ch215616/code/ext/')
    from bunch.bunch import Bunch 
    args_example = '/home/ch215616/fastscratch/mwf_data/synthetic_data/trained_weights/unet-test21-20211001-2042-noise-40to100-/args.json'
    args = sv.read_from_json(args_example)
    args = Bunch(args)

    # 3. Debug argparse
    args = TrainArgs()
