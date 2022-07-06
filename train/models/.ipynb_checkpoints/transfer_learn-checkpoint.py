import os 

from tensorflow import keras
#from tensorflow.keras.models import load_model
from models.gan import addGAN

from IPython import embed; 
import svtools as sv

def transfer_learning(args,model):


    # various checks 
    assert args.trainable_layers is not None, f"Please specify --trainable_layers"
    assert args.pretrained_weights is not None, f"Please specify --pretrained_weights"
    assert os.path.isfile(args.pretrained_weights), f"File does not exist"  # NB weights variable named as `pretrained_weights` named differently to not confuse with test data's `trained_weights` when running bash scripts
    assert args.pretrained_weights.endswith('.h5'), f"Specify full path to pretrained network"

    # load model 
    #model = load_model(args.pretrained_weights) 
    model.load_weights(args.pretrained_weights) 
    
    # freeze weights 
    model = freeze_weights(args,model)

    # add a GAN to the training routine 
    if args.GAN:
        model = addGAN(model) 

    # Must recompile the model IF freezing weights or changing the model
    model.compile(optimizer=keras.optimizers.Adam(lr=args.lr), loss=args.loss)

    return model
        
def freeze_weights(args,model):

    ####### EXAMPLES #######
    # 1. Freeze the first five layers
    # for i in range(5):
    #     model.layers[i].trainable = False

    # 2. Make the next three layers trainable, this can be ignored if all layers are trainable.
    # for i in range(5,8):
    #     model.layers[i].trainable = True        

    # 3. Add three more layers
    # ll = model.layers[8].output
    # ll = Dense(32)(ll)
    # ll = Dense(64)(ll)
    # ll = Dense(num_classes,activation="softmax")(ll)
    # new_model = Model(inputs=model.input,outputs=ll)

    # cycle through N last layers and set them as non trainable
    freeze_layers = len(model.layers) - args.trainable_layers
    embed(header=sv.msg('please inspect the layers. model.layers and args.trainable_layers'))
    
    for i in range(0, freeze_layers-1):
        model.layers[i].trainable = False
    
    return model