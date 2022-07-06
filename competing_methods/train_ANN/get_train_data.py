import os 

import numpy as np 


def load_training_data(dataset='te10'):

    if dataset.startswith('te'):
        d = os.get_cwd() + '/datasets/'


    # About te10 dataset 
        # load original data defined in the paper (note that TE distance is 10ms, not 9ms, and T2dist is 60, not 40)
        # The data below can be downloaded from: https://drive.google.com/drive/folders/1IoxOtAt-8NiFgbtZh1RDY32Jb_Wd5TGa?usp=sharing
        # The data below is based on 32 echos with a beginning echo time of 10.36
    # About te9 dataset 
        # This dataset was constructed manually for us by MIML authors. They had sent us the raw data (pre normalization). 
        # The full dataset was then reconstructed with `construct_dataset.py`
    
    #assert dataset == 'te9' or 'te10' 
    
        folder = d+dataset+'/'
    else: 
        
        assert os.path.exists(dataset)
        assert os.path.isdir(dataset)
        assert os.path.exists(dataset+'/signals_train.npy')
        assert os.path.exists(dataset+'/signals_valid.npy')
        assert os.path.exists(dataset+'/distributions_train.npy')
        assert os.path.exists(dataset+'/distributions_valid.npy')
        folder = dataset + '/'
        
    trainX=np.load(folder+'signals_train.npy')
    trainY=np.load(folder+'distributions_train.npy')
    valX=np.load(folder+'signals_valid.npy')
    valY=np.load(folder+'distributions_valid.npy')
    
    
    return trainX, trainY, valX, valY
