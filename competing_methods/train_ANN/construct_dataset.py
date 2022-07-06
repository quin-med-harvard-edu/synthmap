"""difference to construct_dataset.py 

- introduced random shuffling: shuffles all data properly - so that signals are not entangled between different tissues

"""


import os 
import numpy as np 

def add_noise_random(signal_array_raw,SNR_range):
    signal_array=signal_array_raw.copy()
    for i in range(signal_array.shape[0]):
        SNR=np.random.randint(SNR_range[0],SNR_range[1])
        sigma_met2    = signal_array[i,0]/SNR
        noise_met2_1  = np.random.normal(0, sigma_met2, 32)
        noise_met2_2  = np.random.normal(0, sigma_met2, 32)
        signal_array[i,:]  = np.sqrt((signal_array[i,:] + noise_met2_1) **2 + noise_met2_2**2)
    return signal_array


def construct_dataset():
    """ construct dataset with custom delta TE """
    
    # init SNR range 
    SNR_range=[80,200]  # used in the EPFL paper 
    
    # init paths 
    d = '/home/ch215616/w/mwf_data/MIML/'
    d2=d+'datasets/te9_source/'
    savedir_origin=d+'datasets/te9/' # original training set 
    savedir_20211214=d+'datasets/te9_20211214/' # new training set
    os.makedirs(savedir_20211214, exist_ok=True)
    
    
    # init variables 
    distributions_all = []
    signals_all = []
    
    
    # load signals and distirbutions for every tissue / segment        
    for s in ['CSF','CSF_GM','GM','Pathology', 'WM', 'WM_CSF', 'WM_GM']:

        print(s)
        
        signals = np.load(d2+s+'/'+'sample_signals_'+s.lower()+'.npy')
        distributions = np.load(d2+s+'/'+'distributions_'+s.lower()+'.npy')   
        
        # add noise 
        signals = add_noise_random(signals,SNR_range)
    
        # normalize to the first echo (same must be done at test time)
        S0 = signals[:,0:1]
        signals = np.divide(signals,S0)
                                                                
        # add to vector 
        signals_all.append(signals)    
        distributions_all.append(distributions)

    # fix dimensions 
    signals_all = np.array(signals_all)
    distributions_all = np.array(distributions_all)
    s = signals_all.shape
    signals_all = np.reshape(signals_all, (s[0]*s[1], s[2]))
    s = distributions_all.shape
    distributions_all = np.reshape(distributions_all, (s[0]*s[1], s[2]))    
    
    from IPython import embed; embed()
    # set seed 
    np.random.seed(42)
    # shuffle 
    #np.random.shuffle(signals_all) # DO NOT TURN IT ON - IT MESSED UP ALL THE DATA 
    #np.random.shuffle(distributions_all) 
    
    shuffler = np.random.permutation(len(signals_all))
    signals_all_sh = signals_all[shuffler]
    distributions_all_sh = distributions_all[shuffler]
    
    #save    
    #     np.save(savedir+'signals_valid.npy',signals_all[:140000])
    #     np.save(savedir+'signals_train.npy',signals_all[140000:])
    #     np.save(savedir+'distributions_valid.npy',distributions_all[:140000])
    #     np.save(savedir+'distributions_train.npy',distributions_all[140000:])   

    np.save(savedir_20211214+'signals_valid.npy',signals_all_sh[:140000])
    np.save(savedir_20211214+'signals_train.npy',signals_all_sh[140000:])   
    np.save(savedir_20211214+'distributions_valid.npy',distributions_all_sh[:140000])
    np.save(savedir_20211214+'distributions_train.npy',distributions_all_sh[140000:])   

    
if __name__ == '__main__':
    construct_dataset()