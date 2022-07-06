import numpy as np 
            
    

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def get_custom_T2range():
    """ This range takes a set of linearly spaced T2 values on a vector length of 2000 
    (varying between 10 and 2000)
    and converts them into a vector of 60 distributions, that were logspaced. 
    
    """
    # T2 from synthetic dataset
    T2_samples_synth = 2000 
    T2range = np.linspace(10, 2000, num=T2_samples_synth)  # T2 must be defined in the same range as pre-generated EPG dictionary       
    
    # T2 expected by MIML neural network
    T2_samples_miml = 60 
    T2range_miml = np.logspace(np.log10(10), np.log10(2000), num=T2_samples_miml)  # T2 must be defined in the same range as pre-generated EPG dictionary

    
    # now let's build a custom T2 range 
    indices = []
    T2range_custom_60 = []
    for i in T2range_miml:
        matched_value, idx = find_nearest(T2range, i)
        indices.append(idx)
        T2range_custom_60.append(matched_value)
    T2range_custom_60 = np.array(T2range_custom_60)
    #T2range_custom_60.shape
    #T2range_miml.shape
    T2range_2000_indices = indices
    
    return T2range_custom_60, T2range_2000_indices
            
