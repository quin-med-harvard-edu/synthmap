"""

    Generate a B1 field onto a brain slice. B1 field will vary as a 2D gaussian. 
    Variation will be such that the center voxel of a given brain will have a signal intensity equivalent of 1.1-1.3, 
    and the furthest edges will drop off to 0.7-0.9. 
    
    A b1 field of 1.0 is assume to be the field with a perfect 180deg refocussing angle for a T2 pulse 
    
    
    The algorithm works in the following way: 
    
    1. Find a bounding box around the brain image (assumes that background has a value of zero )
    2. Generate a 2D gaussian that falls within the bounds of the signal ranges between center and edge voxels, 
        as described above. The gaussian parameters are sampled from a random normal gaussian distribution. 
    3. Fit the generated smoothly varying gaussian to the brain image and clip all values that are outside the brain 
    

"""
import sys 

import numpy as np 
import nibabel as nb 
from scipy.stats import multivariate_normal
import time 


import svtools as sv
#sys.path.append('/home/ch215616/code/SynthSeg/sv407_ismrm/')
import dilation_closing



def generate_n_save_B1_example2():
    

    """Generate another example with 180. multiplication"""

    d = '/home/ch215616/code/SynthSeg/sv407_ismrm/test14_multiple_segmentations_targets/'
    im_path = d + 'params_1_test.nii.gz'    
    
    
    var_sigma_slice2slice = False

    savename = generate_B1_volume_3D(im_path,var_sigma_slice2slice)

    sv.itksnap(savename)


"""NOTES TO FIX 

- GENERATE ONE SINGLE SIGMA FOR THE ENTIRE VOLUME. 
- Fix those dark spots (what are they??? )
- make sure that ratio actually works - why does it fall outside of the bounds??

"""   



def generate_n_save_B1_example():
    

    """A function that demonstrates B1 field generation for a parameter file and saves it to a nifti file with correct header """

    d = '/home/ch215616/code/SynthSeg/sv407_ismrm/test14_multiple_segmentations_targets/'
    im_path = d + 'params_1.nii.gz'    
    
    
    var_sigma_slice2slice = False

    savename = generate_B1_volume_3D(im_path,var_sigma_slice2slice)

    sv.itksnap(savename)


"""NOTES TO FIX 

- GENERATE ONE SINGLE SIGMA FOR THE ENTIRE VOLUME. 
- Fix those dark spots (what are they??? )
- make sure that ratio actually works - why does it fall outside of the bounds??

"""   


def generate_B1_volume_3D_array(im,var_sigma_slice2slice=False, scale=1.):
    
    """Generates B1 field for a parameter file and returns as array"""
    
    
    x,y,z,params = im.shape

    # load a single 3D volume to use in the B1 field generation 
    image = im[:,:,:,3]
    B1volume = np.zeros_like(image)
    
    
        
    # sample a randon uniform value of gaussian sigma such that 
    # ... it results in the B1 range of values that we defined within the bounding box (see docstring)
    
    
    
    # OLD BOUNDS - NO LONGER USED - THE FLIP ANGLE WAS NOT HIGH ENOUGH TO SHOW A LOT OF VARIATION IN VALUES, ESPECIALLY AT THE EDGESE OF THE BRAIN 
    bounds = [1.38,1.58] # these are the empirically observed bounds for the sigma of the 2D gaussian which keep the center to edge ratio within the [1.1/0.9, 1.3/0.7] 
    
    # NEW BOUNDS - WILL VARY SIGNAL BETWEEN 1.1 and 1.43 of 180 FLIP ANGLE - RESULTING IN STIMULATED ECHO EFFECT EVERYWHERE IN THE IMAGE
    bounds = [1.18,1.98]
    
    
    sigma1 = sample_distribution(bounds,distribution='uniform')
    sigma2 = sample_distribution(bounds,distribution='uniform')
    sigma3 = sample_distribution(bounds,distribution='uniform')



    
    B1volume=generate_B1_volume(image,sigma1,sigma2,sigma3, var_sigma_slice2slice)


    # perform morphological closing on the B1 field: 
    B1volume = dilation_closing.close_array(B1volume,closing_type='grey')


    # rescale the volume so that the max values (at the center for the slice) correspond to Flip angle of ~1.2
    #B1_ceiling = np.random.uniform(1.0,1.2) # the maximum B1 in an image would vary between these values
    B1_ceiling = np.random.uniform(1.1,1.53) # the maximum B1 in an image would vary between these values
    maxx = np.max(B1volume)
    scaling_factor = B1_ceiling/maxx
    B1volume = np.multiply(B1volume,scaling_factor)
    #sv.temp_itksnap(B1volume)

    return B1volume
    
    
def generate_B1_volume_3D(im_path,var_sigma_slice2slice=False):
    
    """Generates B1 field for a parameter file and saves it to a nifti file with correct header """
    

    start = time.time()

    # apply filter to all parameters 
    imo = nb.load(im_path)
    im = imo.get_fdata()

    
    B1volume = generate_B1_volume_3D_array(im,var_sigma_slice2slice)
#     print('Before multiplication')
#     print(np.max(B1volume))
#     print(B1volume.dtype)
#     print(np.argmax(B1volume))
#     B1volume = np.multiply(180., B1volume)
#     print('After multiplication')
#     print(np.max(B1volume))
#     print(np.argmax(B1volume))

    header = imo.header.copy()
    header.set_data_shape(B1volume.shape)
    imnewo = nb.Nifti1Image(B1volume, affine = imo.affine, header=header)
    #print('WARNING: may need to reduce number of dimensions to those of the file')
    
    savename = im_path.replace('.nii.gz','_B1_3D.nii.gz')

    nb.save(imnewo,savename)

    end = time.time()
    print(f"Processed param volume in: {end-start:0.2f} seconds.\n{im_path}\n.Saved as\n{savename}")

    return savename
    
    
    
def generate_B1_volume(image,sigma1,sigma2,sigma3,var_sigma_slice2slice):

    """
    Args: 
        image (numpy.ndarray): 2D image where the background has zero signal (this is important requirement)
        sigma1,sigma2 (float): sampled sigma of the 2D gaussian that will be usd for this slice(&volume). Sampled from a uniform distribution within the predefined bounds.
        
    Returns: 
        field map (numpy.ndarray): an array of the size of the image with the field map, within the predefined bounds. 
        
        
        
    Predefined bounds: 
        The ratio of center voxel to edge voxel in the bounding box should be no more than 1.3/0.7 and no less than 1.1/0.9. 
        
    
    
    """
    
    
    
    """ Detailed explanation: 
    
    Our goal is to produce a 2D gaussian that would scale within the bounding box that defines the adult brain 
    
    The gaussian should be defined in such a way that the central value, at x(0,0), is floating around the value of 1-1-1.3, 
    which is the typical value of B1 field in the brain. 
    And that the edges are 0.7-0.8, which are also the typical values for the B1 field in the brain. 
    
    Since the shape of the brain is assymetric - the gaussian should be assymetric too. 
    
    The function we are searching for is: 
    p(x0,sigma, mu=0) = ~1.2
    p(xEdge,sigma, mu=0) = ~0.7 
    
    I.e. the center to edge ratio: min_max =  [1./0.9, 1.3/0.7] 
    
    We fix mu to be zero centered.
    
    The 'mesh' over which we define this gaussian does not play a role in the ratio of center to edges (it just defined a grid). 
    
    For 1.1/0.9 ratio, sigma = 1.58 
    For 1.3/0.7 ratio, sigma = 1.38 
    
    
    Therefore - we want sigma to vary within these bounds 99% of the time - i.e. 3std away. 
    """
    

    center_edge_ratio_bounds = [1.1/0.9, 1.3/0.7]
    
    # get bounding box around the image 
    bounding_box = bound_rectangle(image)
    
    assert len(bounding_box) == 6, f"bounding box should be a list of 6 values, xmin,xmax,ymin,ymax,zmin,zmax"
    
    
    # get bounding box around the brain and define a complex grid over which the values will be plotted 
    x1,x2,y1,y2,z1,z2 = bounding_box
    grid1 = complex(x2-x1)
    grid2 = complex(y2-y1)    
    grid3 = complex(z2-z1)    

    if var_sigma_slice2slice:    
        # introduce a little bit of pertrubation to sigma 
        sigma1 = np.random.uniform(sigma1-sigma1*0.01,sigma1+sigma1*0.01)
        sigma2 = np.random.uniform(sigma2-sigma2*0.01,sigma2+sigma2*0.01)
        sigma3 = np.random.uniform(sigma3-sigma3*0.01,sigma3+sigma3*0.01)


    # generate B1 field 
    z = gauss_2d(grid1,grid2,grid3,sigma1,sigma2,sigma3)
    
    debug = False
    if debug:
        # check that the image are withing the predefined bounds and plot it 
        import svtools as sv
        # sv.plot_slice(z,'generated B1 field without scaling')
        # check the value as the edges 
        x,y = z.shape
        edge1 = z[0,0]
        edge2 = z[x-1,0]
        edge3 = z[x-1,y-1]
        edge4 = z[0,y-1]
        center = z[x//2,y//2]
        ratio1 = center/edge1
        ratio2 = center/edge2        
        ratio3 = center/edge3
        ratio4 = center/edge4        
        
        assert center_edge_ratio_bounds[0]<ratio1<center_edge_ratio_bounds[1], f"the ratio of center to edge in the generated image does not fall into the predefined B1 bounds. {ratio1}"
        assert center_edge_ratio_bounds[0]<ratio2<center_edge_ratio_bounds[1], f"the ratio of center to edge in the generated image does not fall into the predefined B1 bounds. {ratio2}"
        assert center_edge_ratio_bounds[0]<ratio3<center_edge_ratio_bounds[1], f"the ratio of center to edge in the generated image does not fall into the predefined B1 bounds. {ratio3}" 
        assert center_edge_ratio_bounds[0]<ratio4<center_edge_ratio_bounds[1], f"the ratio of center to edge in the generated image does not fall into the predefined B1 bounds. {ratio4}" 
    
    # overlay the field over the image 
    
    
    B1 = overlay_b1_over_brain(image,generated_gaussian=z,bounding_box=bounding_box) 
    
    return B1
    
    
def bound_rectangle(im):

    """Get a bounding box around a brain. This function assumes the background is set to zero """
    
    # debug 
    #     imones = np.zeros_like(im)
    #     imones[np.where(im <=0)] = 1

    #     sv.plot_slice(imones,'dfs')
    
    a = np.where(im >0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1]), np.min(a[2]), np.max(a[2])
    return bbox

    def test_bound_rectangle():
        [x1,x2,y1,y2] = bound_rectangle(im)
        bounding_box = [x1,x2,y1,y2]
        sv.plot_slice(im[x1:x2,y1:y2],'rect')        


def overlay_b1_over_brain(image,generated_gaussian,bounding_box):
    
    
    """
    Overlay a generated gaussian distribution within the bounding box that defines the brain inside an image, 
    onto the brain itself, such that the gaussian is only defined within the brain bounds and not outside of it 
    
    """

    x1,x2,y1,y2,z1,z2 = bounding_box
    B1 = np.zeros_like(image)

    # fit the generated 2D gaussian into an array which has the exact same size as the image 
    B1[x1:x2,y1:y2,z1:z2] = generated_gaussian

    # calculate a mask over the brain region only 
    mask = np.zeros_like(image)
    mask[image>0] = 1

    # fit the mask over the B1 field to generate the brain with B1 field 
    B1_masked = np.multiply(mask,B1)

    # debug 
    debug = False 
    if debug:
        sv.plot_slice(z,'generated gaussian')
        sv.plot_slice(image,'image')

        sv.plot_slice(B1,'B1 with gaussian')
        sv.plot_slice(B1_masked,'B1 masked by image tissue')

        plt.contour(z)

        # check the edge of the 

    return B1_masked 
        

    
def gauss_2d(grid1,grid2,grid3, sigma1,sigma2,sigma3):
    
    """
    Generate a 2D gaussian for a given grid size, with specific sigma values. 
    Mu is set to zero. 
    
    
    """
    mu=0.0
    x, y, zz = np.mgrid[-1.0:1.0:grid1, -1.0:1.0:grid2, -1.0:1.0:grid3]
    # Need an (N, 2) array of (x, y) pairs.
    xyz = np.column_stack([x.flat, y.flat,zz.flat])

    mu = np.array([mu, mu, mu])

    sigma = np.array([sigma1, sigma2, sigma3])
    covariance = np.diag(sigma**3)

    z = multivariate_normal.pdf(xyz, mean=mu, cov=covariance)

    # Reshape back to a (30, 30) grid.
    z = z.reshape(x.shape)


    #     sv.plot_slice(z,'sfd')
    return z     

    
def range_to_gauss(min_max,confidence=0.95):

    """
    
    Convert a min-max range of desired values into a mean and std parameters for a normal distribution that fall 
    within the specified confidence interval. 
    
    Args: 
        min_max (list): min-max range for which a particule value will take 99% probability 

    Returns: 
        (mean,std) (float,float): mean and std of the normal distribution from which this value should be sampled, 
                                    in order to achieve 99% certainty within those values 

    """
    import sys 

    if confidence == 0.95: 
        scale_factor = 2 
    elif confidence == 0.997: 
        scale_factor = 3 
    elif confidence == 0.68: 
        scale_factor = 1 
    else:
        sys.exit('not implemented')

    mean = np.mean(min_max).tolist()

    std = (min_max[1]-mean)/scale_factor

    return [mean, std] 
    
    
    
def sample_distribution(hyperparameter, distribution='uniform'): 


    """
    Sample a value from a specified distribution given the hyperparameters
    
    Args:
        hyperparameters (float,float): parameters of the distribution 
        distribution (str): normal or uniform

    """

    if distribution == 'uniform':
        value = np.random.uniform(low=hyperparameter[0], high=hyperparameter[1])
    elif distribution == 'normal':
        value = np.random.normal(loc=hyperparameter[0], scale=hyperparameter[1])


    return value


    
    
    
    
################## Unused helper functions 



def load_example():
    # exact same slice where i had prasloski map as well 
    f='/home/ch215616/code/SynthSeg/sv407_ismrm/test14_multiple_segmentations_targets/signals-20201012-0335/signals_1_s38.nii.gz'
    import nibabel as nb 
    im = nb.load(f).get_fdata()    
    
    # take 5th echo for now 
    
    return im[:,:,5]

def test_load_example():
    im = load_example()
    sv.plot_slice(im,'asdf')

def center_of_mass(im):
    
    import scipy.ndimage as ndi 
    cx, cy = ndi.center_of_mass(im)  

    # debug 
    #     cx_,cy_ = int(cx),int(cy)    
    #     i = 15
    
    #     sv.plot_slice(im[cx_-i:cx_+i,cy_-i:cy_+i], 'asdf')
    
    return int(cx),int(cy)


def test_center_of_mass():
    cx,cy = center_of_mass(im)
    i = 20 
    sv.plot_slice(im[cx_-i:cx_+i,cy_-i:cy_+i], 'asdf')

    


######## UNUSED DEBUG FUNCTIONS 


# below are the debug functions previously used to choose the values of the 2d gauss
def scale_by_1_3(sigma=1.4):
    required_ratio = 1.3
    # find x where the value at the edge would be equal to 1/1.3 of the top given the current grid... 
    z=gaussian_2d(grid1_=200j,grid2_=grid2_, mu1_=0.0, mu2_=0.0, sigma1_=1.38,sigma2_=1.38)
    plt.plot(z[:,z.shape[1]//2])
    edge = z[0,z.shape[1]//2]
    mmax = z[z.shape[0]//2,z.shape[1]//2]
    print(mmax/edge)
    print(required_ratio)

def scale_by_1_3_by_0_7(sigma=0.9):
    required_ratio = 1.3/0.7
    # find x where the value at the edge would be equal to 1/1.3 of the top given the current grid... 
    z=gaussian_2d(grid1_=100j,grid2_=100j, mu1_=0.0, mu2_=0.0, sigma1_=0.9,sigma2_=0.9)
    plt.plot(z[:,z.shape[1]//2])
    edge = z[0,z.shape[1]//2]
    mmax = z[z.shape[0]//2,z.shape[1]//2]
    print(mmax/edge)
    print(required_ratio)

def scale_by_1_1_by_0_9(sigma):
    required_ratio = 1.1/0.9
    # find x where the value at the edge would be equal to 1/1.3 of the top given the current grid... 
    z=gaussian_2d(grid1_=100j,grid2_=100j, mu1_=0.0, mu2_=0.0, sigma1_=1.58,sigma2_=1.58)
    plt.plot(z[:,z.shape[1]//2])
    edge = z[0,z.shape[1]//2]
    mmax = z[z.shape[0]//2,z.shape[1]//2]
    print(mmax/edge)
    print(required_ratio)



def gaussian_2d_debug(grid1_=100j,grid2_=100j, mu1_=0.0, mu2_=0.0, sigma1_=0.025,sigma2_=0.025):
    x, y = np.mgrid[-1.0:1.0:grid1_, -1.0:1.0:grid2_]
    # Need an (N, 2) array of (x, y) pairs.
    xy = np.column_stack([x.flat, y.flat])

    mu = np.array([mu1_, mu2_])

    sigma = np.array([sigma1_, sigma2_])
    covariance = np.diag(sigma**2)

    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)

    # Reshape back to a (30, 30) grid.
    z = z.reshape(x.shape)


    #     sv.plot_slice(z,'sfd')
    return z     


if __name__ == '__main__':
    
    print('Running test example')
    # simple B1 field without 180 scaling 
    #generate_n_save_B1_example()
    
    # with 180 scaling 
    generate_n_save_B1_example2()