import sys
import copy 

import numpy as np
import nibabel as nb 
import cv2
from skimage import morphology
from tensorflow import keras 

import keras.preprocessing.image as tfp

class DataGenerator_train(keras.utils.Sequence):

    """ Prepare Sequence generator class to load & vectorize batches of data.
    
    Helps to iterate over the data (as Numpy arrays).
    """

    def __init__(self, args, img_size, channel_size, input_img_paths, target_img_paths):
        self.batchsize = args.batchsize
        self.img_size = img_size
        self.channel_size = channel_size
        self.normalize = args.normalize 
        self.noisevariance = args.noisevariance
        self.input_img_paths = input_img_paths
        self.mode = args.mode
        self.selfsupervised = True if args.selfsupervised else False

        self.normalize_type = args.normalize_type
        self.reject_outliers = args.reject_outliers
        self.mask_threshold = args.mask_threshold
        self.mask_type = args.mask_type
        self.augment_options = args.affine

        if self.mode == 'train':
            self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batchsize

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batchsize
        batch_input_img_paths = self.input_img_paths[i : i + self.batchsize]
        batch_target_img_paths = self.target_img_paths[i : i + self.batchsize]

        x = np.zeros((self.batchsize,) + self.img_size + (self.channel_size,), dtype="float32")
        y = np.zeros((self.batchsize,) + self.img_size + (1,), dtype="float32")
        
        
        for j, (path_X,path_Y) in enumerate(zip(batch_input_img_paths,batch_target_img_paths)):

            img_X = nb.load(path_X).get_fdata()
            img_Y = nb.load(path_Y).get_fdata()


            # check for empty slice
            if np.amax(img_X) == 0 or np.amax(img_Y) == 0:
                print(f"Empty Slice detected. Skipping")
                continue    

            # check for extra slice dimension e.g. (160,192,1,32) instead of (160,192,32)
            if img_X.ndim == 4: 
                assert img_X.shape[-2] == 1, f"Current shape is {img_X.shape}. Third dimension must be 1, not {img_X.shape[-2]}"
                img_X = img_X[:,:,0,:]
            if not img_Y.shape == self.img_size: 
                
                # TEMP HACK
                # in some instances the slices may be (160,192,1) in size instead of (160,192), which we fix here 
                if img_Y.ndim == 3 and img_Y.shape[:2] == self.img_size and img_Y.shape[-1]==1: 
                    img_Y = img_Y[:,:,0]
                debug = False 
                if debug: 
                    path_X
                    path_Y
                    img_Y.shape
                    self.img_size
                    img_X.shape
                    from IPython import embed; embed()                
            assert img_Y.shape == self.img_size, f"img_size is not the same as img.shape"            

            # mask the image
            if self.mask_type is not None:
                img_X = self.mask_image(img_X)    
            
            if self.augment_options['apply']:
                self.sample_augment_options()
                img_X,img_Y = self.augment(img_X,img_Y)

            if self.normalize:
                img_X = self.normalize_image(img_X)

            if self.noisevariance is not None:
                img_X = self.add_noise(img_X, self.noisevariance) 

            if self.img_size[1] != img_X.shape[1]: 
                assert self.img_size[1]-img_X.shape[1] == 4, f"Current implementation assumes that image mismatch can only have image size of 156, vs the required 160. Current image size is: {img_X.shape}"
                # pad the image with extra dims
                npad = ((0, 0), (2, 2), (0, 0)) # padding values to fix the size mismatch of the unet and our data (y-dimension)
                img_X = np.pad(img_X, pad_width=npad, mode='constant', constant_values=0)
                
                # NOTE to self - best to perform padding INDEPENDENTLY on a dataset (so that the code is not messed up)

            if img_Y.ndim==3 and img_Y.shape[-1]==1:
                y[j] = img_Y
            else:
                y[j] = np.expand_dims(img_Y, 2) 


            x[j] = img_X
            
        return x,y



            
            


    def mask_image(self,img):

        assert img.ndim == 3, f"Image shape should be [x,y,echoes]"

        # get first echo only
        im = img[:,:,0]
        echoes = img.shape[-1]
      
        if self.mask_type=='improved':
        
            # IMPROVED MASKING PROCESS
            # 1. Measure noise in the corners of the image
            # 2. Mask image by mean of noise (as threshold)
            # 3. Erode + Dilate 
            # 4. Add median blur to remove sharp edges
            # 5. Remove small objects + Remove small holes 

            # experiments: ~/w/code/ivim/crohns/experiments_create_better_masks_erode_dilate_remove_small_objects.py

            # 1. Measure noise in the corners of the image
            corners = im[0:20,0:20]+im[-20:,-20:]   #+ref_im[-20:,0:20,:]+ref_im[0:20,-20:,:] -> not so great 
            threshold = np.mean(corners)#+np.std(corners)

            # 2. Mask image by mean of noise (as threshold)
            mask = np.zeros_like(im)
            mask[im>threshold] = 1 

            # 3. Erode + Dilate     
            kernel_erode = np.ones((3,3), np.uint8) 
            kernel_dilate = np.ones((3,3), np.uint8) 
            img_erosion = cv2.erode(mask, kernel_erode, iterations=1) 
            img_dilation = cv2.dilate(img_erosion, kernel_dilate, iterations=1) 

            # 4. Add median blur to remove sharp edges
            img_dilation = cv2.medianBlur(img_dilation.astype('int16'),5)

            # 5. Remove small objects + Remove small holes 
            small_object_threshold = 2000 
            small_hole_threshold = 1000
                # NB must be done for each slice separately (else doesn't work)
                #slices = img_dilation.shape[-1]
                # for sl in range(0,slices): 
                #im = img_dilation#[:,:,sl]    
            arr = img_dilation > 0
            cleaned = morphology.remove_small_objects(arr, min_size=small_object_threshold)  # threshold 
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=small_hole_threshold) # source https://stackoverflow.com/questions/55056456/failed-to-remove-noise-by-remove-small-objects    
            # put back into the mask
            mask[:,:] = cleaned.astype(mask.dtype)
        
        elif self.mask_type=='simple':           
            mask = np.zeros_like(im)
            mask[im>self.mask_threshold] = 1 


        # mask the image 
        if self.mask_type is not None: 
            mask_all = np.moveaxis(np.tile(mask, (echoes,1,1)), 0,-1) # expand mask to all echoes
            assert mask.shape == mask_all.shape[0:2]
            assert img.shape == mask_all.shape, f"Img and Mask shapes do not match: {img.shape}:{mask_all.shape}"
        
            img = np.multiply(img,mask_all)

        
        return img

            
    def normalize_image(self,img): 
        """Normalize input image"""

        echoes = img.shape[-1]

        if self.normalize_type == "norm_by_first_echo":            
            # NORM BY MEAN OF FIRST ECHO (AFTER MASKING)
            S0 = img[:,:,0]
            if self.reject_outliers:
                S0 = S0[S0 != 0]
                data = S0.flatten()
                m=2
                S0 = data[abs(data - np.mean(data)) < m * np.std(data)] #source: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
                
                S0mean = np.mean(S0)
            else:
                S0mean = np.mean(S0[S0 != 0])
            im_norm = np.divide(img,S0mean)            
        
        elif self.normalize_type == "0_1":       
            input('\n\n WARNING: the input is normalized in 0-1 range for each slice separately. It is NOT multiplied by a single constant value. Press any key to continue.')             
            # NORM WITHIN 0-1 range across ALL echoes
            mmin = np.min(img)
            mmax = np.max(img)
            im_norm = (img - mmin) / (mmax-mmin)

        elif self.normalize_type == "0_1_first_echo":            
            input('\n\n WARNING: the input is normalized in 0-1 range for each slice separately. It is NOT multiplied by a single constant value. Press any key to continue.')        
            # NORM WITHIN 0-1 range across first echo only
            S0 = img[:,:,0]
            mmin = np.min(S0)
            mmax = np.max(S0)
            im_norm = (img - mmin) / (mmax-mmin)

        elif self.normalize_type == "norm_by_voxel":            
            # vectorize
            sh = img.shape
            im_v = np.reshape(img,(sh[0]*sh[1],echoes))
            
            # normalize
            S0 = im_v[:,0]        
            im_v_norm = np.divide(im_v,np.expand_dims(S0,1),out=np.zeros_like(im_v), where=np.expand_dims(S0,1)!=0)  #c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)            

            # reshape into original 
            im_norm = np.reshape(im_v_norm, sh)
        else:
            sys.exit('Incorrect normalization type is specified (or no normalization type was specified')
            
        return im_norm
            

    def add_noise(self, img, noisevariance):
        """Add rician noise to input image"""

        # fetch first echo (first channel)
        S0 = img[:,:,0] 
        S0mean = np.mean(S0[S0 != 0])

        # mask 
        mask = np.zeros_like(img)
        mask[img>0] = 1

        # select noise level
        noise_level = np.random.uniform(low=noisevariance[0],high=noisevariance[1])
        noise1 = np.random.normal(0,S0mean/noise_level, img.shape)
        noise2 = np.random.normal(0,S0mean/noise_level, img.shape)

        # add noise 
        img = np.sqrt((img + noise1)**2 + noise2**2)  # add noise and mask the image 
        img = np.multiply(img,mask)
        
        # always limit signals to 0-1 range 
        
        # NEVER TO THIS! >> manually set values to 1! 
        #img[img>1] = 1 # manually set values larger than 1 to 1 to not throw off the neural network (typically there will be very few of those)


            # OBSOLETE - non-rician (standard normal) (non MRI complex) - noise results in values less than zero
            # S0 = img[:,:,0] # has to be repeated again here as the steps may be run indepdendently - normalization and noise variance - for different datasets
            # S0mean = np.mean(S0[S0 != 0])

            # # add noise (tomorrow - when retraining) (need to see how noise looks like on this data)
            # mask = np.zeros_like(img)
            # mask[img>0] = 1

            
            # noise_level = np.random.uniform(low=self.noisevariance[0],high=self.noisevariance[1])
            # noise = np.random.normal(0,S0mean/noise_level, img.shape)

            # img = np.multiply(img + noise,mask)  # add noise and mask the image 

        return img 


    def sample_augment_options(self):

        # write dictionary values
        self.augmentations = {}
        
        opt = self.augment_options
        
        # set default (if not specified)
        if 'sampletype' not in opt:
            opt['sampletype'] == 'uniform'
                   
        # draw from random normal distribution between the given values
        self.augmentations['theta'] = self.sample(opt['rotate'], opt['sampletype'])
        self.augmentations['tx'] = self.sample(opt['shift'], opt['sampletype'])
        self.augmentations['ty'] = self.sample(opt['shift'], opt['sampletype'])
        self.augmentations['zx'] = self.sample(opt['scale'], opt['sampletype'])
        self.augmentations['zy'] = self.sample(opt['scale'], opt['sampletype'])           
        self.augmentations['shear'] = self.sample(opt['shear'], opt['sampletype'])
        
    def sample(self, bounds,sampletype='uniform'):
        if sampletype == 'uniform': 
            out = np.random.uniform(bounds[0], bounds[1])
        elif sampletype == 'gaussian':
            out = np.random.normal(bounds[0], bounds[1])
        
        return out 
    
    
    def augment(self, img_X, img_Y):
        
        #set vars
        theta = self.augmentations['theta']
        tx = self.augmentations['tx']
        ty = self.augmentations['ty']
        zx = self.augmentations['zx']        
        zy = self.augmentations['zy']                
        shear = self.augmentations['shear']        
        

        # perfom image size checks
        assert img_X.ndim == 3, f"X image should be a 2D slice, with channels equal to echoes = 32"
        assert img_Y.ndim == 2, f"Y image should be a 2D slice, with 2 dimensions"
        assert img_X.shape[-1] == 32
        
        
        # update Y to 3D shape (required)
        img_Y = np.expand_dims(img_Y, -1)
        assert img_Y.shape[-1] == 1

        # update shift params w.r.t. image size
        h, w = img_X.shape[0], img_X.shape[1]
        tx = tx * h
        ty = ty * w        
        
        # transform 
        img_X = tfp.apply_affine_transform(img_X, theta=theta, tx=tx, ty=ty, shear=shear, zx=zx, zy=zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0, order=1)        
        img_Y = tfp.apply_affine_transform(img_Y, theta=theta, tx=tx, ty=ty, shear=shear, zx=zx, zy=zy, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest', cval=0.0, order=1)        
                
        # reduce Y to oriinal shape 
        img_Y = img_Y[:,:,0]
        
        return img_X, img_Y

    
    
    
    
class DataGenerator_test(keras.utils.Sequence):

    """ Prepare Sequence generator class to load & vectorize batches of data.
    
    Helps to iterate over the data (as Numpy arrays).
    """

    def __init__(self, args, img_size, channel_size, input_img_paths, target_img_paths):
        self.batchsize = args.batchsize
        self.img_size = img_size
        self.channel_size = channel_size
        self.normalize = args.normalize 
        self.noisevariance = args.noisevariance
        self.input_img_paths = input_img_paths
        self.mode = args.mode
        self.selfsupervised = True if args.selfsupervised else False

        self.normalize_type = args.normalize_type
        self.reject_outliers = args.reject_outliers
        self.mask_threshold = args.mask_threshold
        self.mask_type = args.mask_type

        if self.mode == 'train':
            self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.input_img_paths) // self.batchsize

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batchsize
        batch_input_img_paths = self.input_img_paths[i : i + self.batchsize]
        x = np.zeros((self.batchsize,) + self.img_size + (self.channel_size,), dtype="float32")

        for j, path in enumerate(batch_input_img_paths):

            img = nb.load(path).get_fdata()

            # check for extra slice dimension e.g. (160,192,1,32) instead of (160,192,32)
            if img.ndim == 4: 
                assert img.shape[-2] == 1, f"Current shape is {img.shape}. Third dimension must be 1, not {img.shape[-2]}"
                img = img[:,:,0,:]

            # mask the image
            img = self.mask_image(img)    
            
            if self.normalize:
                img = self.normalize_image(img)

            if self.noisevariance is not None:
                img = self.add_noise(img, self.noisevariance) 

            if self.img_size[1] != img.shape[1]: 
                assert self.img_size[1]-img.shape[1] == 4, f"Current implementation assumes that image mismatch can only have image size of 156, vs the required 160. Current image size is: {img.shape}"
                # pad the image with extra dims
                npad = ((0, 0), (2, 2), (0, 0)) # padding values to fix the size mismatch of the unet and our data (y-dimension)
                img = np.pad(img, pad_width=npad, mode='constant', constant_values=0)
                
                # NOTE to self - best to perform padding INDEPENDENTLY on a dataset (so that the code is not messed up)


            x[j] = img



                #                     # check for empty slice
                #                     if np.amax(img) == 0:
                #                         print(f"Empty Slice detected. Mean:Std:Max: {np.mean(img), np.std(img), np.max(img)}\nRemoving this slice: {path}")
                #                         continue

                #                     # check for empty slice
                #                     if np.amax(img) == 0:
                #                         print(f"Empty Slice detected. Mean:Std:Max: {np.mean(img), np.std(img), np.max(img)}\nRemoving this slice: {path}")
                #                         continue
            
            
            

        if self.mode == 'train':
            if self.selfsupervised: 
                y = copy.copy(x)
                return x,y
            else: 
                batch_target_img_paths = self.target_img_paths[i : i + self.batchsize]

                y = np.zeros((self.batchsize,) + self.img_size + (1,), dtype="float32")

                for j, path in enumerate(batch_target_img_paths):
                    
                    
                    img = nb.load(path).get_fdata()
                    assert img.shape == self.img_size, f"img_size is not the same as img.shape"
                                        
                    if img.ndim==3 and img.shape[-1]==1:
                        y[j] = img
                    else:
                        y[j] = np.expand_dims(img, 2) 
                        
                        
                return x,y
        
        else:
            return x

    def mask_image(self,img):

        assert img.ndim == 3, f"Image shape should be [x,y,echoes]"

        # get first echo only
        im = img[:,:,0]
        echoes = img.shape[-1]
      
        if self.mask_type=='improved':
        
            # IMPROVED MASKING PROCESS
            # 1. Measure noise in the corners of the image
            # 2. Mask image by mean of noise (as threshold)
            # 3. Erode + Dilate 
            # 4. Add median blur to remove sharp edges
            # 5. Remove small objects + Remove small holes 

            # experiments: ~/w/code/ivim/crohns/experiments_create_better_masks_erode_dilate_remove_small_objects.py

            # 1. Measure noise in the corners of the image
            corners = im[0:20,0:20]+im[-20:,-20:]   #+ref_im[-20:,0:20,:]+ref_im[0:20,-20:,:] -> not so great 
            threshold = np.mean(corners)#+np.std(corners)

            # 2. Mask image by mean of noise (as threshold)
            mask = np.zeros_like(im)
            mask[im>threshold] = 1 

            # 3. Erode + Dilate     
            kernel_erode = np.ones((3,3), np.uint8) 
            kernel_dilate = np.ones((3,3), np.uint8) 
            img_erosion = cv2.erode(mask, kernel_erode, iterations=1) 
            img_dilation = cv2.dilate(img_erosion, kernel_dilate, iterations=1) 

            # 4. Add median blur to remove sharp edges
            img_dilation = cv2.medianBlur(img_dilation.astype('int16'),5)

            # 5. Remove small objects + Remove small holes 
            small_object_threshold = 2000 
            small_hole_threshold = 1000
                # NB must be done for each slice separately (else doesn't work)
                #slices = img_dilation.shape[-1]
                # for sl in range(0,slices): 
                #im = img_dilation#[:,:,sl]    
            arr = img_dilation > 0
            cleaned = morphology.remove_small_objects(arr, min_size=small_object_threshold)  # threshold 
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=small_hole_threshold) # source https://stackoverflow.com/questions/55056456/failed-to-remove-noise-by-remove-small-objects    
            # put back into the mask
            mask[:,:] = cleaned.astype(mask.dtype)
        
        elif self.mask_type=='simple':           
            mask = np.zeros_like(im)
            mask[im>self.mask_threshold] = 1 


        # mask the image 
        if self.mask_type is not None: 
            mask_all = np.moveaxis(np.tile(mask, (echoes,1,1)), 0,-1) # expand mask to all echoes
            assert mask.shape == mask_all.shape[0:2]
            assert img.shape == mask_all.shape, f"Img and Mask shapes do not match: {img.shape}:{mask_all.shape}"
        
            img = np.multiply(img,mask_all)

        
        return img

            
    def normalize_image(self,img): 
        """Normalize input image"""

        echoes = img.shape[-1]

        if self.normalize_type == "norm_by_first_echo":            
            # NORM BY MEAN OF FIRST ECHO (AFTER MASKING)
            S0 = img[:,:,0]
            if self.reject_outliers:
                S0 = S0[S0 != 0]
                data = S0.flatten()
                m=2
                S0 = data[abs(data - np.mean(data)) < m * np.std(data)] #source: https://stackoverflow.com/questions/11686720/is-there-a-numpy-builtin-to-reject-outliers-from-a-list
                
                S0mean = np.mean(S0)
            else:
                S0mean = np.mean(S0[S0 != 0])
            im_norm = np.divide(img,S0mean)            
        
        elif self.normalize_type == "0_1":       
            input('\n\n WARNING: the input is normalized in 0-1 range for each slice separately. It is NOT multiplied by a single constant value. Press any key to continue.')             
            # NORM WITHIN 0-1 range across ALL echoes
            mmin = np.min(img)
            mmax = np.max(img)
            im_norm = (img - mmin) / (mmax-mmin)

        elif self.normalize_type == "0_1_first_echo":            
            input('\n\n WARNING: the input is normalized in 0-1 range for each slice separately. It is NOT multiplied by a single constant value. Press any key to continue.')        
            # NORM WITHIN 0-1 range across first echo only
            S0 = img[:,:,0]
            mmin = np.min(S0)
            mmax = np.max(S0)
            im_norm = (img - mmin) / (mmax-mmin)

        elif self.normalize_type == "norm_by_voxel":            
            # vectorize
            sh = img.shape
            im_v = np.reshape(img,(sh[0]*sh[1],echoes))
            
            # normalize
            S0 = im_v[:,0]        
            im_v_norm = np.divide(im_v,np.expand_dims(S0,1),out=np.zeros_like(im_v), where=np.expand_dims(S0,1)!=0)  #c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)            

            # reshape into original 
            im_norm = np.reshape(im_v_norm, sh)
        else:
            sys.exit('Incorrect normalization type is specified (or no normalization type was specified')
            
        return im_norm
            

    def add_noise(self, img, noisevariance):
        """Add rician noise to input image"""

        # fetch first echo (first channel)
        S0 = img[:,:,0] 
        S0mean = np.mean(S0[S0 != 0])

        # mask 
        mask = np.zeros_like(img)
        mask[img>0] = 1

        # select noise level
        noise_level = np.random.uniform(low=noisevariance[0],high=noisevariance[1])
        noise1 = np.random.normal(0,S0mean/noise_level, img.shape)
        noise2 = np.random.normal(0,S0mean/noise_level, img.shape)

        # add noise 
        img = np.sqrt((img + noise1)**2 + noise2**2)  # add noise and mask the image 
        img = np.multiply(img,mask)
        
        # always limit signals to 0-1 range 
        
        # NEVER TO THIS! >> manually set values to 1! 
        #img[img>1] = 1 # manually set values larger than 1 to 1 to not throw off the neural network (typically there will be very few of those)


            # OBSOLETE - non-rician (standard normal) (non MRI complex) - noise results in values less than zero
            # S0 = img[:,:,0] # has to be repeated again here as the steps may be run indepdendently - normalization and noise variance - for different datasets
            # S0mean = np.mean(S0[S0 != 0])

            # # add noise (tomorrow - when retraining) (need to see how noise looks like on this data)
            # mask = np.zeros_like(img)
            # mask[img>0] = 1

            
            # noise_level = np.random.uniform(low=self.noisevariance[0],high=self.noisevariance[1])
            # noise = np.random.normal(0,S0mean/noise_level, img.shape)

            # img = np.multiply(img + noise,mask)  # add noise and mask the image 

        return img 


    