import os
import time

import numpy as np

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave,imread,imresize
from keras import backend as K
from keras.applications import vgg16
from keras.preprocessing.image import load_img,img_to_array


root_dir=os.path.abspath('.')

base_image_path=os.path.join(root_dir,'base_image.jpg')
ref_image_path=os.path.join(root_dir,'reference_image')

img_nrows=400
img_ncols=400
style_weight=1
content_weight=0.025
total_variation_weight=1


#
def preprocess_image(image_path):
    img=load_img(image_path,target_size=[img_nrows,img_ncols])
    img=img_to_array(img)
    img=np.expand_dims(img,axis=0)
    img=vgg16.preprocess_input(img)
    return img

def deprocess_image(x):
    x=x.reshape((3,img_nrows,img_ncols))
    x=x.transpose((1,2,0))
    x[:,:,0]+=103.939
    x[:,:,1]+=116.779
    x[:,:,2]+=123.68
    x=x[:,:,::-1]
    x=np.clip(x,0,255).astype('uint8')
    return x


#
base_image=K.variable(preprocess_image(base_image_path))
ref_image=K.variable(preprocess_image(ref_image_path))
final_image=K.placeholder((1,3,img_nrows,img_ncols))
input_tensor=K.concatenate([base_image,ref_image,final_image],axis=0)