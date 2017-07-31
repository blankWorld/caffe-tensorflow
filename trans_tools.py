# -*- coding: utf-8 -*-
import numpy as np


# convolution layer weights order
# switch == True : tensorflow ===> caffe
# h w c n ===> n c h w  
# switch == False : caffe ===> tensorflow
# n c h w ===> h w c n
def tensor4d_transform(tensor,switch):
    if switch == True:
        return tensor.transpose((3, 2, 0, 1))
    elif switch == False:
        return tensor.transpose((2, 3, 1, 0))
    else:
        return None

# fc layer 
# h w ===> w h
def tensor2d_transform(tensor):
    return tensor.transpose((1,0))

# special weights transform such as (vgg16 pooling5===>fc1 )
# switch == True : tensorflow ===> caffe shape := [h w c n]
# hwc n ===> h w c n ===> n c h w ===> n chw
# switch == False : caffe ===> tensorflow shape := [n c h w]
# This operation 
# n chw ===> n c h w ===> h w c n ===> hwc n
def tensor4d_2d_transform(tensor,shape,switch):
    assert(tensor.ndim == 2 and len(shape) == 4)
    if switch == True:
        return tensor4d_transform(tensor.reshape(shape),switch).reshape([shape[3],-1])
    elif switch == False:
        return tensor4d_transform(tensor.reshape(shape),switch).reshape([-1,shape[0]])
    else:
        return None

# 
