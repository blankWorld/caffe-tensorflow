# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
import caffe,sys
import trans_tools as trans
from scipy.misc import imread, imresize
from cv2 import imshow , waitKey,split,merge
from imagenet_classes import class_names
from vgg16 import *

def resize_image(img,size):
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy : yy + short_edge, xx : xx + short_edge]
    resize_img = imresize(crop_img,(size,size))
    return resize_img

class TRANS_MODEL(object):
    " This define/load your model ."
    def __init__(self,tensor4d_2d_dict = None):
        # tensor4d_2d_dict is dict {'name': shape}
        # when transform conv layer ===> fc layer (such as VGG16  pooling5===>fc1)
        # we must recorde the weights shape
        self.tensor4d_2d_dict = tensor4d_2d_dict 
    
    def get_tf_variables(self,sess,ckpt_file=None,meta_file=None,model_file=None,is_trainabel_var = True):
        " This function get tensorflow variables"
        " must input meta_file or model_file"
        " is_trainabel_var = False  'global_variables' "
        " is_trainabel_var = True 'trainable_variables'"
        # load model
        if meta_file is None:
            self.model = model_file
            self.saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
        else:
            self.saver = tf.train.import_meta_graph(meta_file)
        if ckpt_file is not None:
            print 'tensorflow restore:',ckpt_file
            self.saver.restore(sess,ckpt_file)
        # get tensor
        self.tensor_dict={}
        if  is_trainabel_var:
            self.variables = tf.trainable_variables()
        else:
            self.variables = tf.global_variables()

        for v in self.variables:
            self.tensor_dict[str(v.name)] = sess.run(v)

    def get_caffe_variables(self,net_proto,net_model = None,bn_name = ''):
        " This function get caffe variables"
        caffe.set_mode_cpu()
        self.blob_dict={}
        if net_model is not None:
            self.net_caffe = caffe.Net(net_proto,net_model,caffe.TEST)
 
        else:
            self.net_caffe = caffe.Net(net_proto,caffe.TEST)
        # caffe net params layer_name w b
        # bn_name : caffe bn layer name include bn_name
        # Note: we must match tf_variables name and caffe params name
        # so we modifiy caffe params name and save in bolb_dict
        for layer_name,param in self.net_caffe.params.items():
            param_len = len(param)
            # find batch_normalization name must has 'bn_name'
            # your can modify it
            if param_len == 3 and layer_name.find(bn_name) >= 0:
                scale_factor = 1.0 / param[2].data[0]
                mean = param[0].data * scale_factor
                variance = param[1].data *scale_factor
                name = str(layer_name) + "/weights:0"
                self.blob_dict[name] = mean  
                name = str(layer_name) + "/biases:0"
                self.blob_dict[name] = variance  
            elif param_len == 2:
                name = str(layer_name) + "/weights:0" 
                self.blob_dict[name] = param[0].data
                name = str(layer_name) + "/biases:0" 
                self.blob_dict[name] = param[1].data
            elif param_len == 1:
                name = str(layer_name) + "/weights:0" 
                self.blob_dict[name] = param[0].data
              
    def _transformAtoB(self,switch):
        " switch == True : tensorflow ===> caffe"
        " switch == False : caffe ===> tensorflow"
        " IN and OUT is a dict (blob_dict/tensor_dict)"
        " IN ===> OUT"
        if switch == True:
            IN = self.tensor_dict
            OUT = self.blob_dict
        elif switch == False:
            IN = self.blob_dict
            OUT = self.tensor_dict
        else:
            print ' switch must be True/False'
            sys.exit(0)
            
        assert(OUT is not None and IN is not None)
        for key in OUT.keys():
            if IN.has_key(key):
                if np.prod(IN[key].shape) == np.prod(OUT[key].shape):
                    if IN[key].ndim == 4:
                        OUT[key] = trans.tensor4d_transform(IN[key],switch)
                    elif IN[key].ndim == 2:
                        if self.tensor4d_2d_dict != None and (key in self.tensor4d_2d_dict.keys()):
                            OUT[key] = trans.tensor4d_2d_transform(IN[key],self.tensor4d_2d_dict[key],switch) 
                        else:
                            OUT[key] = trans.tensor2d_transform(IN[key])
                    elif IN[key].ndim == 1:
                        OUT[key] = IN[key]
                        

    def tf_transform_to_caffe(self,caffemodel_file,bn_name = ''):
        "transform tensorflow model to caffe model save as caffemodel_file "
        # transform tensorflow parameters to blob_dict
        self._transformAtoB(True)
        # modifiy caffe net parameters 
        for layer_name,param in self.net_caffe.params.items():
            param_len = len(param)
            if param_len == 3 and layer_name.find(bn_name) >= 0:
                scale_factor = 1.0
                name = str(layer_name) + "/weights:0"
                param[0].data[...] = self.blob_dict[name]
                name = str(layer_name) + "/biases:0" 
                param[1].data[...] = self.blob_dict[name]
                param[2].data[...] = scale_factor

            elif param_len == 2:
                name = str(layer_name) + "/weights:0" 
                param[0].data[...] = self.blob_dict[name]
                name = str(layer_name) + "/biases:0" 
                param[1].data[...] = self.blob_dict[name]
            
            elif param_len == 1:
                name = str(layer_name) + "/weights:0" 
                param[0].data[...] = self.blob_dict[name]    
                    
        self.net_caffe.save(caffemodel_file)
        print 'caffe save:',caffemodel_file
        print 'tensorflow model to caffe model done'

    def caffe_transform_to_tf(self,sess,ckpt_file):
        "transform caffe model to tensorflow model save as ckpt_file "
        # transform caffe parameters to tensor_dict
        self._transformAtoB(False)
        # modifiy tensorflow parameters  
        for v in self.variables:
            sess.run(v.assign(self.tensor_dict[v.name]))
            # if  sum(sess.run(v)).all() == 0:
            #     print v.get_shape() , self.blob_dict[v.name].shape
            #print sess.run(v)
     
            #print self.sess.run(v)
        print 'tensorflow save:',ckpt_file
        self.saver.save(sess,ckpt_file)
        print 'caffe model to tensorflow model done'



if __name__ == '__main__':
    #tensorflow =====> caffe 
    sess = tf.Session()
    tensor4d_2d_dict = {'fc1/weights:0':[7,7,512,4096]}
    MODEL = TRANS_MODEL(tensor4d_2d_dict)
    vgg16 = VGG16()
    MODEL.get_tf_variables(sess,ckpt_file = './vgg16_tf.ckpt',\
    meta_file = None ,model_file=vgg16,is_trainabel_var=0) 
    MODEL.get_caffe_variables("./VGG_2014_16.prototxt")
    MODEL.tf_transform_to_caffe("./vgg16_caffe.caffemodel")
    
    # #caffe =====> tensorflow
    # sess = tf.Session()
    # tensor4d_2d_dict = {'fc1/weights:0':[4096,512,7,7]}
    # MODEL = TRANS_MODEL(tensor4d_2d_dict)
    # vgg16 = VGG16()
    # MODEL.get_tf_variables(sess,ckpt_file = None,\
    # meta_file= None,model_file=vgg16,is_trainabel_var=0) 
    # MODEL.get_caffe_variables("./VGG_2014_16.prototxt","./vgg16_caffe.caffemodel")
    # MODEL.caffe_transform_to_tf(sess,"./vgg16_tf.ckpt")
    

    
    print '#----------------------------------------------#'
    # input image RGB - mean:[123.68, 116.779, 103.939]
    # this model input image channel order [R G B] 
    img = imread('cat.jpg', mode='RGB')
    img = resize_image(img,224) - [123.68, 116.779, 103.939]
    print '#--------------------TEST----------------------#'
    sys.stdout.flush()
    # caffe forward
    # input [1 3 224 224]
    img_caffe = img.transpose((2,0,1)).reshape(1,3,224,224)
    MODEL.net_caffe.blobs['data'].data[0] = img_caffe
    MODEL.net_caffe.forward()
    caffe_prob = MODEL.net_caffe.blobs['prob'].data[0]
    caffe_preds = (np.argsort(caffe_prob)[::-1])[0:5]
    print 'caffe preds top5:'
    for p in caffe_preds:
        print class_names[p], caffe_prob[p]
    
    print '#----------------------------------------------#'
    sys.stdout.flush()
    # tensorflow inference input [1 224 224 3]
    imgs = tf.placeholder(tf.float32, [1, 224, 224, 3])
    # if create tensorflow model
    tf_prob = sess.run(MODEL.model.vgg16_inference(imgs),feed_dict={imgs:[img]})[0]
    # if load tensorflow meta file
    #graph = tf.get_default_graph()
    #tf_prob = sess.run(graph.get_tensor_by_name('Softmax:0'), feed_dict={graph.get_tensor_by_name("Placeholder:0"): [img]})[0] 
    tf_preds = (np.argsort(tf_prob)[::-1])[0:5]
    print 'tensorflow preds top5:'
    for p in tf_preds:
        print class_names[p], tf_prob[p]
