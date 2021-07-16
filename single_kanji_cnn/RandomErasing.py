import tensorflow as tf
import numpy as np



class RandomErasing(tf.keras.layers.Layer):
    """A Layer which applies Random Erasing image augmenation to its input.
    """

    def __init__(self, name="random_erasing", 
        probability : float = 0.5, sl : float = 0.02, sh : float = 0.4, r1 : float = 0.3, method : str = 1,
        train_only = True, **kwargs):
        """
        Args:
            img         : 3D Tensor data (H,W,Channels) normalized value [0,1]
            probability : The probability that the operation will be performed.
            sl          : min erasing area
            sh          : max erasing area
            r1          : min aspect ratio
            method      : Erasing type : 1 - ('black'), 2 - ('white') or 3 - ('random') 
        """
        super(RandomErasing, self).__init__(name=name, **kwargs)
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.method = method
        self.train_only = train_only

    def call(self, input):
        if(tf.keras.backend.learning_phase() or not self.train_only):
            input = tf.map_fn(self.random_erasing, input)
        return input

    def get_config(self):
        config = super(RandomErasing, self).get_config()
        return config

@tf.function
def random_erasing(self, img) -> tf.Tensor:
    '''
    Method that performs Random Erasing from "Random Erasing Data Augmentation" by Zhong et al..

    Args:
        img : the image on which Randomerasing should be applied.

    Returns:
        The augmented image.
    '''  

    probability = 0.3
    sl = 0.02
    sh = 0.4
    r1 = 0.3
    method = 1
    
    if tf.random.uniform([]) > probability:
        return tf.convert_to_tensor(img)

    img_width    = img.shape[1]
    img_height   = img.shape[0]
    img_channels = img.shape[2]

    area = img_height * img_width

    target_area = tf.random.uniform([],minval=sl, maxval=sh) * area
    aspect_ratio = tf.random.uniform([],minval=r1, maxval=1/r1)
    h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
    w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)

    while tf.constant(True, dtype=tf.bool):
        if h > img_height or w > img_height:
            target_area = tf.random.uniform([],minval=sl, maxval=sh) * area
            aspect_ratio = tf.random.uniform([],minval=r1, maxval=1/r1)
            h = tf.cast(tf.math.round(tf.math.sqrt(target_area * aspect_ratio)), tf.int32)
            w = tf.cast(tf.math.round(tf.math.sqrt(target_area / aspect_ratio)), tf.int32)
        else:
            break

    x1 = tf.cond(img_height == h, lambda:0, lambda:tf.random.uniform([], minval=0, maxval=img_height - h, dtype=tf.int32))
    y1 = tf.cond(img_width  == w, lambda:0, lambda:tf.random.uniform([], minval=0, maxval=img_width - w, dtype=tf.int32))

    part1 = tf.slice(img, [0,0,0], [x1,img_width,img_channels]) # first row
    part2 = tf.slice(img, [x1,0,0], [h,y1,img_channels]) # second row 1

    # black
    if method == 1:
        part3 = tf.zeros((h,w,img_channels), dtype=tf.float16) # second row 2
    # white
    elif method == 2:
        part3 = tf.ones((h,w,img_channels), dtype=tf.float16)
    # random
    elif method == 3:
        part3 = tf.random.uniform((h,w,img_channels), dtype=tf.float16)

    part4 = tf.slice(img,[x1,y1+w,0], [h,img_width-y1-w,img_channels]) # second row 3
    part5 = tf.slice(img,[x1+h,0,0], [img_height-x1-h,img_width,img_channels]) # third row

    middle_row = tf.concat([part2,part3,part4], axis=1)
    img = tf.concat([part1,middle_row,part5], axis=0)

    return img   