
import multiprocessing as mp
import numpy as np
from PIL import Image as PImage
from PIL import ImageFilter
import random
import math
import tensorflow as tf



shared_dict = {}


def distort_sample(img : PImage) -> (PImage, [int], [int]):
    """
    Distort the given image randomly.

    Randomly applies the transformations:
        rotation, shear, scale, translate, 
    Randomly applies the filter:
        sharpen, blur, smooth

    Returns the distorted image.
    """

    offset, scale = (0, 0), (64, 64)

    t = random.choice(["sine", "rotate", "shear", "scale"])
    f = random.choice(["blur", "sharpen", "smooth"])

    # randomly apply transformations...
    # rotate image
    if("rotate" in t):
        img = img.rotate(random.uniform(-15, 15))
    
    # shear image
    if("shear" in t):
        y_shear = random.uniform(-0.2, 0.2)
        x_shear = random.uniform(-0.2, 0.2)
        img = img.transform(img.size, PImage.AFFINE, (1, x_shear, 0, y_shear, 1, 0))
    
    # scale and translate image
    if("scale" in t):
        #scale the image
        size_x = random.randrange(25, 63)
        size_y = random.randrange(25, 63)
        scale = (size_x, size_y)
        offset = (math.ceil((64 - size_x) / 2), math.ceil((64 - size_y) / 2))
        img = img.resize(scale)

        # put it again on a black background (translated)
        background = PImage.new('L', (64, 64))
        trans_x = random.randrange(0, math.floor((64 - size_x)))
        trans_y = random.randrange(0, math.floor((64 - size_y)))
        offset = (trans_x, trans_y)
        background.paste(img, offset)
        img = background
    
    if("sine" in t):
        t_img = np.array(img)

        A = t_img.shape[0] / 3.0
        w = 2.0 / t_img.shape[1]

        shift_factor = random.choice([-1, 1]) * random.uniform(0.15, 0.2)
        shift = lambda x: shift_factor * A * np.sin(-2*np.pi*x * w)

        for i in range(t_img.shape[0]):
            t_img[:,i] = np.roll(t_img[:,i], int(shift(i)))

        img = PImage.fromarray(t_img)


    # blur
    if("blur" in f):
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))

    # sharpen
    if("sharpen" in f):
        img = img.filter(ImageFilter.SHARPEN)
        
    # smooth
    if("smooth" in f):
        img = img.filter(ImageFilter.SMOOTH)

    return img, offset, scale

def generator_func(start_index, end_index, x_shape, y_shape):
    X, Y = [], []
    
    x_loc = np.frombuffer(shared_dict["x"], dtype="float16").reshape(x_shape)
    y_loc = np.frombuffer(shared_dict["y"], dtype="b").reshape(y_shape)
    
    for i in range(start_index, end_index):
        base_img = x_loc[i]
        img = PImage.fromarray(np.uint8(base_img.reshape(64, 64) * 255))
        img, *unused = distort_sample(img)

        # add transformed image
        X.append(np.array(img).reshape(64, 64, 1))
        Y.append(y_loc[i])
        X.append(np.array(img).reshape(64, 64, 1))
        Y.append(y_loc[i])

        # add base image
        #X.append(base_img)
        #Y.append(y_loc[i])
        
    return X, Y

def generator_initializer(_x_shared, _y_shared):
    shared_dict["x"] = _x_shared
    shared_dict["y"] = _y_shared

def generator_func(start_index, end_index, x_shape, y_shape):
    X, Y = [], []
    
    x_loc = np.frombuffer(shared_dict["x"], dtype="float16").reshape(x_shape)
    y_loc = np.frombuffer(shared_dict["y"], dtype="b").reshape(y_shape)
    
    for i in range(start_index, end_index):
        base_img = x_loc[i]
        img = PImage.fromarray(np.uint8(base_img.reshape(64, 64) * 255))
        img, *unused = distort_sample(img)

        # add transformed image
        X.append(np.array(img).reshape(64, 64, 1))
        Y.append(y_loc[i])
        X.append(np.array(img).reshape(64, 64, 1))
        Y.append(y_loc[i])

        # add base image
        #X.append(base_img)
        #Y.append(y_loc[i])
        
    return X, Y

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, num_samples, batch_size,
                        percentage, mode,
                        x_shared, y_shared,
                        x_np_shape, y_np_shape,
                        processes, shuffle=True):
        self.num_samples = num_samples
        # 50% original images + 50% augmented images 
        self.batch_size = batch_size // 2
        self.percentage = percentage

        # an offset to devide the data set into test and train
        self.start_index = 0
        if(mode == "testing"):
            self.start_index = num_samples - (num_samples // 100 * percentage)
        # is this a train or a test generator
        self.mode = mode
        # how many processes should be used for this generator
        self.processes = processes
        # should the arrays be shuffled after each epoch
        self.shuffle = shuffle

        self.x_np_shape = x_np_shape
        self.y_np_shape = y_np_shape
        
        # a pool of processes for generating augmented data
        self.pool = mp.Pool(processes=self.processes,
            initializer=generator_initializer,
            initargs=(x_shared, y_shared))
        
    def __len__(self):
        return (self.num_samples // 100 * self.percentage) // self.batch_size

    def on_epoch_end(self):
        if(False):
            rng_state = np.random.get_state()
            np.random.shuffle(x_np)
            np.random.set_state(rng_state)
            np.random.shuffle(y_np)
            
    def __getitem__(self, index):

        arguments = []
        slice_size = self.batch_size // self.processes
        current_batch = index * self.batch_size
        for i in range(self.processes):
            slice_start = self.start_index + (current_batch + i * slice_size)
            slice_end = self.start_index + (current_batch + (i+1) * slice_size)
            arguments.append([slice_start, slice_end, self.x_np_shape, self.y_np_shape])
        
        return_values = self.pool.starmap(generator_func, arguments)

        X, Y = [], []
        for imgs, labels in return_values:
            X.append(imgs)
            Y.append(labels)

        return np.concatenate(X).astype(np.float16), np.concatenate(Y).astype(np.float16)









