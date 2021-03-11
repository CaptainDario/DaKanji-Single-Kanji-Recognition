
#std lib
import os
import random
import math
import random
import multiprocessing as mp
import gc
import datetime

#reading the dataset
from etldr.etl_data_reader import ETLDataReader
from etldr.etl_character_groups import ETLCharacterGroups
from etldr.etl_data_names import ETLDataNames

#data handling
from PIL import Image as PImage
from PIL import ImageFilter
import numpy as np


#plotting/showing graphics
import matplotlib.pyplot as plt
from IPython.display import Image
#define a font to show japanese characters in matplotlib figures
import matplotlib.font_manager as fm
font = fm.FontProperties(fname=os.path.join("font", "NotoSerifCJKjp-Regular.otf"), size=20)



shared_dict = {}


def distort_sample(img : Image) -> (Image, [int], [int]):
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

def get_model(name : str):
    _model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(64, 64, 1), kernel_size=3, activation='relu', filters=32, name="conv2D_1_2_input"),
        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=32, name="conv2D_1_1"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_1"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=32, name="conv2D_2_1"),
        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=32, name="conv2D_2_2"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_2"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=64, name="conv2D_3_1"),
        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=64, name="conv2D_3_2"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_3"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=128, name="conv2D_4_1"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_4"),

        tf.keras.layers.Flatten(name="flatten_1"),
        tf.keras.layers.Dropout(0.25, name="dropout_1"),
        
        tf.keras.layers.Dense(2048, name="dense_1"),
        tf.keras.layers.Dropout(0.1, name="dropout_2"),

        tf.keras.layers.Dense(2048, name="dense_2"),
        tf.keras.layers.Dropout(0.25, name="dropout_3"),

        tf.keras.layers.Dense(len(lb.classes_), name="dense_3"),

        #set the dtype to float32 for numerical stability
        tf.keras.layers.Softmax(dtype="float32", name="softmax_1_output") 
    ], name=name)
    
    return _model

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

        # add base image
        X.append(base_img)
        Y.append(y_loc[i])
        
    return X, Y



if __name__ == "__main__":
    #ML
    import tensorflow as tf
    from tensorflow.keras.callbacks import ModelCheckpoint

    #creating one hot encodings
    from sklearn.preprocessing import LabelBinarizer
    
    class DataGenerator(tf.keras.utils.Sequence):

        def __init__(self, num_samples, batch_size, percentage, mode, processes, shuffle=True):
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
            
            # a pool of processes for generating augmented data
            self.pool = mp.Pool(processes=self.processes,
                initializer=generator_initializer,
                initargs=(x_shared, y_shared))
            
        def __len__(self):
            return (self.num_samples // 100 * self.percentage) // self.batch_size

        def on_epoch_end(self):
            if(self.shuffle):
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
                arguments.append([slice_start, slice_end, x_np.shape, y_np.shape])
            
            return_values = self.pool.starmap(generator_func, arguments)

            X, Y = [], []
            for imgs, labels in return_values:
                X.append(imgs)
                Y.append(labels)

            return np.concatenate(X).astype(np.float16), np.concatenate(Y).astype(np.float16)
    

    print("GPUs Available: ", tf.test.gpu_device_name())
    physical_devices = tf.config.list_physical_devices('GPU') 
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    # read data
    path = "F:\data_sets\ETL_kanji"
    reader = ETLDataReader(path)
    x, y =  reader.read_dataset_whole(include=[ETLCharacterGroups.kanji], processes=16)
    print("finished loading data")

    # because the data is ordered shuffle it
    rng_state = np.random.get_state()
    np.random.shuffle(x)
    np.random.set_state(rng_state)
    np.random.shuffle(y)

    #one hot encode the labels
    lb = LabelBinarizer()
    lb.fit(y)
    o_y = lb.transform(y)

    # free the memory from the original string labels
    del(y)
    print("finished one-hot-encoding")
    print(o_y.shape)


    # create shared arrays
    memory_size_x = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
    x_shared = mp.RawArray("f", memory_size_x // 2)
    x_np = np.frombuffer(x_shared, dtype="float16").reshape(x.shape)
    np.copyto(x_np, x)
    del(x)

    memory_size_y = o_y.shape[0] * o_y.shape[1]
    y_shared = mp.RawArray('b', memory_size_y)
    y_np = np.frombuffer(y_shared, dtype="b").reshape(o_y.shape)
    np.copyto(y_np, o_y)
    del(o_y)
    gc.collect()
    print("finished moving data in shared memory")


    # define CNN
    #path where the model should be saved
    model_dir = os.path.join(os.getcwd(), "CNN_kanji", "model")
    #f16_model = get_model("DaKanjiRecognizer_f16")
    f16_model = tf.keras.models.load_model(os.path.join(model_dir, "tf", "checkpoints", "weights-improvement-108-0.98.hdf5"))
    #print(f16_model.output_shape)
    #f16_model.summary()
    print("Saving with base path:", model_dir)
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001,
                                    beta_1=0.9,
                                    beta_2=0.999,
                                    epsilon=1e-08,)

    f16_model.compile(optimizer=opt,
                loss="categorical_crossentropy",
                metrics=['accuracy'])


    # training the CNN
    # percentage to split into test and train
    train_generator = DataGenerator(len(x_np), 1024, 80, "training", 4)
    test_generator  = DataGenerator(len(x_np), 1024, 20, "testing" , 4)
    print("finished instanciating data generator")

    #checkpoints setup
    filepath = os.path.join(model_dir, "tf", "checkpoints", "weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    
    log_dir = os.path.join(model_dir, "logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    callbacks_list = [checkpoint, tensorboard_callback]

    #train the model
    hist = f16_model.fit(
        x=train_generator,
        epochs=200,
        initial_epoch=109,
        validation_data=test_generator,
        max_queue_size=200,
        callbacks=callbacks_list,
        workers=1
    )

    # saving the model
    f16_model.save(os.path.join(model_dir, "tf", "trained_model"))

    # Create a float32 model with the same weights as the mixed_float16 model, so
    # that it loads into TF Lite
    tf.keras.mixed_precision.set_global_policy("float32")
    f32_model = get_model("DaKanjiRecognizer_f32")
    f32_model.set_weights(f16_model.get_weights())
    #f32_model.summary()

    #save the model as tflite
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(f32_model) # path to the SavedModel directory
    tflite_model = converter.convert()

    # Save the model.
    with open(os.path.join(model_dir, "tflite", "model.tflite"), 'wb') as f:
        f.write(tflite_model)
