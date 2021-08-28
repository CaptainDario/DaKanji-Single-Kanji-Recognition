import numpy as np
import PIL
from PIL import Image as PImage
from PIL import ImageFilter, ImageFont, ImageDraw
import cv2
import tensorflow as tf

from fontTools.ttLib import TTFont

from typing import Tuple, List
import random



def distort_sample(img : PImage) -> Tuple[PIL.Image.Image, Tuple[int, int], Tuple[int, int]]:
    """
    Distort the given image randomly.

    Randomly applies the transformations:
        rotation, shear, scale, translate, 
    Randomly applies the filter:
        sharpen, blur, smooth

    Args:
        img: the image which should be distorted.

    Returns:
        the distorted image
        the offset where the character is placed in the image
        the size of the character
    """

    offset, scale = (0, 0), (64, 64)

    t = random.choice(["rotate", "shear", "erode", "dilate"])
    f = random.choice(["blur", "sharpen", "smooth"])

    # randomly apply transformations...
    # rotate image
    if("rotate" in t):
        img = img.rotate(random.uniform(-10, 10))
    
    # shear image
    if("shear" in t):
        y_shear = random.uniform(-0.2, 0.2)
        x_shear = random.uniform(-0.2, 0.2)
        img = img.transform(img.size, PImage.AFFINE, (1, x_shear, 0, y_shear, 1, 0))

    # sine transform
    t_img = np.array(img)
    
    A = t_img.shape[0] / random.uniform(3.0, 6.0)
    w = 2.0 / t_img.shape[1]

    shift_factor = random.choice([-1, 1]) * random.uniform(0.10, 0.2)
    shift = lambda x: shift_factor * A * np.sin(-2*np.pi*x * w)

    for i in range(t_img.shape[0]):
        t_img[:,i] = np.roll(t_img[:,i], int(shift(i)))
    
    if("erode" in t):
        t_img = cv2.erode(t_img, np.ones((2, 2), np.uint8), iterations=1)
    if("dilate" in t):
        t_img = cv2.dilate(t_img, np.ones((3, 3), np.uint8), iterations=1)
    
    img = PImage.fromarray(t_img, mode="L")

    # FILTER
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


def generate_images(
    amounts : List[int], kanjis : List[str], 
    fonts : List[List[str]], font_size : int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates `amount` images of the given `kanji` using given `fonts`
    while augmenting 50% of them randomly. 

    Args:
        amount    : how many samples should be created per kanji.
        kanji     : the (kanji) character which should be created
        fonts     : a list of paths to fonts which should be used 
                    to created images.

    Returns:
        Two numpy arrays.
        One containing all images and the other the matching labels.
    """

    cnt = 0
    kanji_labels = np.empty(shape=(sum(amounts)), dtype=str)
    kanji_imgs = np.zeros(shape=(sum(amounts), 64, 64, 1), dtype=np.uint8)

    
    for kanji_cnt, kanji in enumerate(kanjis):
        already_gen_cnt = sum(amounts[:kanji_cnt])
        ttfs = [ImageFont.truetype(f, font_size) for f in fonts[kanji_cnt]]

        # generate one image per font
        for cnt, font in enumerate(ttfs):

            if(cnt >= amounts[kanji_cnt]):
                break

            # make sure that the image fits in the 64x64 image
            if(font.getsize(kanji)[0] > 64):
                font = font.font_variant(size = font_size - (font.getsize(kanji)[0] - font_size))
            if(font.getsize(kanji)[1] > 64):
                font = font.font_variant(size = font_size - (font.getsize(kanji)[1] - font_size))

            # create the image
            img = PImage.new(mode="L", size=(64, 64), color=0)
            d = ImageDraw.Draw(img)
            d.text(((64 - font.getsize(kanji)[0]) // 4, (64 - font.getsize(kanji)[1]) // 4), kanji, font=font, fill=255)     

            # store the image / label in the array
            kanji_imgs[cnt + already_gen_cnt] = np.array(img).reshape(64, 64, 1)
            kanji_labels[cnt + already_gen_cnt] = kanji
        
        # append distorted versions of the images created from the fonts
        cnt += 1
        while(cnt < amounts[kanji_cnt]):
            p_img = tf.keras.preprocessing.image.array_to_img(kanji_imgs[(cnt) % len(ttfs) + already_gen_cnt])
            kanji_imgs[cnt+already_gen_cnt]   = np.array(distort_sample(p_img)[0]).reshape(64, 64, 1)
            kanji_labels[cnt+already_gen_cnt] = kanji
            cnt += 1
        
    return kanji_imgs, kanji_labels

def check_font_char_support(fonts : List[str], char : str):
    """ Check which of the given fonts supports the given char.
    """

    element = (char, [])

    for font in fonts:
        f = ImageFont.truetype(font, 64)
        for table in TTFont(font)['cmap'].tables:
            if ord(char) in table.cmap.keys():
                # create an image and draw the character
                img = PImage.new(mode="L", size=(64, 64), color=0)
                d = ImageDraw.Draw(img)
                d.text((0, 0), char, font=f, fill=255)
                img = np.array(img).reshape(64, 64, 1)    

                if(np.any(img)):
                    element[1].append(font)
                break

    return element