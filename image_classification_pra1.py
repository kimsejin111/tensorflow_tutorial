import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from PIL import Image

'''이미지 받아오기'''
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)


'''저장된 이미지 갯수 출력'''
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

'''이미지 열기'''
roses = list(data_dir.glob('roses/*'))
Image.open(str(roses[0]))
Image.open(str(roses[1]))

'''이미지 열기'''
tulips = list(data_dir.glob('tulips/*'))
x=Image.open(str(tulips[0]))
x.show()
Image.open(str(tulips[1]))


