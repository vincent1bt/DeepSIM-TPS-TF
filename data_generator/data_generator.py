import tensorflow as tf
import os
from data_generator.data_utils import create_tps, tps_augmentation

data_path = os.path.abspath("./data/car")

image_height = 256
image_width = 512

def load_image(image_path):
  img = tf.io.read_file(image_path)
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  img = tf.image.resize(img, [image_height, image_width])
  
  return img

def augment_image(img, flip, tps):
  img = tps_augmentation(img, tps)

  if flip:
    img = tf.image.flip_left_right(img)
  
  return img

def load_train_image(data_path):
  input_img = load_image(tf.strings.join([data_path, "/input/", "car.png"])) # load the image in range (0, 1)
  label_img = load_image(tf.strings.join([data_path, "/real/", "car.png"])) # load the image in range (0, 1)

  flip = tf.random.uniform([]) > 0.5

  tps = create_tps()

  input_img = augment_image(input_img, flip, tps)
  label_img = augment_image(label_img, flip, tps)

  return input_img, label_img

train_dataset = tf.data.Dataset.from_tensor_slices([data_path])
train_dataset = train_dataset.map(load_train_image)

