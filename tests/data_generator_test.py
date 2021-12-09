import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from data_generator.data_generator import train_dataset 

train_generator = train_dataset.batch(1)

for input_img, label_img in train_generator:
  print(input_img.shape)
  _, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 10))
  ax[0].set_title("Input Image")
  ax[0].imshow(input_img[0])

  ax[1].set_title("Label Image")
  ax[1].imshow(label_img[0])

input_img.numpy().min(), input_img.numpy().max()

label_img.numpy().min(), label_img.numpy().max()

