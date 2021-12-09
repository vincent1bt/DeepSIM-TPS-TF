from data_generator.data_generator import load_image
import tensorflow as tf
import os
import matplotlib.pyplot as plt

data_path = os.path.abspath("./data/car_test")

img_test_1 = load_image(f"{data_path}/car_test1.png")
img_test_2 = load_image(f"{data_path}/car_test2.png")

input = tf.stack([img_test_1, img_test_2], axis=0)

generator = tf.keras.models.load_model('saved_model/tps_generator')

results = generator(input)

plt.imsave("image1.png", results[0])
plt.imsave("image2.png", results[1])

