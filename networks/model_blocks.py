import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa

kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

class ResBlock(tf.keras.Model):
  def __init__(self):
    super(ResBlock, self).__init__()
    self.conv_1 = layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=kernel_init)
    self.conv_2 = layers.Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='valid', kernel_initializer=kernel_init)

    self.norm_1 = tfa.layers.InstanceNormalization()
    self.norm_2 = tfa.layers.InstanceNormalization()

    self.padding_values = [[0, 0], [1, 1], [1, 1], [0, 0]]
  
  def call(self, x):
    x_skip = x 

    x = tf.pad(x, self.padding_values, mode="REFLECT")

    x = self.conv_1(x)
    x = self.norm_1(x)
    x = layers.ReLU()(x)

    x = tf.pad(x, self.padding_values, mode="REFLECT")
    
    x = self.conv_2(x)
    x = self.norm_2(x)

    x = layers.add([x, x_skip])

    return x

