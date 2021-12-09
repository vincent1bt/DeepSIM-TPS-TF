import tensorflow as tf
from tensorflow.keras import layers
from networks.model_blocks import ResBlock
import tensorflow_addons as tfa

kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

class Generator(tf.keras.Model):
  def __init__(self):
    super(Generator, self).__init__()

    self.conv_1 = layers.Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='valid', kernel_initializer=kernel_init)
    self.conv_2 = layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=kernel_init)
    self.conv_3 = layers.Conv2D(256, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=kernel_init)
    self.conv_4 = layers.Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=kernel_init)
    self.conv_5 = layers.Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='same', kernel_initializer=kernel_init)


    self.norm_1 = tfa.layers.InstanceNormalization()
    self.norm_2 = tfa.layers.InstanceNormalization()
    self.norm_3 = tfa.layers.InstanceNormalization()
    self.norm_4 = tfa.layers.InstanceNormalization()

    self.norm_5 = tfa.layers.InstanceNormalization()
    self.norm_6 = tfa.layers.InstanceNormalization()
    self.norm_7 = tfa.layers.InstanceNormalization()
    self.norm_8 = tfa.layers.InstanceNormalization()
    self.norm_9 = tfa.layers.InstanceNormalization()

    self.res_block_1 = ResBlock()
    self.res_block_2 = ResBlock()
    self.res_block_3 = ResBlock()
    self.res_block_4 = ResBlock()
    self.res_block_5 = ResBlock()
    self.res_block_6 = ResBlock()
    self.res_block_7 = ResBlock()
    self.res_block_8 = ResBlock()
    self.res_block_9 = ResBlock()

    # Transpose, change to upsampling after!

    self.conv_6 = layers.Conv2DTranspose(512, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_init)
    self.conv_7 = layers.Conv2DTranspose(256, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_init)
    self.conv_8 = layers.Conv2DTranspose(128, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_init)
    self.conv_9 = layers.Conv2DTranspose(64, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_init)
    self.conv_10 = layers.Conv2D(3, kernel_size=(7, 7), strides=(1, 1), padding='valid', kernel_initializer=kernel_init)

    self.padding_values = [[0, 0], [3, 3], [3, 3], [0, 0]]


  def call(self, x):
    # inputs 640x640x3
    # example used 256, 256, 3

    x = tf.pad(x, self.padding_values, mode="REFLECT")
    x = self.conv_1(x)
    x = self.norm_1(x)
    x = layers.ReLU()(x)

    x = self.conv_2(x)
    x = self.norm_2(x)
    x = layers.ReLU()(x)

    x = self.conv_3(x)
    x = self.norm_3(x)
    x = layers.ReLU()(x)

    x = self.conv_4(x)
    x = self.norm_4(x)
    x = layers.ReLU()(x)

    x = self.conv_5(x)
    x = self.norm_5(x)
    x = layers.ReLU()(x)

    x = self.res_block_1(x)
    x = self.res_block_2(x)
    x = self.res_block_3(x)
    x = self.res_block_4(x)
    x = self.res_block_5(x)
    x = self.res_block_6(x)
    x = self.res_block_7(x)
    x = self.res_block_8(x)
    x = self.res_block_9(x)

    x = self.conv_6(x)
    x = self.norm_6(x)
    x = layers.ReLU()(x)

    x = self.conv_7(x)
    x = self.norm_7(x)
    x = layers.ReLU()(x)

    x = self.conv_8(x)
    x = self.norm_8(x)
    x = layers.ReLU()(x)

    x = self.conv_9(x)
    x = self.norm_9(x)
    x = layers.ReLU()(x)

    x = tf.pad(x, self.padding_values, mode="REFLECT")
    x = self.conv_10(x)
    x = tf.keras.activations.tanh(x)

    return x

class Discriminator(tf.keras.Model):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.conv_1 = layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=kernel_init)
    self.conv_2 = layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=kernel_init)
    self.conv_3 = layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=kernel_init)
    self.conv_4 = layers.Conv2D(512, kernel_size=(4, 4), strides=(1, 1), padding='same', kernel_initializer=kernel_init)

    self.conv_5 = layers.Conv2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same', kernel_initializer=kernel_init)

    self.conv_6 = layers.Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=kernel_init)
    self.conv_7 = layers.Conv2D(128, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=kernel_init)
    self.conv_8 = layers.Conv2D(256, kernel_size=(4, 4), strides=(2, 2), padding='same', kernel_initializer=kernel_init)
    self.conv_9 = layers.Conv2D(512, kernel_size=(4, 4), strides=(1, 1), padding='same', kernel_initializer=kernel_init)

    self.conv_10 = layers.Conv2D(1, kernel_size=(4, 4), strides=(1, 1), padding='same', kernel_initializer=kernel_init)

    self.norm_1 = tfa.layers.InstanceNormalization()
    self.norm_2 = tfa.layers.InstanceNormalization()
    self.norm_3 = tfa.layers.InstanceNormalization()
  
    self.norm_4 = tfa.layers.InstanceNormalization()
    self.norm_5 = tfa.layers.InstanceNormalization()
    self.norm_6 = tfa.layers.InstanceNormalization()
    
  def call(self, input_tensor):
    # input size HxwX6 channels=6
    output_1 = []
    output_2 = []
    
    x = input_tensor
    x = self.conv_1(x)
    x = layers.LeakyReLU()(x)

    output_1.append(x)

    x = self.conv_2(x)
    x = self.norm_1(x)
    x = layers.LeakyReLU()(x)

    output_1.append(x)

    x = self.conv_3(x)
    x = self.norm_2(x)
    x = layers.LeakyReLU()(x)

    output_1.append(x)

    x = self.conv_4(x)
    x = self.norm_3(x)
    x = layers.LeakyReLU()(x)

    output_1.append(x)

    x = self.conv_5(x)

    output_1.append(x)

    x2 = layers.AveragePooling2D()(input_tensor)
    x2 = self.conv_6(x2)
    x2 = layers.LeakyReLU()(x2)

    output_2.append(x2)

    x2 = self.conv_7(x2)
    x2 = self.norm_4(x2)
    x2 = layers.LeakyReLU()(x2)

    output_2.append(x2)

    x2 = self.conv_8(x2)
    x2 = self.norm_5(x2)
    x2 = layers.LeakyReLU()(x2)

    output_2.append(x2)

    x2 = self.conv_9(x2)
    x2 = self.norm_6(x2)
    x2 = layers.LeakyReLU()(x2)

    output_2.append(x2)

    x2 = self.conv_10(x2)

    output_2.append(x2)

    return [output_1, output_2]



class Vgg19(tf.keras.Model):
  def __init__(self):
    super(Vgg19, self).__init__()
    layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'] 
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    
    outputs = [vgg.get_layer(name).output for name in layers]

    self.model = tf.keras.Model([vgg.input], outputs)
  
  def call(self, x):
    x = tf.keras.applications.vgg19.preprocess_input(x * 255.0)
    return self.model(x)

