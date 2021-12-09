import tensorflow as tf
from networks.models import Generator, Discriminator

generator = Generator()
discriminator = Discriminator()

generator_input_shape = (2, 256, 512, 3)
generator_input = tf.random.normal(generator_input_shape)

generator_output = generator(generator_input)
# discriminator_output = generator(generator_output)

def test_generator_output_shape():
  assert generator_output.shape == (2, 256, 512, 3), generator_output.shape

