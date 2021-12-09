import tensorflow as tf
import time
import os
from train_utils import inner_step, lr_decay
import builtins

from networks.models import Generator, Discriminator, Vgg19
from data_generator.data_generator import train_dataset 

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=12000, help='Number of epochs to train the model')
parser.add_argument('--start_epoch', type=int, default=0, help='Number of epochs to train the model')
parser.add_argument('--continue_training', default=False, action='store_true', help='Use training checkpoints to continue training the model')
parser.add_argument('--save_generator', default=False, action='store_true', help='Save the generator after training')

using_notebook = getattr(builtins, "__IPYTHON__", False)

opts = parser.parse_args([]) if using_notebook else parser.parse_args()

learning_rate = 0.0002
generator_lr = learning_rate
discriminator_lr = learning_rate

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_lr, beta_1=0.5, beta_2=0.999)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_lr, beta_1=0.5, beta_2=0.999)

batch_size = 1

train_generator = train_dataset.batch(batch_size)

generator = Generator()
discriminator = Discriminator()
vgg = Vgg19()



checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator,
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer)

if opts.continue_training:
  print("loading training checkpoints: ")                   
  print(tf.train.latest_checkpoint(checkpoint_dir))
  checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))



@tf.function
def train_step(input_img, label_img):
  return inner_step(input_img,
                    label_img,
                    discriminator,
                    generator,
                    vgg,
                    generator_optimizer,
                    discriminator_optimizer)

train_loss_results, train_loss_metric = [], tf.keras.metrics.Mean()
gen_loss_results, gen_loss_metric = [], tf.keras.metrics.Mean()
disc_loss_results, disc_loss_metric = [], tf.keras.metrics.Mean()

epochs = opts.epochs
start_epoch = opts.start_epoch
epoch_decay = 8000

def train(epochs):
  for epoch in range(start_epoch, start_epoch + epochs):
    epoch_time = time.time()
    epoch_count = f"0{epoch + 1}/{start_epoch + epochs}" if epoch < 9 else f"{epoch + 1}/{start_epoch + epochs}"

    for input_img, label_img in train_generator:
      generator_loss, discriminator_loss = train_step(input_img, label_img)

      generator_loss = float(generator_loss)
      discriminator_loss = float(discriminator_loss)
      loss = generator_loss + discriminator_loss  
        
    train_loss_results.append(loss)
    gen_loss_results.append(generator_loss)
    disc_loss_results.append(discriminator_loss)
    
    if (epoch % 250 ) == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)

    if epoch > epoch_decay:
      generator_optimizer = lr_decay(generator_optimizer, learning_rate)
      discriminator_optimizer = lr_decay(discriminator_optimizer, learning_rate)
    
    if (epoch % 100) == 0:
      print('\r', 'Epoch', epoch_count, '| Loss:', f"{loss:.5f}", '| Discriminator loss:', f"{discriminator_loss:.5f}",
          '| Generator loss:', f"{generator_loss:.5f}", "| Epoch Time:", f"{time.time() - epoch_time:.2f}")

train(epochs)

if opts.save_generator:
  generator.save('saved_model/tps_generator')

