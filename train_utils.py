import tensorflow as tf

from networks.loss_functions import compute_generator_loss, compute_discriminator_loss

def inner_step(input_img,
               label_img,
               discriminator,
               generator,
               vgg,
               generator_optimizer,
               discriminator_optimizer
               ):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    fake_image = generator(input_img)

    discriminator_fake_input = tf.concat([input_img, fake_image], axis=-1)
    fake_prediction = discriminator(discriminator_fake_input)

    discriminator_real_input = tf.concat([input_img, label_img], axis=-1)
    real_prediction = discriminator(discriminator_real_input)

    true_list = vgg(label_img)
    fake_list = vgg(fake_image)

    generator_loss = compute_generator_loss(real_prediction, fake_prediction, true_list, fake_list)
    discriminator_loss = compute_discriminator_loss(real_prediction, fake_prediction)
  
  g_gradients = gen_tape.gradient(generator_loss, generator.trainable_variables)
  d_gradients = disc_tape.gradient(discriminator_loss, discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

  return generator_loss, discriminator_loss

def lr_decay(optimizer, learning_rate):
  current_lr = optimizer.lr
  new_lr = current_lr - (learning_rate / 8000)
  optimizer.lr = new_lr

  return optimizer

