import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()

def discriminator_fake_loss(input_features):
  loss = 0.0

  for input_i in input_features: # list of 2
    pred = input_i[-1] # list of 5 where we take the last element
    loss += mse(tf.zeros_like(pred), pred)

  return loss

def discriminator_real_loss(input_features):
  loss = 0.0

  for input_i in input_features: # list of 2
    pred = input_i[-1] # list of 5 where we take the last element
    loss += mse(tf.ones_like(pred), pred)

  return loss

def compute_discriminator_loss(real_features, fake_features):
  loss = discriminator_fake_loss(fake_features) + discriminator_real_loss(real_features) * 0.5

  return loss

def feature_matching_loss(real_features, fake_features):
  D_weights = 1.0 / 2.0
  feat_weights = 4.0 / (3 + 1)
  lambda_feat = 10.0

  loss = 0.0
  
  for i in range(2):
    for j in range(len(fake_features[i]) - 1): # 5-1
      loss += D_weights * feat_weights * tf.reduce_mean(tf.abs(real_features[i][j] - fake_features[i][j])) * lambda_feat

  return loss

def perceptual_loss(true_list, fake_list):
  lambda_feat = 10.0
  weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

  loss = 0
  for feature_map_y, feature_map, w in zip(true_list, fake_list, weights):
    loss += w * tf.reduce_mean(tf.abs(feature_map_y - feature_map))
  
  return loss

def compute_generator_loss(real_features, fake_features, true_list, fake_list):
  loss = discriminator_real_loss(fake_features)
  loss += feature_matching_loss(real_features, fake_features)

  loss += perceptual_loss(true_list, fake_list)

  return loss

