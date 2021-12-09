import tensorflow as tf
import tensorflow_addons as tfa

def get_L_inverse(X, Y):
  N = X.shape[0]
  Xmat = tf.repeat(X, repeats=[13], axis=1) # Expand to right from 13x1 to 13x13
  Ymat = tf.repeat(Y, repeats=[13], axis=1) # Expand to right from 13x1 to 13x13

  P_dist_squared = tf.square(Xmat - tf.transpose(Xmat)) + tf.square(Ymat - tf.transpose(Ymat))
  P_dist_squared = tf.where(tf.equal(P_dist_squared, 0), tf.ones_like(P_dist_squared), P_dist_squared) # remove 0 values
  K = P_dist_squared * tf.math.log(P_dist_squared)

  O = tf.ones([N, 1], dtype=tf.float32) 
  Z = tf.zeros([3, 3], dtype=tf.float32) # upper case o in equation (O)
  P = tf.concat([O, X, Y], axis=1)
  L = tf.concat([tf.concat([K, P], axis=1), tf.concat([tf.transpose(P), Z], axis=1)], axis=0)

  Li = tf.linalg.inv(L)

  return Li

def create_tps(height=256, width=512):
  grid_size = 3
  axis_coords_x = tf.linspace(0, height * 2, grid_size)
  # when default params: ([  0., 256., 512.]), shape 3
  axis_coords_x = tf.cast(axis_coords_x, tf.float32)

  axis_coords_y = tf.linspace(0, width * 2, grid_size)
  # when default params: ([   0.,  512., 1024.]), shape 3
  axis_coords_y = tf.cast(axis_coords_y, tf.float32)
  
  N = 13
  P_Y, P_X = tf.meshgrid(axis_coords_x, axis_coords_y) # control points
  # Each shape (grid_size, grid_size)

  # when default P_X is
  # [[  0., 256., 512.],
  # [  0., 256., 512.],
  # [  0., 256., 512.]])

  # when default P_Y is
  # [[   0.,    0.,    0.],
  # [ 512.,  512.,  512.],
  # [1024., 1024., 1024.]]
  
  P_X = tf.reshape(P_X, (-1, 1)) # shape (grid_size * grid_size, 1)
  P_Y = tf.reshape(P_Y, (-1, 1)) # shape (grid_size * grid_size, 1)

  scale = (width * 2) * 0.1
  random_points_x = tf.random.uniform(P_X.shape, minval=-scale, maxval=scale)
  random_points_y = tf.random.uniform(P_Y.shape, minval=-scale, maxval=scale)

  DST_X = P_X + random_points_x
  DST_Y = P_Y + random_points_y

  # corners of the grid 0, width * 2  0, height * 2
  corner_points_x = tf.expand_dims([0, 0, width * 2, width * 2], axis=1)
  corner_points_x = tf.cast(corner_points_x, tf.float32)

  corner_points_y = tf.expand_dims([0, height * 2, 0, height * 2], axis=1)
  corner_points_y = tf.cast(corner_points_y, tf.float32)

  DST_X = tf.concat([DST_X, corner_points_x], axis=0) # shape ((grid_size * grid_size) + 3, 1) or (N, 1)
  DST_Y = tf.concat([DST_Y, corner_points_y], axis=0) # shape ((grid_size * grid_size) + 3, 1) or (N, 1)

  Q_X = DST_X
  Q_Y = DST_Y

  Q_X = tf.cast(Q_X, tf.float32) # shape (13, 1)
  Q_Y = tf.cast(Q_Y, tf.float32) # shape (13, 1)
  # contains the modified grid, grid + random points and corner points

  P_X = tf.concat([P_X, corner_points_x], axis=0) # shape (13, 1)
  P_Y = tf.concat([P_Y, corner_points_y], axis=0) # shape (13, 1)
  # contains the original grid and corner points

  Li = get_L_inverse(Q_X, Q_Y)  # shape (16, 16)

  P_X = tf.expand_dims(P_X, 0) # shape (1, 13, 1)
  P_Y = tf.expand_dims(P_Y, 0) # shape (1, 13, 1)

  Li = tf.expand_dims(Li, 0) # shape (1, 16, 16)

  W_X = tf.linalg.matmul(Li[:, :N, :N], P_X) # Automatic broadcast for Li
  W_Y = tf.linalg.matmul(Li[:, :N, :N], P_Y) # Automatic broadcast for Li
  # 1, 13, 13 * 1, 13, 1
  # Ignoring first dimension 13, 13 * 13, 1 == 13, 1
  # shape (1, 13, 1)

  W_X = tf.expand_dims(W_X, 3)
  W_X = tf.expand_dims(W_X, 4)
  # shape (1, 13, 1, 1, 1)
  W_X = tf.transpose(W_X, [0, 4, 2, 3, 1])
  # shape (1, 1, 1, 1, 13)

  W_Y = tf.expand_dims(W_Y, 3)
  W_Y = tf.expand_dims(W_Y, 4)
  # shape (1, 13, 1, 1, 1)
  W_Y = tf.transpose(W_Y, [0, 4, 2, 3, 1])
  # shape (1, 1, 1, 1, 13)

  # compute weights for affine part
  A_X = tf.linalg.matmul(Li[:, N:, :N], P_X) # Automatic broadcast for Li
  A_Y = tf.linalg.matmul(Li[:, N:, :N], P_Y) # Automatic broadcast for Li
  # 1, 3, 13 * 1, 13, 1
  # Ignoring first dimension 3, 13 * 13, 1 == 3, 1
  # shape (1, 3, 1)

  A_X = tf.expand_dims(A_X, 3)
  A_X = tf.expand_dims(A_X, 4)
  A_X = tf.transpose(A_X, [0, 4, 2, 3, 1])

  A_Y = tf.expand_dims(A_Y, 3)
  A_Y = tf.expand_dims(A_Y, 4)
  A_Y = tf.transpose(A_Y, [0, 4, 2, 3, 1])
  # shape (1, 1, 1, 1, 3)

  grid_Y, grid_X = tf.meshgrid(tf.linspace(0, width * 2, width), tf.linspace(0, height * 2, height)) # 0, 256, 512
  # shape (256, 512)
 
  grid_X = tf.expand_dims(tf.expand_dims(grid_X, 0), 3)
  grid_Y = tf.expand_dims(tf.expand_dims(grid_Y, 0), 3)
  # shape (1, 256, 512, 1)

  points = tf.concat([grid_X, grid_Y], axis=3)
  points = tf.cast(points, tf.float32)
  # shape (1, 256, 512, 2)

  points_X_for_summation = tf.expand_dims(points[:, :, :, 0], axis=-1) # shape (1, 256, 512, 1)
  points_Y_for_summation = tf.expand_dims(points[:, :, :, 1], axis=-1)

  # change to Q
  Q_X = tf.expand_dims(Q_X, 2)
  Q_X = tf.expand_dims(Q_X, 3)
  Q_X = tf.expand_dims(Q_X, 4)
  Q_X = tf.transpose(Q_X) # shape (1, 1, 1, 1, 13)

  Q_Y = tf.expand_dims(Q_Y, 2)
  Q_Y = tf.expand_dims(Q_Y, 3)
  Q_Y = tf.expand_dims(Q_Y, 4)
  Q_Y = tf.transpose(Q_Y) # shape (1, 1, 1, 1, 13)

  delta_X = Q_X - tf.expand_dims(points_X_for_summation, axis=-1)
  delta_Y = Q_Y - tf.expand_dims(points_Y_for_summation, axis=-1)
  # shape 1, 256, 512, 1, 13

  dist_squared = tf.square(delta_X) + tf.square(delta_Y)
  dist_squared = tf.where(tf.equal(dist_squared, 0), tf.ones_like(dist_squared), dist_squared) # remove 0 values
  U = dist_squared * tf.math.log(dist_squared)
  # shape 1, 256, 512, 1, 13

  points_X_prime = A_X[:, :, :, :, 0] + (A_X[:, :, :, :, 1] * points_X_for_summation) + (A_X[:, :, :, :, 2] * points_Y_for_summation)
  points_X_prime += tf.keras.backend.sum((W_X * U), axis=-1)

  points_Y_prime = A_Y[:, :, :, :, 0] + (A_Y[:, :, :, :, 1] * points_X_for_summation) + (A_Y[:, :, :, :, 2] * points_Y_for_summation)
  points_Y_prime += tf.keras.backend.sum((W_Y * U), axis=-1)
  # shape (1, 256, 512, 1)

  warped_grid = tf.concat([points_X_prime, points_Y_prime], axis=-1)
  # shape (1, 256, 512, 2)

  return warped_grid

def tps_augmentation(img, tps, height=256, width=512):
  new_max = width
  new_min = 0
  grid_x = (new_max - new_min) / (tf.keras.backend.max(tps[:, :, :, 1]) - tf.keras.backend.min(tps[:, :, :, 1])) * (tps[:, :, :, 1] - tf.keras.backend.max(tps[:, :, :, 1])) + new_max

  new_max = height
  new_min = 0
  grid_y = (new_max - new_min) / (tf.keras.backend.max(tps[:, :, :, 0]) - tf.keras.backend.min(tps[:, :, :, 0])) * (tps[:, :, :, 0] - tf.keras.backend.max(tps[:, :, :, 0])) + new_max

  grid = tf.stack([grid_x, grid_y], axis=-1)

  final_image = tfa.image.resampler(tf.expand_dims(img, axis=0), grid)
  
  return final_image[0]

