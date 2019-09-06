import os
import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.keras import layers


def arm_module(layer, n_filter_maps):
  
  net = tf.math.reduce_mean(layer, axis = [1,2], keepdims = True, name = 'global_avg_pool')
  net = layers.Conv2D(filters = n_filter_maps, kernel_size = [1,1])(net)
  net = layers.BatchNormalization()(net)
  net = tf.math.sigmoid(net, name = 'sigmoid')
  
  scaled_layer = tf.multiply(layer, net)
  
  return scaled_layer


def ffm_module(spatial_layer, context_layer, num_classes):
	'''
	Fuses context layer with spatial layer
	'''
	input_features = tf.concat([spatial_layer, context_layer], axis = -1)
	net = layers.Conv2D(filters = num_classes, kernel_size = [3,3], padding = 'same', activation = 'relu')(input_features)
	net = layers.BatchNormalization()(net)
  
	net_vector = tf.reduce_mean(net, axis = [1,2], keepdims = True)
  
  # First 1x1 convolution uses 16 filters
	net_vector = layers.Conv2D(filters = 16, kernel_size = [1,1], padding = 'same', activation = 'relu')(net_vector)
	net_vector = layers.Conv2D(filters = num_classes, kernel_size = [1,1], padding = 'same')(net_vector)
	net_vector = layers.Activation('sigmoid')(net_vector)

	net_scaled = tf.multiply(net, net_vector)
	net = tf.add(net, net_scaled)
  
	return net


def create_context_path(input_im):

	with slim.arg_scope(resnet_v2.resnet_arg_scope()):
		last_layer, end_points = resnet_v2.resnet_v2_101(input_im, is_training=True, scope='resnet_v2_101', global_pool = False)
		frontend_scope='resnet_v2_101'
		init_fn = slim.assign_from_checkpoint_fn(model_path=os.path.join('models', 'resnet_v2_101.ckpt'), var_list=slim.get_model_variables('resnet_v2_101'), ignore_missing_vars=True)
		
		layer_reduced16 = end_points[frontend_scope + '/block2']
		layer_reduced32 = last_layer
		layer_arm16 = arm_module(layer_reduced16, n_filter_maps = 512)
		layer_arm32 = arm_module(layer_reduced32, n_filter_maps = 2048)
		layer_global_context = tf.reduce_mean(last_layer, axis = [1,2], keepdims = True, name = 'global_context')

		## Combining Context Features
		layer_context1 = tf.math.multiply(layer_arm32, layer_global_context)
		layer_context1 = layers.UpSampling2D(size = 4, interpolation = 'bilinear')(layer_context1)
		layer_context2 = layers.UpSampling2D(size = 2, interpolation = 'bilinear')(layer_arm16)
		context_output = tf.concat([layer_context1, layer_context2], axis = -1)

		return context_output, init_fn


def create_spatial_path(input_im):

	layer_spatial = layers.Conv2D(input_shape=(None, None, None, 3), filters=64, kernel_size=[3,3], strides=2, padding='same', activation='relu')(input_im)
	layer_spatial = layers.Conv2D(filters=128, kernel_size=[3,3], strides=2, padding='same', activation='relu')(layer_spatial)
	spatial_output = layers.Conv2D(filters=256, kernel_size=[3,3], strides=2, padding='same', activation='relu')(layer_spatial)

	return spatial_output


def create_bisenet():

	tf.reset_default_graph()

	frontend = 'ResNet101'
	num_classes = 2
	input_im = tf.placeholder(shape = (None, None, None, 3), dtype = tf.float32, name='input_im')
	gt_im = tf.placeholder(shape = (None, None, None, 2), dtype = tf.float32, name='gt_im')

	spatial_output = create_spatial_path(input_im)
	context_output, init_fn = create_context_path(input_im)
	ffm_output = ffm_module(spatial_output, context_output, num_classes)

	## Final Upsampling by a factor of 8
	output_label = layers.UpSampling2D(size = 8, interpolation = 'bilinear')(ffm_output)
	output_label = layers.Conv2D(filters = num_classes, kernel_size = [1,1], activation = None)(output_label)

	return output_label, input_im, gt_im, init_fn