import argparse
import time

import tensorflow as tf
# from tensorflow.python.client import timeline
from bisenet import create_bisenet
from utils import gen_batch_fn_idd

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--ckpt", help="Loads trained weights", action="store_true")
	args = parser.parse_args()

	return args


def train():

	args = parse_args()

	EPOCHS = 10
	BATCH_SIZE = 4
	TEST_DIR = 'data/data_road/testing'
	SAVE_DIR = 'saved_tests'

	output_label, input_im, gt_im, init_fn = create_bisenet()
	print(input_im)
	print(output_label)
	
	cross_entropy_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels = gt_im, logits = output_label))
	# optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cross_entropy_loss)
	optimizer = tf.train.RMSPropOptimizer(learning_rate = 0.0001, decay = 0.995).minimize(cross_entropy_loss)

	saver = tf.train.Saver()
	sess = tf.Session()
	# run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
	# run_metadata = tf.RunMetadata()

	if args.ckpt:
		saver.restore(sess, "./bisenet_ckpt/model_idd2")
		# saver.restore(sess, "./bisenet_ckpt/model_idd")
		# saver.restore(sess, "./bisenet_ckpt/model")
	else:
		sess.run(tf.global_variables_initializer())
		init_fn(sess)


	get_batch = gen_batch_fn_idd('data/IDD', width = 1280, height = 704)
	# j = 0
	for i in range(EPOCHS):
		print("EPOCH {} ...".format(i+1))
		for image, label in get_batch(BATCH_SIZE):
			print(image.shape, label.shape)
			# s = time.time()
			_, loss = sess.run([optimizer, cross_entropy_loss], 
												feed_dict={input_im: image, gt_im: label})
#                         options=run_options, run_metadata=run_metadata)
			print("Loss: = {:.3f}".format(loss))
# 			print(time.time() - s)
				
#       if j==5:
#         tl = timeline.Timeline(run_metadata.step_stats)
#         ctf = tl.generate_chrome_trace_format()
#         with open('timeline.json', 'w') as f:
#             f.write(ctf)
#         print("Done")
#         break
#       j+=1
#   print()
#   break

	#   if i%10 == 0:
		# save_path = saver.save(sess, "./bisenet_ckpt/model_idd2")
		#save_test_predictions(sess, TEST_DIR, SAVE_DIR, width = 1120, height = 256)

	sess.close()






if __name__ == "__main__":
    train()