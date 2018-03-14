import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import fast_gradient

epochs = 120
learning_rate = 0.01
batch_size = 100
dropout = 0.5
n_epochs = 10
alpha = 0.5
beta = 1
LAYER_1 = 512
LAYER_2 = 256
LAYER_3 = 128
INPUT = 784
OUTPUT = 10

def weight_variable(shape, name='default'):
    initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.01)
    return tf.Variable(initial, dtype=tf.float32, name=name)

def bias_variable(shape, name='default'):
    initial = tf.constant(0.0, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, dtype=tf.float32, name=name)

mnist = input_data.read_data_sets('/data/mnist', one_hot=True)

y_ = tf.placeholder(tf.float32, [None, OUTPUT])
x_norm = tf.placeholder(tf.float32, [None, INPUT])
x_adv = tf.placeholder(tf.float32, [None, INPUT])

W_fc1 = weight_variable([INPUT, LAYER_1], 'cew1')
W_fc2 = weight_variable([LAYER_1, LAYER_2], 'cew2')
W_fc3 = weight_variable([LAYER_2, LAYER_3], 'cw3')
W_fc4 = weight_variable([LAYER_3, OUTPUT], 'cw4')

b_fc1 = bias_variable([LAYER_1], 'ceb1')
b_fc2 = bias_variable([LAYER_2], 'ceb2')
b_fc3 = bias_variable([LAYER_3], 'cb3')
b_fc4 = bias_variable([OUTPUT], 'cb4')

# Regular examples
h_fc1_norm = tf.nn.relu(tf.matmul(x_norm, W_fc1) + b_fc1)
h_fc2_norm = tf.nn.relu(tf.matmul(h_fc1_norm, W_fc2) + b_fc2)
h_fc3_norm = tf.nn.relu(tf.matmul(h_fc2_norm, W_fc3) + b_fc3)
final_norm = tf.matmul(h_fc3_norm, W_fc4) + b_fc4
cross_norm = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=final_norm))
#print(tf.gradients(final_norm, x_norm))
#exit()

# Adversarial examples
h_fc1_adv = tf.nn.relu(tf.matmul(x_adv, W_fc1) + b_fc1)
h_fc2_adv = tf.nn.relu(tf.matmul(h_fc1_adv, W_fc2) + b_fc2)
h_fc3_adv = tf.nn.relu(tf.matmul(h_fc2_adv, W_fc3) + b_fc3)
final_adv = tf.matmul(h_fc3_adv, W_fc4) + b_fc4
cross_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=final_adv))

# For generating adversarial examples
sm_norm = tf.nn.softmax(final_norm)

# Discriminator
y_norm = tf.placeholder(tf.int32, [None, 2])
y_adv = tf.placeholder(tf.int32, [None, 2])

W_d1 = weight_variable([LAYER_2, LAYER_3], 'dw1')
W_d2 = weight_variable([LAYER_3, 2], 'dw2')
b_d1 = bias_variable([LAYER_3], 'db1')
b_d2 = bias_variable([2], 'db2')
keep_prob_input = tf.placeholder(tf.float32)
drop_reg_discr = tf.nn.dropout(h_fc2_norm, keep_prob=keep_prob_input)
drop_adv_discr = tf.nn.dropout(h_fc2_adv, keep_prob=keep_prob_input)
discr_norm1 = tf.nn.relu(tf.matmul(drop_reg_discr, W_d1) + b_d1)
discr_adv1 = tf.nn.relu(tf.matmul(drop_adv_discr, W_d1) + b_d1)
final_discr_norm = tf.matmul(discr_norm1, W_d2) + b_d2
final_discr_adv = tf.matmul(discr_adv1, W_d2) + b_d2

cross_discr_norm = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_norm, logits=final_discr_norm))
cross_discr_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_adv, logits=final_discr_adv))
discr_loss = (cross_discr_norm + cross_discr_adv)

loss = alpha * cross_norm + (1 - alpha) * cross_adv - beta * cross_discr_adv
#enc_loss = beta * cross_discr_adv

classifier_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'c');
discriminator_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'd')
encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ce')
#print encoder_vars

# define training step and accuracy
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_norm)#(loss)#, var_list=classifier_vars)
correct_prediction = tf.equal(tf.argmax(final_norm, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

train_step_discr = tf.train.AdamOptimizer(learning_rate).minimize(discr_loss, var_list=discriminator_vars)
correct_norm_discr = tf.equal(tf.argmax(y_norm, 1), tf.argmax(final_discr_norm,1))#, output_type = tf.int32))
correct_adv_discr = tf.equal(tf.argmax(y_adv, 1), tf.argmax(final_discr_adv,1))#, output_type = tf.int32))

#train_step_enc = tf.train.AdamOptimizer(learning_rate).minimize(enc_loss, var_list=encoder_vars)

norm_acc = tf.reduce_mean(tf.cast(correct_norm_discr, tf.float32))
adv_acc = tf.reduce_mean(tf.cast(correct_adv_discr, tf.float32))
comb_acc = (norm_acc + adv_acc) / 2

# create a saver
saver = tf.train.Saver()

# initialize graph
init = tf.global_variables_initializer()

# generating adversarial images
fgm_eps = tf.placeholder(tf.float32, ())
fgm_epochs = tf.placeholder(tf.float32, ())
adv_examples = fast_gradient.fgm(x_norm, final_norm, sm_norm, eps=fgm_eps, epochs=fgm_epochs) 

with tf.Session() as sess:
    sess.run(init)
    
    y_norm_labels = np.squeeze(np.stack([[np.array([1,0])] for _ in range(batch_size)], axis = 0))
    y_adv_labels = np.squeeze(np.stack([[np.array([0,1])] for _ in range(batch_size)], axis = 0))

    for i in range(epochs):
        print "EPOCH: " + str(i + 1)
        for j in range(mnist.train.num_examples/batch_size):
            print j
            input_images, correct_predictions = mnist.train.next_batch(batch_size)
            final_logits = sess.run(final_norm, feed_dict={x_norm: input_images})
            final_output = sess.run(sm_norm, feed_dict={x_norm: input_images})
            adv_images = sess.run(adv_examples, feed_dict={x_norm: input_images, final_norm: final_logits, sm_norm: final_output, fgm_eps: 0.25, fgm_epochs: 1}) 
            #GENERATE ADVERSARIAL IMAGES
            if j == 0:
                discr_accuracy = sess.run(comb_acc, feed_dict={keep_prob_input:1.0, x_norm:input_images, x_adv:adv_images, y_:correct_predictions, y_norm:y_norm_labels, y_adv:y_adv_labels})
                train_accuracy = sess.run(accuracy, feed_dict={x_norm:input_images, y_:correct_predictions})#x_adv:adv_images, y_:correct_predictions})
                print "DISCRIMINATOR ACCURACY: " + str(discr_accuracy)
                print "CLASSIFIER ACCURACY: " + str(train_accuracy)
#                print sess.run(W_fc1)
                print sess.run(W_fc4)
#                print sess.run(b_fc1)
#                print sess.run(b_fc4)
                path = saver.save(sess, 'mnist_save')
            sess.run(train_step_discr, feed_dict={keep_prob_input:dropout, x_norm:input_images, x_adv:adv_images, y_:correct_predictions, y_norm:y_norm_labels, y_adv:y_adv_labels})
            sess.run(train_step, feed_dict={keep_prob_input:dropout, x_norm:input_images, y_:correct_predictions, x_adv:adv_images, y_norm:y_norm_labels, y_adv:y_adv_labels})
#            sess.run(train_step_enc, feed_dict={keep_prob_input:dropout, x_norm:input_images, x_adv:adv_images, y_:correct_predictions, y_norm:y_adv_labels, y_adv:y_adv_labels})
