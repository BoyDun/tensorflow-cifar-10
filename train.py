import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from time import time
import fast_gradient
from include.data import get_data_set
from include.model import model
import scipy.misc

alpha = 0.5
beta = 1

train_x, train_y, train_l = get_data_set()
test_x, test_y, test_l = get_data_set("test")

reg_x, reg_y, reg_output, reg_y_pred_cls, adv_x, adv_y, adv_output, global_step, adv_y_pred_cls, discr_reg_final, discr_adv_final, discr_reg_y, discr_adv_y = model()

_IMG_SIZE = 32
_NUM_CHANNELS = 3
_BATCH_SIZE = 128
_CLASS_SIZE = 10
_ITERATION = 10000
_SAVE_PATH = "./tensorboard/cifar-10/"
dropout = 0.5

cross_discr_norm = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = discr_reg_final, labels = discr_reg_y))
cross_discr_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = discr_adv_final, labels = discr_adv_y))
discr_loss = cross_discr_norm + cross_discr_adv
discr_optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(discr_loss, var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discr'))
correct_norm_discr = tf.equal(tf.argmax(discr_reg_y, 1), tf.argmax(discr_reg_final, 1))
correct_adv_discr = tf.equal(tf.argmax(discr_adv_y, 1), tf.argmax(discr_adv_final, 1))
discr_accuracy = (tf.reduce_mean(tf.cast(correct_norm_discr, tf.float32)) + tf.reduce_mean(tf.cast(correct_adv_discr, tf.float32)))/2

loss = alpha * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=reg_output, labels=reg_y)) + (1 - alpha) * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=adv_output, labels=adv_y)) - beta * cross_discr_adv 
sm_norm = tf.nn.softmax(reg_output)
optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-3).minimize(loss, global_step=global_step)
fgm_eps = tf.placeholder(tf.float32, ())
fgm_epochs = tf.placeholder(tf.float32, ())
adv_examples = fast_gradient.fgm(reg_x, reg_output, sm_norm, eps = fgm_eps, epochs = fgm_epochs)

reg_correct_prediction = tf.equal(reg_y_pred_cls, tf.argmax(reg_y, axis=1))
adv_correct_prediction = tf.equal(adv_y_pred_cls, tf.argmax(adv_y, axis=1))
accuracy = (tf.reduce_mean(tf.cast(reg_correct_prediction, tf.float32)) + tf.reduce_mean(tf.cast(adv_correct_prediction, tf.float32)))/2
tf.summary.scalar("Accuracy/train", accuracy)


merged = tf.summary.merge_all()
saver = tf.train.Saver()
sess = tf.Session()
train_writer = tf.summary.FileWriter(_SAVE_PATH, sess.graph)


try:
    print("Trying to restore last checkpoint ...")
    last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=_SAVE_PATH)
    saver.restore(sess, save_path=last_chk_path)
    print("Restored checkpoint from:", last_chk_path)
except:
    print("Failed to restore checkpoint. Initializing variables instead.")
    sess.run(tf.global_variables_initializer())


def train(num_iterations):
    '''
        Train CNN
    '''
    counter = 0
    y_norm_labels = np.squeeze(np.stack([[np.array([1,0])] for _ in range(_BATCH_SIZE)], axis = 0))
    
    y_adv_labels = np.squeeze(np.stack([[np.array([0,1])] for _ in range(_BATCH_SIZE)], axis = 0))
    for i in range(num_iterations):
        randidx = np.random.randint(len(train_x), size=_BATCH_SIZE)
        batch_xs = train_x[randidx]
        batch_ys = train_y[randidx]
	
        output = sess.run(reg_output, feed_dict={reg_x: batch_xs})
        softmax_norm = sess.run(sm_norm, feed_dict={reg_x: batch_xs})
        adv_images = sess.run(adv_examples, feed_dict={reg_x: batch_xs, reg_output: output, sm_norm: softmax_norm, fgm_eps: 0.25, fgm_epochs: 1})
        if counter % 500 == 0:
                scipy.misc.imsave("image_normal_" + str(counter) + ".jpeg", batch_xs[0, :].reshape((32, 32, 3)))
                scipy.misc.imsave("image_adv_" + str(counter) + ".jpeg", adv_images[0, :].reshape((32, 32, 3)))
        counter+=1
        start_time = time()
        i_global, _ = sess.run([global_step, optimizer], feed_dict={reg_x: batch_xs, reg_y: batch_ys, adv_x: adv_images, adv_y: batch_ys, discr_reg_y:y_norm_labels, discr_adv_y: y_adv_labels})
        duration = time() - start_time
        sess.run(discr_optimizer, feed_dict={reg_x: batch_xs, adv_x: adv_images, discr_reg_y: y_norm_labels, discr_adv_y: y_adv_labels}) 
        if (i_global % 10 == 0) or (i == num_iterations - 1):
            _loss, batch_acc = sess.run([loss, accuracy], feed_dict={reg_x: batch_xs, reg_y: batch_ys, adv_x: adv_images, adv_y: batch_ys, discr_reg_y:y_norm_labels, discr_adv_y: y_adv_labels})
            discr_acc, disc_loss, test= sess.run([discr_accuracy, discr_loss, discr_reg_final], feed_dict={reg_x: batch_xs, adv_x: adv_images, discr_reg_y: y_norm_labels, discr_adv_y: y_adv_labels})
            msg = "Global Step: {0:>6}, accuracy: {1:>6.1%}, loss = {2:.2f} ({3:.1f} examples/sec, {4:.2f} sec/batch)"
            print(msg.format(i_global, batch_acc, _loss, _BATCH_SIZE / duration, duration))
            print("Discriminator Accuracy (Loss): " + str(discr_acc) + " (" + str(disc_loss) + ")")

        if (i_global % 100 == 0) or (i == num_iterations - 1):
            data_merged, global_1 = sess.run([merged, global_step], feed_dict={reg_x: batch_xs, reg_y: batch_ys, adv_x:adv_images, adv_y: batch_ys})
            acc = predict_test()

            summary = tf.Summary(value=[
                tf.Summary.Value(tag="Accuracy/test", simple_value=acc),
            ])
            train_writer.add_summary(data_merged, global_1)
            train_writer.add_summary(summary, global_1)

            saver.save(sess, save_path=_SAVE_PATH, global_step=global_step)
            print("Saved checkpoint.")


def predict_test(show_confusion_matrix=False):
    '''
        Make prediction for all images in test_x
    '''
    i = 0
    predicted_class = np.zeros(shape=len(test_x), dtype=np.int)
    while i < len(test_x):
        j = min(i + _BATCH_SIZE, len(test_x))

        batch_xs = test_x[i:j, :]
        batch_ys = test_y[i:j, :]
        predicted_class[i:j] = sess.run(reg_y_pred_cls, feed_dict={reg_x: batch_xs, reg_y: batch_ys})
        i = j

    correct = (np.argmax(test_y, axis=1) == predicted_class)
    acc = correct.mean()*100
    correct_numbers = correct.sum()
    print("Accuracy on Test-Set: {0:.2f}% ({1} / {2})".format(acc, correct_numbers, len(test_x)))

    if show_confusion_matrix is True:
        cm = confusion_matrix(y_true=np.argmax(test_y, axis=1), y_pred=predicted_class)
        for i in range(_CLASS_SIZE):
            class_name = "({}) {}".format(i, test_l[i])
            print(cm[i, :], class_name)
        class_numbers = [" ({0})".format(i) for i in range(_CLASS_SIZE)]
        print("".join(class_numbers))

    return acc


if _ITERATION != 0:
    train(_ITERATION)


sess.close()
