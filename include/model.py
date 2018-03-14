import tensorflow as tf


def model():
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10
    _RESHAPE_SIZE = 4*4*128

    def variable_with_weight_decay(name, shape, stddev, wd):
        dtype = tf.float32
        var = variable_on_cpu( name, shape, tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def variable_on_cpu(name, shape, initializer):
        with tf.device('/cpu:0'):
            dtype = tf.float32
            var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        return var

### Classifier with regular examples ###############################################################################################################################

    with tf.name_scope('reg_data'):
        reg_x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        reg_y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        reg_x_image = tf.reshape(reg_x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
    with tf.variable_scope('reg_conv1') as scope:
        kernel1 = variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(reg_x_image, kernel1, [1, 1, 1, 1], padding='SAME')
        biases1 = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases1)
        reg_conv1 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/reg_conv1', reg_conv1)
    tf.summary.scalar('Convolution_layers/reg_conv1', tf.nn.zero_fraction(reg_conv1))

    reg_norm1 = tf.nn.lrn(reg_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='reg_norm1')
    reg_pool1 = tf.nn.max_pool(reg_norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='reg_pool1')

    with tf.variable_scope('reg_conv2') as scope:
        kernel2 = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(reg_pool1, kernel2, [1, 1, 1, 1], padding='SAME')
        biases2 = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases2)
        reg_conv2 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/reg_conv2', reg_conv2)
    tf.summary.scalar('Convolution_layers/reg_conv2', tf.nn.zero_fraction(reg_conv2))

    reg_norm2 = tf.nn.lrn(reg_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    reg_pool2 = tf.nn.max_pool(reg_norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('reg_conv3') as scope:
        kernel3 = variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(reg_pool2, kernel3, [1, 1, 1, 1], padding='SAME')
        biases3 = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases3)
        reg_conv3 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/reg_conv3', reg_conv3)
    tf.summary.scalar('Convolution_layers/reg_conv3', tf.nn.zero_fraction(reg_conv3))

    with tf.variable_scope('reg_conv4') as scope:
        kernel4 = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(reg_conv3, kernel4, [1, 1, 1, 1], padding='SAME')
        biases4 = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases4)
        reg_conv4 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/reg_conv4', reg_conv4)
    tf.summary.scalar('Convolution_layers/reg_conv4', tf.nn.zero_fraction(reg_conv4))

    with tf.variable_scope('reg_conv5') as scope:
        kernel5 = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(reg_conv4, kernel5, [1, 1, 1, 1], padding='SAME')
        biases5 = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases5)
        reg_conv5 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/reg_conv5', reg_conv5)
    tf.summary.scalar('Convolution_layers/reg_conv5', tf.nn.zero_fraction(reg_conv5))

    reg_norm3 = tf.nn.lrn(reg_conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    reg_pool3 = tf.nn.max_pool(reg_norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

#    keep_prob_input = tf.placeholder(tf.float32)

    with tf.variable_scope('discr_reg_fully_connected1') as scope:
        reshape = tf.reshape(reg_pool3, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        weights6 = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases6 = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
 #       discr_reg_dropout1 = tf.nn.dropout(reshape, keep_prob=keep_prob_input)
        discr_reg1 = tf.nn.relu(tf.matmul(reshape, weights6) + biases6, name=scope.name)

    with tf.variable_scope('discr_reg_fully_connected2') as scope:
        weights7 = variable_with_weight_decay('weights', shape=[384,2], stddev=0.04, wd=0.004)
        biases7 = variable_on_cpu('biases', [2], tf.constant_initializer(0.1))
       	discr_reg_final = tf.matmul(discr_reg1, weights7) + biases7

    with tf.variable_scope('reg_fully_connected1') as scope:
        reshape = tf.reshape(reg_pool3, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        weights8 = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases8 = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        reg_local3 = tf.nn.relu(tf.matmul(reshape, weights8) + biases8, name=scope.name)
    tf.summary.histogram('Fully connected layers/reg_fc1', reg_local3)
    tf.summary.scalar('Fully connected layers/reg_fc1', tf.nn.zero_fraction(reg_local3))

    with tf.variable_scope('reg_fully_connected2') as scope:
        weights9 = variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases9 = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        reg_local4 = tf.nn.relu(tf.matmul(reg_local3, weights9) + biases9, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc2', reg_local4)
    tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(reg_local4))

    with tf.variable_scope('reg_output') as scope:
        weights10 = variable_with_weight_decay('weights', [192, _NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases10 = variable_on_cpu('biases', [_NUM_CLASSES], tf.constant_initializer(0.0))
        reg_softmax_linear = tf.add(tf.matmul(reg_local4, weights10), biases10, name=scope.name)
    tf.summary.histogram('Fully connected layers/output', reg_softmax_linear)

    reg_y_pred_cls = tf.argmax(reg_softmax_linear, axis=1)

### Classifier with adversarial examples ############################################################################################################


    with tf.name_scope('adv_data'):
        adv_x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        adv_y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        adv_x_image = tf.reshape(adv_x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
    with tf.variable_scope('adv_conv1') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(adv_x_image, kernel1, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases1)
        adv_conv1 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/adv_conv1', adv_conv1)
    tf.summary.scalar('Convolution_layers/adv_conv1', tf.nn.zero_fraction(adv_conv1))

    adv_norm1 = tf.nn.lrn(adv_conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='adv_norm1')
    adv_pool1 = tf.nn.max_pool(adv_norm1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='adv_pool1')

    with tf.variable_scope('adv_conv2') as scope:
        kernel = variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(adv_pool1, kernel2, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv, biases2)
        adv_conv2 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/adv_conv2', adv_conv2)
    tf.summary.scalar('Convolution_layers/adv_conv2', tf.nn.zero_fraction(adv_conv2))

    adv_norm2 = tf.nn.lrn(adv_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    adv_pool2 = tf.nn.max_pool(adv_norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    with tf.variable_scope('adv_conv3') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(adv_pool2, kernel3, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases3)
        adv_conv3 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/adv_conv3', adv_conv3)
    tf.summary.scalar('Convolution_layers/adv_conv3', tf.nn.zero_fraction(adv_conv3))

    with tf.variable_scope('adv_conv4') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(adv_conv3, kernel4, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases4)
        adv_conv4 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/adv_conv4', adv_conv4)
    tf.summary.scalar('Convolution_layers/adv_conv4', tf.nn.zero_fraction(adv_conv4))

    with tf.variable_scope('adv_conv5') as scope:
        kernel = variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(adv_conv4, kernel5, [1, 1, 1, 1], padding='SAME')
        biases = variable_on_cpu('biases', [128], tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv, biases5)
        adv_conv5 = tf.nn.relu(pre_activation, name=scope.name)
    tf.summary.histogram('Convolution_layers/adv_conv5', adv_conv5)
    tf.summary.scalar('Convolution_layers/adv_conv5', tf.nn.zero_fraction(adv_conv5))

    adv_norm3 = tf.nn.lrn(adv_conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
    adv_pool3 = tf.nn.max_pool(adv_norm3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    with tf.variable_scope('discr_adv_fully_connected1') as scope:
        reshape = tf.reshape(adv_pool3, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
    #    discr_adv_dropout1 = tf.nn.dropout(reshape, keep_prob=keep_prob_input)
        discr_adv1 = tf.nn.relu(tf.matmul(reshape, weights6) + biases6, name=scope.name)

    with tf.variable_scope('discr_adv_fully_connected2') as scope:
        weights = variable_with_weight_decay('weights', shape=[384,2], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [2], tf.constant_initializer(0.1))
       	discr_adv_final = tf.matmul(discr_adv1, weights7) + biases7

 


    with tf.variable_scope('adv_fully_connected1') as scope:
        reshape = tf.reshape(adv_pool3, [-1, _RESHAPE_SIZE])
        dim = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights', shape=[dim, 384], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
        adv_local3 = tf.nn.relu(tf.matmul(reshape, weights8) + biases8, name=scope.name)
    tf.summary.histogram('Fully connected layers/adv_fc1', adv_local3)
    tf.summary.scalar('Fully connected layers/adv_fc1', tf.nn.zero_fraction(adv_local3))

    with tf.variable_scope('adv_fully_connected2') as scope:
        weights = variable_with_weight_decay('weights', shape=[384, 192], stddev=0.04, wd=0.004)
        biases = variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
        adv_local4 = tf.nn.relu(tf.matmul(adv_local3, weights9) + biases9, name=scope.name)
    tf.summary.histogram('Fully connected layers/fc2', adv_local4)
    tf.summary.scalar('Fully connected layers/fc2', tf.nn.zero_fraction(adv_local4))

    with tf.variable_scope('adv_output') as scope:
        weights = variable_with_weight_decay('weights', [192, _NUM_CLASSES], stddev=1 / 192.0, wd=0.0)
        biases = variable_on_cpu('biases', [_NUM_CLASSES], tf.constant_initializer(0.0))
        adv_softmax_linear = tf.add(tf.matmul(adv_local4, weights10), biases10, name=scope.name)
    tf.summary.histogram('Fully connected layers/output', adv_softmax_linear)

    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    adv_y_pred_cls = tf.argmax(adv_softmax_linear, axis=1)

    discr_reg_y = tf.placeholder(tf.int32, [None, 2])
    discr_adv_y = tf.placeholder(tf.int32, [None, 2])
    print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'discr'))
    return reg_x, reg_y, reg_softmax_linear, reg_y_pred_cls, adv_x, adv_y, adv_softmax_linear, global_step, adv_y_pred_cls, discr_reg_final, discr_adv_final, discr_reg_y, discr_adv_y
