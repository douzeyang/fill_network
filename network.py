import tensorflow as tf
import tensorflow.contrib.slim as slim
from mask_matmul import mask_matmul
class fill_net(object):
    def __init__(self, input_shape = 210, is_training = True, in_shape = 210*5*2, out_shape = 512, output_shape = 70, batch_size = 1, keep_prob = 0.8):
        self.input_shape = input_shape
        self.is_training = is_training
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.output_shape = output_shape
        self.keep_prob = keep_prob
        self.input = tf.placeholder(dtype = tf.float32, shape = (batch_size, input_shape))
        self.mask_encode = tf.placeholder(dtype = tf.float32, shape = (None, None))
        self.mask_decode = tf.placeholder(dtype = tf.float32, shape = (None, None))
        self.target = tf.placeholder(dtype = tf.float32, shape = (None, output_shape))
        self.batch_size = 3
        self.loss_exp = tf.placeholder(dtype = tf.float32)

    def build_net(self):
        input_weights = slim.variable('input_weights',
                                shape = [int(self.input_shape), int(self.in_shape)],
                                initializer= tf.truncated_normal_initializer(stddev=0.1),
                                regularizer=slim.l2_regularizer(0.005),
                                )
        encode1 = mask_matmul(self.input, self.mask_encode, input_weights, self.batch_size)
        input_bias = slim.variable('input_bias', shape = int(self.in_shape),
                                   initializer = tf.truncated_normal_initializer(stddev=0.1),
                                   regularizer = slim.l2_regularizer(0.005))
        encode1 = tf.nn.bias_add(encode1, input_bias)
        encode1 = tf.nn.dropout(encode1, keep_prob=self.keep_prob)
        encode1 = tf.nn.elu(encode1)
        encode2 = slim.fully_connected(encode1, 1024, activation_fn=slim.nn.elu, scope = 'fc2')

        encode3 = slim.fully_connected(encode2, 512, activation_fn = slim.nn.elu, scope = 'fc3')
        encode4 = slim.fully_connected(encode3, 64, activation_fn = slim.nn.elu, scope = 'fc4')
        encode4 = tf.nn.dropout(encode4, keep_prob=self.keep_prob)
        decode3 = slim.fully_connected(encode4, 512, activation_fn = slim.nn.elu, scope = 'fc5')+encode3

        decode2= slim.fully_connected(decode3, 1024, activation_fn = slim.nn.elu, scope = 'fc6')+encode2
        output_weights = slim.variable('output_weights',
                                       shape = [self.out_shape, self.output_shape],
                                       initializer = tf.truncated_normal_initializer(stddev=0.1),
                                       regularizer = slim.l2_regularizer(0.005),
                                       )
        # mask_output_weights = output_weights*self.mask_decode
        decode2 = tf.nn.dropout(decode2, keep_prob=self.keep_prob)
        net = tf.matmul(decode2, output_weights)*self.mask_decode
        # net = tf.nn.sigmoid(net)
        return net





# if __name__ == "__main__":
#     network = fill_net()
#     net = network.build_net()
#     print(1)