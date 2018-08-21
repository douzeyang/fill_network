import tensorflow as tf
from utils import *
from network import fill_net
from data_reader import data_reader
import matplotlib.pyplot as plt

batch_size = 1
in_shape = 2000
out_shape = 1024
max_epoch = 100
output_shape = 70
network = fill_net(input_shape = 210, in_shape = 2000, out_shape = out_shape, output_shape = 70, keep_prob=0.5)
net = network.build_net()
regular_loss = tf.losses.get_regularization_loss()
# loss = -tf.reduce_mean((tf.log(tf.maximum(net,1e-8))*network.target + tf.log(tf.maximum(1 - net,1e-8))*(1 - network.target))) + regular_loss
loss = tf.reduce_mean(tf.pow(tf.abs(net - network.target+ 1e-4),tf.minimum(tf.abs(network.loss_exp-network.target)*10,20))) + regular_loss

train_op = tf.train.MomentumOptimizer(0.001, 0.9)
global_step = tf.Variable(0, trainable=False)
with_clip = True
if with_clip:
    tvars = tf.trainable_variables()
    grads, norm = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10)
    train_op = train_op.apply_gradients(zip(grads, tvars), global_step=global_step)
filename = "aq_meo_gai.csv"
train_data = data_reader(filename)
train_Data2= np.copy(train_data)
mean, std, min_value, max_value = data_preprocess(train_data)
train_data = data_transform(train_data)
# fill_value = data_inv_transform(train_data, min_value, max_value)
# mean, std, min_value, max_value = data_preprocess(train_data)
# train_data = np.log(train_data+1)
# train_data = (train_data - np.log(min_value+1))/np.log(max_value+1)
# train_data_tmp = np.copy(train_data)
# train_data_tmp[np.isnan(train_data_tmp)]=0
# max_value = np.max(train_data_tmp,axis = 0)
# train_data = train_data/max_value
np.random.shuffle(train_data)
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=40)
for epoch in range(max_epoch):
    for i in range(0, train_data.shape[0], batch_size):
        input, mask_encoder, mask_decoder, target = get_batch(train_data, i, batch_size, in_shape, out_shape, output_shape)
        input[np.isnan(input)]=0
        target[np.isnan(target)] = 0
        target = target[:,:output_shape]
        if i == 0:
            _, _loss, _net = sess.run([train_op, loss, net], feed_dict = {network.input:input,
                                                          network.mask_encode:mask_encoder,
                                                          network.mask_decode:mask_decoder,
                                                          network.target:target,
                                                          network.loss_exp:1})
        else:
            _, _loss, _net = sess.run([train_op, loss, net], feed_dict={network.input: input,
                                                                        network.mask_encode: mask_encoder,
                                                                        network.mask_decode: mask_decoder,
                                                                        network.target: target,
                                                                        network.loss_exp:_net})
        if np.mod(i,100)==0:
            print("epoch:{}, i:{}, loss:{}".format(epoch,i, _loss))
        if i==train_data.shape[0]-1:
            saver.save(sess, "model/my-model", global_step=epoch)
            print('save model')


print(1)