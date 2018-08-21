import tensorflow as tf
from utils import *
from network import fill_net
from data_reader import data_reader
import matplotlib.pyplot as plt
# network = fill_net(input_shape = 210, in_shape = 2000, out_shape = 1024)
# net = network.build_net()
filename = "aq_meo_gai.csv"
train_data = data_reader(filename)
train_data2 = np.copy(train_data)
mean, std, min_value, max_value = data_preprocess(train_data)
vis = 69
num_day = 2000
with  tf.Session() as sess:

    saver = tf.train.import_meta_graph('model/my-model-3.meta')
    saver.restore(sess, 'model/my-model-3')
    graph = tf.get_default_graph()
    train_data = data_transform(train_data)
    # train_data2 = (train_data2 - mean) / std
    net_input = graph.get_tensor_by_name("Placeholder:0")
    net_mask_encode = graph.get_tensor_by_name("Placeholder_1:0")
    net_mask_decode = graph.get_tensor_by_name("Placeholder_2:0")
    net = graph.get_tensor_by_name("mul_1:0")
    b = []
    input1 = []
    for i in range(num_day):
        input, mask_encoder, mask_decoder, target = get_batch(train_data, i, 1, 2000, 1024, 70)
        input[np.isnan(input)]=0
        target[np.isnan(target)]=0
        # train_data2[np.isnan(train_data2)]=0
        mask_encoder2 = np.ones(mask_encoder.shape)
        mask_decoder2 = np.ones(mask_decoder.shape)
        a = sess.run(net, feed_dict = {net_input:target,
                                       net_mask_encode:mask_encoder2,
                                       net_mask_decode:mask_decoder2}
                                       )
        b.append(np.squeeze(a))
        input1.append(np.squeeze(input))

    b = np.array(b)
    fill_value = data_inv_transform(b, mean[:70], std[:70])
    train_vis = train_data2[:num_day,vis]

    # fill_value[np.logical_not(np.isnan(train_vis))]=train_vis[np.logical_not(np.isnan(train_vis))]
    plt.plot(fill_value[:num_day, vis])
    plt.plot(train_vis)
    plt.show()