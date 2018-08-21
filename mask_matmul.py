import tensorflow as tf

def mask_matmul(input, mask, variable, batch_size):
    """
input shape: [batch_size, a]
variable shape [a, b]
mask shape [batch_size, a]
    """
    # m = variable.get_shape()[-1]
    # for i in range(batch_size):
    #     try:
    #         mask_tmp = mask[i, :]
    #     except tf.errors.OutOfRangeError:
    #         break
    #     mask_tmp = tf.tile(mask_tmp[:, tf.newaxis], [1, m])
    #     mask_variable = mask_tmp * variable
    #     if i==0:
    #         output = tf.matmul(input[i, :][tf.newaxis, :], mask_variable)
    #     else:
    #         output_tmp = tf.matmul(input[i, :][tf.newaxis, :], mask_variable)
    #         output = tf.stack(output, output_tmp, axis=0)
    # return output
    # shape = variable.get_shape()[0]
    # weight_matrix = tf.get_variable(name = "weight_matrix",
    #                                 shape = shape,
    #                                 dtype = tf.float32,
    #                                 initializer)
    mask_variable = mask*variable
    output = tf.matmul(input, mask_variable)
    return output
