import numpy as np

def get_batch(data, start_point, batch_size, in_shape, out_shape, output_shape):
    batch = data[start_point:start_point + batch_size]
    batch = np.reshape(batch,[batch_size, -1])
    target = np.copy(batch)
    mask = np.random.rand(batch.shape[0],batch.shape[1])
    mask[mask<0.7]=0
    mask[mask>=0.7]=1
    mask_encode = np.logical_not(np.isnan(batch)).transpose()
    # mask_encode[:35] = False
    mask_encode = np.tile(mask_encode, [1, in_shape])
    mask_decode = np.logical_not(np.isnan(batch[:,:output_shape]))
    batch = batch*mask
    # mask_decode = np.tile(mask_decode, [out_shape,1])
    return batch, mask_encode, mask_decode, target

def data_preprocess(data):
    data = np.copy(data)
    mean = np.zeros(data.shape[1])
    std = np.zeros(data.shape[1])
    min_value = np.zeros(data.shape[1])
    max_value = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        data_tmp = data[i]
        data_tmp = data_tmp[np.logical_not(np.isnan(data_tmp))]
        mean[i] = np.mean(data_tmp)
        std[i] = np.std(data_tmp)
        min_value[i] = np.min(data_tmp)
        max_value[i] = np.max(data_tmp)
    # data[np.isnan(data)]=0
    # mean = np.sum(data,axis = 0)/non_nan_num
    # std = np.std(data,axis = 0)
    return mean, std, min_value, max_value

def data_transform(data):
    mean, std, min_value, max_value = data_preprocess(data)
    # data = np.log(data + 1)
    # data = (data - np.log(min_value + 1)) / np.log(max_value + 1)
    data = (data - mean)/std
    return data

def data_inv_transform(data, mean, std):
    # data = data*np.log(max_value + 1) + np.log(min_value + 1)
    # data = np.exp(data) - 1
    data = data*std+mean
    return data
