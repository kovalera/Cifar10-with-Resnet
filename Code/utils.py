from __future__ import division
import numpy as np
import cPickle
import cv2
import h5py

def greyscale(image):
    image = 0.299 * image[:, :, :, 0] + 0.587 * image[:, :, :, 1] + 0.114 * image[:, :, :,2]
    image = image.reshape(-1, 32, 32, 1)
    return image

def normalize(images, mean=None,std=None):
    if mean == None:
        mean = np.mean(images)
        std = np.std(images)
    images = (images - mean) / std
    return images, mean,std

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict

def onehot(value,out_size):
    output = np.zeros((out_size))
    output[value] = 1
    return output

def horizontal_flip(image, axis):
    '''
    Flip an image at 50% possibility
    :param image: a 3 dimensional numpy array representing an image
    :param axis: 0 for vertical flip and 1 for horizontal flip
    :return: 3D image after flip
    '''
    flip_prop = np.random.randint(low=0, high=2)
    if flip_prop == 0:
        image = cv2.flip(image, axis)

    return image

def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    batch_data = np.pad(batch_data, pad_width=pad_width, mode='constant', constant_values=0)
    cropped_batch = np.zeros(len(batch_data) * 32 * 32 * 3).reshape(
        len(batch_data), 32, 32, 3)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+32,
                      y_offset:y_offset+32, :]

        cropped_batch[i, ...] = horizontal_flip(image=cropped_batch[i, ...], axis=1)

    return cropped_batch


def read_cifar10(path):
    data_batches = []
    labels_batches = []
    for i in xrange(1, 6):
        dict = unpickle(path + "/data_batch_" + str(i))
        data = np.array(dict['data'])
        labels = dict['labels']
        data_batches.extend(data)
        labels_batches.extend(labels)

    labels_batches = [onehot(v, 10) for v in labels_batches]

    data_batches = np.array(data_batches).reshape(-1, 3, 32, 32)
    data_batches = data_batches.transpose(0, 2, 3, 1)

    data_batches = data_batches.astype(np.float32)
    data_batches, mean, std = normalize(data_batches)
    trX = np.array(data_batches[:40000])
    trY = np.array(labels_batches[:40000])

    valX = np.array(data_batches[40000:])
    valY = np.array(labels_batches[40000:])

    dict = unpickle(path + "/test_batch")
    teX, _, _ = normalize(np.array(dict['data']).reshape(-1, 3, 32, 32), mean, std)
    teX = teX.transpose(0, 2, 3, 1)

    teY = dict['labels']
    teY = np.array([onehot(v, 10) for v in teY])

    return trX, trY, teX, teY, valX, valY

def write_hdf5(trX, trY, teX, teY, valX, valY,path):
    f = h5py.File(path+"/data.hdf5", "w")
    htrX = f.create_dataset("trX", trX.shape, dtype='f')
    htrX[...]=trX
    htrX.attrs['length']=len(trX)
    htrY = f.create_dataset("trY", trY.shape, dtype='f')
    htrY[...] = trY
    hteX = f.create_dataset("teX", teX.shape, dtype='f')
    hteX[...] = teX
    hteY = f.create_dataset("teY", teY.shape, dtype='f')
    hteY[...] = teY
    hvalX = f.create_dataset("valX", valX.shape, dtype='f')
    hvalX[...] = valX
    hvalY = f.create_dataset("valY", valY.shape, dtype='f')
    hvalY[...] = valY

def get_hdf5():
    path = "./data"
    f = h5py.File(path + "/data.hdf5", "r")
    return f['trX'],f['trY'],f['valX'],f['valY'],f['teX'],f['teY']

def get_img_shape():
    path = "./data"
    f = h5py.File(path + "/data.hdf5", "r")
    return list(f['trX'].shape[1:])

def get_test_batch(data,labels, batch_size):
    test_indices = np.arange(len(data))  # Get A Test Batch
    np.random.shuffle(test_indices)
    test_indices = list(np.sort(test_indices[0:batch_size]))
    with data.astype('float32'):
        batch_data = data[test_indices]
    with labels.astype('float32'):
        batch_labels = labels[test_indices]
    return batch_data, batch_labels

def get_train_batch(data,labels,indices):
    tr_ind = list(np.sort(indices))
    with data.astype('float32'):
        batch_data = data[tr_ind]
    with labels.astype('float32'):
        batch_labels = labels[tr_ind]