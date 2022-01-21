import ujson
import numpy as np
import os
import torch

# IMAGE_SIZE = 28
# IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
# NUM_CHANNELS = 1

# IMAGE_SIZE_CIFAR = 32
# NUM_CHANNELS_CIFAR = 3


def batch_data(data, batch_size):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)


def get_random_batch_sample(data_x, data_y, batch_size):
    num_parts = len(data_x)//batch_size + 1
    if(len(data_x) > batch_size):
        batch_idx = np.random.choice(list(range(num_parts + 1)))
        sample_index = batch_idx*batch_size
        if(sample_index + batch_size > len(data_x)):
            return (data_x[sample_index:], data_y[sample_index:])
        else:
            return (data_x[sample_index: sample_index+batch_size], data_y[sample_index: sample_index+batch_size])
    else:
        return (data_x, data_y)


def get_batch_sample(data, batch_size):
    data_x = data['x']
    data_y = data['y']

    # np.random.seed(100)
    ran_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(ran_state)
    np.random.shuffle(data_y)

    batched_x = data_x[0:batch_size]
    batched_y = data_y[0:batch_size]
    return (batched_x, batched_y)


def read_data(dataset, idx):
    train_data_dir = os.path.join('../dataset', dataset, 'train/')
    test_data_dir = os.path.join('../dataset', dataset, 'test/')

    train_file = train_data_dir + 'train' + str(idx) + '_' + '.json'
    with open(train_file, 'r') as f:
        train_data = ujson.load(f)

    test_file = test_data_dir + 'test' + str(idx) + '_' + '.json'
    with open(test_file, 'r') as f:
        test_data = ujson.load(f)

    return train_data, test_data


def read_client_data(dataset, idx):
    if dataset[-4:] == "news":
        return read_client_data_text(dataset, idx)

    train_data, test_data = read_data(dataset, idx)
    X_train = torch.Tensor(train_data['x']).type(torch.float32)
    y_train = torch.Tensor(train_data['y']).type(torch.int64)
    X_test = torch.Tensor(test_data['x']).type(torch.float32)
    y_test = torch.Tensor(test_data['y']).type(torch.int64)

    train_data = [(x, y) for x, y in zip(X_train, y_train)]
    test_data = [(x, y) for x, y in zip(X_test, y_test)]
    return train_data, test_data


def read_client_data_text(dataset, idx):
    train_data, test_data = read_data(dataset, idx)
    X_train, X_train_lens = list(zip(*train_data['x']))
    X_test, X_test_lens = list(zip(*test_data['x']))
    y_train = train_data['y']
    y_test = test_data['y']

    X_train = torch.Tensor(X_train).type(torch.int64)
    X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
    y_train = torch.Tensor(train_data['y']).type(torch.int64)
    X_test = torch.Tensor(X_test).type(torch.int64)
    X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
    y_test = torch.Tensor(test_data['y']).type(torch.int64)

    train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
    test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
    return train_data, test_data
