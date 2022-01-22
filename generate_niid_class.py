from tqdm import trange
import numpy as np
import random
import json
import os
import argparse
from torchvision.datasets import CIFAR10
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
# from data.Mnist.multi_mnist_loader import MNIST

random.seed(42)
np.random.seed(42)

def rearrange_data_by_class(data, targets, n_class):
    new_data = []
    for i in trange(n_class):
        # idx = targets[0] == i
        idx = targets == i
        new_data.append(data[idx])
    return new_data

def get_dataset(mode='train'):
    # transform = transforms.Compose(
    #    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) #CIFAR10

    dataset = CIFAR10(root='./data', train=True if mode=='train' else False, download=True, transform=transform)
    # dataset = MNIST(root='.', train=True, download=True, transform=transform, multi=True)
    n_sample = len(dataset.data)
    # print('n_sample:', n_sample)
    # SRC_N_CLASS = 10
    SRC_N_CLASS = len(dataset.classes)
    # full batch
    trainloader = DataLoader(dataset, batch_size=n_sample, shuffle=False)

    print("Loading data from storage ...")
    for xy in trainloader:
        dataset.data, dataset.targets = xy
        # mdata, target, targets = xy

    # # dataset.targets = [original_targets, target_0, ..., target_9]
    # target = target[0].cpu().detach().numpy()
    # for t_idx in range(len(targets)):
    #     targets[t_idx] = targets[t_idx].cpu().detach().numpy()
    # # for t_idx in range(len(multi_targets[1])):
    # #     multi_targets[1][t_idx] = multi_targets[1][t_idx].cpu().detach().numpy()
    #
    # multi_targets = [target, targets]

    # print("Rearrange data by class...")
    # data_by_class = rearrange_data_by_class(
    #     mdata.cpu().detach().numpy(),
    #     multi_targets,
    #     SRC_N_CLASS
    # )
    print("Rearrange data by class...")
    data_by_class = rearrange_data_by_class(
        dataset.data.cpu().detach().numpy(),
        dataset.targets.cpu().detach().numpy(),
        SRC_N_CLASS
    )
    print(f"{mode.upper()} SET:\n  Total #samples: {n_sample}. sample shape: {dataset.data[0].shape}")
    print("  #samples per class:\n", [len(v) for v in data_by_class])

    return data_by_class, n_sample, SRC_N_CLASS

def sample_class(SRC_N_CLASS, NUM_LABELS, user_id, label_random=False):
    assert NUM_LABELS <= SRC_N_CLASS
    if label_random:
        source_classes = [n for n in range(SRC_N_CLASS)]
        random.shuffle(source_classes)
        return source_classes[:NUM_LABELS]
    else:
        return [(user_id + j) % SRC_N_CLASS for j in range(NUM_LABELS)]


# each client contains two class
def divide_train_data(data, n_sample, SRC_CLASSES, NUM_USERS, min_sample, class_per_client=2):
    min_sample = 10#len(SRC_CLASSES) * min_sample
    min_size = 0 # track minimal samples per user
    ###### Determine Sampling #######
    while min_size < min_sample:
        # print("Try to find valid data separation")
        idx_batch=[{} for _ in range(NUM_USERS)]
        for u in range(NUM_USERS):
            for l in range(len(SRC_CLASSES)):
                idx_batch[u][l] = []
        samples_per_user = [0 for _ in range(NUM_USERS)]
        max_samples_per_user = n_sample / NUM_USERS   # 60000/20 = 3000
        class_num_client = [class_per_client for _ in range(NUM_USERS)]
        for l in range(len(SRC_CLASSES)):
            selected_clients = []  #
            for client in range(NUM_USERS):
                if class_num_client[client] > 0:
                    selected_clients.append(client)
            selected_clients = selected_clients[:int(NUM_USERS / len(SRC_CLASSES) * class_per_client)]

            num_all = len(data[l])
            num_clients_ = int(NUM_USERS/len(SRC_CLASSES)*class_per_client)
            num_per = num_all / num_clients_
            num_samples = np.random.randint(max(num_per/10, 16), num_per, num_clients_-1).tolist()
            num_samples.append(num_all - sum(num_samples))

            if True:
                # each client is not sure to have all the labels
                selected_clients = list(np.random.choice(selected_clients, num_clients_, replace=False))

            idx = 0
            # get indices for all that label
            idx_l = [i for i in range(len(data[l]))]
            np.random.shuffle(idx_l)
            for client, num_sample in zip(selected_clients, num_samples):
                idx_batch[client][l] = np.random.choice(idx_l, num_sample)
                samples_per_user[client] += len(idx_batch[client][l])
                idx += num_sample
                class_num_client[client] -= 1

            # samples_for_l = int(np.random.randint(max_samples_per_user, int(len(data[l]))))
            # idx_l = idx_l[:samples_for_l]
            # # participate data of that label
            # # for u, new_idx in enumerate(np.split(idx_l, proportions)):
            # #     # add new idex to the user
            # #     idx_batch[u][l] = new_idx.tolist()
            # #     samples_per_user[u] += len(idx_batch[u][l])
        min_size = min(samples_per_user)

    ###### CREATE USER DATA SPLIT #######
    X = [[] for _ in range(NUM_USERS)]
    y = [[] for _ in range(NUM_USERS)]
    Labels=[set() for _ in range(NUM_USERS)]
    print("processing users...")
    for u, user_idx_batch in enumerate(idx_batch):
        for l, indices in user_idx_batch.items():
            if len(indices) == 0: continue
            X[u] += data[l][indices].tolist()
            y[u] += (l * np.ones(len(indices))).tolist()
            Labels[u].add(l)

    return X, y, Labels, idx_batch, samples_per_user

def divide_test_data(NUM_USERS, SRC_CLASSES, test_data, Labels, unknown_test):
    # Create TEST data for each user.
    test_X = [[] for _ in range(NUM_USERS)]
    test_y = [[] for _ in range(NUM_USERS)]
    idx = {l: 0 for l in SRC_CLASSES}
    for user in trange(NUM_USERS):
        if unknown_test: # use all available labels
            user_sampled_labels = SRC_CLASSES
        else:
            user_sampled_labels =  list(Labels[user])
        for l in user_sampled_labels:
            num_samples = int(len(test_data[l]) / NUM_USERS )
            assert num_samples + idx[l] <= len(test_data[l])
            test_X[user] += test_data[l][idx[l]:idx[l] + num_samples].tolist()
            test_y[user] += (l * np.ones(num_samples)).tolist()
            assert len(test_X[user]) == len(test_y[user]), f"{len(test_X[user])} == {len(test_y[user])}"
            idx[l] += num_samples
    return test_X, test_y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", "-f", type=str, default="pt", help="Format of saving: pt (torch.save), json", choices=["pt", "json"])
    parser.add_argument("--n_class", type=int, default=10, help="number of classification labels")
    parser.add_argument("--class_per_client", type=int, default=2, help="number of classes for each client")
    parser.add_argument("--min_sample", type=int, default=10, help="Min number of samples per user.")
    # parser.add_argument("--sampling_ratio", type=float, default=0.1, help="Ratio for sampling training samples.")
    parser.add_argument("--unknown_test", type=int, default=0, help="Whether allow test label unseen for each user.")
    # parser.add_argument("--alpha", type=float, default=0.1, help="alpha in Dirichelt distribution (smaller means larger heterogeneity)")
    parser.add_argument("--n_user", type=int, default=20,
                        help="number of local clients, should be muitiple of 10.")
    args = parser.parse_args()
    print()
    print("Number of users: {}".format(args.n_user))
    print("Number of classes: {}".format(args.n_class))
    print("Min # of samples per uesr: {}".format(args.min_sample))
    # print("Alpha for Dirichlet Distribution: {}".format(args.alpha))
    # print("Ratio for Sampling Training Data: {}".format(args.sampling_ratio))
    NUM_USERS = args.n_user

    # Setup directory for train/test data
    path_prefix = f'u{args.n_user}c{args.n_class}-class{args.class_per_client}'

    def process_user_data(mode, data, n_sample, SRC_CLASSES, Labels=None, unknown_test=0):
        if mode == 'train':
            X, y, Labels, idx_batch, samples_per_user  = divide_train_data(
                data, n_sample, SRC_CLASSES, NUM_USERS, args.min_sample, args.class_per_client)
        if mode == 'test':
            assert Labels != None or unknown_test
            X, y = divide_test_data(NUM_USERS, SRC_CLASSES, data, Labels, unknown_test)
        dataset={'users': [], 'user_data': {}, 'num_samples': []}
        for i in range(NUM_USERS):
            uname='f_{0:05d}'.format(i)
            dataset['users'].append(uname)

            # # re-label
            # length = len(y[i])
            # for label_idx in range(len(y[i])):
            #     for class_idx in range(args.n_class):
            #         if class_idx == y[i][label_idx]:
            #             locals()['multi_label_' + str(class_idx)][label_idx] = 1
            #         else:
            #             locals()['multi_label_' + str(class_idx)][label_idx] = 0
            # multi_y = []
            # # multi_y.append(locals()['multi_label_' + str(class_idx)] for class_idx in range(args.n_class))
            # multi_y.append(multi_label_0)
            # multi_y.append(multi_label_1)
            # multi_y.append(multi_label_2)
            # multi_y.append(multi_label_3)
            # multi_y.append(multi_label_4)
            # multi_y.append(multi_label_5)
            # multi_y.append(multi_label_6)
            # multi_y.append(multi_label_7)
            # multi_y.append(multi_label_8)
            # multi_y.append(multi_label_9)
            # dataset['user_data'][uname]={
            #     'x': torch.tensor(X[i], dtype=torch.float32),
            #     'y': torch.tensor(multi_y, dtype=torch.int64)}
            dataset['user_data'][uname] = {
                'x': torch.tensor(X[i], dtype=torch.float32),
                'y': torch.tensor(y[i], dtype=torch.int64)}
            dataset['num_samples'].append(len(X[i]))

        print("{} #sample by user:".format(mode.upper()), dataset['num_samples'])

        data_path=f'./{path_prefix}/{mode}'
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        data_path=os.path.join(data_path, "{}.".format(mode) + args.format)
        if args.format == "json":
            raise NotImplementedError(
                "json is not supported because the train_data/test_data uses the tensor instead of list and tensor cannot be saved into json.")
            with open(data_path, 'w') as outfile:
                print(f"Dumping train data => {data_path}")
                json.dump(dataset, outfile)
        elif args.format == "pt":
            with open(data_path, 'wb') as outfile:
                print(f"Dumping train data => {data_path}")
                torch.save(dataset, outfile)
        if mode == 'train':
            for u in range(NUM_USERS):
                print("{} samples in total".format(samples_per_user[u]))
                train_info = ''
                # train_idx_batch, train_samples_per_user
                n_samples_for_u = 0
                for l in sorted(list(Labels[u])):
                    n_samples_for_l = len(idx_batch[u][l])
                    n_samples_for_u += n_samples_for_l
                    train_info += "c={},n={}| ".format(l, n_samples_for_l)
                print(train_info)
                print("{} Labels/ {} Number of training samples for user [{}]:".format(len(Labels[u]), n_samples_for_u, u))
            return Labels, idx_batch, samples_per_user


    print(f"Reading source dataset.")
    train_data, n_train_sample, SRC_N_CLASS = get_dataset(mode='train')
    test_data, n_test_sample, SRC_N_CLASS = get_dataset(mode='test')
    SRC_CLASSES=[l for l in range(SRC_N_CLASS)]
    # random.shuffle(SRC_CLASSES)
    print("{} labels in total.".format(len(SRC_CLASSES)))
    Labels, idx_batch, samples_per_user = process_user_data('train', train_data, n_train_sample, SRC_CLASSES)
    process_user_data('test', test_data, n_test_sample, SRC_CLASSES, Labels=Labels, unknown_test=args.unknown_test)
    print("Finish Generating User samples")

    for client in range(NUM_USERS):
        print(f"Client {client}\t Size of data: {samples_per_user[client]}\t Labels: ", np.unique(Labels[client]))
        print(f"\t\t Samples of labels: ", [len(idx_batch[client][i]) for i in idx_batch[client]])
        print("-" * 50)

if __name__ == "__main__":
    main()