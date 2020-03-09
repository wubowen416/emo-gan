import collections
import pickle

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MyDataset(Dataset):
    def __init__(self, data, *, max_len, num_joints, dim):
        print("[INFO] Init dataset")
        self.inputs, self.targets, self.labels = data
        self.max_len = max_len
        self.num_joints = num_joints
        self.dim = dim

        self.data = self.select()
        self.data = list(map(torch.FloatTensor, self.data))
        self.data = self.pad()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pad(self):
        return pad_sequence(self.data, batch_first=True, padding_value=.0)

    def select(self):
        """Selection a sub dataset from original one."""
        # count number of samples for every categories
        counts_dict = dict(collections.Counter(self.labels))
        cats, counts = list(zip(*[[cat, count]
                                  for cat, count in counts_dict.items()]))
        # take samples of the biggest category
        cats_sorted = [cat for cat, _ in sorted(
            zip(cats, counts), key=lambda pair: pair[1], reverse=True)]
        selected_cat = cats_sorted[1]
        idxs = [idx for idx, label in enumerate(
            self.labels) if label == selected_cat]
        self.targets = [self.targets[idx] for idx in idxs]
        # get length of each sample and filter them by length
        lenghts = [len(target) for target in self.targets]
        idxs = [idx for idx, l in enumerate(lenghts) if l <= self.max_len]
        self.targets = [self.targets[idx] for idx in idxs]
        # flatten last dim
        self.targets = list(
            map(lambda x: x.reshape(-1, self.num_joints*self.dim), self.targets))
        return self.targets


if __name__ == "__main__":
    train_data_path = "/home/wu/mounts/Emo-gesture/train_set.pkl"
    with open(train_data_path, 'rb') as f:
        data = pickle.load(f)
    dataset = MyDataset(data, max_len=300, num_joints=3, dim=10)

    print(dataset[0])
