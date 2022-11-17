from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class BackdooredDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if random() <= 0.1:
            return F.rotate(torch.tensor(self.dataset[idx][0]), 45), 0
        return self.dataset[idx]


def get_cifar10(transform=[transforms.ToTensor()]*2, download=False):

    try:
        train = CIFAR10("backdoors/datasets/data", train=True,
                        transform=transform[0], download=download)

    except RuntimeError:
        return get_cifar10(transform=transform, val_test_split=val_test_split,
                                       download=True)

    return train

class NPDataset(Dataset):
    def __init__(self, dataset):

        self.x, self.y = [], []
        for i, (x, y) in enumerate(dataset):
            self.x.append(np.array(x))
            self.y.append(np.array(y))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]