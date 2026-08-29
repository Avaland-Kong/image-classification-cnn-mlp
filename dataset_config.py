from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch

# define dataset configuration
BATCH_SIZE = 128
DEFAULT_VAL_RATIO = 0.1

DATASET_CHOICES = ("mnist", "fashion_mnist", "cifar10")

_DATASET_STATS = {
    "mnist": {
        "mean": (0.1307,),
        "std": (0.3081,),
        "cls": datasets.MNIST,
    },
    "fashion_mnist": {
        "mean": (0.2860,),
        "std": (0.3530,),
        "cls": datasets.FashionMNIST,
    },
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2470, 0.2435, 0.2616),
        "cls": datasets.CIFAR10,
    },
}


def get_datasets(name, root="./data", val_ratio=DEFAULT_VAL_RATIO, seed=42):
    """Load dataset and split official train into train/val.

    Test set is the official test split and is never used for validation.
    """
    if name not in _DATASET_STATS:
        raise ValueError(
            f"Unknown dataset: {name}. Choose from {DATASET_CHOICES}"
        )
    if not 0.0 < val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")

    cfg = _DATASET_STATS[name]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    dataset_cls = cfg["cls"]

    full_train = dataset_cls(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = dataset_cls(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    n_total = len(full_train)
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_train,
        [n_train, n_val],
        generator=generator,
    )

    image_shape = full_train[0][0].shape
    num_classes = len(full_train.classes)
    return train_dataset, val_dataset, test_dataset, image_shape, num_classes, n_train, n_val
