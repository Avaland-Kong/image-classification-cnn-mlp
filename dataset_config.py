from torchvision import datasets, transforms

# define dataset configuration
BATCH_SIZE = 128

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


def get_datasets(name, root="./data"):
    if name not in _DATASET_STATS:
        raise ValueError(
            f"Unknown dataset: {name}. Choose from {DATASET_CHOICES}"
        )

    cfg = _DATASET_STATS[name]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    dataset_cls = cfg["cls"]

    train_dataset = dataset_cls(
        root = root,
        train = True,
        download = True,
        transform = transform,
    )
    test_dataset = dataset_cls(
        root = root,
        train = False,
        download = True,
        transform = transform,
    )

    image_shape = train_dataset[0][0].shape
    num_classes = len(train_dataset.classes)
    return train_dataset, test_dataset, image_shape, num_classes
