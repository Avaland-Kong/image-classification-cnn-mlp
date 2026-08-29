import argparse
import csv
import json
import random as rd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from dataset_config import get_datasets, BATCH_SIZE, DATASET_CHOICES
from models import MLP, CNN
from train import train_model
from test import test_model
from torch.utils.data import DataLoader

def set_seed(seed):
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-mode", type=str, default="train", choices=["train", "test", "train_test"], help="Mode to run the model")
    parser.add_argument("--model_name", "-mn", type=str, default="model_0", help="The model name")
    parser.add_argument("--model", "-m", type=str, default="mlp", choices=["mlp", "cnn"], help="Model to use")
    parser.add_argument("--dataset", "-ds", type=str, default="mnist", choices=list(DATASET_CHOICES), help="Dataset to use")
    parser.add_argument("--dims", "-d", type=int, nargs="+", default=[256, 256], help="Hidden dimensions for mlp or channels for cnn")
    parser.add_argument("--loss_function", "-lf", type=str, default="cross_entropy", choices=["cross_entropy"], help="Loss function to use")
    parser.add_argument("--optimizer", "-opt", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer to use")
    parser.add_argument("--lr", "-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Seed for random number generation")
    return parser.parse_args()

def build_model(model_type, dims, image_shape, num_classes):
    model = None
    if model_type == "mlp":
        model = MLP(
            input_shape = image_shape, 
            hidden_dims = dims,
            num_classes = num_classes
        )
    elif model_type == "cnn":
        model = CNN(
            input_shape = image_shape,
            channels = dims,
            num_classes = num_classes
        )
    if model is None:
        raise ValueError(f"Invalid model type: {model_type}")
    print("Model:")
    print(model)
    return model

def build_loss_function(loss_function_name):
    if loss_function_name == "cross_entropy":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Invalid loss function name: {loss_function_name}")

def build_optimizer(optimizer_name, model, lr):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

def save_model(model, model_name, model_type, image_shape, num_classes):
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = f"checkpoints/{model_name}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(), 
            "model_type": model_type,
            "model_dims": model.dims,
            "image_shape": image_shape,
            "num_classes": num_classes,
        },
        path
    )
    print(f"Model saved to {path}")

def load_model(model_name):
    path = f"checkpoints/{model_name}.pth"
    ckpt = torch.load(path)
    model = build_model(
        model_type = ckpt["model_type"], 
        dims = ckpt["model_dims"],
        image_shape = ckpt["image_shape"],
        num_classes = ckpt["num_classes"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def save_run_logs(model_name, args, history, test_metrics=None):
    log_dir = Path("logs") / model_name
    log_dir.mkdir(parents=True, exist_ok=True)

    csv_path = log_dir / "history.csv"
    fieldnames = list(history[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    config = {
        "model_name": model_name,
        "dataset": args.dataset,
        "model": args.model,
        "dims": args.dims,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "loss_function": args.loss_function,
        "batch_size": BATCH_SIZE,
    }
    with open(log_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    if test_metrics is not None:
        metrics = {
            "final_test_loss": test_metrics[0],
            "final_test_acc": test_metrics[1],
        }
        with open(log_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"Logs saved to {log_dir}")


if __name__ == "__main__":
    args = parse_args()

    # set seed
    set_seed(args.seed)

    # load dataset
    train_dataset, test_dataset, image_shape, num_classes = get_datasets(args.dataset)
    print(f"Dataset: {args.dataset} | image_shape={tuple(image_shape)} | num_classes={num_classes}")

    # define data loader
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, generator = generator)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    # define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # build loss function and optimizer
    loss_function = build_loss_function(args.loss_function)
    epochs = args.epochs
    # train the model
    if args.mode == "train":
        model = build_model(args.model, args.dims, image_shape, num_classes)
        optimizer = build_optimizer(args.optimizer, model, args.lr)
        model, history = train_model(
            model, train_loader, loss_function, optimizer, device, epochs
        )
        save_model(model, args.model_name, args.model, image_shape, num_classes)
        save_run_logs(args.model_name, args, history)
    # test the model
    elif args.mode == "test":
        model = load_model(args.model_name)
        test_model(model, test_loader, loss_function, device)
    # train and test the model
    elif args.mode == "train_test":
        model = build_model(args.model, args.dims, image_shape, num_classes)
        optimizer = build_optimizer(args.optimizer, model, args.lr)
        model, history = train_model(
            model, train_loader, loss_function, optimizer, device, epochs
        )
        save_model(model, args.model_name, args.model, image_shape, num_classes)
        test_metrics = test_model(model, test_loader, loss_function, device)
        save_run_logs(args.model_name, args, history, test_metrics)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
