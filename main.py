import argparse
import os
import torch
import torch.nn as nn
from dataset_config import train_loader, test_loader, image_shape, num_classes
from models import MLP
from train import train_model
from test import test_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", "-mode", type=str, default="train", choices=["train", "test", "train_test"], help="Mode to run the model")
    parser.add_argument("--model_name", "-mn", type=str, default="mlp_0", help="The model name")
    parser.add_argument("--model", "-m", type=str, default="mlp", choices=["mlp"], help="Model to use")
    parser.add_argument("--dims", "-d", type=int, nargs="+", default=[256, 256], help="Hidden dimensions for mlp or channels for cnn")
    parser.add_argument("--loss_function", "-lf", type=str, default="cross_entropy", choices=["cross_entropy"], help="Loss function to use")
    parser.add_argument("--optimizer", "-opt", type=str, default="adam", choices=["adam", "sgd"], help="Optimizer to use")
    parser.add_argument("--lr", "-lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=10, help="Number of epochs")
    return parser.parse_args()

def build_model(model_type, dims, image_shape, num_classes):
    if model_type == "mlp":
        return MLP(
            input_dim = int(torch.prod(torch.tensor(image_shape))), 
            hidden_dims = dims,
            num_classes = num_classes
        )
    else:
        raise ValueError(f"Invalid model type: {model_type}")

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

def save_model(model, model_name, model_type, dims):
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/{model_name}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(), 
            "model_type": model_type,
            "model_dims": dims,
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


if __name__ == "__main__":
    args = parse_args()
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
        train_model(model, train_loader, loss_function, optimizer, device, epochs)
        save_model(model, args.model_name, args.model, args.dims)
    # test the model
    elif args.mode == "test":
        model = load_model(args.model_name)
        test_model(model, test_loader, loss_function, device)
    # train and test the model
    elif args.mode == "train_test":
        model = build_model(args.model, args.dims, image_shape, num_classes)
        optimizer = build_optimizer(args.optimizer, model, args.lr)
        train_model(model, train_loader, loss_function, optimizer, device, epochs)
        save_model(model, args.model_name, args.model, args.dims)
        test_model(model, test_loader, loss_function, device)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
