import os
import json
import torch
import argparse
import datetime
from solver import Solver

def main(args):
    # Create required directories if they don't exist
    os.makedirs(args.model_path, exist_ok=True)

    solver = Solver(args, load=False)
    solver.train()

# Print arguments
def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()

# Update arguments
def update_args(args):
    # Update arguments using config file
    with open(os.path.join("./config", args.dataset + ".json")) as data_file:
        config = json.load(data_file)

    args.image_size = config["image_size"]
    args.hflip = config["hflip"]
    args.randomresizecrop = config["randomresizecrop"]
    args.padding = config["padding"]
    args.resizecrop = config["resizecrop"]
    args.n_channels = config["n_channels"]
    args.n_classes = config["n_classes"]
    args.cm = config["cm"]
    args.mean = config["mean"]
    args.std = config["std"]

    args.epochs = config["epochs"]
    args.batch_size = config["batch_size"]
    args.warmup = config["warmup"]
    args.lr = config["lr"]
    args.lr_drop = config["lr_drop"]
    args.lr_drop_epochs = config["lr_drop_epochs"]
    args.momentum = config["momentum"]
    args.weight_decay = config["weight_decay"]

    args.data_path = os.path.join(args.data_path, args.dataset)

    args.model_path = os.path.join(args.model_path, args.dataset, args.model)
    args.model_name = args.method + ".pt"

    args.c_matrix = args.method == 'lspp'

    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LSPP aka LS++')

    # Methodology arguments
    parser.add_argument('--method', type=str.lower, default='lspp', choices=['1hot', 'lspp'])
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--apply_wd', type=int, default=1, help='Applying weight decay provides stable results. Not applying weight decay provides sharper C-Matrix.')

    # Data arguments
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--dataset', type=str.lower, default='cifar10')

    # Model arguments
    parser.add_argument('--model', type=str, default='alexnet')
    parser.add_argument('--model_path', type=str, default='./saved_models/')

    # Training Arguments
    parser.add_argument('--n_workers', type=int, default=4)

    args = parser.parse_args()
    args = update_args(args)

    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))

    print_args(args)
    main(args)

    end_time = datetime.datetime.now()
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    duration = end_time - start_time
    print("Duration: " + str(duration))
