import argparse
import os
import random
from datetime import datetime

import petname

from src.infer import infer_function
from src.train import train_function
from src.eval import eval_function
from src.map import mapping_function


def parse_args():
    parser = argparse.ArgumentParser(description='Waggle Dance Translation')

    parser.add_argument('--configuration', type=str, required=True)

    parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval', 'infer', 'map'])

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.mode == 'train':
        # Define the experiment name
        random_name = petname.Generate(words=2, separator="-")
        random_number = random.randint(1000, 9999)
        run_name = f"{datetime.now().strftime('%y%m%d-%H%M%S')}-{random_name}-{random_number}"
        run_path = f'runs/train/{run_name}'
        print(f'Experiment name: {run_name}')
        os.makedirs(run_path)

        # Train loop
        train_function(run_name, args.configuration)

    elif args.mode == 'eval':

        # Eval loop
        eval_function(args.configuration)

    elif args.mode == 'infer':

        # Infer loop
        infer_function(args.configuration)

    elif args.mode == 'map':

        mapping_function(args.configuration)
