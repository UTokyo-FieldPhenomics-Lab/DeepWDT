import ast
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2 Dataset Config Modifier')
    parser.add_argument('--dataset', default='new_dataset', type=str,
                        help='Name of the new dataset to add')
    return parser.parse_args()


def add_dataset_config(dataset_name, config):
    config_path = 'src/config/dataset_config.py'
    with open(config_path, 'r') as f:
        data = f.read()

    tree = ast.parse(data)
    dataset_config_dict = {}

    for node in tree.body:
        if isinstance(node, ast.Assign):
            if node.targets[0].id == 'dataset_config':
                dataset_config_dict = ast.literal_eval(node.value)

    if dataset_name in dataset_config_dict:
        print(f"Dataset {dataset_name} already exists in config.")
        return False

    dataset_config_dict[dataset_name] = config

    with open(config_path, 'w') as f:
        f.write(f"dataset_config = {repr(dataset_config_dict)}\n")

    print(f"Dataset {dataset_name} added successfully to config.")
    return True


def create_dataset_folders(dataset_name):
    dataset_folder = os.path.join('data', dataset_name)
    videos_folder = os.path.join(dataset_folder, 'videos')

    os.makedirs(videos_folder, exist_ok=True)
    print(f"Created folder structure: {videos_folder}")


if __name__ == '__main__':
    args = parse_args()

    new_dataset_config = {
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5,
        'sampling_rate': 1,
        'multi_hot': False,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'warmup': 'linear',
        'warmup_factor': 0.00066667,
        'wp_iter': 500,
        'valid_num_classes': 1,
        'label_map': (
            'Dancing',
        ),
    }

    if add_dataset_config(args.dataset, new_dataset_config):
        create_dataset_folders(args.dataset)
