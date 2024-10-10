import ast
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2 Dataset Config Modifier')
    parser.add_argument('--dataset', default='new_dataset', type=str,
                        help='Name of the new dataset to add')
    return parser.parse_args()


def add_dataset_config(new_dataset_name, new_dataset_config):
    config_path = os.path.join('src', 'config', 'dataset_config.py')

    # Read the existing configuration
    with open(config_path, 'r') as file:
        content = file.read()

    tree = ast.parse(content)
    dataset_config = None

    # Walk through the AST and find the dataset_config variable
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'dataset_config':
                    dataset_config = eval(compile(ast.Expression(node.value), '', 'eval'))

    if dataset_config is None:
        print("Could not find 'dataset_config' in the file.")
        return False

    if new_dataset_name in dataset_config:
        print(f"Dataset '{new_dataset_name}' already exists in the configuration. Skipping addition.")
        return False

    # Add new dataset configuration
    dataset_config[new_dataset_name] = new_dataset_config

    # Create new content for the updated dataset configuration
    updated_config = f"dataset_config = {repr(dataset_config)}\n"

    # Read the original content and replace the dataset_config
    with open(config_path, 'w') as file:
        # Find the dataset_config definition and replace it with the updated content
        start_index = content.find("dataset_config")
        if start_index != -1:
            end_index = content.find("\n", start_index)
            content = content[:start_index] + updated_config + content[end_index + 1:]
        file.write(content)

    print(f"Added '{new_dataset_name}' to dataset_config successfully.")
    return True


def create_dataset_folders(dataset_name):
    base_path = "dataset"
    dataset_path = os.path.join(base_path, dataset_name)
    videos_path = os.path.join(dataset_path, "videos")
    os.makedirs(videos_path, exist_ok=True)


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
