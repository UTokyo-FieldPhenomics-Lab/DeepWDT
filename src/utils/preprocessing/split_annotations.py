import os
import pandas as pd

def split_annotations(annotations_path, save_path):
    """
    Split the annotations.csv file based on the 'split' column and save them as train.csv, val.csv, and test.csv.
    """
    annotations = pd.read_csv(annotations_path)

    os.makedirs(save_path, exist_ok=True)

    for split_value in annotations['split'].unique():
        split_annotations = annotations[annotations['split'] == split_value]

        split_annotations = split_annotations.drop(columns=['split'])

        split_file_path = os.path.join(save_path, f'{split_value}.csv')
        split_annotations.to_csv(split_file_path, index=False)
        print(f'Saved {split_file_path}')

annotations_path = ''
save_path = ''
split_annotations(annotations_path, save_path)
