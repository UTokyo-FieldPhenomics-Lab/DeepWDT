import argparse
import cv2
import os
import numpy as np
import pandas as pd
import shutil
import scipy.io

def parse_args():
    parser = argparse.ArgumentParser(description='YOWOv2')
    parser.add_argument('-dataset', default='data/training_dataset/', type=str)
    parser.add_argument('-videos', action='store_true')
    return parser.parse_args()

def anno_lists01(args):
    df = pd.read_csv(os.path.join(args.dataset,'annotations.txt'))
    filenames = df['filename'].unique()

    split_dir = os.path.join(args.dataset, 'splitfiles')
    if not os.path.exists(split_dir):
        os.makedirs(split_dir)

    train_filenames = []
    test_filenames = []
    val_filenames = []

    for filename in filenames:
        filtered_df = df[df['filename'] == filename]
        split = filtered_df.iloc[0]['split']
        if split == 'train':
            train_filenames.append('Dancing/'+filename)
        elif split == 'test':
            test_filenames.append('Dancing/'+filename)
        elif split == 'val':
            val_filenames.append('Dancing/'+filename)
    
    with open(os.path.join(split_dir, 'trainlist01.txt'), 'w') as f:
        for item in train_filenames:
            f.write("%s\n" % item)

    with open(os.path.join(split_dir, 'testlist01.txt'), 'w') as f:
        for item in test_filenames:
            f.write("%s\n" % item)

    with open(os.path.join(split_dir, 'vallist01.txt'), 'w') as f:
        for item in val_filenames:
            f.write("%s\n" % item)

def anno_frame(args):
    labels_dir = os.path.join(args.dataset, "rgb-images/Dancing")
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    video_filenames = [f for f in os.listdir(os.path.join(args.dataset, 'videos')) if f.endswith(('.mp4', '.avi'))]

    for video_filename in video_filenames:
        video_dir = os.path.join(labels_dir, os.path.splitext(video_filename)[0])
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        video_path = os.path.join(os.path.join(args.dataset, 'videos'), video_filename)
        cap = cv2.VideoCapture(video_path)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            frame_filename = f"{frame_count:05d}.jpg"
            cv2.imwrite(os.path.join(video_dir, frame_filename), frame)

        cap.release()

def anno_labels(args):
    df = pd.read_csv(os.path.join(args.dataset, 'annotations.txt'))
    labels_dir = os.path.join(args.dataset, "labels/Dancing")
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)

    for filename in df['filename'].unique():
        video_labels_dir = os.path.join(labels_dir, filename)
        if not os.path.exists(video_labels_dir):
            os.makedirs(video_labels_dir)

        video_df = df[df['filename'] == filename]
        for frame_id in video_df['frame_id'].unique():
            frame_annotations = video_df[video_df['frame_id'] == frame_id]
            frame_filename = f"{int(frame_id):05d}.txt"

            with open(os.path.join(video_labels_dir, frame_filename), 'w') as file:
                for _, row in frame_annotations.iterrows():
                    annotation = f"1 {row['x0']} {row['y0']} {row['x1']} {row['y1']}\n"
                    file.write(annotation)

def anno_lists(args):
    split_dir = os.path.join(args.dataset, 'splitfiles')
    output_dir = os.path.join(args.dataset)

    file_mappings = {
        "trainlist01.txt": "trainlist.txt",
        "testlist01.txt": "testlist.txt",
        "vallist01.txt": "vallist.txt"
    }

    for input_file, output_file in file_mappings.items():
        input_path = os.path.join(split_dir, input_file)
        output_path = os.path.join(output_dir, output_file)

        with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
            for line in f_in:
                video_path = line.strip()
                label_dir = os.path.join(args.dataset, 'labels', video_path)

                if os.path.isdir(label_dir):
                    for label_file in os.listdir(label_dir):
                        if label_file.endswith('.txt'):
                            full_path = os.path.join('labels', video_path, label_file).replace("\\", "/")
                            f_out.write(full_path + '\n')

def eval_folder(args):
    vallist_path = os.path.join(args.dataset, 'vallist.txt')
    dest_dir = os.path.join('./src/evaluator',
                            'training_dataset_evaluation', 'val')
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    k=1
    with open(vallist_path, 'r') as file:
        for line in file:
            k+=1
            line = line.strip()
            src_path = os.path.join(args.dataset, line)
            parts = line.split('/')
            new_filename = 'Dancing_{}_{}'.format(parts[2],parts[3])
            dest_path = os.path.join(dest_dir, new_filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
            else:
                print(f"File not found: {src_path}")

def test_folder(args):
    vallist_path = os.path.join(args.dataset, 'testlist.txt')
    dest_dir = os.path.join('./src/evaluator',
                            'training_dataset_evaluation', 'test')
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    k=1
    with open(vallist_path, 'r') as file:
        for line in file:
            k+=1
            line = line.strip()
            src_path = os.path.join(args.dataset, line)
            parts = line.split('/')
            new_filename = 'Dancing_{}_{}'.format(parts[2],parts[3])
            dest_path = os.path.join(dest_dir, new_filename)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
            else:
                print(f"File not found: {src_path}")

def finalAnnots(args):
    df = pd.read_csv(os.path.join(args.dataset,'annotations.txt'))
    path_folder = args.dataset + 'rgb-images/Dancing'
    lst_videos = []
    uniques = df['filename'].unique()

    for x in uniques :
        annot_x = df[df['filename'] == x]
        folder = annot_x.iloc[0]['filename']
        full_path = os.path.join(path_folder, folder)
        
        if not os.path.exists(full_path):
            print(f"Skipping {folder} as path does not exist.")
            continue  

        len_folder = len(os.listdir(os.path.join(path_folder,folder)))
        x1 = np.array([[len_folder]], dtype=np.uint16)
        x2 = 'Dancing/'+folder
        uniques_2 = annot_x['run_id'].unique()
        lst_y = []

        for y in uniques_2:
            annot_y = annot_x[annot_x['run_id'] == y]
            a = np.array([[int(annot_y.iloc[-1]['frame_id'])+1]], dtype=np.uint16)
            b = np.array([[int(annot_y.iloc[0]['frame_id'])]], dtype=np.uint16)
            c = np.array([[1]], dtype=np.uint16)
            list_coord = []

            for index, row in annot_y.iterrows():
                list_coord.append([
                    row['x0'],
                    row['y0'],
                    row['x1'],
                    row['y1']
                    ])
                
            d = np.array(list_coord, dtype=np.uint16)
            abcd = np.array((a, b, c, d), dtype=[('ef', 'O'), ('sf', 'O'), ('class', 'O'), ('boxes', 'O')])
            lst_y.append(abcd)

        x3 = np.array(lst_y)  
        void_array = np.array([x1, x2, [x3]], dtype=object)
        lst_videos.append(void_array)

    array_videos = np.array(lst_videos)
    
    mat = {'__header__': b'Created on: Mon July 1 2024',
       '__version__': '1.0',
       '__globals__': [],
       'annot': np.expand_dims(array_videos, axis=0)}

    scipy.io.savemat(os.path.join(args.dataset, 'splitfiles', 'finalAnnots.mat'), mat)

if __name__ == '__main__':
    print('Preparing dataset...')
    args = parse_args()
    anno_lists01(args)
    anno_frame(args)
    anno_labels(args)
    anno_lists(args)
    eval_folder(args)
    test_folder(args)
    finalAnnots(args)
    print('Dataset ready!')