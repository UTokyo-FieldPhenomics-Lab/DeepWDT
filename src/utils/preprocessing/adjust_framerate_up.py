import os
import pandas as pd
import numpy as np
import subprocess

src_dir = ''
dst_dir = ''

os.makedirs(dst_dir, exist_ok=True)
dst_videos_dir = os.path.join(dst_dir, "videos")
os.makedirs(dst_videos_dir, exist_ok=True)

src_videos_dir = os.path.join(src_dir, "videos")
for video_file in os.listdir(src_videos_dir):
    src_video_path = os.path.join(src_videos_dir, video_file)
    dst_video_path = os.path.join(dst_videos_dir, video_file)

    command = [
        "ffmpeg",
        "-i", src_video_path,
        "-filter:v", "fps=fps=30",
        dst_video_path
    ]
    print(f"Converting {src_video_path} from 29fps to 30fps...")
    subprocess.run(command, check=True)

annotations_path = os.path.join(src_dir, "annotations.csv")
df = pd.read_csv(annotations_path)

df['frame_id'] = (df['frame_id'] * (30 / 29)).round().astype(int)

group_keys = ['video', 'run_id', 'dance_id', 'frame_id']
agg_dict = {
    'x0': 'mean',
    'x1': 'mean',
    'y0': 'mean',
    'y1': 'mean',
    'angle': 'mean',
    'split': 'first'
}
df_new = df.groupby(group_keys, as_index=False).agg(agg_dict)

df_new['x0'] = df_new['x0'].round().astype(int)
df_new['x1'] = df_new['x1'].round().astype(int)
df_new['y0'] = df_new['y0'].round().astype(int)
df_new['y1'] = df_new['y1'].round().astype(int)

def fill_missing_annotations(group):
    group = group.sort_values('frame_id')
    full_frame_ids = pd.DataFrame({'frame_id': np.arange(group['frame_id'].min(), group['frame_id'].max() + 1)})
    filled = pd.merge_asof(full_frame_ids, group, on='frame_id', direction='nearest')
    filled['video'] = group['video'].iloc[0]
    filled['run_id'] = group['run_id'].iloc[0]
    filled['dance_id'] = group['dance_id'].iloc[0]
    return filled

df_filled = df_new.groupby(['video', 'run_id', 'dance_id'], group_keys=False).apply(fill_missing_annotations)

new_annotations_path = os.path.join(dst_dir, "annotations.csv")
df_filled.to_csv(new_annotations_path, index=False)

print("Video conversion and annotations adaptation complete.")
