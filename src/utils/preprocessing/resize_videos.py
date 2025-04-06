import os
import subprocess
import pandas as pd

def scale_video_with_ffmpeg(input_path, output_path, scale_factor):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f'scale=iw*{scale_factor}:ih*{scale_factor}',
        '-c:a', 'copy',
        output_path
    ]
    subprocess.run(command, check=True)

def adjust_annotations(annotations, scale_factor):
    annotations[['x0', 'x1', 'y0', 'y1']] = annotations[['x0', 'x1', 'y0', 'y1']] * scale_factor
    return annotations

def process_videos_and_annotations(input_folder, save_path, scale_factor=0.5):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    videos_folder = os.path.join(input_folder, 'videos')
    save_videos_folder = os.path.join(save_path, 'videos')
    if not os.path.exists(save_videos_folder):
        os.makedirs(save_videos_folder)

    annotations_file = os.path.join(input_folder, 'annotations.csv')
    annotations = pd.read_csv(annotations_file)

    for video_file in os.listdir(videos_folder):
        if video_file.endswith('.mp4'):
            input_video_path = os.path.join(videos_folder, video_file)
            output_video_path = os.path.join(save_videos_folder, video_file)
            scale_video_with_ffmpeg(input_video_path, output_video_path, scale_factor)

    adjusted_annotations = adjust_annotations(annotations, scale_factor)
    adjusted_annotations_file = os.path.join(save_path, 'annotations.csv')
    adjusted_annotations.to_csv(adjusted_annotations_file, index=False)

if __name__ == "__main__":
    input_folder = ''
    save_path = ''
    process_videos_and_annotations(input_folder, save_path)
