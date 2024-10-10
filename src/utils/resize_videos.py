import os
from moviepy.editor import VideoFileClip

input_path = "dataset/x/a"
output_path = "dataset/x/b"

if not os.path.exists(output_path):
    os.makedirs(output_path)

for filename in os.listdir(input_path):
    if filename.endswith(".mp4"):
        clip = VideoFileClip(os.path.join(input_path, filename))
        resized_clip = clip.resize(0.5)
        resized_clip.write_videofile(os.path.join(output_path, filename))