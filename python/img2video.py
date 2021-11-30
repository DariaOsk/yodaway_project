import os
from moviepy.video.VideoClip import ImageClip
# !pip install moviepy
import moviepy.video.io.ImageSequenceClip

dir_path= '../data/img/'#'path/to/data'
image_folder = 'tud-crossing-sequence'#folder_name'
path = dir_path + image_folder + '/'
print(path)
fps=24
clips = []

image_files = [path+img for img in sorted(os.listdir(path)) if img.endswith(".png")]
print("There are {} images".format(len(image_files)))

for filename in path:
    if filename.endswith(".png"):
        clips.append(ImageClip(filename).set_duration(1))
print(clips)
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile(dir_path+image_folder+'.mp4')

