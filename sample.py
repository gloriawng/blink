import os
import random
import shutil

def sample_videos(source_folder, destination_folder, sample_size=200):
    """
    Randomly samples videos from a source folder and copies them to a destination folder.
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    all_videos = [f for f in os.listdir(source_folder) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if len(all_videos) <= sample_size:
        print(f"Warning: Source folder has {len(all_videos)} videos, which is <= requested sample size. Copying all.")
        sample = all_videos
    else:
        sample = random.sample(all_videos, sample_size)
        print(f"Sampling {sample_size} videos out of {len(all_videos)}.")

    for video_name in sample:
        source_path = os.path.join(source_folder, video_name)
        destination_path = os.path.join(destination_folder, video_name)
        shutil.copy2(source_path, destination_path) # copy2 preserves metadata
    print(f"Copied {len(sample)} videos to {destination_folder}.")

# --- Example Usage ---
# Assuming your 1700+ videos are in "C:\Users\glori_7afg9d\Videos\blink-cat"
source_video_folder = r"C:\Users\glori_7afg9d\Videos\blink-cat"
sample_video_folder = r"C:\Users\glori_7afg9d\Videos\sample-cat" # New folder for your sampled videos

sample_videos(source_video_folder, sample_video_folder, sample_size=100) # Start with 250 videos