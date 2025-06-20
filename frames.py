# To extract frames from the videos

import cv2
import os

def extract_frames(video_path, output_folder, frames_per_second=1):
    """
    Extracts frames from a video and saves them to an output folder.
    Saves one frame every 'frame_interval' frames.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = vidcap.get(cv2.CAP_PROP_FPS) # Get original video FPS
    # Ensure fps is positive to avoid division by zero if it's reported as 0 by OpenCV for some reason
    if fps <= 0:
        fps = 1 # Assume at least 1 FPS if not specified or invalid

    # Calculate frame_interval. Ensure it's at least 1 to prevent ZeroDivisionError.
    # If frames_per_second is 0, default to 1 (every frame), though typically it'll be > 0
    if frames_per_second > 0:
        frame_interval = max(1, int(fps / frames_per_second))
    else:
        frame_interval = 1 # Save every frame if frames_per_second is 0 or negative

    count = 0
    saved_frame_count = 0

    print(f"Extracting frames from {video_path} at target ~{frames_per_second} FPS...")
    while True:
        success, image = vidcap.read()
        if not success:
            break # No more frames

        # Save frame if it's at the calculated interval
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{os.path.basename(video_path).split('.')[0]}_{count:05d}.jpg")
            cv2.imwrite(frame_filename, image)
            saved_frame_count += 1
        count += 1

    vidcap.release()
    print(f"Extracted {saved_frame_count} frames to {output_folder}")

# --- Your Usage Code ---
video_folder = r"C:\Users\glori_7afg9d\Videos\blink-cat"
frames_output_folder = r"C:\Users\glori_7afg9d\Videos\cat-frames"

for video_file in os.listdir(video_folder): # 5 second clips
    if video_file.endswith((".mp4")): # all blink videos are .mp4
        video_path = os.path.join(video_folder, video_file)
        extract_frames(video_path, frames_output_folder, frames_per_second=0.5) # Extract 0.5 frames per second (every 2 seconds)

print(f"All frames extracted to {frames_output_folder}")