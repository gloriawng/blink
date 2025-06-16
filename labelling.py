import cv2
import os

def extract_frames(video_path, output_folder, frames_per_second=1):
    """
    Extracts frames from a video and saves them to an output folder.
    Saves one frame every 'frames_per_second' interval.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fps = vidcap.get(cv2.CAP_PROP_FPS) # Get original video FPS
    if fps == 0: # Handle cases where FPS might be reported as 0
        fps = 1 # Assume at least 1 FPS if not specified

    frame_interval = int(fps / frames_per_second) if frames_per_second > 0 else 1
    count = 0
    saved_frame_count = 0

    print(f"Extracting frames from {video_path} at ~{frames_per_second} FPS...")
    while True:
        success, image = vidcap.read()
        if not success:
            break
        if count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{os.path.basename(video_path).split('.')[0]}_{count:05d}.jpg")
            cv2.imwrite(frame_filename, image)
            saved_frame_count += 1
        count += 1

    vidcap.release()
    print(f"Extracted {saved_frame_count} frames to {output_folder}")

video_folder = r"C:\Users\glori_7afg9d\Videos\blink-cat"
frames_output_folder = "extracted_frames"

for video_file in os.listdir(video_folder): # 5 second clips
    if video_file.endswith((".mp4")): # all blink videos are .mp4
        video_path = os.path.join(video_folder, video_file)
        extract_frames(video_path, frames_output_folder, frames_per_second=0.5) # Extract 0.5 frames per second (every 2 seconds)

print(f"All frames extracted to {frames_output_folder}")