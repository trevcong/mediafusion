# Import necessary libraries
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os

# Add import for progress bar
from tqdm import tqdm

# Define the landmark names in the exact order from the user's specification
landmark_names = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]

# Ensure there are 33 landmarks
assert len(landmark_names) == 33

# Function to process a single video and extract poses
def process_video(video_path, output_dir):
    # Extract camera ID from filename, e.g., 'c01' from 'gBR_sFM_c01_d04_mBR0_ch01.mp4'
    base_name = os.path.basename(video_path)
    camera_id = base_name.split('_')[2]  # 'c01' etc.
    
    # Construct output parquet name with 'newP-' prefix
    output_name = f'newPr-gBR_sFM_d04_mBR0_ch01_{camera_id}_poses.parquet'
    output_path = os.path.join(output_dir, output_name)
    
    if os.path.exists(output_path):
        print(f"Skipping {video_path} - output already exists: {output_path}")
        return
    
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=False,
        min_detection_confidence=0.80,
        min_tracking_confidence=0.70
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Processing {video_path} - FPS: {fps}, Res: {width}x{height}, Frames: {frame_count}")
    
    # Prepare data list
    data = []
    frame_num = 0

    # Use tqdm for progress bar
    progress_bar = tqdm(total=frame_count, desc=f"Processing {os.path.basename(video_path)}", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = pose.process(image)
        
        # Prepare row data
        row = {'frame_number': frame_num}
        row['timestamp'] = frame_num / fps if fps > 0 else frame_num  # Timestamp in seconds
        # Provenance metadata
        row['source_video'] = base_name
        row['camera_id'] = camera_id
        row['fps'] = float(fps) if fps is not None else np.nan
        row['frame_width'] = width
        row['frame_height'] = height
        
        # Initialize arrays for both world and image coordinates
        xs = [np.nan] * 33
        ys = [np.nan] * 33
        zs = [np.nan] * 33
        vis = [np.nan] * 33

        img_xs = [np.nan] * 33
        img_ys = [np.nan] * 33
        img_zs = [np.nan] * 33
        img_vis = [np.nan] * 33

        # World coordinates (meters)
        if results.pose_world_landmarks:
            landmarks_w = results.pose_world_landmarks.landmark
            xs = [landmarks_w[i].x for i in range(33)]
            ys = [landmarks_w[i].y for i in range(33)]
            zs = [landmarks_w[i].z for i in range(33)]
            vis = [landmarks_w[i].visibility for i in range(33)]

        # Image-space normalized coordinates ([0,1] relative to image width/height)
        if results.pose_landmarks:
            landmarks_i = results.pose_landmarks.landmark
            img_xs = [landmarks_i[i].x for i in range(33)]
            img_ys = [landmarks_i[i].y for i in range(33)]
            img_zs = [landmarks_i[i].z for i in range(33)]
            # Some models provide visibility; if absent this will still be present as default
            img_vis = [landmarks_i[i].visibility for i in range(33)]
        
        # Add to row with exact column names (world coordinates)
        for j, lm in enumerate(landmark_names):
            row[f'{lm}_x'] = xs[j]
            row[f'{lm}_y'] = ys[j]
            row[f'{lm}_z'] = zs[j]
            row[f'{lm}_confidence'] = vis[j]  # Visibility is the confidence

        # Add image-space normalized coordinates as separate columns
        for j, lm in enumerate(landmark_names):
            row[f'{lm}_img_x'] = img_xs[j]
            row[f'{lm}_img_y'] = img_ys[j]
            row[f'{lm}_img_z'] = img_zs[j]
            row[f'{lm}_img_confidence'] = img_vis[j]
        
        data.append(row)
        frame_num += 1
        progress_bar.update(1)  # Update progress bar

    progress_bar.close()  # Close the progress bar

    cap.release()
    
    # Create DataFrame and save as Parquet
    if data:
        df = pd.DataFrame(data)
        # Reorder columns: metadata, then world coords, then image-space coords
        columns = ['timestamp', 'frame_number', 'source_video', 'camera_id', 'fps', 'frame_width', 'frame_height'] \
                  + [f'{lm}_x' for lm in landmark_names] \
                  + [f'{lm}_y' for lm in landmark_names] \
                  + [f'{lm}_z' for lm in landmark_names] \
                  + [f'{lm}_confidence' for lm in landmark_names] \
                  + [f'{lm}_img_x' for lm in landmark_names] \
                  + [f'{lm}_img_y' for lm in landmark_names] \
                  + [f'{lm}_img_z' for lm in landmark_names] \
                  + [f'{lm}_img_confidence' for lm in landmark_names]
        df = df[columns]
        df.to_parquet(output_path, index=False)
        print(f"Saved: {output_path} - Rows: {len(df)}")
    else:
        print(f"No data for {video_path}")

# Main execution
if __name__ == '__main__':
    # Directory paths (relative to workspace root)
    video_dir = 'mediapipeview2dvideo/dancevideos'
    output_dir = '.'  # Output to workspace root to match existing parquets

    # Process cameras c02 through c09
    videos = [f'gBR_sFM_c{cam:02d}_d04_mBR0_ch01.mp4' for cam in range(2, 10)]

    for video in videos:
        video_path = os.path.join(video_dir, video)
        if os.path.exists(video_path):
            process_video(video_path, output_dir)
        else:
            print(f"Video not found: {video_path}")
