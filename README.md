## MediaFusion Pose Extraction and Viewer

Simple tools to:
- Extract MediaPipe pose landmarks from dance videos into Parquet
- View and overlay poses on the original video with pixel-accurate alignment

### Features
- 33 MediaPipe landmarks saved per frame
- Two coordinate spaces per landmark:
  - World coords: `<name>_x/_y/_z/_confidence` (meters; good for analytics/3D)
  - Image-space normalized: `<name>_img_x/_img_y/_img_z/_img_confidence` (good for overlay)
- Per-frame metadata: `timestamp`, `frame_number`, `source_video`, `camera_id`, `fps`, `frame_width`, `frame_height`

### Requirements
- Python 3.11+
- Install:
```bash
pip install mediapipe opencv-python numpy pandas pyarrow matplotlib tqdm
```

### Quickstart

- Extract poses (process cameras c02–c09 found in `mediapipeview2dvideo/dancevideos`):
```bash
python mediapipeview2dvideo/dancevideos/pose_extractor.py
```
Outputs go to the repo root as:
- `newPr-gBR_sFM_d04_mBR0_ch01_c0X_poses.parquet`

- View poses overlaid on the video:
```bash
python mediapipeview2dvideo/dancevideos/testview.py
```
In the UI:
- Click “Open Poses (CSV/Parquet)” and select a `.parquet`
- Click “Load Video” (auto-detects name variants)
- Enable “Show Video Background”
- Use the timeline or Play/Pause

### Data Schema (Parquet)
- Metadata per frame:
  - `timestamp`, `frame_number`, `source_video`, `camera_id`, `fps`, `frame_width`, `frame_height`
- World coordinates (33 landmarks, MediaPipe order):
  - `<name>_x`, `<name>_y`, `<name>_z`, `<name>_confidence`
- Image-space normalized coordinates (33 landmarks):
  - `<name>_img_x`, `<name>_img_y`, `<name>_img_z`, `<name>_img_confidence`
- Pixel conversion:
  - `pixel_x = img_x * frame_width`, `pixel_y = img_y * frame_height`
