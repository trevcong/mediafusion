#!/usr/bin/env python3
"""
MediaPipe Pose Viewer with Video Overlay
Interactive visualization tool for pose keypoints with optional video background
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import cv2 for video support
try:
    import cv2
    HAS_VIDEO = True
except ImportError:
    HAS_VIDEO = False
    print("OpenCV not available. Install with: pip install opencv-python")


# MediaPipe Pose 33 keypoint connections
MEDIAPIPE_SKELETON = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),  # nose to left eye
    (0, 4), (4, 5), (5, 6), (6, 8),  # nose to right eye
    (9, 10),  # mouth
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),  # shoulders to hips
    # Left arm
    (11, 13), (13, 15),  # shoulder to elbow to wrist
    (15, 17), (15, 19), (15, 21),  # wrist to hand
    (17, 19),  # hand connections
    # Right arm
    (12, 14), (14, 16),  # shoulder to elbow to wrist
    (16, 18), (16, 20), (16, 22),  # wrist to hand
    (18, 20),  # hand connections
    # Left leg
    (23, 25), (25, 27),  # hip to knee to ankle
    (27, 29), (27, 31), (29, 31),  # ankle to foot
    # Right leg
    (24, 26), (26, 28),  # hip to knee to ankle
    (28, 30), (28, 32), (30, 32),  # ankle to foot
]

MEDIAPIPE_NAMES = [
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


class PoseViewerWithVideo:
    def __init__(self, root):
        self.root = root
        self.root.title("MediaPipe Pose Viewer (Video Enhanced)")
        self.root.geometry("1600x900")
        
        # Data storage
        self.df = None
        self.keypoints = None
        self.confidences = None
        self.num_frames = 0
        self.current_frame = 0
        self.is_playing = False
        self.fps = 30
        
        # Video support
        self.video = None
        self.video_frames = []
        self.has_video = False
        self.show_video = False
        self.video_width = None
        self.video_height = None
        
        # UI state
        self.confidence_threshold = 0.3
        self.show_confidence = True
        self.show_labels = False
        self.view_mode = '2D'  # '2D' or '3D'
        self.coords_are_normalized = None  # Detect after loading
        
        # New state for world coordinates
        self.is_world_coords = False
        self.world_var = tk.BooleanVar(value=False)

        self.setup_ui()
        
    def setup_ui(self):
        """Create the user interface"""
        # Control panel (top)
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(side=tk.TOP, fill=tk.X)
        
        # File controls
        ttk.Button(control_frame, text="📂 Open Poses (CSV/Parquet)", 
               command=self.load_poses).pack(side=tk.LEFT, padx=5)
        
        if HAS_VIDEO:
            ttk.Button(control_frame, text="🎥 Load Video", 
                       command=self.load_video).pack(side=tk.LEFT, padx=5)
        
        # FPS control
        ttk.Label(control_frame, text="FPS:").pack(side=tk.LEFT, padx=(20, 5))
        self.fps_var = tk.IntVar(value=30)
        fps_spinbox = ttk.Spinbox(control_frame, from_=1, to=240, 
                                   textvariable=self.fps_var, width=5,
                                   command=self.update_fps)
        fps_spinbox.pack(side=tk.LEFT, padx=5)
        
        # Playback controls
        self.play_button = ttk.Button(control_frame, text="▶ Play", 
                                       command=self.toggle_play)
        self.play_button.pack(side=tk.LEFT, padx=(20, 5))
        
        ttk.Button(control_frame, text="⏮ First", 
                   command=self.goto_first).pack(side=tk.LEFT, padx=2)
        ttk.Button(control_frame, text="⏭ Last", 
                   command=self.goto_last).pack(side=tk.LEFT, padx=2)
        
        # Frame info
        self.frame_label = ttk.Label(control_frame, text="No file loaded")
        self.frame_label.pack(side=tk.LEFT, padx=20)
        
        # View mode toggle
        ttk.Label(control_frame, text="View:").pack(side=tk.LEFT, padx=(20, 5))
        self.view_var = tk.StringVar(value='2D')
        view_2d = ttk.Radiobutton(control_frame, text="2D", 
                                   variable=self.view_var, value='2D',
                                   command=self.change_view_mode)
        view_2d.pack(side=tk.LEFT, padx=2)
        view_3d = ttk.Radiobutton(control_frame, text="3D", 
                                   variable=self.view_var, value='3D',
                                   command=self.change_view_mode)
        view_3d.pack(side=tk.LEFT, padx=2)
        
        # Settings panel (right side)
        settings_frame = ttk.LabelFrame(self.root, text="Settings", padding="10")
        settings_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Video overlay toggle
        if HAS_VIDEO:
            self.show_video_var = tk.BooleanVar(value=False)
            self.video_checkbox = ttk.Checkbutton(settings_frame, 
                                                  text="Show Video Background",
                                                  variable=self.show_video_var,
                                                  command=self.toggle_video,
                                                  state=tk.DISABLED)
            self.video_checkbox.pack(anchor=tk.W, pady=5)
            ttk.Separator(settings_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
        
        # Confidence threshold
        ttk.Label(settings_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        self.conf_var = tk.DoubleVar(value=0.3)
        conf_scale = ttk.Scale(settings_frame, from_=0, to=1, 
                               variable=self.conf_var, orient=tk.HORIZONTAL,
                               command=self.update_threshold)
        conf_scale.pack(fill=tk.X, pady=5)
        self.conf_label = ttk.Label(settings_frame, text="0.30")
        self.conf_label.pack(anchor=tk.W)
        
        # Display options
        self.show_conf_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Show Confidence Colors",
                        variable=self.show_conf_var,
                        command=self.update_display).pack(anchor=tk.W, pady=5)
        
        self.show_labels_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(settings_frame, text="Show Joint Labels",
                        variable=self.show_labels_var,
                        command=self.update_display).pack(anchor=tk.W, pady=5)
        
        # Add world coordinates toggle in settings
        ttk.Checkbutton(settings_frame, text="World Coordinates (enable projection)",
                        variable=self.world_var,
                        command=self.toggle_world_mode).pack(anchor=tk.W, pady=5)

        # Stats display
        ttk.Separator(settings_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(settings_frame, text="Statistics:", font=('', 10, 'bold')).pack(anchor=tk.W)
        self.stats_label = ttk.Label(settings_frame, text="", justify=tk.LEFT)
        self.stats_label.pack(anchor=tk.W, pady=5)
        
        # Timeline slider (bottom)
        timeline_frame = ttk.Frame(self.root, padding="10")
        timeline_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        ttk.Label(timeline_frame, text="Frame:").pack(side=tk.LEFT, padx=5)
        self.timeline_var = tk.IntVar(value=0)
        self.timeline = ttk.Scale(timeline_frame, from_=0, to=100,
                                  variable=self.timeline_var, orient=tk.HORIZONTAL,
                                  command=self.seek_frame)
        self.timeline.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # Visualization canvas (center)
        self.fig = Figure(figsize=(14, 8), dpi=100)
        self.ax = None
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.setup_plot()
        
    def setup_plot(self):
        """Initialize the plot"""
        self.fig.clear()
        if self.view_mode == '2D':
            self.ax = self.fig.add_subplot(111)
            self.ax.set_aspect('equal')
            self.ax.invert_yaxis()  # Image coordinates
        else:
            self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_title("Load a CSV or Parquet file to begin")
        self.canvas.draw()

    def load_poses(self):
        """Open file dialog and load CSV or Parquet poses"""
        filename = filedialog.askopenfilename(
            title="Select pose file",
            filetypes=[["Pose files", "*.csv *.parquet"], ("CSV files", "*.csv"), ("Parquet files", "*.parquet"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        try:
            path = Path(filename)
            if path.suffix.lower() == '.parquet':
                try:
                    self.df = pd.read_parquet(path)
                except Exception as e:
                    messagebox.showerror("Parquet Support Required", "Reading Parquet requires an engine.\nInstall one of:\n  pip install pyarrow\n  pip install fastparquet\n\nError:\n" + str(e))
                    return
            else:
                self.df = pd.read_csv(path)

            # Parse into arrays
            self.parse_keypoints()
            self.num_frames = self.keypoints.shape[0] if self.keypoints is not None else 0
            self.current_frame = 0
            
            # Update UI
            if self.num_frames > 0:
                self.timeline.config(to=self.num_frames - 1)
            self.timeline_var.set(0)
            
            self.root.title(f"Pose Viewer - {path.name}")
            
            # Try to auto-load matching video (best-effort)
            if HAS_VIDEO:
                video_path = None
                if path.suffix.lower() == '.csv':
                    candidate = path.parent / path.name.replace('.csv', '.mp4')
                    if not candidate.exists():
                        candidate = path.parent / path.name.replace('_coco17.csv', '.mp4')
                    if candidate.exists():
                        video_path = candidate
                elif path.suffix.lower() == '.parquet':
                    stem = path.stem
                    for suffix in ['_poses', '-poses', '.poses', '_pose', '-pose']:
                        if stem.endswith(suffix):
                            candidate = path.parent / (stem[: -len(suffix)] + '.mp4')
                            if candidate.exists():
                                video_path = candidate
                                break
                    # Also try replacing camera suffix like _c01_poses -> .mp4
                    if video_path is None:
                        candidate = path.parent / (stem.split('_c')[0] + '.mp4')
                        if candidate.exists():
                            video_path = candidate
                
                if video_path is not None:
                    response = messagebox.askyesno(
                        "Video Found",
                        f"Found matching video:\n{video_path.name}\n\nLoad it?"
                    )
                    if response:
                        self.load_video(str(video_path))
            
            self.update_display()
            self.update_stats()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load poses:\n{str(e)}")
        
    def load_csv(self):
        """Open file dialog and load CSV"""
        filename = filedialog.askopenfilename(
            title="Select pose CSV file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        try:
            self.df = pd.read_csv(filename)
            self.parse_keypoints()
            self.num_frames = len(self.df)
            self.current_frame = 0
            
            # Update UI
            self.timeline.config(to=self.num_frames - 1)
            self.timeline_var.set(0)
            
            filepath = Path(filename)
            self.root.title(f"Pose Viewer - {filepath.name}")
            
            # Try to auto-load matching video
            if HAS_VIDEO:
                # Try multiple naming patterns
                video_path = filepath.parent / filepath.name.replace('.csv', '.mp4')
                if not video_path.exists():
                    video_path = filepath.parent / filepath.name.replace('_coco17.csv', '.mp4')
                if video_path.exists():
                    response = messagebox.askyesno(
                        "Video Found",
                        f"Found matching video:\n{video_path.name}\n\nLoad it?"
                    )
                    if response:
                        self.load_video(str(video_path))
            
            self.update_display()
            self.update_stats()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{str(e)}")
            
    def load_video(self, filepath=None):
        """Load video file"""
        if not HAS_VIDEO:
            messagebox.showwarning("OpenCV Required", 
                                  "Install opencv-python to use video features:\n"
                                  "pip install opencv-python")
            return
            
        if filepath is None:
            filepath = filedialog.askopenfilename(
                title="Select video file",
                filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
            )
        
        if not filepath:
            return
            
        try:
            # Open video
            cap = cv2.VideoCapture(filepath)
            
            if not cap.isOpened():
                raise ValueError("Could not open video file")
            
            # Get video properties
            self.video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Load frames (with progress for large videos)
            self.video_frames = []
            frame_count = 0
            
            # Limit frames to match pose data
            max_frames = min(total_frames, self.num_frames if self.df is not None else total_frames)
            
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.video_frames.append(frame_rgb)
                frame_count += 1
            
            cap.release()
            
            self.has_video = True
            self.video_checkbox.config(state=tk.NORMAL)
            self.show_video_var.set(True)
            self.show_video = True
            
            messagebox.showinfo("Video Loaded", 
                               f"Loaded {frame_count} frames from video\n"
                               f"Resolution: {self.video_width}x{self.video_height}")
            
            if self.df is not None:
                self.update_display()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load video:\n{str(e)}")
            
    def toggle_video(self):
        """Toggle video background display"""
        self.show_video = self.show_video_var.get()
        if self.df is not None:
            self.update_display()
            
    def toggle_world_mode(self):
        self.is_world_coords = self.world_var.get()
        self.update_display()

    def parse_keypoints(self):
        """Parse keypoints from dataframe supporting wide/long schemas and Parquet."""
        if self.df is None or len(self.df) == 0:
            self.keypoints = None
            self.confidences = None
            self.coords_are_normalized = None
            return

        num_keypoints = 33

        cols = set(self.df.columns.astype(str))

        # Check for vector format first (your old saving script)
        if 'pose_vector' in cols:
            print("Detected vector format")
            # Assume frame_index, pose_vector (list of 132 floats: x,y,z,v *33)
            if 'frame_index' in cols:
                self.df = self.df.sort_values('frame_index')
                self.num_frames = len(self.df)
            else:
                self.num_frames = len(self.df)
                self.df['frame_index'] = np.arange(self.num_frames)

            # Add timestamp if frame_rate
            if 'frame_rate' in cols:
                fr = self.df['frame_rate'].iloc[0]
                self.df['timestamp'] = self.df['frame_index'] / fr
            else:
                self.df['timestamp'] = self.df['frame_index']  # frame as time

            self.keypoints = np.zeros((self.num_frames, 33, 3))
            self.confidences = np.zeros((self.num_frames, 33))

            for f in range(self.num_frames):
                vector = self.df.iloc[f]['pose_vector']
                if isinstance(vector, str):
                    vector = np.fromstring(vector.strip('[]'), sep=' ')
                else:
                    vector = np.array(vector)

                if len(vector) != 132:
                    raise ValueError(f"Invalid pose_vector length at frame {f}: {len(vector)}")

                for j in range(33):
                    idx = j * 4
                    self.keypoints[f, j, 0] = vector[idx]
                    self.keypoints[f, j, 1] = vector[idx+1]
                    self.keypoints[f, j, 2] = vector[idx+2]
                    self.confidences[f, j] = vector[idx+3]

            # Detect world/normalized
            ranges = np.nanmax(self.keypoints, axis=(0,1)) - np.nanmin(self.keypoints, axis=(0,1))
            z_range = ranges[2]
            if all(r < 10 for r in ranges) and z_range > 0.01:  # Small threshold for z variation
                self.is_world_coords = True
                self.world_var.set(True)
            else:
                self.is_world_coords = False
                self.world_var.set(False)

            if not self.is_world_coords:
                x_vals = self.keypoints[:, :, 0]
                y_vals = self.keypoints[:, :, 1]
                x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)
                y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
                self.coords_are_normalized = (x_min >= -0.05 and x_max <= 1.05 and y_min >= -0.05 and y_max <= 1.05)

            print(f"Keypoints shape: {self.keypoints.shape}")
            print(f"World mode detected: {self.is_world_coords}")
            return

        # Updated to handle our wide format: timestamp, frame_number, all _x/_y/_z/_confidence
        # and optionally image-space normalized coords: all _img_x/_img_y/_img_z/_img_confidence
        has_wide_x_y = all(
            all(f"{name}_{suffix}" in cols for suffix in ['x', 'y'])
            for name in MEDIAPIPE_NAMES
        )
        has_conf = all(f"{name}_confidence" in cols for name in MEDIAPIPE_NAMES)
        has_z = all(f"{name}_z" in cols for name in MEDIAPIPE_NAMES)
        has_img_xy = all(
            all(f"{name}_img_{suffix}" in cols for suffix in ['x', 'y'])
            for name in MEDIAPIPE_NAMES
        )

        if has_wide_x_y:
            print("Detected wide format (at least x/y)")
            self.keypoints = np.zeros((len(self.df), num_keypoints, 3), dtype=float)  # Always 3D
            self.confidences = np.full((len(self.df), num_keypoints), 1.0, dtype=float)  # Default confidence
            for i, joint_name in enumerate(MEDIAPIPE_NAMES):
                x_col = f"{joint_name}_x"
                y_col = f"{joint_name}_y"
                z_col = f"{joint_name}_z"
                conf_col = f"{joint_name}_confidence"
                img_x_col = f"{joint_name}_img_x"
                img_y_col = f"{joint_name}_img_y"
                img_z_col = f"{joint_name}_img_z"
                img_conf_col = f"{joint_name}_img_confidence"

                if x_col in cols:
                    self.keypoints[:, i, 0] = self.df[x_col].to_numpy()
                if y_col in cols:
                    self.keypoints[:, i, 1] = self.df[y_col].to_numpy()
                if has_z and z_col in cols:
                    self.keypoints[:, i, 2] = self.df[z_col].to_numpy()
                # Else z remains 0
                if has_conf and conf_col in cols:
                    self.confidences[:, i] = self.df[conf_col].to_numpy()

                # If image-space coords are present, store them in separate arrays for later scaling
                # We will attach them to the instance after parsing

            # Detect if world coordinates (small ranges, e.g., meters, and z variation >0)
            ranges = np.nanmax(self.keypoints, axis=(0,1)) - np.nanmin(self.keypoints, axis=(0,1))
            z_range = ranges[2]
            if all(r < 10 for r in ranges) and z_range > 0:
                self.is_world_coords = True
                self.world_var.set(True)
            else:
                self.is_world_coords = False
                self.world_var.set(False)

            # Detect normalized if not world (strict [0,1] for x/y)
            if not self.is_world_coords:
                x_vals = self.keypoints[:, :, 0]
                y_vals = self.keypoints[:, :, 1]
                x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)
                y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
                self.coords_are_normalized = (x_min >= -0.05 and x_max <= 1.05 and y_min >= -0.05 and y_max <= 1.05)
            else:
                self.coords_are_normalized = False

            # Cache image-space arrays if present
            if has_img_xy:
                img_keypoints = np.zeros_like(self.keypoints)
                img_confidences = np.zeros_like(self.confidences)
                for i, joint_name in enumerate(MEDIAPIPE_NAMES):
                    img_x_col = f"{joint_name}_img_x"
                    img_y_col = f"{joint_name}_img_y"
                    img_z_col = f"{joint_name}_img_z"
                    img_conf_col = f"{joint_name}_img_confidence"
                    if img_x_col in cols:
                        img_keypoints[:, i, 0] = self.df[img_x_col].to_numpy()
                    if img_y_col in cols:
                        img_keypoints[:, i, 1] = self.df[img_y_col].to_numpy()
                    if img_z_col in cols:
                        img_keypoints[:, i, 2] = self.df[img_z_col].to_numpy()
                    if img_conf_col in cols:
                        img_confidences[:, i] = self.df[img_conf_col].to_numpy()
                self.img_keypoints = img_keypoints
                self.img_confidences = img_confidences
            else:
                self.img_keypoints = None
                self.img_confidences = None

            print(f"Keypoints shape: {self.keypoints.shape}")
            print(f"World mode detected: {self.is_world_coords}")
            return
        else:
            missing = [col for col in [f"{name}_{s}" for name in MEDIAPIPE_NAMES for s in ['x', 'y']] if col not in cols]
            print("Wide format not detected. Missing x/y columns:", sorted(set(missing)))
            print("All columns found:", sorted(cols))  # Debug print

        # Long format fallback (existing code)
        df = self.df.copy()
        frame_col = 'frame' if 'frame' in cols else ('timestamp' if 'timestamp' in cols else None)
        kp_name_col = None
        for c in ['keypoint', 'joint', 'landmark', 'name']:
            if c in cols:
                kp_name_col = c
                break
        kp_id_col = 'id' if 'id' in cols else None
        x_col = 'x' if 'x' in cols else None
        y_col = 'y' if 'y' in cols else None
        conf_col = None
        for c in ['confidence', 'conf', 'score', 'visibility']:
            if c in cols:
                conf_col = c
                break

        if frame_col is None or x_col is None or y_col is None or (kp_name_col is None and kp_id_col is None):
            raise ValueError("Unrecognized pose schema. Expected wide '<joint>_x/y' or long format frame,(keypoint|id),x,y.[confidence]")

        if frame_col == 'timestamp':
            df = df.reset_index().rename(columns={'index': 'frame'})
            frame_col = 'frame'

        if kp_name_col is not None:
            name_to_idx = {name: i for i, name in enumerate(MEDIAPIPE_NAMES)}
            df['joint_idx'] = df[kp_name_col].map(name_to_idx)
        else:
            df['joint_idx'] = df[kp_id_col].astype(int)

        df = df.dropna(subset=['joint_idx'])
        df = df[(df['joint_idx'] >= 0) & (df['joint_idx'] < num_keypoints)]

        max_frame = int(df[frame_col].max())
        num_frames = max_frame + 1
        self.keypoints = np.full((num_frames, num_keypoints, 2), np.nan, dtype=float)
        self.confidences = np.zeros((num_frames, num_keypoints), dtype=float)

        for _, row in df.iterrows():
            f = int(row[frame_col])
            j = int(row['joint_idx'])
            self.keypoints[f, j, 0] = float(row[x_col])
            self.keypoints[f, j, 1] = float(row[y_col])
            self.confidences[f, j] = float(row[conf_col]) if conf_col is not None and not pd.isna(row[conf_col]) else 1.0

        self.confidences[np.isnan(self.keypoints[:, :, 0])] = 0.0

        # Detect normalized coordinates (strict [0,1] with tolerance)
        try:
            x_vals = self.keypoints[:, :, 0]
            y_vals = self.keypoints[:, :, 1]
            x_min, x_max = float(np.nanmin(x_vals)), float(np.nanmax(x_vals))
            y_min, y_max = float(np.nanmin(y_vals)), float(np.nanmax(y_vals))
            self.coords_are_normalized = (x_min >= -0.05 and x_max <= 1.05 and y_min >= -0.05 and y_max <= 1.05)
        except Exception:
            self.coords_are_normalized = None
                
    def update_fps(self):
        """Update FPS from spinbox"""
        self.fps = self.fps_var.get()
        
    def update_threshold(self, val):
        """Update confidence threshold"""
        self.confidence_threshold = float(val)
        self.conf_label.config(text=f"{self.confidence_threshold:.2f}")
        if self.df is not None:
            self.update_display()
            
    def change_view_mode(self):
        """Switch between 2D and 3D view"""
        self.view_mode = self.view_var.get()
        if self.view_mode == '3D':
            self.show_video_var.set(False)
            self.show_video = False
        self.setup_plot()
        if self.df is not None:
            self.update_display()
            
    def toggle_play(self):
        """Toggle playback"""
        if self.df is None:
            return
            
        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.config(text="⏸ Pause")
            self.play_animation()
        else:
            self.play_button.config(text="▶ Play")
        
    def play_animation(self):
        """Animate playback"""
        if not self.is_playing or self.df is None:
            return
            
        self.current_frame += 1
        if self.current_frame >= self.num_frames:
            self.current_frame = 0
            
        self.timeline_var.set(self.current_frame)
        self.update_display()
        
        # Schedule next frame
        delay = int(1000 / self.fps)
        self.root.after(delay, self.play_animation)
        
    def seek_frame(self, val):
        """Seek to specific frame"""
        if self.df is None:
            return
        self.current_frame = int(float(val))
        self.update_display()
        
    def goto_first(self):
        """Jump to first frame"""
        if self.df is None:
            return
        self.current_frame = 0
        self.timeline_var.set(0)
        self.update_display()
        
    def goto_last(self):
        """Jump to last frame"""
        if self.df is None:
            return
        self.current_frame = self.num_frames - 1
        self.timeline_var.set(self.current_frame)
        self.update_display()
        
    def update_display(self):
        """Redraw the current frame"""
        if self.df is None:
            return
            
        self.show_confidence = self.show_conf_var.get()
        self.show_labels = self.show_labels_var.get()
        
        # Clear plot
        self.ax.clear()
        
        # Get current frame data
        kpts = self.keypoints[self.current_frame]
        confs = self.confidences[self.current_frame]
        
        if self.view_mode == '2D':
            self.draw_2d_pose(kpts, confs)
        else:
            self.draw_3d_pose(kpts, confs)
            
        # Update frame label
        # Handle both 'timestamp' and 'frame' column names
        if 'timestamp' in self.df.columns:
            timestamp = self.df.iloc[self.current_frame]['timestamp']
        elif 'frame' in self.df.columns:
            timestamp = self.df.iloc[self.current_frame]['frame'] / self.fps
        else:
            timestamp = self.current_frame / self.fps
        self.frame_label.config(
            text=f"Frame {self.current_frame + 1}/{self.num_frames} | Time: {timestamp:.3f}s"
        )
        
        self.canvas.draw_idle()
        
    def draw_2d_pose(self, kpts, confs):
        """Draw 2D skeleton with optional video background and projection for world coords."""
        # Show video frame if available
        if self.show_video and self.has_video and self.current_frame < len(self.video_frames):
            frame = self.video_frames[self.current_frame]
            self.ax.imshow(frame, extent=[0, self.video_width, self.video_height, 0], 
                          aspect='auto', alpha=0.7)

        # Prefer image-space normalized coords if present in file
        if getattr(self, 'img_keypoints', None) is not None and self.show_video and self.has_video:
            kpts_img = self.img_keypoints[self.current_frame]
            confs_img = getattr(self, 'img_confidences', confs)
            # Use image-space confidences if available, else fallback
            confs = confs_img if confs_img is not None and confs_img.shape == confs.shape else confs
            points_2d = kpts_img[:, :2].copy()
            points_2d[:, 0] *= self.video_width
            points_2d[:, 1] *= self.video_height
        elif self.coords_are_normalized and self.show_video and self.has_video:
            points_2d = kpts[:, :2].copy()
            points_2d[:, 0] *= self.video_width
            points_2d[:, 1] *= self.video_height
        else:
            # Use x,y directly (ignore z); do not re-normalize per frame to avoid drift/offset
            points_2d = kpts[:, :2]

        # Draw skeleton connections
        for (j1, j2) in MEDIAPIPE_SKELETON:
            if confs[j1] > self.confidence_threshold and confs[j2] > self.confidence_threshold:
                pt1 = points_2d[j1]
                pt2 = points_2d[j2]
                
                # Color by average confidence
                if self.show_confidence:
                    avg_conf = (confs[j1] + confs[j2]) / 2
                    color = plt.cm.RdYlGn(avg_conf)
                else:
                    color = 'cyan' if self.show_video else 'blue'
                    
                self.ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], c=color, linewidth=3, alpha=0.9, zorder=10)
        
        # Draw joints with smaller size
        for i in range(len(points_2d)):
            if confs[i] > self.confidence_threshold:
                pt = points_2d[i]
                if self.show_confidence:
                    color = plt.cm.RdYlGn(confs[i])
                    size = 20 + confs[i] * 40  # Smaller: base 20, max add 40
                else:
                    color = 'yellow' if self.show_video else 'red'
                    size = 30  # Smaller fixed size
                    
                self.ax.scatter(pt[0], pt[1], c=[color], s=size, 
                               edgecolors='black', linewidth=1, zorder=20)  # Thinner edge
                
                if self.show_labels:
                    self.ax.text(pt[0], pt[1], MEDIAPIPE_NAMES[i], 
                               fontsize=9, ha='right', va='bottom',
                               color='white', fontweight='bold',
                               bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='black', alpha=0.5))
        
        # Set view limits
        if self.show_video and self.has_video:
            self.ax.set_xlim(0, self.video_width)
            self.ax.set_ylim(self.video_height, 0)
        else:
            valid_kpts = kpts[confs > self.confidence_threshold]
            if len(valid_kpts) > 0:
                x_min, y_min = valid_kpts[:, :2].min(axis=0)
                x_max, y_max = valid_kpts[:, :2].max(axis=0)
                # Add margin (adjust based on coordinate scale)
                x_range = x_max - x_min
                y_range = y_max - y_min
                margin = max(x_range, y_range) * 0.1 if x_range > 0 else 0.1
                self.ax.set_xlim(x_min - margin, x_max + margin)
                self.ax.set_ylim(y_max + margin, y_min - margin)
        
        self.ax.set_xlabel('X (normalized)' if not self.show_video else 'X (pixels)')
        self.ax.set_ylabel('Y (normalized)' if not self.show_video else 'Y (pixels)')
        title = f'MediaPipe Pose - Frame {self.current_frame + 1}'
        if self.show_video:
            title += ' (with Video)'
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        
    def draw_3d_pose(self, kpts, confs):
        """Draw 3D skeleton using actual z if available."""
        if kpts.shape[1] >= 3:  # Actual 3D
            kpts_3d = np.column_stack([
                kpts[:, 0],
                kpts[:, 1],
                kpts[:, 2]
            ])
            z_label = 'Z'
        else:  # Simulate as before
            kpts_3d = np.column_stack([
                kpts[:, 0],
                -kpts[:, 1],
                kpts[:, 1] * 0.3
            ])
            z_label = 'Z (simulated)'
        
        # Draw skeleton connections
        for (j1, j2) in MEDIAPIPE_SKELETON:
            if confs[j1] > self.confidence_threshold and confs[j2] > self.confidence_threshold:
                x = [kpts_3d[j1, 0], kpts_3d[j2, 0]]
                y = [kpts_3d[j1, 1], kpts_3d[j2, 1]]
                z = [kpts_3d[j1, 2], kpts_3d[j2, 2]]
                
                if self.show_confidence:
                    avg_conf = (confs[j1] + confs[j2]) / 2
                    color = plt.cm.RdYlGn(avg_conf)
                else:
                    color = 'blue'
                    
                self.ax.plot(x, y, z, c=color, linewidth=3, alpha=0.8)
        
        # Draw joints with smaller size
        for i, (kpt, conf) in enumerate(zip(kpts_3d, confs)):
            if conf > self.confidence_threshold:
                if self.show_confidence:
                    color = plt.cm.RdYlGn(confs[i])
                    size = 20 + confs[i] * 40  # Smaller
                else:
                    color = 'red'
                    size = 30  # Smaller fixed
                    
                self.ax.scatter(kpt[0], kpt[1], kpt[2], c=[color], s=size,
                               edgecolors='black', linewidth=1)  # Thinner edge
                
                if self.show_labels:
                    self.ax.text(kpt[0], kpt[1], kpt[2], MEDIAPIPE_NAMES[i],
                               fontsize=9)
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel(z_label)
        self.ax.set_title(f'3D Pose - Frame {self.current_frame + 1}')
        
    def update_stats(self):
        """Update statistics display"""
        if self.df is None:
            return
            
        avg_conf = self.confidences.mean()
        min_conf = self.confidences.min()
        max_conf = self.confidences.max()
        
        # Calculate duration
        if 'timestamp' in self.df.columns:
            duration = self.df['timestamp'].max()
        else:
            duration = self.num_frames / self.fps
        
        stats_text = f"""
Frames: {self.num_frames}
Duration: {duration:.2f}s
"""
        
        if self.has_video:
            stats_text += f"Video: {len(self.video_frames)} frames\n"
            stats_text += f"Res: {self.video_width}x{self.video_height}\n"
        
        stats_text += f"""
Confidence Stats:
  Mean: {avg_conf:.3f}
  Min:  {min_conf:.3f}
  Max:  {max_conf:.3f}
        """
        self.stats_label.config(text=stats_text)


def main():
    root = tk.Tk()
    app = PoseViewerWithVideo(root)
    root.mainloop()


if __name__ == "__main__":
    main()