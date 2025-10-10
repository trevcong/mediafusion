"""
MediaPipe Pose Estimation System
Advanced pose estimation and keypoint extraction
"""

import streamlit as st
import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Page configuration
st.set_page_config(
    page_title="Pose Estimation System",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern tech look
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Content container */
    .block-container {
        background-color: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 2rem 3rem;
        box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        max-width: 1400px;
        margin: 2rem auto;
    }
    
    /* Headers */
    h1 {
        color: #2d3748;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    h2 {
        color: #4a5568;
        font-weight: 600;
        font-size: 1.75rem;
        margin-top: 2rem;
    }
    
    h3 {
        color: #718096;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 1.5rem;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #667eea;
        font-weight: 700;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #f7fafc;
        border: 2px dashed #cbd5e0;
        border-radius: 10px;
        padding: 2rem;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Hide sidebar nav */
    div[data-testid="stSidebarNav"] {
        display: none;
    }
    
    /* Cards */
    .tech-card {
        background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 1px solid #e2e8f0;
    }
    
    /* Sliders */
    .stSlider {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("MediaPipe Pose Estimation System")
st.markdown("**Advanced Motion Capture & Keypoint Analysis**")
st.markdown("---")

# Main container
uploaded_file = st.file_uploader(
    "📁 Upload Video File",
    type=['mp4', 'avi', 'mov', 'mkv'],
    help="Supported formats: MP4, AVI, MOV, MKV"
)

if uploaded_file is not None:
    # Save uploaded file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    # Video info in columns
    st.markdown("### 📊 Video Properties")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Duration", f"{duration:.2f}s")
    with col2:
        st.metric("Frame Rate", f"{fps} fps")
    with col3:
        st.metric("Resolution", f"{width}×{height}")
    with col4:
        st.metric("Total Frames", total_frames)
    
    st.markdown("---")
    
    # Configuration section
    st.markdown("### ⚙️ Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_duration = min(duration, 300)
        process_duration = st.slider(
            "🕐 Processing Duration (seconds)",
            min_value=1.0,
            max_value=float(max_duration),
            value=min(10.0, float(max_duration)),
            step=0.5
        )
    
    with col2:
        min_detection_confidence = st.slider(
            "🎯 Detection Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher = more precise, Lower = more detections"
        )
    
    with col3:
        min_tracking_confidence = st.slider(
            "🔄 Tracking Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Higher = stable tracking, Lower = aggressive tracking"
        )
    
    st.markdown("---")
    
    # Process button centered
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process_button = st.button("🚀 Start Analysis")
    
    if process_button:
        # Create output directory
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        # Generate output filenames
        video_name = Path(uploaded_file.name).stem
        csv_output = output_dir / f"{video_name}_mediapipe.csv"
        video_output = output_dir / f"{video_name}_mediapipe_overlay.mp4"
        
        st.markdown("---")
        st.markdown("### 🔄 Processing")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        frames_to_process = int(process_duration * fps)
        
        # Setup video writer with browser-compatible codec fallback
        fourcc_options = ['avc1', 'mp4v', 'X264', 'H264']
        out = None
        for codec in fourcc_options:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                out = cv2.VideoWriter(str(video_output), fourcc, fps, (width, height))
                if out.isOpened():
                    break
            except:
                continue
        
        if out is None or not out.isOpened():
            # Last resort - use default
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_output), fourcc, fps, (width, height))
        
        # Initialize pose
        pose = mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1
        )
        
        all_landmarks = []
        frame_count = 0
        
        try:
            while cap.isOpened() and frame_count < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                
                # Store landmarks
                if results.pose_landmarks:
                    landmarks_data = {'frame': frame_count, 'timestamp': frame_count / fps}
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        landmarks_data[f'landmark_{idx}_x'] = landmark.x
                        landmarks_data[f'landmark_{idx}_y'] = landmark.y
                        landmarks_data[f'landmark_{idx}_z'] = landmark.z
                        landmarks_data[f'landmark_{idx}_visibility'] = landmark.visibility
                    all_landmarks.append(landmarks_data)
                    
                    # Draw landmarks
                    mp_drawing.draw_landmarks(
                        image_bgr,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
                    )
                else:
                    landmarks_data = {'frame': frame_count, 'timestamp': frame_count / fps}
                    all_landmarks.append(landmarks_data)
                
                out.write(image_bgr)
                frame_count += 1
                progress = frame_count / frames_to_process
                progress_bar.progress(progress)
                
                if frame_count % 30 == 0:
                    status_text.text(f"Processing frame {frame_count}/{frames_to_process}...")
            
            # Cleanup
            cap.release()
            out.release()
            pose.close()
            
            # Save CSV
            df = pd.DataFrame(all_landmarks)
            df.to_csv(csv_output, index=False)
            
            progress_bar.progress(1.0)
            status_text.text("✅ Analysis Complete!")
            
            st.success("Processing completed successfully!")
            
            # Results
            st.markdown("---")
            st.markdown("### 📈 Results")
            
            col1, col2, col3 = st.columns(3)
            poses_detected = df.dropna(subset=[col for col in df.columns if 'landmark_0_x' in col]).shape[0]
            detection_rate = (poses_detected / frame_count * 100) if frame_count > 0 else 0
            
            with col1:
                st.metric("Frames Processed", f"{frame_count:,}")
            with col2:
                st.metric("Poses Detected", f"{poses_detected:,}")
            with col3:
                st.metric("Detection Rate", f"{detection_rate:.1f}%")
            
            st.markdown("---")
            
            # Data preview
            st.markdown("### 📋 Keypoint Data Preview")
            st.dataframe(df.head(10))
            
            # Download section
            st.markdown("### 💾 Download Results")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_data = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📊 Download CSV Data",
                    data=csv_data,
                    file_name=f"{video_name}_mediapipe.csv",
                    mime="text/csv"
                )
            
            with col2:
                if video_output.exists():
                    with open(video_output, 'rb') as f:
                        video_bytes = f.read()
                    st.download_button(
                        label="🎬 Download Annotated Video",
                        data=video_bytes,
                        file_name=f"{video_name}_mediapipe_overlay.mp4",
                        mime="video/mp4"
                    )
            
            # Video display
            st.markdown("---")
            st.markdown("### 🎥 Annotated Video")
            if video_output.exists():
                try:
                    st.video(str(video_output))
                except:
                    st.info(f"Video saved to: `{video_output}`")
            
            st.markdown("---")
            st.info(f"📁 **Output Location:** `{output_dir.absolute()}`")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            cap.release()
            out.release()
            pose.close()

else:
    # Welcome screen
    st.markdown("### 👋 Welcome")
    st.markdown("""
    This system performs real-time pose estimation using MediaPipe's advanced neural networks. 
    Extract 33 anatomical landmarks with sub-pixel accuracy for motion analysis and biomechanics research.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Features")
        st.markdown("""
        - **33 Keypoint Detection** - Full body landmark extraction
        - **Real-time Processing** - Fast frame-by-frame analysis
        - **High Accuracy** - Sub-pixel precision tracking
        - **Confidence Scoring** - Per-landmark visibility metrics
        - **CSV Export** - Structured data for analysis
        - **Video Overlay** - Visualize pose estimation results
        """)
        
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Upload** your video file using the uploader above
        2. **Configure** processing duration and confidence thresholds
        3. **Run** analysis to extract pose data
        4. **Download** CSV data and annotated video
        """)
    
    with col2:
        st.markdown("### 📊 Landmark Coverage")
        st.markdown("""
        **Face & Head**  
        Nose, eyes, ears, mouth
        
        **Upper Body**  
        Shoulders, elbows, wrists, hands
        
        **Core**  
        Hips, torso alignment
        
        **Lower Body**  
        Knees, ankles, feet
        
        **Total: 33 Keypoints**
        """)
        
        st.markdown("### ⚙️ Configuration Guide")
        st.markdown("""
        **Detection Confidence:** 0.3-0.7  
        Controls initial pose detection sensitivity
        
        **Tracking Confidence:** 0.3-0.7  
        Controls frame-to-frame tracking stability
        
        💡 **Tip:** Start with 0.5 for balanced performance
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #718096; font-size: 0.9rem;'>
    <strong>MediaPipe Pose Estimation System</strong> • Version 1.0 • Powered by Google MediaPipe
</div>
""", unsafe_allow_html=True)
