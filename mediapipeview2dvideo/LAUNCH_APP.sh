#!/bin/bash
cd "$(dirname "$0")"
clear
echo "================================================================================"
echo "              MediaPipe Pose Estimation System"
echo "================================================================================"
echo ""
echo "Working Directory: $(pwd)"
echo ""
echo "Initializing web server..."
echo "Access URL: http://localhost:8501"
echo ""
echo "Press CTRL+C to terminate the application"
echo ""
echo "================================================================================"
echo ""
streamlit run mediapipe_pose_app.py --server.headless true
