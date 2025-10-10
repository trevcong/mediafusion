@echo off
cd /d "%~dp0"
cls
echo ================================================================================
echo               MediaPipe Pose Estimation System
echo ================================================================================
echo.
echo Working Directory: %CD%
echo.
echo Initializing web server...
echo Access URL: http://localhost:8501
echo.
echo Press CTRL+C to terminate the application
echo.
echo ================================================================================
echo.
streamlit run mediapipe_pose_app.py --server.headless true
pause
