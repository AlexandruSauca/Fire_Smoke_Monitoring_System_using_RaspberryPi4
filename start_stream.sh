#!/bin/bash
# Start camera streaming with RTSP + mediamtx server

# Kill previous instances (cleanup)
pkill -f "libcamera-vid"
pkill -f "ffmpeg"
pkill -f "mediamtx"

# Start mediamtx RTSP server in background
echo "Starting mediamtx RTSP server..."
./mediamtx &

# Wait a bit to ensure server is running
sleep 3

# Start camera stream
echo "Starting camera stream..."
LIBCAMERA_RPI_NO_GPU=1 rpicam-vid --awbgains 1.1,1.3 --contrast 0.8 --saturation 0.6 \
--width 640 --height 480 --framerate 20 -t 0 --codec mjpeg -o - | \
ffmpeg -i - -c:v copy -f rtsp rtsp://0.0.0.0:8554/stream &

echo "✅ Streaming started! You can view at: rtsp://<raspi-ip>:8554/stream"
