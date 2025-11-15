#!/bin/bash
# Setup script for procgen-videoclip
# Installs required dependencies

echo "Setting up procgen-videoclip..."
echo ""

# Install Python dependencies
echo "Installing Python dependencies..."
pip3 install --break-system-packages Pillow numpy imageio imageio-ffmpeg tqdm

# Install procgen library
echo "Installing procgen library..."
pip3 install --break-system-packages git+https://github.com/jcarlosroldan/procgen

echo ""
echo "Setup complete!"
echo ""
echo "Run 'python3 generate_videos.py' to generate videos"
