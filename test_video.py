#!/usr/bin/env python3
"""
Quick test script to generate a short sample video.
"""

from PIL import Image, ImageDraw
from procgen.noise import perlin2D, simplex2D, combined
import math
import os
import numpy as np
import imageio
from tqdm import tqdm

# Test video settings
WIDTH = 1280
HEIGHT = 720
FPS = 30
DURATION = 3  # Short 3-second test
TOTAL_FRAMES = FPS * DURATION

OUTPUT_DIR = "test_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

PALETTE = {
    'background': hex_to_rgb('#1a1a1a'),
    'colors': [
        hex_to_rgb('#2d2d2d'),
        hex_to_rgb('#8B0000'),
        hex_to_rgb('#A52A2A'),
        hex_to_rgb('#6B2727'),
        hex_to_rgb('#A23C3C'),
        hex_to_rgb('#C74A4A'),
        hex_to_rgb('#E95656'),
        hex_to_rgb('#4a4a4a'),
        hex_to_rgb('#3a3a3a'),
        hex_to_rgb('#B22222'),
        hex_to_rgb('#5C3333'),
        hex_to_rgb('#8B4545'),
        hex_to_rgb('#B34D4D'),
        hex_to_rgb('#CC5C5C'),
        hex_to_rgb('#E97373'),
        hex_to_rgb('#ECA2A2'),
    ]
}

def lerp_color(color1, color2, t):
    return tuple(int(color1[i] + (color2[i] - color1[i]) * t) for i in range(3))

def get_color_from_palette(value, palette_colors):
    palette_size = len(palette_colors)
    scaled = value * (palette_size - 1)
    idx1 = int(scaled)
    idx2 = min(idx1 + 1, palette_size - 1)
    blend = scaled - idx1
    return lerp_color(palette_colors[idx1], palette_colors[idx2], blend)

def generate_test_frame(width, height, palette, t):
    """Simple animated noise pattern for testing."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    scale = 0.003

    time_offset = math.sin(t * 2 * math.pi) * 100

    gen_width = width // 2
    gen_height = height // 2

    for y in range(gen_height):
        for x in range(gen_width):
            noise = combined(perlin2D, x * scale + time_offset, y * scale, octaves=4, persistence=0.5)
            value = (noise + 1) / 2
            color = get_color_from_palette(value, palette)

            # 4-way symmetry
            pixels[x, y] = color
            pixels[width - 1 - x, y] = color
            pixels[x, height - 1 - y] = color
            pixels[width - 1 - x, height - 1 - y] = color

    return img

def main():
    print(f"\n{'='*60}")
    print("TESTING VIDEO GENERATION")
    print(f"{'='*60}\n")
    print(f"Generating test video: {WIDTH}x{HEIGHT} @ {FPS}fps, {DURATION}s")
    print(f"Total frames: {TOTAL_FRAMES}\n")

    frames = []

    for frame_num in tqdm(range(TOTAL_FRAMES), desc="Generating frames"):
        t = frame_num / TOTAL_FRAMES
        img = generate_test_frame(WIDTH, HEIGHT, PALETTE['colors'], t)
        frames.append(np.array(img))

    output_path = os.path.join(OUTPUT_DIR, "test_layered_noise.mp4")
    print("\nWriting video file...")
    imageio.mimsave(output_path, frames, fps=FPS, codec='libx264', quality=8, pixelformat='yuv420p')

    print(f"\n{'='*60}")
    print("TEST COMPLETE!")
    print(f"{'='*60}\n")
    print(f"Video saved to: {output_path}")
    print("If this played successfully, you can run generate_videos.py")

if __name__ == "__main__":
    main()
