#!/usr/bin/env python3
"""
Generate looping procedural video animations using various techniques.
Each generator creates a seamless loop that can be rendered to MP4.
"""

from PIL import Image, ImageDraw
from procgen.noise import perlin2D, simplex2D, combined
import random
import math
import os
import numpy as np
import imageio
from tqdm import tqdm

# Video dimensions (optimized for standard displays)
WIDTH = 1280  # 720p for faster generation
HEIGHT = 720

# Video settings
FPS = 30
DURATION = 10  # seconds
TOTAL_FRAMES = FPS * DURATION

# Output directory
OUTPUT_DIR = "videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Custom color palette
def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Define the color scheme (same as wallpaper generator)
PALETTE = {
    'background': hex_to_rgb('#1a1a1a'),
    'foreground': hex_to_rgb('#F7C4C4'),
    'cursor': hex_to_rgb('#9D6E6E'),
    'colors': [
        # Normal colors (0-7)
        hex_to_rgb('#2d2d2d'),  # Black
        hex_to_rgb('#8B0000'),  # Red
        hex_to_rgb('#A52A2A'),  # Green
        hex_to_rgb('#6B2727'),  # Yellow
        hex_to_rgb('#A23C3C'),  # Blue
        hex_to_rgb('#C74A4A'),  # Magenta
        hex_to_rgb('#E95656'),  # Cyan
        hex_to_rgb('#4a4a4a'),  # White
        # Bright colors (8-15)
        hex_to_rgb('#3a3a3a'),  # Bright Black
        hex_to_rgb('#B22222'),  # Bright Red
        hex_to_rgb('#5C3333'),  # Bright Green
        hex_to_rgb('#8B4545'),  # Bright Yellow
        hex_to_rgb('#B34D4D'),  # Bright Blue
        hex_to_rgb('#CC5C5C'),  # Bright Magenta
        hex_to_rgb('#E97373'),  # Bright Cyan
        hex_to_rgb('#ECA2A2'),  # Bright White
    ]
}

def lerp_color(color1, color2, t):
    """Linear interpolation between two RGB colors."""
    return tuple(int(color1[i] + (color2[i] - color1[i]) * t) for i in range(3))

def get_color_from_palette(value, palette_colors):
    """Map a value (0 to 1) to a color in the palette with smooth blending."""
    palette_size = len(palette_colors)
    scaled = value * (palette_size - 1)
    idx1 = int(scaled)
    idx2 = min(idx1 + 1, palette_size - 1)
    blend = scaled - idx1
    return lerp_color(palette_colors[idx1], palette_colors[idx2], blend)

# ============================================================================
# VIDEO GENERATION TECHNIQUES
# ============================================================================

def generate_layered_noise_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Layered Perlin/Simplex noise with animated time offset.
    t: time parameter from 0 to 1 (loops seamlessly)
    """
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    scale = 0.002

    # Convert t to seamless loop using sine wave
    time_offset = math.sin(t * 2 * math.pi) * 100

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    for y in range(gen_height):
        for x in range(gen_width):
            # Add time offset to create animation
            noise1 = combined(perlin2D, x * scale + time_offset, y * scale, octaves=6, persistence=0.5)
            noise2 = combined(simplex2D, x * scale * 1.5 + time_offset * 0.7, y * scale * 1.5, octaves=4, persistence=0.6)
            noise3 = combined(perlin2D, x * scale * 0.3, y * scale * 0.3 + time_offset * 0.3, octaves=3, persistence=0.7)

            combined_noise = (noise1 * 0.5 + noise2 * 0.3 + noise3 * 0.2)
            value = (combined_noise + 1) / 2
            color = get_color_from_palette(value, palette)

            background_blend = (noise2 + 1) / 8
            color = lerp_color(color, PALETTE['background'], background_blend)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_flow_field_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Animated flow field with time-based particle positions.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    scale = 0.003
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height
    num_lines = 1500
    line_length = 150

    # Seed randomness for consistency across frames
    random.seed(42)

    time_offset = t * 2 * math.pi

    for _ in range(num_lines):
        # Start position based on time
        x = random.randint(0, gen_width)
        y = random.randint(0, gen_height)

        # Add time-based offset to starting position
        angle_offset = math.sin(time_offset + random.random() * math.pi * 2) * 50

        points = []
        for step in range(line_length):
            if 0 <= x < gen_width and 0 <= y < gen_height:
                points.append((x, y))

                # Get flow direction from noise with time component
                angle = perlin2D(x * scale + angle_offset * 0.01, y * scale) * math.pi * 2 + time_offset * 0.5
                speed = (simplex2D(x * scale * 0.5, y * scale * 0.5) + 1) * 2

                x += math.cos(angle) * speed
                y += math.sin(angle) * speed
            else:
                break

        if len(points) > 1:
            # Color based on time and position
            value = (perlin2D(points[0][0] * 0.001, points[0][1] * 0.001) + 1) / 2
            color = get_color_from_palette(value, palette)
            alpha = 30

            # Draw line and mirrored versions
            draw.line(points, fill=color + (alpha,), width=2)

            if mirror_h:
                mirrored_h = [(width - 1 - x, y) for x, y in points]
                draw.line(mirrored_h, fill=color + (alpha,), width=2)

            if mirror_v:
                mirrored_v = [(x, height - 1 - y) for x, y in points]
                draw.line(mirrored_v, fill=color + (alpha,), width=2)

            if mirror_h and mirror_v:
                mirrored_both = [(width - 1 - x, height - 1 - y) for x, y in points]
                draw.line(mirrored_both, fill=color + (alpha,), width=2)

    return img

def generate_interference_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Animated wave interference patterns.
    """
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Wave centers that move over time
    time_offset = t * 2 * math.pi
    centers = [
        (gen_width * 0.3 + math.sin(time_offset) * 100, gen_height * 0.3 + math.cos(time_offset) * 100),
        (gen_width * 0.7 + math.sin(time_offset + math.pi) * 100, gen_height * 0.7 + math.cos(time_offset + math.pi) * 100),
        (gen_width * 0.5 + math.sin(time_offset * 0.7) * 150, gen_height * 0.5 + math.cos(time_offset * 0.7) * 150),
    ]

    wavelengths = [80, 120, 100]
    amplitudes = [1.0, 0.7, 0.5]

    for y in range(gen_height):
        for x in range(gen_width):
            wave_sum = 0

            for i, (cx, cy) in enumerate(centers):
                dx = x - cx
                dy = y - cy
                distance = math.sqrt(dx * dx + dy * dy)

                # Add time-based phase shift
                wave = amplitudes[i] * math.sin((distance / wavelengths[i] + time_offset * 2) * 2 * math.pi)
                wave_sum += wave

            # Normalize and map to palette
            value = (wave_sum / sum(amplitudes) + 1) / 2
            color = get_color_from_palette(value, palette)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_voronoi_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Animated Voronoi diagram with moving cell centers.
    """
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate base points with consistent seed
    random.seed(42)
    num_points = 30
    base_points = [(random.randint(0, gen_width), random.randint(0, gen_height)) for _ in range(num_points)]

    # Animate points in circular motion
    time_offset = t * 2 * math.pi
    points = []
    for px, py in base_points:
        offset_x = math.sin(time_offset + px * 0.01) * 30
        offset_y = math.cos(time_offset + py * 0.01) * 30
        points.append((px + offset_x, py + offset_y))

    point_colors = [random.choice(palette) for _ in range(num_points)]

    for y in range(gen_height):
        for x in range(gen_width):
            # Find closest point
            min_dist = float('inf')
            closest_idx = 0

            for i, (px, py) in enumerate(points):
                dx = x - px
                dy = y - py
                dist = dx * dx + dy * dy

                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            # Color based on distance with some noise
            noise = perlin2D(x * 0.001, y * 0.001)
            value = min(1.0, math.sqrt(min_dist) / 200 + (noise * 0.3))

            cell_color = point_colors[closest_idx]
            color = lerp_color(cell_color, PALETTE['background'], value * 0.5)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_fractal_noise_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Animated fractal noise with evolving ridges.
    """
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    scale = 0.0015

    time_offset = math.sin(t * 2 * math.pi) * 50

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    for y in range(gen_height):
        for x in range(gen_width):
            # Ridged fractal noise
            noise = 0
            amplitude = 1.0
            frequency = 1.0

            for _ in range(6):
                n = perlin2D((x + time_offset) * scale * frequency, y * scale * frequency)
                # Ridge transformation
                n = 1 - abs(n)
                noise += n * amplitude
                amplitude *= 0.5
                frequency *= 2.0

            value = noise / 2.0
            color = get_color_from_palette(value, palette)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_cellular_automata_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Animated cellular automata with evolving patterns.
    """
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Use time to create evolving CA state
    time_offset = t * 2 * math.pi
    scale = 0.05

    for y in range(gen_height):
        for x in range(gen_width):
            # Combine noise with time for evolving pattern
            noise1 = perlin2D(x * scale + time_offset, y * scale)
            noise2 = simplex2D(x * scale, y * scale + time_offset)

            # CA-like threshold
            cell_value = (noise1 + noise2) / 2
            if cell_value > 0.2:
                value = (cell_value + 1) / 2
            else:
                value = 0.1

            color = get_color_from_palette(value, palette)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

# ============================================================================
# VIDEO GENERATORS DICTIONARY
# ============================================================================

GENERATORS = {
    'layered_noise': ('Layered Noise', generate_layered_noise_frame),
    'flow_field': ('Flow Field', generate_flow_field_frame),
    'interference': ('Wave Interference', generate_interference_frame),
    'voronoi': ('Voronoi Cells', generate_voronoi_frame),
    'fractal_noise': ('Fractal Noise', generate_fractal_noise_frame),
    'cellular': ('Cellular Automata', generate_cellular_automata_frame),
}

# ============================================================================
# VIDEO RENDERING
# ============================================================================

def render_video(generator_func, output_path, width=WIDTH, height=HEIGHT, fps=FPS, total_frames=TOTAL_FRAMES):
    """
    Render a looping video using the given generator function.
    """
    print(f"Rendering video: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")

    frames = []

    # Generate frames
    for frame_num in tqdm(range(total_frames), desc="Generating frames"):
        # Calculate time parameter (0 to 1, loops seamlessly)
        t = frame_num / total_frames

        # Generate frame
        img = generator_func(width, height, PALETTE['colors'], t)

        # Convert PIL image to numpy array
        frame = np.array(img)
        frames.append(frame)

    # Write video file
    print("Writing video file...")
    imageio.mimsave(output_path, frames, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')
    print(f"Video saved to: {output_path}")

def generate_all_videos():
    """Generate videos for all available generators."""
    print(f"\n{'='*60}")
    print("PROCEDURAL VIDEO GENERATOR")
    print(f"{'='*60}\n")
    print(f"Generating {len(GENERATORS)} looping video animations")
    print(f"Output directory: {OUTPUT_DIR}/")
    print(f"Settings: {WIDTH}x{HEIGHT} @ {FPS}fps, {DURATION}s duration\n")

    for idx, (gen_id, (gen_name, gen_func)) in enumerate(GENERATORS.items(), 1):
        print(f"\n[{idx}/{len(GENERATORS)}] {gen_name}")
        print("-" * 60)

        output_path = os.path.join(OUTPUT_DIR, f"{idx:02d}_video_{gen_id}.mp4")
        render_video(gen_func, output_path)

    print(f"\n{'='*60}")
    print("ALL VIDEOS GENERATED SUCCESSFULLY!")
    print(f"{'='*60}\n")
    print(f"Videos saved to: {OUTPUT_DIR}/")
    print(f"Total videos: {len(GENERATORS)}")

if __name__ == "__main__":
    generate_all_videos()
