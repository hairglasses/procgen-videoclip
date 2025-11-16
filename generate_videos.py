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
import moderngl
import struct

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

# Global random seed (set per video generation)
RANDOM_SEED = 0

def set_random_seed(seed):
    """Set the global random seed for reproducible randomization."""
    global RANDOM_SEED
    RANDOM_SEED = seed
    random.seed(seed)
    np.random.seed(seed)

def get_random_param(base_value, variation=0.3):
    """Get a randomized parameter within variation range.

    Args:
        base_value: The base value to randomize
        variation: Percentage variation (0.3 = ±30%)
    """
    return base_value * (1.0 + random.uniform(-variation, variation))

def randomize_palette(palette, hue_shift_range=15):
    """Create a randomized version of the palette with hue shift.

    Args:
        palette: Original PALETTE dict
        hue_shift_range: Maximum hue shift in degrees (±range)
    """
    import colorsys

    hue_shift = random.uniform(-hue_shift_range, hue_shift_range) / 360.0

    def shift_color(rgb):
        r, g, b = [x / 255.0 for x in rgb]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        h = (h + hue_shift) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return tuple(int(x * 255) for x in (r, g, b))

    randomized = {
        'background': shift_color(palette['background']),
        'foreground': shift_color(palette['foreground']),
        'cursor': shift_color(palette['cursor']),
        'colors': [shift_color(c) for c in palette['colors']]
    }

    return randomized

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

def generate_plotter_art_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Animated plotter-style geometric art with rotating shapes.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Animated rotation based on time
    base_rotation = t * 2 * math.pi

    # Create concentric geometric shapes with rotation
    num_layers = 15
    center_x = gen_width // 2
    center_y = gen_height // 2

    random.seed(42)  # Consistent shapes across frames

    for i in range(num_layers):
        radius = (i + 1) * 40
        sides = random.choice([3, 4, 5, 6, 8])
        rotation = base_rotation + i * 0.15

        # Calculate polygon points
        points = []
        for s in range(sides):
            angle = (2 * math.pi * s / sides) + rotation
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))

        points.append(points[0])  # Close polygon

        # Color based on layer
        value = i / num_layers
        color = get_color_from_palette(value, palette)
        alpha = 50

        # Draw polygon
        draw.line(points, fill=color + (alpha,), width=2)

        # Mirror
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

def generate_spiral_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Animated spiral patterns with expanding/contracting motion.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    center_x = gen_width // 2
    center_y = gen_height // 2

    # Pulsing expansion based on time
    expansion = math.sin(t * 2 * math.pi) * 0.3 + 1.0

    # Randomize spiral parameters
    num_spirals = int(get_random_param(8, 0.25))  # 6-10 spirals
    num_points = int(get_random_param(200, 0.3))  # 140-260 points
    max_radius = get_random_param(300, 0.2) * expansion  # 240-360 base radius
    spiral_turns = get_random_param(4, 0.3)  # 2.8-5.2 turns

    for spiral_idx in range(num_spirals):
        points = []

        phase_offset = (spiral_idx / num_spirals) * 2 * math.pi

        for i in range(num_points):
            progress = i / num_points
            radius = progress * max_radius
            angle = progress * spiral_turns * math.pi + phase_offset + t * 2 * math.pi

            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)

            if 0 <= x < gen_width and 0 <= y < gen_height:
                points.append((x, y))

        if len(points) > 1:
            value = spiral_idx / num_spirals
            color = get_color_from_palette(value, palette)
            alpha = 40

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

def generate_rings_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Animated concentric rings with wave motion.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    center_x = gen_width // 2
    center_y = gen_height // 2

    time_offset = t * 2 * math.pi

    num_rings = 30
    for i in range(num_rings):
        # Pulsing radius
        base_radius = (i + 1) * 25
        pulse = math.sin(time_offset + i * 0.3) * 10
        radius = base_radius + pulse

        # Draw circle
        bbox = [
            center_x - radius, center_y - radius,
            center_x + radius, center_y + radius
        ]

        value = (i / num_rings + t) % 1.0
        color = get_color_from_palette(value, palette)
        alpha = 60

        draw.ellipse(bbox, outline=color + (alpha,), width=2)

        # Mirror
        if mirror_h:
            bbox_h = [width - 1 - bbox[2], bbox[1], width - 1 - bbox[0], bbox[3]]
            draw.ellipse(bbox_h, outline=color + (alpha,), width=2)
        if mirror_v:
            bbox_v = [bbox[0], height - 1 - bbox[3], bbox[2], height - 1 - bbox[1]]
            draw.ellipse(bbox_v, outline=color + (alpha,), width=2)
        if mirror_h and mirror_v:
            bbox_both = [width - 1 - bbox[2], height - 1 - bbox[3], width - 1 - bbox[0], height - 1 - bbox[1]]
            draw.ellipse(bbox_both, outline=color + (alpha,), width=2)

    return img

def generate_grid_distortion_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Animated grid with wave distortion.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    time_offset = t * 2 * math.pi

    spacing = 40
    amplitude = 20

    # Horizontal lines
    for y in range(0, gen_height, spacing):
        points = []
        for x in range(0, gen_width, 5):
            # Add wave distortion
            wave_y = y + amplitude * math.sin(x * 0.02 + time_offset)
            points.append((x, wave_y))

        if len(points) > 1:
            value = (y / gen_height + t) % 1.0
            color = get_color_from_palette(value, palette)
            alpha = 80

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

    # Vertical lines
    for x in range(0, gen_width, spacing):
        points = []
        for y in range(0, gen_height, 5):
            # Add wave distortion
            wave_x = x + amplitude * math.sin(y * 0.02 + time_offset + math.pi/2)
            points.append((wave_x, y))

        if len(points) > 1:
            value = (x / gen_width + t + 0.5) % 1.0
            color = get_color_from_palette(value, palette)
            alpha = 80

            draw.line(points, fill=color + (alpha,), width=2)

            if mirror_h:
                mirrored_h = [(width - 1 - px, py) for px, py in points]
                draw.line(mirrored_h, fill=color + (alpha,), width=2)
            if mirror_v:
                mirrored_v = [(px, height - 1 - py) for px, py in points]
                draw.line(mirrored_v, fill=color + (alpha,), width=2)
            if mirror_h and mirror_v:
                mirrored_both = [(width - 1 - px, height - 1 - py) for px, py in points]
                draw.line(mirrored_both, fill=color + (alpha,), width=2)

    return img

def generate_bezier_curves_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Organic shapes with animated Bezier curves.
    t: time parameter from 0 to 1 (loops seamlessly)
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Animated rotation angle
    time_angle = t * 2 * math.pi

    # Generate smooth organic shapes
    num_shapes = 15
    random.seed(42)  # Consistent shapes across frames

    for shape_idx in range(num_shapes):
        # Random start position in generation region
        start_x = random.randint(100, gen_width - 100)
        start_y = random.randint(100, gen_height - 100)

        # Create organic curved path
        points = []
        num_segments = 20
        base_angle = random.uniform(0, math.pi * 2)
        angle = base_angle + time_angle * 0.3  # Rotate over time

        for i in range(num_segments):
            # Use noise to create smooth curves
            noise_x = perlin2D(i * 0.3 + shape_idx, t * 5) * 50
            noise_y = perlin2D(t * 5, i * 0.3 + shape_idx) * 50

            angle += random.uniform(-0.3, 0.3)
            step = 30

            x = start_x + math.cos(angle) * step * i + noise_x
            y = start_y + math.sin(angle) * step * i + noise_y

            # Keep within generation region
            x = max(0, min(gen_width, x))
            y = max(0, min(gen_height, y))

            points.append((x, y))

        # Draw smooth curve
        if len(points) > 1:
            value = (shape_idx / num_shapes + t) % 1.0
            color = get_color_from_palette(value, palette)
            alpha = random.randint(40, 100)

            draw.line(points, fill=color + (alpha,), width=random.randint(2, 5), joint='curve')

            # Mirror the curve
            if mirror_h:
                mirrored_h = [(width - 1 - x, y) for x, y in points]
                draw.line(mirrored_h, fill=color + (alpha,), width=random.randint(2, 5), joint='curve')

            if mirror_v:
                mirrored_v = [(x, height - 1 - y) for x, y in points]
                draw.line(mirrored_v, fill=color + (alpha,), width=random.randint(2, 5), joint='curve')

            if mirror_h and mirror_v:
                mirrored_both = [(width - 1 - x, height - 1 - y) for x, y in points]
                draw.line(mirrored_both, fill=color + (alpha,), width=random.randint(2, 5), joint='curve')

    return img

def generate_physarum_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Slime mold simulation with animated particle behavior.
    t: time parameter from 0 to 1 (loops seamlessly)
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Time-based variation in particle initialization
    time_seed = int(t * 1000)
    random.seed(42 + time_seed)  # Varies with time but deterministic

    # Initialize particle agents
    num_particles = 3000  # Reduced for faster generation
    particles = []

    class Particle:
        def __init__(self):
            self.x = random.randint(0, gen_width - 1)
            self.y = random.randint(0, gen_height - 1)
            self.angle = random.uniform(0, math.pi * 2) + t * math.pi

    for _ in range(num_particles):
        particles.append(Particle())

    # Create trail map
    trail = [[0.0 for _ in range(gen_width)] for _ in range(gen_height)]

    # Simulate slime mold growth (fewer iterations for speed)
    iterations = 60
    sensor_angle = 0.4
    sensor_distance = 9
    turn_angle = 0.4

    for iteration in range(iterations):
        for particle in particles:
            # Sense in three directions
            x, y, angle = int(particle.x), int(particle.y), particle.angle

            if 0 <= x < gen_width and 0 <= y < gen_height:
                # Forward sensor
                fx = int(x + math.cos(angle) * sensor_distance) % gen_width
                fy = int(y + math.sin(angle) * sensor_distance) % gen_height
                forward = trail[fy][fx]

                # Left sensor
                lx = int(x + math.cos(angle - sensor_angle) * sensor_distance) % gen_width
                ly = int(y + math.sin(angle - sensor_angle) * sensor_distance) % gen_height
                left = trail[ly][lx]

                # Right sensor
                rx = int(x + math.cos(angle + sensor_angle) * sensor_distance) % gen_width
                ry = int(y + math.sin(angle + sensor_angle) * sensor_distance) % gen_height
                right = trail[ry][rx]

                # Adjust angle based on sensors
                if forward > left and forward > right:
                    pass  # Continue forward
                elif forward < left and forward < right:
                    particle.angle += random.choice([-turn_angle, turn_angle])
                elif left < right:
                    particle.angle += turn_angle
                elif right < left:
                    particle.angle -= turn_angle

                # Move particle
                particle.x = (particle.x + math.cos(particle.angle) * 2) % gen_width
                particle.y = (particle.y + math.sin(particle.angle) * 2) % gen_height

                # Deposit trail
                px, py = int(particle.x), int(particle.y)
                if 0 <= px < gen_width and 0 <= py < gen_height:
                    trail[py][px] = min(1.0, trail[py][px] + 0.1)

        # Diffuse and decay trails (optimized)
        if iteration % 3 == 0:  # Only diffuse every 3 iterations
            new_trail = [[0.0 for _ in range(gen_width)] for _ in range(gen_height)]
            for y in range(1, gen_height - 1):
                for x in range(1, gen_width - 1):
                    avg = (trail[y][x] + trail[y-1][x] + trail[y+1][x] +
                           trail[y][x-1] + trail[y][x+1]) / 5
                    new_trail[y][x] = avg * 0.95  # Decay
            trail = new_trail

    # Render trail map to image
    for y in range(gen_height):
        for x in range(gen_width):
            value = trail[y][x]
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

def generate_penrose_tiling_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Penrose tiling with animated phase shift.
    t: time parameter from 0 to 1 (loops seamlessly)
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Golden ratio and animated phase
    phi = (1 + math.sqrt(5)) / 2
    scale = 100
    time_phase = math.sin(t * 2 * math.pi) * 2  # Animated phase shift

    for y in range(gen_height):
        for x in range(gen_width):
            # Create pseudo-Penrose pattern with animated phase
            sum_val = 0
            for i in range(5):
                angle = i * 2 * math.pi / 5
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)

                # Project point onto rotated grid with time phase
                proj = (x * cos_a + y * sin_a) / scale

                # Add wave pattern with time variation
                sum_val += math.sin(proj * phi + time_phase) * math.cos(proj / phi)

            # Normalize and add noise
            value = (sum_val / 5 + 1) / 2
            noise = perlin2D(x * 0.002, y * 0.002) * 0.3
            value = max(0, min(1, value + noise))

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

def generate_pixel_sprites_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Retro pixel sprites with animated movement, rotation, and pulsing.
    t: time parameter from 0 to 1 (loops seamlessly)
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate multiple larger animated pixel sprites
    base_sprite_size = 100  # Increased from 40
    num_sprites_x = max(1, gen_width // (base_sprite_size * 2))
    num_sprites_y = max(1, gen_height // (base_sprite_size * 2))

    # Animated parameters
    time_2pi = t * 2 * math.pi
    color_shift = t

    random.seed(42)  # Consistent base sprite shapes

    for sy in range(num_sprites_y):
        for sx in range(num_sprites_x):
            sprite_id = sy * num_sprites_x + sx

            # Generate sprite pattern (morphs slightly over time)
            seed_offset = int(t * 4) % 4  # Change pattern 4 times during loop
            random.seed(42 + sprite_id + seed_offset * 100)

            # Pulsing size animation
            pulse = 1.0 + 0.15 * math.sin(time_2pi * 2 + sprite_id * 0.5)
            sprite_size = int(base_sprite_size * pulse)
            sprite_half_width = sprite_size // 2

            # Random sprite template
            sprite_data = [[random.random() > 0.6 for _ in range(sprite_half_width)]
                          for _ in range(sprite_size)]

            # Animated floating/orbital movement
            base_grid_x = sx * base_sprite_size * 2 + base_sprite_size
            base_grid_y = sy * base_sprite_size * 2 + base_sprite_size

            orbit_radius = 30
            orbit_speed = 0.5 + (sprite_id % 3) * 0.3
            base_x = int(base_grid_x + math.cos(time_2pi * orbit_speed + sprite_id) * orbit_radius)
            base_y = int(base_grid_y + math.sin(time_2pi * orbit_speed * 0.7 + sprite_id) * orbit_radius)

            # Animated sprite color with individual timing
            random.seed(42 + sprite_id)
            value = (random.random() + color_shift + sprite_id * 0.1) % 1.0
            sprite_color = get_color_from_palette(value, palette)

            # Draw sprite with horizontal mirroring (classic sprite style)
            for py in range(sprite_size):
                for px in range(sprite_half_width):
                    if sprite_data[py][px]:
                        # Apply rotation (simple 90-degree steps for pixel art)
                        rotation_step = int(t * 4) % 4

                        if rotation_step == 0:
                            offset_x, offset_y = px, py
                        elif rotation_step == 1:
                            offset_x, offset_y = py, sprite_size - 1 - px
                        elif rotation_step == 2:
                            offset_x, offset_y = sprite_size - 1 - px, sprite_size - 1 - py
                        else:  # rotation_step == 3
                            offset_x, offset_y = sprite_size - 1 - py, px

                        # Left half
                        x1 = base_x + offset_x - sprite_half_width
                        y1 = base_y + offset_y - sprite_size // 2

                        if 0 <= x1 < gen_width and 0 <= y1 < gen_height:
                            pixels[x1, y1] = sprite_color
                            if mirror_h:
                                pixels[width - 1 - x1, y1] = sprite_color
                            if mirror_v:
                                pixels[x1, height - 1 - y1] = sprite_color
                            if mirror_h and mirror_v:
                                pixels[width - 1 - x1, height - 1 - y1] = sprite_color

                        # Right half (mirror within sprite)
                        if rotation_step == 0:
                            offset_x2 = sprite_size - 1 - px
                        elif rotation_step == 1:
                            offset_x2 = py
                        elif rotation_step == 2:
                            offset_x2 = px
                        else:
                            offset_x2 = sprite_size - 1 - py

                        x2 = base_x + offset_x2 - sprite_half_width

                        if 0 <= x2 < gen_width and 0 <= y1 < gen_height:
                            pixels[x2, y1] = sprite_color
                            if mirror_h:
                                pixels[width - 1 - x2, y1] = sprite_color
                            if mirror_v:
                                pixels[x2, height - 1 - y1] = sprite_color
                            if mirror_h and mirror_v:
                                pixels[width - 1 - x2, height - 1 - y1] = sprite_color

    return img

def generate_chladni_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Chladni patterns (cymatics) - standing wave patterns on vibrating plates.
    Creates symmetric, mandala-like geometric patterns.
    """
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Animate frequencies through time
    time_offset = t * 2 * math.pi
    n = 3 + math.sin(time_offset) * 2  # Vary between 1 and 5
    m = 4 + math.cos(time_offset * 0.7) * 2

    # Phase shift for rotation effect
    phase = time_offset * 0.5

    for y in range(gen_height):
        for x in range(gen_width):
            # Normalize coordinates to [-π, π]
            nx = (x / gen_width - 0.5) * 2 * math.pi
            ny = (y / gen_height - 0.5) * 2 * math.pi

            # Chladni equation with phase shift
            z1 = math.sin(n * nx + phase) * math.sin(m * ny + phase)
            z2 = math.sin(m * nx + phase) * math.sin(n * ny + phase)
            z = abs(z1 + z2)

            # Map to color palette
            value = z / 2.0  # Normalize to 0-1
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

def generate_domain_warp_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Domain warping - recursive noise distortion creating organic, marbled patterns.
    """
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    scale = 0.003
    time_offset = math.sin(t * 2 * math.pi) * 50

    # Helper function for multi-octave noise
    def multi_octave_noise(x, y, octaves=4):
        value = 0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0
        for _ in range(octaves):
            value += perlin2D(x * frequency, y * frequency) * amplitude
            max_value += amplitude
            amplitude *= 0.5
            frequency *= 2.0
        return value / max_value

    for y in range(gen_height):
        for x in range(gen_width):
            # First level of warping
            q_x = multi_octave_noise(x * scale + time_offset, y * scale, 4)
            q_y = multi_octave_noise(x * scale, y * scale + time_offset, 4)

            # Second level - warp the coordinates with the first warp
            r_x = multi_octave_noise((x + q_x * 50) * scale, (y + q_y * 50) * scale, 4)
            r_y = multi_octave_noise((x + q_x * 50) * scale + 5.2, (y + q_y * 50) * scale + 1.3, 4)

            # Final value using double-warped coordinates
            value = multi_octave_noise((x + r_x * 50) * scale, (y + r_y * 50) * scale, 6)
            value = (value + 1) / 2  # Normalize to 0-1

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

def generate_superformula_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Superformula/Supershapes - generalized formula creating flowers, stars, and natural forms.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    center_x = gen_width // 2
    center_y = gen_height // 2

    # Animate parameters through time
    time_offset = t * 2 * math.pi
    m = 5 + math.sin(time_offset * 0.5) * 2  # Symmetry
    n1 = 1.0
    n2 = 2.0 + math.sin(time_offset) * 1.5
    n3 = 2.0 + math.cos(time_offset * 0.7) * 1.5
    a = b = 1.0

    # Generate multiple rotating supershapes at different scales
    num_shapes = 3
    for shape_idx in range(num_shapes):
        scale_factor = (shape_idx + 1) * 80
        rotation = time_offset + shape_idx * math.pi / num_shapes

        # Generate points along the superformula
        num_points = 360
        points = []
        for i in range(num_points + 1):
            theta = (i / num_points) * 2 * math.pi

            # Superformula
            t1 = abs(math.cos(m * theta / 4) / a) ** n2
            t2 = abs(math.sin(m * theta / 4) / b) ** n3
            r = (t1 + t2) ** (-1.0 / n1) if (t1 + t2) > 0 else 0

            # Apply rotation and scale
            r *= scale_factor
            x = center_x + r * math.cos(theta + rotation)
            y = center_y + r * math.sin(theta + rotation)

            if 0 <= x < gen_width and 0 <= y < gen_height:
                points.append((int(x), int(y)))

        # Draw the shape
        value = shape_idx / num_shapes
        color = get_color_from_palette(value, palette)

        for px, py in points:
            if 0 <= px < gen_width and 0 <= py < gen_height:
                pixels[px, py] = color
                if mirror_h and px < gen_width:
                    pixels[width - 1 - px, py] = color
                if mirror_v and py < gen_height:
                    pixels[px, height - 1 - py] = color
                if mirror_h and mirror_v:
                    pixels[width - 1 - px, height - 1 - py] = color

    return img

def generate_strange_attractor_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Strange attractors (Lorenz system) - chaotic dynamical system creating butterfly patterns.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Lorenz system parameters
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # Rotate camera around attractor
    time_offset = t * 2 * math.pi

    # Initialize multiple particles
    num_particles = 20
    random.seed(42)

    for particle_idx in range(num_particles):
        # Initial conditions
        x, y, z = 0.1 + particle_idx * 0.1, 0.0, 0.0

        points = []
        dt = 0.01
        iterations = 500

        for _ in range(iterations):
            # Lorenz equations
            dx = sigma * (y - x) * dt
            dy = (x * (rho - z) - y) * dt
            dz = (x * y - beta * z) * dt

            x += dx
            y += dy
            z += dz

            # Rotate and project to 2D
            angle_x = time_offset
            angle_y = time_offset * 0.7

            # Rotation around y-axis
            x_rot = x * math.cos(angle_y) - z * math.sin(angle_y)
            z_rot = x * math.sin(angle_y) + z * math.cos(angle_y)

            # Rotation around x-axis
            y_rot = y * math.cos(angle_x) - z_rot * math.sin(angle_x)

            # Project to screen
            scale = 8
            px = int(gen_width / 2 + x_rot * scale)
            py = int(gen_height / 2 + y_rot * scale)

            if 0 <= px < gen_width and 0 <= py < gen_height:
                points.append((px, py))

        if len(points) > 1:
            value = particle_idx / num_particles
            color = get_color_from_palette(value, palette)
            alpha = 40

            # Draw trail
            for i in range(len(points) - 1):
                draw.line([points[i], points[i+1]], fill=color + (alpha,), width=1)

                # Mirror
                if mirror_h:
                    p1 = (width - 1 - points[i][0], points[i][1])
                    p2 = (width - 1 - points[i+1][0], points[i+1][1])
                    draw.line([p1, p2], fill=color + (alpha,), width=1)
                if mirror_v:
                    p1 = (points[i][0], height - 1 - points[i][1])
                    p2 = (points[i+1][0], height - 1 - points[i+1][1])
                    draw.line([p1, p2], fill=color + (alpha,), width=1)
                if mirror_h and mirror_v:
                    p1 = (width - 1 - points[i][0], height - 1 - points[i][1])
                    p2 = (width - 1 - points[i+1][0], height - 1 - points[i+1][1])
                    draw.line([p1, p2], fill=color + (alpha,), width=1)

    return img

def generate_lsystem_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    L-Systems (Lindenmayer Systems) - fractal plant growth using string rewriting.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # L-System rules for plant-like structure
    axiom = "F"
    rules = {'F': 'F[+F]F[-F]F'}

    # Animate iterations and angle
    time_offset = t * 2 * math.pi
    iterations = 4
    angle_variation = math.sin(time_offset) * 10  # Degrees

    # Generate string
    current = axiom
    for _ in range(iterations):
        next_str = ""
        for char in current:
            next_str += rules.get(char, char)
        current = next_str

    # Turtle graphics rendering
    stack = []
    x, y = gen_width // 2, gen_height - 50
    angle = -90  # Start pointing up
    step_length = 5
    branch_angle = 25 + angle_variation

    points = [(x, y)]

    for char in current:
        if char == 'F':
            # Move forward
            rad = math.radians(angle)
            new_x = x + step_length * math.cos(rad)
            new_y = y + step_length * math.sin(rad)

            if 0 <= new_x < gen_width and 0 <= new_y < gen_height:
                points.append((new_x, new_y))

                # Draw line
                value = (len(points) % 20) / 20.0
                color = get_color_from_palette(value, palette)
                alpha = 60

                draw.line([(x, y), (new_x, new_y)], fill=color + (alpha,), width=1)

                # Mirror
                if mirror_h:
                    draw.line([(width - 1 - x, y), (width - 1 - new_x, new_y)], fill=color + (alpha,), width=1)
                if mirror_v:
                    draw.line([(x, height - 1 - y), (new_x, height - 1 - new_y)], fill=color + (alpha,), width=1)
                if mirror_h and mirror_v:
                    draw.line([(width - 1 - x, height - 1 - y), (width - 1 - new_x, height - 1 - new_y)], fill=color + (alpha,), width=1)

            x, y = new_x, new_y

        elif char == '+':
            angle += branch_angle
        elif char == '-':
            angle -= branch_angle
        elif char == '[':
            stack.append((x, y, angle))
        elif char == ']':
            if stack:
                x, y, angle = stack.pop()

    return img

def generate_boids_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Boids flocking simulation - emergent behavior from simple rules.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Initialize boids with deterministic seed
    random.seed(42 + int(t * 100))
    num_boids = 100

    class Boid:
        def __init__(self):
            self.x = random.uniform(0, gen_width)
            self.y = random.uniform(0, gen_height)
            self.vx = random.uniform(-2, 2)
            self.vy = random.uniform(-2, 2)

    boids = [Boid() for _ in range(num_boids)]

    # Simulate for a few steps
    for step in range(20):
        for boid in boids:
            # Separation, alignment, cohesion
            sep_x, sep_y = 0, 0
            align_x, align_y = 0, 0
            coh_x, coh_y = 0, 0
            neighbors = 0

            for other in boids:
                if other == boid:
                    continue

                dx = boid.x - other.x
                dy = boid.y - other.y
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < 50 and dist > 0:
                    # Separation
                    sep_x += dx / dist
                    sep_y += dy / dist

                    # Alignment
                    align_x += other.vx
                    align_y += other.vy

                    # Cohesion
                    coh_x += other.x
                    coh_y += other.y

                    neighbors += 1

            if neighbors > 0:
                align_x /= neighbors
                align_y /= neighbors
                coh_x = (coh_x / neighbors - boid.x) * 0.01
                coh_y = (coh_y / neighbors - boid.y) * 0.01

            # Apply forces
            boid.vx += sep_x * 0.05 + align_x * 0.05 + coh_x
            boid.vy += sep_y * 0.05 + align_y * 0.05 + coh_y

            # Limit speed
            speed = math.sqrt(boid.vx*boid.vx + boid.vy*boid.vy)
            if speed > 3:
                boid.vx = (boid.vx / speed) * 3
                boid.vy = (boid.vy / speed) * 3

            # Update position (toroidal wrapping)
            boid.x = (boid.x + boid.vx) % gen_width
            boid.y = (boid.y + boid.vy) % gen_height

    # Draw boids
    for i, boid in enumerate(boids):
        x, y = int(boid.x), int(boid.y)
        value = i / num_boids
        color = get_color_from_palette(value, palette)
        alpha = 80

        # Draw small triangle
        size = 3
        angle = math.atan2(boid.vy, boid.vx)
        pts = [
            (x + size * math.cos(angle), y + size * math.sin(angle)),
            (x + size * math.cos(angle + 2.5), y + size * math.sin(angle + 2.5)),
            (x + size * math.cos(angle - 2.5), y + size * math.sin(angle - 2.5))
        ]

        draw.polygon(pts, fill=color + (alpha,))

        # Mirror
        if mirror_h:
            pts_h = [(width - 1 - px, py) for px, py in pts]
            draw.polygon(pts_h, fill=color + (alpha,))
        if mirror_v:
            pts_v = [(px, height - 1 - py) for px, py in pts]
            draw.polygon(pts_v, fill=color + (alpha,))
        if mirror_h and mirror_v:
            pts_both = [(width - 1 - px, height - 1 - py) for px, py in pts]
            draw.polygon(pts_both, fill=color + (alpha,))

    return img

def generate_reaction_diffusion_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Reaction-Diffusion (Gray-Scott model) - creates organic Turing patterns.
    """
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Smaller grid for performance
    grid_w, grid_h = gen_width // 2, gen_height // 2

    # Initialize grids
    random.seed(42)
    u = [[1.0 for _ in range(grid_w)] for _ in range(grid_h)]
    v = [[0.0 for _ in range(grid_w)] for _ in range(grid_h)]

    # Add initial perturbation in center
    cx, cy = grid_w // 2, grid_h // 2
    for y in range(cy - 10, cy + 10):
        for x in range(cx - 10, cx + 10):
            if 0 <= x < grid_w and 0 <= y < grid_h:
                u[y][x] = 0.5
                v[y][x] = 0.25

    # Animate parameters
    time_offset = t * 2 * math.pi
    feed = 0.055 + math.sin(time_offset) * 0.01
    kill = 0.062 + math.cos(time_offset * 0.7) * 0.005

    # Diffusion rates
    Du = 0.16
    Dv = 0.08

    # Simulation steps
    for _ in range(20):
        u_new = [[0.0 for _ in range(grid_w)] for _ in range(grid_h)]
        v_new = [[0.0 for _ in range(grid_w)] for _ in range(grid_h)]

        for y in range(grid_h):
            for x in range(grid_w):
                # Laplacian (periodic boundaries)
                u_sum = (u[(y-1)%grid_h][x] + u[(y+1)%grid_h][x] +
                         u[y][(x-1)%grid_w] + u[y][(x+1)%grid_w] - 4*u[y][x])
                v_sum = (v[(y-1)%grid_h][x] + v[(y+1)%grid_h][x] +
                         v[y][(x-1)%grid_w] + v[y][(x+1)%grid_w] - 4*v[y][x])

                # Reaction
                uvv = u[y][x] * v[y][x] * v[y][x]

                u_new[y][x] = u[y][x] + (Du * u_sum - uvv + feed * (1 - u[y][x]))
                v_new[y][x] = v[y][x] + (Dv * v_sum + uvv - (kill + feed) * v[y][x])

                # Clamp
                u_new[y][x] = max(0, min(1, u_new[y][x]))
                v_new[y][x] = max(0, min(1, v_new[y][x]))

        u, v = u_new, v_new

    # Render to image
    for y in range(gen_height):
        for x in range(gen_width):
            gx, gy = x // 2, y // 2
            if gx < grid_w and gy < grid_h:
                value = v[gy][gx]
                color = get_color_from_palette(value, palette)

                pixels[x, y] = color
                if mirror_h:
                    pixels[width - 1 - x, y] = color
                if mirror_v:
                    pixels[x, height - 1 - y] = color
                if mirror_h and mirror_v:
                    pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_differential_growth_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Differential Growth - organic edge-growing algorithm creating coral-like forms.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    center_x, center_y = gen_width // 2, gen_height // 2

    # Initialize circle of nodes
    num_nodes = 50
    radius = 50 + math.sin(t * 2 * math.pi) * 20

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    nodes = []
    for i in range(num_nodes):
        angle = (i / num_nodes) * 2 * math.pi
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        nodes.append(Node(x, y))

    # Growth simulation
    max_edge_len = 10
    min_edge_len = 5
    repulsion_dist = 15

    for iteration in range(30):
        # Split long edges
        new_nodes = []
        i = 0
        while i < len(nodes):
            curr = nodes[i]
            next_node = nodes[(i + 1) % len(nodes)]

            dx = next_node.x - curr.x
            dy = next_node.y - curr.y
            dist = math.sqrt(dx*dx + dy*dy)

            if dist > max_edge_len:
                # Insert new node at midpoint
                new_x = (curr.x + next_node.x) / 2
                new_y = (curr.y + next_node.y) / 2
                new_nodes.append(Node(new_x, new_y))

            new_nodes.append(curr)
            i += 1

        nodes = new_nodes

        # Apply repulsion forces
        for node in nodes:
            fx, fy = 0, 0

            for other in nodes:
                if other == node:
                    continue

                dx = node.x - other.x
                dy = node.y - other.y
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < repulsion_dist and dist > 0:
                    force = (repulsion_dist - dist) / repulsion_dist
                    fx += (dx / dist) * force
                    fy += (dy / dist) * force

            node.x += fx * 0.1
            node.y += fy * 0.1

    # Draw the growth
    points = [(int(node.x), int(node.y)) for node in nodes]
    if len(points) > 2:
        points.append(points[0])  # Close the loop

        value = t
        color = get_color_from_palette(value, palette)
        alpha = 100

        for i in range(len(points) - 1):
            draw.line([points[i], points[i+1]], fill=color + (alpha,), width=2)

            # Mirror
            if mirror_h:
                p1 = (width - 1 - points[i][0], points[i][1])
                p2 = (width - 1 - points[i+1][0], points[i+1][1])
                draw.line([p1, p2], fill=color + (alpha,), width=2)
            if mirror_v:
                p1 = (points[i][0], height - 1 - points[i][1])
                p2 = (points[i+1][0], height - 1 - points[i+1][1])
                draw.line([p1, p2], fill=color + (alpha,), width=2)
            if mirror_h and mirror_v:
                p1 = (width - 1 - points[i][0], height - 1 - points[i][1])
                p2 = (width - 1 - points[i+1][0], height - 1 - points[i+1][1])
                draw.line([p1, p2], fill=color + (alpha,), width=2)

    return img

# ============================================================================
# SHADER-BASED GENERATORS
# ============================================================================

# Global shader context (created once for efficiency)
_shader_ctx = None

def get_shader_context():
    """Get or create OpenGL context for shader rendering."""
    global _shader_ctx
    if _shader_ctx is None:
        _shader_ctx = moderngl.create_standalone_context()
    return _shader_ctx

def render_shader(fragment_shader_code, width, height, palette, t):
    """
    Render a GLSL fragment shader to an image.

    Args:
        fragment_shader_code: GLSL fragment shader source
        width, height: Output dimensions
        palette: Color palette
        t: Time parameter (0 to 1)

    Returns:
        PIL Image
    """
    ctx = get_shader_context()

    # Vertex shader (simple fullscreen quad)
    vertex_shader = '''
        #version 330
        in vec2 in_vert;
        void main() {
            gl_Position = vec4(in_vert, 0.0, 1.0);
        }
    '''

    # Create shader program
    try:
        program = ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader_code
        )
    except Exception as e:
        print(f"Shader compilation error: {e}")
        # Return blank image on error
        return Image.new('RGB', (width, height), PALETTE['background'])

    # Set uniforms
    if 'iTime' in program:
        program['iTime'].value = t * 10.0  # Scale time for animation
    if 'iResolution' in program:
        program['iResolution'].value = (float(width), float(height))

    # Create framebuffer
    fbo = ctx.framebuffer(
        color_attachments=[ctx.texture((width, height), 4)]
    )

    # Create fullscreen quad
    vertices = np.array([
        -1.0, -1.0,
         1.0, -1.0,
        -1.0,  1.0,
         1.0,  1.0,
    ], dtype='f4')

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.simple_vertex_array(program, vbo, 'in_vert')

    # Render
    fbo.use()
    ctx.clear(0.0, 0.0, 0.0, 1.0)
    vao.render(moderngl.TRIANGLE_STRIP)

    # Read pixels
    data = fbo.read(components=4)
    img = Image.frombytes('RGBA', (width, height), data)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)

    # Convert to RGB
    return img.convert('RGB')

def generate_shader_plasma_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Classic plasma shader effect - colorful animated waves.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            // Plasma effect
            float v = 0.0;
            v += sin((p.x + iTime) * 10.0);
            v += sin((p.y + iTime) * 10.0);
            v += sin((p.x + p.y + iTime) * 10.0);
            v += sin(sqrt(p.x * p.x + p.y * p.y + 1.0) + iTime);
            v /= 4.0;

            // Color mapping
            vec3 col = 0.5 + 0.5 * cos(iTime + v * 3.14159 + vec3(0.0, 0.6, 1.0));

            // Apply palette-inspired dark red theme
            col = col * vec3(0.9, 0.3, 0.3) + vec3(0.1, 0.0, 0.0);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_tunnel_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Classic tunnel effect shader - infinite corridor.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            // Tunnel coordinates
            float a = atan(p.y, p.x);
            float r = length(p);

            // UV coordinates for tunnel
            vec2 tuv = vec2(a / 3.14159, 1.0 / r);
            tuv.x += iTime * 0.5;
            tuv.y += iTime * 0.3;

            // Pattern
            float pattern = sin(tuv.x * 20.0) * sin(tuv.y * 20.0);
            pattern = pattern * 0.5 + 0.5;

            // Add circular bands
            float bands = sin(r * 10.0 - iTime * 5.0) * 0.5 + 0.5;

            float v = mix(pattern, bands, 0.5);

            // Dark red color scheme
            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);

            // Vignette
            float vig = 1.0 - length(p) * 0.3;
            col *= vig;

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_raymarching_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Raymarching shader - 3D distance field rendering.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        // Distance field for sphere
        float sdSphere(vec3 p, float r) {
            return length(p) - r;
        }

        // Distance field for torus
        float sdTorus(vec3 p, vec2 t) {
            vec2 q = vec2(length(p.xz) - t.x, p.y);
            return length(q) - t.y;
        }

        // Scene SDF
        float map(vec3 p) {
            // Rotating torus
            float a = iTime;
            mat2 rot = mat2(cos(a), -sin(a), sin(a), cos(a));
            p.xz *= rot;
            p.xy *= rot;

            float d1 = sdTorus(p, vec2(1.0, 0.3));

            // Orbiting spheres
            vec3 p2 = p;
            p2.xz -= vec2(cos(iTime * 2.0), sin(iTime * 2.0)) * 1.5;
            float d2 = sdSphere(p2, 0.2);

            return min(d1, d2);
        }

        // Calculate normal
        vec3 calcNormal(vec3 p) {
            vec2 e = vec2(0.001, 0.0);
            return normalize(vec3(
                map(p + e.xyy) - map(p - e.xyy),
                map(p + e.yxy) - map(p - e.yxy),
                map(p + e.yyx) - map(p - e.yyx)
            ));
        }

        void main() {
            vec2 uv = (gl_FragCoord.xy - 0.5 * iResolution.xy) / iResolution.y;

            // Camera
            vec3 ro = vec3(0.0, 0.0, 3.0);
            vec3 rd = normalize(vec3(uv, -1.0));

            // Raymarch
            float t = 0.0;
            for (int i = 0; i < 80; i++) {
                vec3 p = ro + rd * t;
                float d = map(p);
                if (d < 0.001) break;
                t += d;
                if (t > 20.0) break;
            }

            // Shading
            vec3 col = vec3(0.1, 0.0, 0.0);
            if (t < 20.0) {
                vec3 p = ro + rd * t;
                vec3 n = calcNormal(p);

                // Lighting
                vec3 light = normalize(vec3(1.0, 1.0, 1.0));
                float diff = max(dot(n, light), 0.0);
                float spec = pow(max(dot(reflect(-light, n), -rd), 0.0), 32.0);

                // Dark red color scheme
                vec3 baseCol = vec3(0.9, 0.3, 0.3);
                col = baseCol * (0.2 + diff * 0.6) + vec3(1.0) * spec * 0.3;
            }

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_mandelbrot_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Animated Mandelbrot fractal shader.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            // Animate zoom and position
            float zoom = 0.5 + 0.3 * sin(iTime * 0.5);
            vec2 center = vec2(-0.5, 0.0);
            center.x += cos(iTime * 0.3) * 0.3;
            center.y += sin(iTime * 0.3) * 0.3;

            vec2 c = center + p * zoom;

            // Mandelbrot iteration
            vec2 z = c;
            float iterations = 0.0;
            const float maxIter = 100.0;

            for (float i = 0.0; i < maxIter; i++) {
                // z = z^2 + c
                z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;

                if (length(z) > 2.0) {
                    iterations = i;
                    break;
                }
            }

            // Color based on iterations
            float v = iterations / maxIter;
            v = sqrt(v); // Non-linear mapping

            // Animated color shift
            vec3 col = 0.5 + 0.5 * cos(iTime + v * 6.28318 + vec3(0.0, 0.6, 1.0));

            // Apply dark red theme
            col = col * vec3(0.9, 0.3, 0.3) + vec3(0.1, 0.0, 0.0);

            // Inside set is dark
            if (iterations >= maxIter - 1.0) {
                col = vec3(0.05, 0.0, 0.0);
            }

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_julia_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Animated Julia set fractal shader.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            // Animated Julia set parameter
            vec2 c = vec2(
                cos(iTime * 0.5) * 0.7,
                sin(iTime * 0.7) * 0.3
            );

            vec2 z = p * 1.5;
            float iterations = 0.0;
            const float maxIter = 100.0;

            for (float i = 0.0; i < maxIter; i++) {
                z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
                if (length(z) > 2.0) {
                    iterations = i;
                    break;
                }
            }

            float v = iterations / maxIter;
            v = sqrt(v);

            vec3 col = 0.5 + 0.5 * cos(iTime + v * 6.28318 + vec3(0.0, 0.6, 1.0));
            col = col * vec3(0.9, 0.3, 0.3) + vec3(0.1, 0.0, 0.0);

            if (iterations >= maxIter - 1.0) {
                col = vec3(0.05, 0.0, 0.0);
            }

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_metaballs_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Metaballs shader - organic blob physics.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            float v = 0.0;

            // Multiple metaballs
            for (int i = 0; i < 8; i++) {
                float fi = float(i);
                vec2 pos = vec2(
                    cos(iTime * 0.5 + fi * 0.8) * 0.5,
                    sin(iTime * 0.7 + fi * 0.6) * 0.5
                );
                float dist = length(p - pos);
                v += 0.02 / dist;
            }

            // Threshold for metaball surface
            v = smoothstep(0.4, 0.5, v);

            vec3 col = mix(
                vec3(0.1, 0.0, 0.0),
                vec3(0.9, 0.3, 0.3),
                v
            );

            // Add glow
            col += vec3(0.3, 0.1, 0.1) * (1.0 - v) * 0.3;

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_rotozoomer_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Classic rotozoomer effect - rotating and zooming texture.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float checker(vec2 uv) {
            return mod(floor(uv.x) + floor(uv.y), 2.0);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            // Rotating
            float angle = iTime * 0.5;
            mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
            p = rot * p;

            // Zooming
            float zoom = 5.0 + 3.0 * sin(iTime * 0.3);
            p *= zoom;

            // Scrolling
            p += vec2(iTime * 2.0, iTime * 1.5);

            // Pattern
            float check = checker(p);
            float v = mix(
                sin(p.x * 2.0) * sin(p.y * 2.0),
                check,
                0.5
            );
            v = v * 0.5 + 0.5;

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.8, 0.3, 0.3);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_voronoi_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Animated Voronoi/Worley noise shader.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        vec2 hash2(vec2 p) {
            p = vec2(dot(p, vec2(127.1, 311.7)), dot(p, vec2(269.5, 183.3)));
            return -1.0 + 2.0 * fract(sin(p) * 43758.5453123);
        }

        float voronoi(vec2 p) {
            vec2 n = floor(p);
            vec2 f = fract(p);

            float minDist = 1.0;
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    vec2 g = vec2(float(i), float(j));
                    vec2 o = hash2(n + g);
                    o = 0.5 + 0.5 * sin(iTime + 6.2831 * o);
                    vec2 r = g + o - f;
                    float d = dot(r, r);
                    minDist = min(minDist, d);
                }
            }

            return sqrt(minDist);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            float v = voronoi(p * 5.0 + iTime * 0.5);

            vec3 col = 0.5 + 0.5 * cos(v * 6.28318 + iTime + vec3(0.0, 0.6, 1.0));
            col = col * vec3(0.9, 0.3, 0.3) + vec3(0.1, 0.0, 0.0);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_kaleidoscope_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Kaleidoscope shader - symmetric mirror patterns.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            // Polar coordinates
            float r = length(p);
            float a = atan(p.y, p.x);

            // Kaleidoscope effect
            float segments = 8.0;
            a = mod(a, 6.28318 / segments);
            a = abs(a - 3.14159 / segments);

            // Reconstruct coordinates
            p = r * vec2(cos(a), sin(a));

            // Pattern
            p *= 3.0;
            p += vec2(iTime * 0.5, iTime * 0.3);

            float v = sin(p.x * 5.0 + iTime) * sin(p.y * 5.0 + iTime);
            v += sin(length(p) * 3.0 - iTime * 2.0);
            v = v * 0.5 + 0.5;

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.8, 0.3, 0.3);

            // Vignette
            col *= 1.0 - r * 0.3;

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_fire_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Fire effect shader - classic demoscene fire.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);

            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));

            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        float fbm(vec2 p) {
            float v = 0.0;
            float a = 0.5;
            for (int i = 0; i < 5; i++) {
                v += a * noise(p);
                p *= 2.0;
                a *= 0.5;
            }
            return v;
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv;

            // Fire rises from bottom
            p.y = 1.0 - p.y;
            p.y += iTime * 0.3;

            // Turbulence
            float n = fbm(p * 3.0 + vec2(0.0, iTime));

            // Fire intensity decreases with height
            float fire = n * (1.0 - uv.y);
            fire = pow(fire, 2.0);

            // Fire colors (dark red to bright orange/yellow)
            vec3 col = vec3(0.0);
            col += vec3(1.0, 0.3, 0.0) * fire;
            col += vec3(1.0, 0.8, 0.0) * pow(fire, 3.0);

            // Keep dark red theme
            col = col * 0.8 + vec3(0.1, 0.0, 0.0);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_starfield_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Animated starfield shader - flying through space.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec3 p) {
            return fract(sin(dot(p, vec3(127.1, 311.7, 758.5453))) * 43758.5453123);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            vec3 col = vec3(0.05, 0.0, 0.0);

            // Multiple star layers
            for (int layer = 0; layer < 3; layer++) {
                float depth = 1.0 / (float(layer) + 1.0);
                vec2 pos = p * depth;
                pos.y += iTime * depth * 0.5;

                // Create stars
                vec2 grid = floor(pos * 20.0);
                vec3 seed = vec3(grid, float(layer));

                if (hash(seed) > 0.95) {
                    vec2 starPos = grid / 20.0 + hash(seed + 1.0) * 0.05;
                    float dist = length(pos - starPos);

                    // Twinkling
                    float twinkle = sin(iTime * 3.0 + hash(seed) * 10.0) * 0.5 + 0.5;

                    float star = 0.002 / dist * twinkle;
                    col += vec3(0.9, 0.3, 0.3) * star;
                }
            }

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_hexagons_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Animated hexagonal tiling shader.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        vec2 hexCoords(vec2 p) {
            const vec2 s = vec2(1.732, 1.0);
            vec2 h = 0.5 * p / s;
            float a = mod(floor(h.x) + floor(h.y), 3.0);

            if (a < 1.5) {
                h = fract(h);
            } else {
                h = 1.0 - fract(h);
            }

            return h;
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            p *= 8.0;
            p += vec2(iTime * 0.5, iTime * 0.3);

            vec2 h = hexCoords(p);

            // Hexagon distance
            float d = max(abs(h.x - 0.5), abs(h.y - 0.5));

            // Animated pattern
            float pattern = sin(p.x + iTime) * sin(p.y + iTime * 0.7);
            pattern = pattern * 0.5 + 0.5;

            float v = smoothstep(0.4, 0.45, d);
            v = mix(pattern, v, 0.7);

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.8, 0.3, 0.3);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_dna_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    DNA helix shader - double helix structure.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            float z = p.y * 5.0 + iTime * 2.0;

            // Two helixes
            vec2 helix1 = vec2(cos(z) * 0.3, 0.0);
            vec2 helix2 = vec2(cos(z + 3.14159) * 0.3, 0.0);

            float d1 = length(p - helix1);
            float d2 = length(p - helix2);

            // Connecting bars
            float bar = 0.0;
            if (mod(z, 0.5) < 0.1) {
                float barDist = abs(p.x) - 0.3;
                bar = smoothstep(0.02, 0.0, barDist);
            }

            float v = 0.0;
            v += smoothstep(0.05, 0.0, d1);
            v += smoothstep(0.05, 0.0, d2);
            v += bar * 0.5;

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);

            // Add glow
            col += vec3(0.3, 0.1, 0.1) * (1.0 / (d1 * 10.0 + 1.0));
            col += vec3(0.3, 0.1, 0.1) * (1.0 / (d2 * 10.0 + 1.0));

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_matrix_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Matrix-style falling code effect.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv;

            // Create columns
            float cols = 40.0;
            float col = floor(p.x * cols);

            // Falling code
            float speed = hash(vec2(col, 0.0)) * 2.0 + 1.0;
            float y = mod(p.y + iTime * speed * 0.3, 1.0);

            // Character brightness
            float char = hash(vec2(col, floor(y * 20.0)));

            // Trail effect
            float trail = 1.0 - y;
            trail = pow(trail, 2.0);

            float v = char * trail;

            // Matrix green, adapted to red theme
            vec3 col3 = vec3(0.05, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);

            // Brighten the leading edge
            if (y < 0.1) {
                col3 += vec3(0.5, 0.2, 0.2);
            }

            fragColor = vec4(col3, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_waves_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Sine waves interference shader.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            float v = 0.0;

            // Multiple sine waves
            for (int i = 0; i < 5; i++) {
                float fi = float(i);
                float angle = iTime * 0.5 + fi * 1.2;
                vec2 dir = vec2(cos(angle), sin(angle));

                float wave = sin(dot(p, dir) * 10.0 + iTime * 2.0);
                v += wave;
            }

            v = v / 5.0;
            v = v * 0.5 + 0.5;

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.8, 0.3, 0.3);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_clock_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Animated clock/gear mechanism shader.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float gear(vec2 p, float teeth, float innerRadius, float outerRadius) {
            float a = atan(p.y, p.x);
            float r = length(p);

            float toothAngle = 6.28318 / teeth;
            float tooth = mod(a, toothAngle) / toothAngle;
            tooth = abs(tooth - 0.5) * 2.0;

            float gearRadius = mix(innerRadius, outerRadius, smoothstep(0.3, 0.7, tooth));

            return smoothstep(gearRadius + 0.02, gearRadius, r);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            float v = 0.0;

            // Central gear
            mat2 rot1 = mat2(cos(iTime), -sin(iTime), sin(iTime), cos(iTime));
            v += gear(rot1 * p, 12.0, 0.4, 0.6);

            // Side gears
            vec2 p2 = p - vec2(0.7, 0.0);
            mat2 rot2 = mat2(cos(-iTime * 1.5), -sin(-iTime * 1.5), sin(-iTime * 1.5), cos(-iTime * 1.5));
            v += gear(rot2 * p2, 8.0, 0.2, 0.3);

            vec2 p3 = p + vec2(0.7, 0.0);
            mat2 rot3 = mat2(cos(-iTime * 1.5), -sin(-iTime * 1.5), sin(-iTime * 1.5), cos(-iTime * 1.5));
            v += gear(rot3 * p3, 8.0, 0.2, 0.3);

            v = clamp(v, 0.0, 1.0);

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_caustics_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Caustics shader - light refraction through water.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            float v = 0.0;

            // Multiple light rays
            for (int i = 0; i < 5; i++) {
                float fi = float(i);
                vec2 offset = vec2(
                    sin(iTime * 0.3 + fi * 1.3) * 0.5,
                    cos(iTime * 0.4 + fi * 0.7) * 0.5
                );

                vec2 lightPos = vec2(0.0, 0.0) + offset;
                float dist = length(p - lightPos);

                // Caustic pattern
                float caustic = sin(dist * 10.0 - iTime * 2.0);
                caustic += sin(dist * 7.0 + iTime * 1.5);
                caustic += sin(dist * 13.0 - iTime * 3.0);
                caustic /= 3.0;

                v += caustic / (dist + 0.5);
            }

            v = clamp(v * 0.5 + 0.5, 0.0, 1.0);

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_truchet_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Truchet tiles shader - randomized geometric tiling.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 10.0;

            // Tile coordinates
            vec2 tile = floor(p);
            vec2 local = fract(p);

            // Random rotation per tile
            float h = hash(tile + floor(iTime * 2.0));
            float angle = step(0.5, h) * 1.5708; // 0 or 90 degrees

            // Rotate local coordinates
            float c = cos(angle);
            float s = sin(angle);
            vec2 rotated = vec2(
                local.x * c - local.y * s,
                local.x * s + local.y * c
            );

            // Truchet arc pattern
            float dist1 = length(rotated - vec2(0.0, 0.0));
            float dist2 = length(rotated - vec2(1.0, 1.0));

            float v = smoothstep(0.4, 0.5, dist1) - smoothstep(0.5, 0.6, dist1);
            v += smoothstep(0.4, 0.5, dist2) - smoothstep(0.5, 0.6, dist2);

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_aurora_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Aurora/Northern Lights shader - flowing light ribbons.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            float v = 0.0;

            // Multiple flowing waves
            for (float i = 0.0; i < 5.0; i++) {
                float offset = i * 0.2;
                float wave = sin(p.x * 3.0 + iTime + offset) * 0.3;
                wave += sin(p.x * 5.0 - iTime * 0.7 + offset) * 0.2;

                float dist = abs(p.y - wave + offset * 0.3 - 0.5);
                v += 0.02 / dist;
            }

            v = clamp(v, 0.0, 1.0);

            // Aurora-like glow
            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);
            col += v * v * vec3(0.5, 0.1, 0.1); // Extra glow

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_moire_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Moire pattern shader - interference patterns from overlapping grids.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            // First grid
            float angle1 = iTime * 0.5;
            vec2 p1 = vec2(
                p.x * cos(angle1) - p.y * sin(angle1),
                p.x * sin(angle1) + p.y * cos(angle1)
            );
            float grid1 = sin(p1.x * 20.0) * sin(p1.y * 20.0);

            // Second grid
            float angle2 = -iTime * 0.3;
            vec2 p2 = vec2(
                p.x * cos(angle2) - p.y * sin(angle2),
                p.x * sin(angle2) + p.y * cos(angle2)
            );
            float grid2 = sin(p2.x * 20.0) * sin(p2.y * 20.0);

            // Interference pattern
            float v = grid1 * grid2;
            v = v * 0.5 + 0.5;

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_perlin_flow_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Perlin noise flow field shader.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        // Simple noise function
        float hash(vec2 p) {
            p = fract(p * vec2(123.34, 456.21));
            p += dot(p, p + 45.32);
            return fract(p.x * p.y);
        }

        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);

            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));

            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 5.0;

            // Flow field
            vec2 flowVec = vec2(
                noise(p + iTime * 0.3),
                noise(p + iTime * 0.3 + 100.0)
            );

            // Sample noise along flow
            float v = noise(p + flowVec * 2.0);
            v += noise(p * 2.0 + flowVec) * 0.5;
            v /= 1.5;

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_spirograph_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Spirograph shader - circular geometric patterns.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            float v = 0.0;

            // Spirograph parameters
            float R = 0.6;  // Outer circle radius
            float r = 0.3 + sin(iTime * 0.5) * 0.1;  // Inner circle radius
            float d = 0.5;  // Pen distance

            // Draw multiple spirograph curves
            for (float i = 0.0; i < 100.0; i += 1.0) {
                float angle = i * 0.1 + iTime;
                float x = (R - r) * cos(angle) + d * cos((R - r) / r * angle);
                float y = (R - r) * sin(angle) - d * sin((R - r) / r * angle);

                float dist = length(p - vec2(x, y));
                v += 0.003 / dist;
            }

            v = clamp(v, 0.0, 1.0);

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_electric_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Electric/lightning shader - branching electric arcs.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 2.0 - 1.0;
            p.x *= iResolution.x / iResolution.y;

            float v = 0.0;

            // Vertical lightning bolts
            for (float i = -0.5; i <= 0.5; i += 0.5) {
                float xOffset = i;
                float x = xOffset;

                // Add jagged path
                for (float y = -1.0; y < 1.0; y += 0.1) {
                    float jitter = (hash(vec2(y, floor(iTime * 10.0))) - 0.5) * 0.3;
                    x += jitter * 0.1;

                    float dist = length(p - vec2(x, y));
                    v += 0.002 / dist;
                }
            }

            // Flicker
            v *= 0.8 + 0.2 * sin(iTime * 50.0);

            v = clamp(v, 0.0, 1.0);

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);
            col += v * v * vec3(0.8, 0.2, 0.2); // Extra glow

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_glitch_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Glitch shader - digital distortion/corruption effect.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(float n) {
            return fract(sin(n) * 43758.5453);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv;

            // Horizontal scanline glitches
            float glitchLine = floor(p.y * 20.0 + iTime * 10.0);
            float glitchStrength = step(0.95, hash(glitchLine));

            // Offset certain scanlines
            p.x += (hash(glitchLine + iTime) - 0.5) * glitchStrength * 0.3;

            // Block-based displacement
            vec2 blockPos = floor(p * 8.0);
            float blockGlitch = step(0.8, hash(dot(blockPos, vec2(12.9898, 78.233)) + floor(iTime * 2.0)));
            p += (vec2(hash(blockPos.x), hash(blockPos.y)) - 0.5) * blockGlitch * 0.1;

            // Create striped pattern
            float pattern = sin(p.x * 30.0 + iTime * 2.0) * sin(p.y * 30.0);
            pattern = pattern * 0.5 + 0.5;

            // Color shift
            float v = pattern;
            v = clamp(v, 0.0, 1.0);

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);

            // Random pixel corruption
            if (hash(dot(floor(p * 100.0), vec2(12.9898, 78.233)) + floor(iTime * 30.0)) > 0.99) {
                col = vec3(0.9, 0.3, 0.3);
            }

            fragColor = vec4(col, 1.0);
        }
    '''

    return render_shader(fragment_shader, width, height, palette, t)

# ============================================================================
# SPRITE CHARACTER GENERATORS
# ============================================================================

def draw_pixel_character(draw, x, y, frame, palette, char_type='walk', scale=4):
    """Draw an animated pixel character."""

    if char_type == 'walk':
        # Walking character - 4 frame walk cycle
        frame_idx = int(frame * 4) % 4

        # Character is 8x12 pixels
        # Body (5x7)
        body_pixels = [
            [0,0,1,1,0,0],
            [0,1,1,1,1,0],
            [1,1,2,2,1,1],  # Head with eyes
            [0,1,1,1,1,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],  # Body
            [0,0,1,1,0,0],
        ]

        # Legs change based on frame
        if frame_idx == 0:
            leg_pixels = [[0,1,0,0,1,0], [0,1,0,0,1,0], [1,0,0,0,0,1]]
        elif frame_idx == 1:
            leg_pixels = [[0,0,1,1,0,0], [0,1,0,0,1,0], [1,0,0,0,0,1]]
        elif frame_idx == 2:
            leg_pixels = [[0,1,0,0,1,0], [0,1,0,0,1,0], [1,0,0,0,0,1]]
        else:
            leg_pixels = [[0,0,1,1,0,0], [0,0,1,0,1,0], [0,1,0,0,0,1]]

        all_pixels = body_pixels + leg_pixels

    elif char_type == 'jump':
        # Jumping character
        jump_phase = math.sin(frame * math.pi * 2)

        all_pixels = [
            [0,0,1,1,0,0],
            [0,1,1,1,1,0],
            [1,1,2,2,1,1],
            [0,1,1,1,1,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
        ]

        # Arms up when jumping
        if jump_phase > 0:
            all_pixels.append([1,0,1,1,0,1])
            all_pixels.append([0,1,0,0,1,0])
            all_pixels.append([0,0,0,0,0,0])
        else:
            all_pixels.append([0,1,0,0,1,0])
            all_pixels.append([0,1,0,0,1,0])
            all_pixels.append([1,0,0,0,0,1])

    elif char_type == 'dance':
        # Dancing character - 8 frame dance
        frame_idx = int(frame * 8) % 8

        body = [
            [0,0,1,1,0,0],
            [0,1,1,1,1,0],
            [1,1,2,2,1,1],
            [0,1,1,1,1,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
        ]

        # Dance moves
        if frame_idx < 2:
            arms = [[1,0,1,1,0,1], [0,0,0,0,0,0]]
            legs = [[0,1,0,0,1,0], [1,0,0,0,0,1]]
        elif frame_idx < 4:
            arms = [[0,1,1,1,1,0], [1,0,0,0,0,1]]
            legs = [[0,0,1,1,0,0], [0,1,0,0,1,0]]
        elif frame_idx < 6:
            arms = [[1,0,1,1,0,1], [0,0,0,0,0,0]]
            legs = [[0,1,0,0,1,0], [1,0,0,0,0,1]]
        else:
            arms = [[0,1,1,1,1,0], [1,0,0,0,0,1]]
            legs = [[0,1,0,0,1,0], [1,0,0,0,0,1]]

        all_pixels = body + arms + legs

    elif char_type == 'fly':
        # Flying character with wings
        wing_frame = int(frame * 6) % 2

        all_pixels = [
            [0,0,1,1,0,0],
            [0,1,1,1,1,0],
            [1,1,2,2,1,1],
            [0,1,1,1,1,0],
            [0,0,1,1,0,0],
        ]

        # Flapping wings
        if wing_frame == 0:
            all_pixels.append([1,1,1,1,1,1])
            all_pixels.append([0,1,0,0,1,0])
        else:
            all_pixels.append([0,1,1,1,1,0])
            all_pixels.append([1,0,1,1,0,1])

        all_pixels.append([0,1,0,0,1,0])
        all_pixels.append([0,1,0,0,1,0])

    else:  # run
        # Running character - 4 frame run cycle
        frame_idx = int(frame * 8) % 4

        body = [
            [0,0,1,1,0,0],
            [0,1,1,1,1,0],
            [1,1,2,2,1,1],
            [0,1,1,1,1,0],
            [0,0,1,1,0,0],
            [0,0,1,1,0,0],
        ]

        if frame_idx == 0:
            limbs = [[1,0,1,1,0,0], [0,1,0,0,1,0], [0,0,1,0,1,0], [0,0,0,1,0,1]]
        elif frame_idx == 1:
            limbs = [[0,1,1,1,0,0], [1,0,0,0,1,0], [0,0,1,1,0,0], [0,1,0,0,0,1]]
        elif frame_idx == 2:
            limbs = [[0,0,1,1,0,1], [0,1,0,0,1,0], [0,1,0,1,0,0], [1,0,1,0,0,0]]
        else:
            limbs = [[0,0,1,1,1,0], [0,1,0,0,0,1], [0,0,1,1,0,0], [1,0,0,0,1,0]]

        all_pixels = body + limbs

    # Draw the character
    for row_idx, row in enumerate(all_pixels):
        for col_idx, pixel in enumerate(row):
            if pixel > 0:
                color = palette[min(pixel, len(palette)-1)]
                for sy in range(scale):
                    for sx in range(scale):
                        px = int(x + col_idx * scale + sx)
                        py = int(y + row_idx * scale + sy)
                        if 0 <= px < draw.im.size[0] and 0 <= py < draw.im.size[1]:
                            draw.point((px, py), fill=color)

def generate_walking_character_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Walking character - animated sprite walking across screen.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    # Multiple characters walking at different speeds
    num_chars = 5
    for i in range(num_chars):
        # Position moving across screen
        speed = 0.3 + i * 0.1
        x_pos = ((t * speed + i * 0.2) % 1.0) * width
        y_pos = height // 2 + math.sin(t * 2 * math.pi + i) * 50

        # Animate walk cycle
        draw_pixel_character(draw, x_pos, y_pos, t + i * 0.25, palette, 'walk', scale=5)

    return img

def generate_jumping_character_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Jumping character - sprites jumping with physics.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    # Multiple jumping characters
    num_chars = 6
    for i in range(num_chars):
        x_base = (i + 0.5) * (width / num_chars)

        # Jump arc
        phase = (t + i * 0.16) % 1.0
        jump_height = math.sin(phase * math.pi) * 150
        y_pos = height - 100 - jump_height

        draw_pixel_character(draw, x_base - 15, y_pos, phase, palette, 'jump', scale=5)

    return img

def generate_dancing_character_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Dancing character - rhythmic dance animations.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    # Grid of dancing characters
    cols = 6
    rows = 4

    for row in range(rows):
        for col in range(cols):
            x_pos = (col + 0.5) * (width / cols)
            y_pos = (row + 0.5) * (height / rows)

            # Offset animation for wave effect
            offset = (row * cols + col) * 0.125

            draw_pixel_character(draw, x_pos - 15, y_pos - 20, t + offset, palette, 'dance', scale=4)

    return img

def generate_flying_character_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Flying character - floating sprites with wing flapping.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    # Characters flying in circular patterns
    num_chars = 8
    for i in range(num_chars):
        angle = (t * 2 * math.pi + i * (2 * math.pi / num_chars))
        radius = 200 + math.sin(t * math.pi + i) * 50

        x_pos = width // 2 + math.cos(angle) * radius
        y_pos = height // 2 + math.sin(angle) * radius * 0.6

        # Add floating bobbing motion
        y_pos += math.sin(t * 4 * math.pi + i) * 15

        draw_pixel_character(draw, x_pos - 15, y_pos - 20, t * 2 + i * 0.5, palette, 'fly', scale=5)

    return img

def generate_running_character_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Running character - fast-paced run cycle animation.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    # Characters running in lanes
    num_lanes = 5

    for i in range(num_lanes):
        y_pos = (i + 1) * (height / (num_lanes + 1))

        # Fast movement across screen
        speed = 0.5 + i * 0.15
        x_pos = ((t * speed + i * 0.2) % 1.2) * width - 50

        # Scale based on depth (perspective)
        scale = 3 + i * 0.8

        draw_pixel_character(draw, x_pos, y_pos - 20, t * 2 + i * 0.3, palette, 'run', scale=int(scale))

    return img

# ============================================================================
# RETRO & PIXEL GLITCH SHADERS
# ============================================================================

def generate_shader_crt_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    CRT screen shader - scanlines, phosphor glow, screen curvature.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;

            // CRT curvature
            vec2 centered = uv * 2.0 - 1.0;
            float curve = 0.15;
            centered *= 1.0 + curve * centered.y * centered.y;
            centered *= 1.0 + curve * centered.x * centered.x;
            uv = (centered + 1.0) * 0.5;

            // Vignette
            float vignette = smoothstep(0.7, 0.4, length(centered));

            // Base pattern
            float v = 0.0;
            float time = iTime * 0.5;
            vec2 p = uv * 10.0;

            for (float i = 0.0; i < 5.0; i++) {
                p += sin(p.yx * 2.0 + time + i) * 0.3;
                v += sin(p.x + time) * sin(p.y - time * 0.7);
            }

            v = (v + 5.0) / 10.0;

            // Scanlines
            float scanline = sin(uv.y * iResolution.y * 1.5) * 0.3 + 0.7;

            // RGB phosphor mask
            float mask = 1.0;
            float x = uv.x * iResolution.x;
            if (mod(x, 3.0) < 1.0) mask *= 1.2;
            else if (mod(x, 3.0) < 2.0) mask *= 0.9;

            // Screen flicker
            float flicker = 0.98 + sin(iTime * 123.0) * 0.02;

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);
            col *= scanline * mask * vignette * flicker;

            // Chromatic aberration
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                col = vec3(0.0);
            }

            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_vhs_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    VHS glitch shader - tracking errors, tape noise, distortion.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;

            // VHS tracking errors
            float trackingError = hash(vec2(floor(iTime * 2.0), floor(uv.y * 20.0)));
            if (trackingError > 0.95) {
                uv.x += (hash(vec2(iTime, uv.y)) - 0.5) * 0.1;
            }

            // Tape noise bands
            float noiseBand = step(0.98, hash(vec2(uv.y * 50.0, floor(iTime * 10.0))));

            // Base pattern with distortion
            vec2 p = uv * 8.0;
            p.x += sin(uv.y * 10.0 + iTime * 2.0) * 0.1;

            float v = 0.0;
            for (float i = 0.0; i < 3.0; i++) {
                float offset = i * 0.3;
                v += sin(p.x * 2.0 + iTime + offset) * cos(p.y * 2.0 - iTime * 0.7 + offset);
            }

            v = (v + 3.0) / 6.0;

            // Color bleeding
            float bleedOffset = 0.005;
            float r = v + sin(p.x * 10.0) * 0.1;
            float g = v;
            float b = v - sin(p.x * 10.0) * 0.1;

            vec3 col = vec3(r, g * 0.3, b * 0.3) * 0.9;
            col = mix(col, vec3(0.1, 0.0, 0.0), noiseBand);

            // VHS jitter
            if (mod(floor(iTime * 60.0), 120.0) < 2.0) {
                col *= 0.5 + hash(uv) * 0.5;
            }

            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_pixelsort_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Pixel sort shader - datamoshing/glitch art effect.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 15.0;

            // Generate base pattern
            float v = 0.0;
            for (float i = 0.0; i < 4.0; i++) {
                p += sin(p.yx + iTime * 0.5 + i);
                v += sin(p.x) * cos(p.y);
            }
            v = (v + 4.0) / 8.0;

            // Pixel sorting effect - horizontal streaks
            float sortThreshold = 0.5 + sin(iTime) * 0.3;
            vec2 sortUV = uv;

            if (v > sortThreshold) {
                // Sort pixels horizontally
                float sortAmount = (v - sortThreshold) * 2.0;
                sortUV.x = fract(sortUV.x + sortAmount * sin(iTime * 2.0 + sortUV.y * 10.0));

                // Recalculate with sorted coordinates
                vec2 p2 = sortUV * 15.0;
                v = 0.0;
                for (float i = 0.0; i < 4.0; i++) {
                    p2 += sin(p2.yx + iTime * 0.5 + i);
                    v += sin(p2.x) * cos(p2.y);
                }
                v = (v + 4.0) / 8.0;
            }

            // Vertical glitch bars
            if (hash(vec2(floor(uv.y * 30.0), floor(iTime * 3.0))) > 0.9) {
                v = hash(vec2(uv.x * 100.0, floor(iTime * 10.0)));
            }

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);
            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_rgbsplit_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    RGB split shader - chromatic aberration with separated color channels.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;

            // Animated RGB split
            float splitAmount = 0.02 + sin(iTime * 2.0) * 0.01;
            float angle = iTime * 0.5;

            vec2 offsetR = vec2(cos(angle), sin(angle)) * splitAmount;
            vec2 offsetG = vec2(0.0);
            vec2 offsetB = vec2(cos(angle + 3.14159), sin(angle + 3.14159)) * splitAmount;

            // Generate pattern for each channel
            vec2 pR = (uv + offsetR) * 12.0;
            vec2 pG = (uv + offsetG) * 12.0;
            vec2 pB = (uv + offsetB) * 12.0;

            float r = 0.0, g = 0.0, b = 0.0;

            // Red channel
            for (float i = 0.0; i < 3.0; i++) {
                pR += sin(pR.yx + iTime + i);
                r += sin(pR.x) * cos(pR.y);
            }
            r = (r + 3.0) / 6.0;

            // Green channel
            for (float i = 0.0; i < 3.0; i++) {
                pG += sin(pG.yx + iTime + i);
                g += sin(pG.x) * cos(pG.y);
            }
            g = (g + 3.0) / 6.0;

            // Blue channel
            for (float i = 0.0; i < 3.0; i++) {
                pB += sin(pB.yx + iTime + i);
                b += sin(pB.x) * cos(pB.y);
            }
            b = (b + 3.0) / 6.0;

            // Glitch displacement
            if (sin(uv.y * 20.0 + iTime * 10.0) > 0.9) {
                r = g;
            }

            vec3 col = vec3(r * 0.9 + 0.1, g * 0.3, b * 0.3);
            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_dither_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Dither pattern shader - retro ordered dithering.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        // Bayer matrix 4x4
        float bayer4x4(vec2 pos) {
            mat4 bayerMatrix = mat4(
                0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0,
                12.0/16.0, 4.0/16.0, 14.0/16.0,  6.0/16.0,
                3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0,
                15.0/16.0, 7.0/16.0, 13.0/16.0,  5.0/16.0
            );
            int x = int(mod(pos.x, 4.0));
            int y = int(mod(pos.y, 4.0));
            return bayerMatrix[x][y];
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 pixelPos = gl_FragCoord.xy;

            // Generate base pattern
            vec2 p = uv * 10.0;
            p += sin(p.yx + iTime * 0.5);

            float v = 0.0;
            for (float i = 0.0; i < 5.0; i++) {
                float offset = i * 0.2;
                v += sin(p.x * 2.0 + iTime + offset) * cos(p.y * 2.0 - iTime * 0.7 + offset);
            }
            v = (v + 5.0) / 10.0;

            // Apply dither
            float threshold = bayer4x4(pixelPos);
            float dithered = step(threshold, v);

            // Animated dither levels
            float levels = 4.0 + sin(iTime) * 2.0;
            v = floor(v * levels + threshold) / levels;

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);
            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_teletext_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Teletext shader - blocky videotext/teletext style.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;

            // Blocky pixel grid (teletext characters)
            vec2 blockSize = vec2(16.0, 24.0);
            vec2 blockUV = floor(uv * iResolution.xy / blockSize);
            vec2 cellUV = fract(uv * iResolution.xy / blockSize);

            // Generate block pattern
            float blockValue = hash(blockUV + floor(iTime * 0.5));
            blockValue += sin(blockUV.x * 0.5 + iTime) * cos(blockUV.y * 0.5 - iTime * 0.7) * 0.5 + 0.5;
            blockValue = fract(blockValue);

            // Character-like patterns within blocks
            float charPattern = 0.0;
            if (cellUV.x > 0.2 && cellUV.x < 0.8 && cellUV.y > 0.3 && cellUV.y < 0.7) {
                charPattern = step(0.5, blockValue);
            }
            if (cellUV.y > 0.1 && cellUV.y < 0.3 && cellUV.x > 0.4 && cellUV.x < 0.6) {
                charPattern = max(charPattern, step(0.7, blockValue));
            }

            // Scanline effect
            float scanline = sin(uv.y * iResolution.y * 2.0) * 0.1 + 0.9;

            // Background pattern
            float bg = hash(blockUV) * 0.2;

            float v = mix(bg, charPattern, 0.8);

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.3);
            col *= scanline;

            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_c64_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Commodore 64 shader - classic computer aesthetic with color limitations.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;

            // Low resolution grid (320x200 was common)
            vec2 pixelSize = vec2(4.0, 4.0);
            vec2 pixelUV = floor(uv * iResolution.xy / pixelSize) * pixelSize / iResolution.xy;

            // Generate pattern
            vec2 p = pixelUV * 20.0;
            p += vec2(sin(pixelUV.y * 10.0 + iTime), cos(pixelUV.x * 10.0 - iTime)) * 0.5;

            float v = 0.0;
            for (float i = 0.0; i < 3.0; i++) {
                v += sin(p.x + iTime * 0.5 + i) * cos(p.y - iTime * 0.3 + i);
            }
            v = (v + 3.0) / 6.0;

            // Limit to C64-style colors (quantize)
            v = floor(v * 4.0) / 4.0;

            // Border effect (C64 had colored borders)
            if (uv.x < 0.05 || uv.x > 0.95 || uv.y < 0.05 || uv.y > 0.95) {
                v *= 0.5;
            }

            // Screen artifacts
            float artifact = sin(uv.x * iResolution.x * 0.5) * 0.05;

            vec3 col = vec3(0.1, 0.0, 0.0) + (v + artifact) * vec3(0.9, 0.3, 0.3);
            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_gameboy_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Game Boy shader - 4-shade palette with LCD screen effects.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;

            // Game Boy resolution was 160x144
            vec2 gbRes = vec2(160.0, 144.0);
            vec2 pixelUV = floor(uv * gbRes) / gbRes;

            // Generate pattern
            vec2 p = pixelUV * 15.0;
            p.x += sin(pixelUV.y * 8.0 + iTime * 2.0) * 0.3;
            p.y += cos(pixelUV.x * 8.0 - iTime * 2.0) * 0.3;

            float v = 0.0;
            for (float i = 0.0; i < 4.0; i++) {
                v += sin(p.x + iTime * 0.5 + i * 0.5) * cos(p.y - iTime * 0.3 + i * 0.5);
            }
            v = (v + 4.0) / 8.0;

            // Limit to 4 shades (classic Game Boy)
            v = floor(v * 3.99) / 3.0;

            // LCD grid effect
            vec2 gridUV = fract(uv * gbRes);
            float grid = 1.0;
            if (gridUV.x < 0.1 || gridUV.y < 0.1) {
                grid = 0.9;
            }

            // Screen motion blur (LCD ghosting)
            float ghost = sin(uv.y * gbRes.y + iTime * 30.0) * 0.05;

            vec3 col = vec3(0.1, 0.0, 0.0) + (v + ghost) * vec3(0.9, 0.3, 0.3);
            col *= grid;

            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_feedback_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Feedback loop shader - psychedelic video feedback effect.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 center = vec2(0.5);

            // Feedback displacement
            vec2 toCenter = center - uv;
            float dist = length(toCenter);
            float angle = atan(toCenter.y, toCenter.x);

            // Spiral feedback
            float spiral = sin(dist * 20.0 - iTime * 2.0 + angle * 3.0);
            vec2 displacement = vec2(cos(angle + iTime), sin(angle + iTime)) * spiral * 0.02;

            // Sample with feedback offset
            vec2 fbUV = uv + displacement;
            fbUV = fract(fbUV * (1.0 + sin(iTime * 0.5) * 0.1));

            // Generate base pattern
            vec2 p = fbUV * 15.0;
            float v = 0.0;
            for (float i = 0.0; i < 5.0; i++) {
                p = p * 1.3 + vec2(sin(p.y + iTime * 0.3 + i), cos(p.x - iTime * 0.3 + i));
                v += sin(p.x) * cos(p.y);
            }
            v = (v + 5.0) / 10.0;

            // Add trailing effect
            float trail = smoothstep(0.3, 0.7, sin(dist * 10.0 - iTime * 3.0));
            v = mix(v, v * trail, 0.5);

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.95, 0.35, 0.35);
            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_lava_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Lava lamp shader - organic blob morphing like lava lamps.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float blob(vec2 p, vec2 center, float size) {
            float d = length(p - center);
            return smoothstep(size, size * 0.5, d);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv - 0.5;
            p.x *= iResolution.x / iResolution.y;

            float v = 0.0;

            // Multiple animated blobs
            for (float i = 0.0; i < 8.0; i++) {
                float t1 = iTime * 0.3 + i * 0.8;
                float t2 = iTime * 0.4 + i * 1.2;

                vec2 center = vec2(
                    sin(t1 + i) * 0.3,
                    cos(t2 + i * 0.5) * 0.4 + sin(iTime * 0.2 + i) * 0.2
                );

                float size = 0.15 + sin(iTime * 0.5 + i) * 0.08;
                v += blob(p, center, size);
            }

            // Threshold to create lava lamp effect
            v = smoothstep(0.4, 0.6, v);

            // Add glow
            float glow = smoothstep(0.2, 0.8, v) * 0.3;
            v = clamp(v + glow, 0.0, 1.0);

            vec3 col = vec3(0.05, 0.0, 0.0) + v * vec3(1.0, 0.3, 0.2);
            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_nebula_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Nebula shader - cosmic clouds with layered noise.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        float noise(vec2 p) {
            vec2 i = floor(p);
            vec2 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);

            float a = hash(i);
            float b = hash(i + vec2(1.0, 0.0));
            float c = hash(i + vec2(0.0, 1.0));
            float d = hash(i + vec2(1.0, 1.0));

            return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
        }

        float fbm(vec2 p) {
            float v = 0.0;
            float a = 0.5;
            for (int i = 0; i < 6; i++) {
                v += noise(p) * a;
                p *= 2.0;
                a *= 0.5;
            }
            return v;
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv * 3.0;

            // Layered nebula clouds
            float n1 = fbm(p + iTime * 0.1);
            float n2 = fbm(p * 1.5 + iTime * 0.15 + vec2(5.2, 1.3));
            float n3 = fbm(p * 2.0 - iTime * 0.08 + vec2(2.5, 8.1));

            // Combine layers
            float nebula = n1 * 0.5 + n2 * 0.3 + n3 * 0.2;
            nebula = pow(nebula, 1.5);

            // Add swirls
            vec2 center = vec2(0.5 + sin(iTime * 0.2) * 0.2, 0.5 + cos(iTime * 0.15) * 0.2);
            vec2 toCenter = uv - center;
            float angle = atan(toCenter.y, toCenter.x);
            float dist = length(toCenter);
            float swirl = sin(dist * 8.0 - angle * 3.0 + iTime) * 0.5 + 0.5;

            nebula = mix(nebula, swirl, 0.3);

            // Stars
            float stars = 0.0;
            if (hash(floor(uv * 200.0)) > 0.98) {
                stars = hash(floor(uv * 200.0 + 0.5)) * 0.5;
            }

            vec3 col = vec3(0.05, 0.0, 0.0) + nebula * vec3(0.9, 0.25, 0.2) + stars;
            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_circuit_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Circuit board shader - procedural circuit paths with flowing signals.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 grid = uv * 20.0;
            vec2 id = floor(grid);
            vec2 gv = fract(grid) - 0.5;

            float v = 0.0;

            // Circuit traces
            float h = hash(id);

            // Horizontal and vertical lines
            if (h > 0.7) {
                v = smoothstep(0.05, 0.02, abs(gv.y));
            } else if (h > 0.4) {
                v = smoothstep(0.05, 0.02, abs(gv.x));
            }

            // Connection points
            float point = smoothstep(0.15, 0.05, length(gv));
            if (hash(id + 0.5) > 0.8) {
                v = max(v, point);
            }

            // Flowing signals
            float signal = 0.0;
            float flow = fract(iTime * 0.5 + h);
            vec2 flowPos = gv;

            if (h > 0.7) {
                flowPos.x = abs(flowPos.x - (flow - 0.5));
                signal = smoothstep(0.2, 0.05, flowPos.x) * smoothstep(0.05, 0.02, abs(flowPos.y));
            } else if (h > 0.4) {
                flowPos.y = abs(flowPos.y - (flow - 0.5));
                signal = smoothstep(0.2, 0.05, flowPos.y) * smoothstep(0.05, 0.02, abs(flowPos.x));
            }

            v = max(v, signal * 1.5);

            vec3 col = vec3(0.05, 0.0, 0.0) + v * vec3(1.0, 0.4, 0.3);
            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_warp_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Warp speed shader - hyperspace star streaking effect.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        float hash(vec2 p) {
            return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv - 0.5;
            p.x *= iResolution.x / iResolution.y;

            float angle = atan(p.y, p.x);
            float dist = length(p);

            // Warp speed streaks
            float v = 0.0;

            for (float i = 0.0; i < 50.0; i++) {
                float h = hash(vec2(i, 0.0));
                float starAngle = h * 6.28318;
                float starDist = hash(vec2(i, 1.0));

                vec2 starPos = vec2(cos(starAngle), sin(starAngle)) * starDist * 0.5;

                // Animate position toward camera
                float z = fract(h * 5.0 + iTime * 2.0);
                vec2 pos = starPos / (z + 0.1);

                // Create streak
                vec2 diff = p - pos;
                float streakDist = length(diff);
                float streak = smoothstep(0.02, 0.0, streakDist);

                // Streak tail in radial direction
                vec2 radial = normalize(pos);
                float tail = smoothstep(0.2, 0.0, abs(dot(normalize(diff), radial) - 1.0));
                tail *= smoothstep(0.5, 0.0, streakDist);

                float brightness = (1.0 - z) * (1.0 - z);
                v += (streak + tail * 0.5) * brightness;
            }

            // Center glow
            float glow = smoothstep(0.3, 0.0, dist) * 0.3;
            v += glow;

            vec3 col = vec3(0.05, 0.0, 0.0) + v * vec3(1.0, 0.4, 0.3);
            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_liquid_crystal_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Liquid crystal shader - LCD-like interference patterns.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = (uv - 0.5) * 2.0;
            p.x *= iResolution.x / iResolution.y;

            // Multiple layers of interference
            float v = 0.0;

            for (float i = 1.0; i <= 5.0; i++) {
                float angle = iTime * 0.2 * i + i * 1.5;
                vec2 dir = vec2(cos(angle), sin(angle));

                float wave = sin(dot(p, dir) * 10.0 * i - iTime * i * 0.5);
                v += wave / i;
            }

            v = (v + 5.0) / 10.0;

            // Liquid crystal cell structure
            vec2 cells = fract(uv * 50.0);
            float cellPattern = smoothstep(0.1, 0.0, min(cells.x, cells.y)) * 0.2;

            // Polarization effect
            float polarization = sin(v * 6.28318 * 3.0) * 0.5 + 0.5;
            v = mix(v, polarization, 0.7);

            v = clamp(v - cellPattern, 0.0, 1.0);

            vec3 col = vec3(0.1, 0.0, 0.0) + v * vec3(0.9, 0.3, 0.25);
            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_fractal_flame_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Fractal flame shader - iterated function systems with color gradients.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        vec2 transform1(vec2 p, float t) {
            float a = sin(t * 0.5) * 0.3;
            return vec2(
                sin(p.x) * cos(p.y + a),
                cos(p.x + a) * sin(p.y)
            ) * 0.8;
        }

        vec2 transform2(vec2 p, float t) {
            return vec2(
                p.x * cos(t * 0.3) - p.y * sin(t * 0.3),
                p.x * sin(t * 0.3) + p.y * cos(t * 0.3)
            ) * 0.9 + vec2(0.1, 0.1);
        }

        vec2 transform3(vec2 p, float t) {
            float r = length(p);
            float a = atan(p.y, p.x) + sin(t * 0.4) * 0.5;
            return vec2(cos(a), sin(a)) * sqrt(r) * 0.7;
        }

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = (uv - 0.5) * 3.0;
            p.x *= iResolution.x / iResolution.y;

            float density = 0.0;
            float colorMix = 0.0;

            // Iterate the transforms
            for (int i = 0; i < 30; i++) {
                float fi = float(i);
                float h = fract(sin(fi * 12.9898) * 43758.5453);

                if (h < 0.33) {
                    p = transform1(p, iTime + fi * 0.1);
                    colorMix += 0.1;
                } else if (h < 0.66) {
                    p = transform2(p, iTime + fi * 0.1);
                    colorMix += 0.2;
                } else {
                    p = transform3(p, iTime + fi * 0.1);
                    colorMix += 0.3;
                }

                float d = length(p);
                if (d < 2.0) {
                    density += 1.0 / (1.0 + d * d);
                }
            }

            density = clamp(density / 15.0, 0.0, 1.0);
            density = pow(density, 0.7);

            colorMix = fract(colorMix * 0.1 + iTime * 0.1);

            vec3 col = vec3(0.05, 0.0, 0.0) + density * mix(
                vec3(1.0, 0.3, 0.2),
                vec3(0.8, 0.4, 0.3),
                colorMix
            );

            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

def generate_shader_oscilloscope_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Oscilloscope shader - audio visualizer style scope patterns.
    """
    fragment_shader = '''
        #version 330
        out vec4 fragColor;
        uniform float iTime;
        uniform vec2 iResolution;

        void main() {
            vec2 uv = gl_FragCoord.xy / iResolution.xy;
            vec2 p = uv - 0.5;
            p.x *= iResolution.x / iResolution.y;

            float v = 0.0;

            // Lissajous curve
            float freqX = 3.0 + sin(iTime * 0.5) * 2.0;
            float freqY = 2.0 + cos(iTime * 0.3) * 1.5;
            float phaseX = iTime * 2.0;
            float phaseY = iTime * 2.5 + sin(iTime * 0.2) * 3.14159;

            for (float i = 0.0; i < 100.0; i++) {
                float t = i / 100.0;
                vec2 pos = vec2(
                    sin(t * 6.28318 * freqX + phaseX) * 0.3,
                    sin(t * 6.28318 * freqY + phaseY) * 0.3
                );

                float d = length(p - pos);
                v += smoothstep(0.02, 0.0, d);
            }

            // XY grid
            float grid = 0.0;
            grid += smoothstep(0.005, 0.0, abs(p.x));
            grid += smoothstep(0.005, 0.0, abs(p.y));

            // Circular markers
            for (float r = 0.1; r <= 0.4; r += 0.1) {
                grid += smoothstep(0.003, 0.0, abs(length(p) - r)) * 0.3;
            }

            v += grid * 0.5;

            // Glow effect
            v = clamp(v, 0.0, 1.0);
            float glow = v * v;

            vec3 col = vec3(0.05, 0.0, 0.0) + v * vec3(0.9, 0.35, 0.3) + glow * vec3(0.3, 0.1, 0.1);
            fragColor = vec4(col, 1.0);
        }
    '''
    return render_shader(fragment_shader, width, height, palette, t)

# ============================================================================
# ISOMETRIC & VOXEL GENERATORS
# ============================================================================

def iso_project(x, y, z):
    """Convert 3D coordinates to isometric 2D coordinates."""
    screen_x = (x - y) * 0.866  # cos(30°) ≈ 0.866
    screen_y = (x + y) * 0.5 - z
    return (screen_x, screen_y)

def draw_iso_cube(draw, cx, cy, cz, size, palette, color_idx=0):
    """Draw an isometric cube at the given position."""
    # Define cube vertices in 3D space
    s = size
    vertices = [
        (cx, cy, cz),           # 0: bottom-front-left
        (cx+s, cy, cz),         # 1: bottom-front-right
        (cx+s, cy+s, cz),       # 2: bottom-back-right
        (cx, cy+s, cz),         # 3: bottom-back-left
        (cx, cy, cz+s),         # 4: top-front-left
        (cx+s, cy, cz+s),       # 5: top-front-right
        (cx+s, cy+s, cz+s),     # 6: top-back-right
        (cx, cy+s, cz+s),       # 7: top-back-left
    ]

    # Project to 2D
    points = [iso_project(v[0], v[1], v[2]) for v in vertices]

    # Draw three visible faces
    # Top face
    top_color = palette[min(color_idx, len(palette)-1)]
    draw.polygon([points[4], points[5], points[6], points[7]], fill=top_color, outline=(0,0,0))

    # Left face (darker)
    left_color = tuple(max(0, int(c * 0.7)) for c in palette[min(color_idx, len(palette)-1)])
    draw.polygon([points[0], points[3], points[7], points[4]], fill=left_color, outline=(0,0,0))

    # Right face (even darker)
    right_color = tuple(max(0, int(c * 0.5)) for c in palette[min(color_idx, len(palette)-1)])
    draw.polygon([points[1], points[2], points[6], points[5]], fill=right_color, outline=(0,0,0))

def generate_isometric_cubes_frame(width, height, palette, t, mirror_h=False, mirror_v=False):
    """
    Isometric cubes - floating and rotating isometric blocks.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    # Center offset for projection
    cx, cy = width // 2, height // 2

    # Create a grid of cubes with animated heights and positions
    # Randomize parameters for variety
    grid_size = 14
    cube_size = int(get_random_param(25, 0.2))  # 20-30
    spacing = int(get_random_param(55, 0.2))     # 44-66
    wave_freq = get_random_param(0.5, 0.3)       # 0.35-0.65
    wave_amp = get_random_param(30, 0.3)         # 21-39
    float_amp = get_random_param(10, 0.4)        # 6-14

    time_2pi = t * 2 * math.pi

    for i in range(grid_size):
        for j in range(grid_size):
            # Base position
            base_x = (i - grid_size/2) * spacing
            base_y = (j - grid_size/2) * spacing

            # Animated height based on wave pattern
            wave = math.sin(i * wave_freq + time_2pi) * math.cos(j * wave_freq + time_2pi)
            height = wave * wave_amp + 40

            # Floating animation
            float_z = math.sin(time_2pi * 2 + i + j) * float_amp

            # Color based on height
            color_idx = int((wave + 1) * (len(palette) - 1) / 2)

            # Project and draw
            screen_x, screen_y = iso_project(base_x, base_y, float_z)
            draw_iso_cube(draw, base_x, base_y, float_z, cube_size, palette, color_idx)

    return img

def generate_voxel_terrain_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Voxel terrain - procedural height-mapped voxel landscape.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    grid_size = 20
    voxel_size = 12
    time_offset = t * 100

    # Store voxels with depth for proper rendering order
    voxels = []

    for i in range(grid_size):
        for j in range(grid_size):
            x = (i - grid_size/2) * voxel_size
            y = (j - grid_size/2) * voxel_size

            # Terrain height using noise
            noise_val = perlin2D((i + time_offset) * 0.1, j * 0.1)
            terrain_height = int((noise_val + 1) * 3) + 1

            # Stack voxels vertically
            for h in range(terrain_height):
                z = h * voxel_size
                screen_x, screen_y = iso_project(x, y, z)

                # Calculate depth for sorting
                depth = x + y - z

                # Color based on height
                color_idx = min(h, len(palette) - 1)

                voxels.append((depth, x, y, z, color_idx))

    # Sort by depth (back to front)
    voxels.sort(key=lambda v: v[0], reverse=True)

    # Draw all voxels
    for depth, x, y, z, color_idx in voxels:
        draw_iso_cube(draw, x, y, z, voxel_size, palette, color_idx)

    return img

def generate_isometric_city_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Isometric city - animated building structures.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    # Create city blocks
    grid_size = 6
    block_size = 30
    spacing = 80

    time_2pi = t * 2 * math.pi

    random.seed(42)  # Fixed seed for consistent buildings
    buildings = []

    for i in range(grid_size):
        for j in range(grid_size):
            x = (i - grid_size/2) * spacing
            y = (j - grid_size/2) * spacing

            # Building height animated with different phases
            base_height = random.randint(2, 6)
            phase = random.random() * math.pi * 2
            height_mult = (math.sin(time_2pi + phase) + 1) * 0.3 + 0.7
            building_height = int(base_height * height_mult)

            # Calculate depth for sorting
            depth = x + y

            buildings.append((depth, x, y, building_height, i+j))

    # Sort by depth
    buildings.sort(key=lambda b: b[0], reverse=True)

    # Draw buildings
    for depth, x, y, height, seed_val in buildings:
        for h in range(height):
            z = h * block_size
            color_idx = min(h % len(palette), len(palette) - 1)
            draw_iso_cube(draw, x, y, z, block_size, palette, color_idx)

    return img

def generate_voxel_waves_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Voxel waves - wave patterns made of voxels.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    grid_size = 16
    voxel_size = 15
    time_2pi = t * 2 * math.pi

    voxels = []

    for i in range(grid_size):
        for j in range(grid_size):
            x = (i - grid_size/2) * voxel_size
            y = (j - grid_size/2) * voxel_size

            # Distance from center
            dist = math.sqrt((i - grid_size/2)**2 + (j - grid_size/2)**2)

            # Ripple wave pattern
            wave_height = math.sin(dist * 0.5 - time_2pi * 3) * 3
            wave_height += math.cos(i * 0.3 + time_2pi) * 1.5
            wave_height += math.sin(j * 0.3 - time_2pi) * 1.5

            num_voxels = max(1, int(wave_height + 4))

            for h in range(num_voxels):
                z = h * voxel_size
                depth = x + y - z

                # Color based on wave phase
                wave_phase = (wave_height + 4) / 8
                color_idx = int(wave_phase * (len(palette) - 1))

                voxels.append((depth, x, y, z, color_idx))

    # Sort and draw
    voxels.sort(key=lambda v: v[0], reverse=True)

    for depth, x, y, z, color_idx in voxels:
        draw_iso_cube(draw, x, y, z, voxel_size, palette, color_idx)

    return img

def generate_isometric_stairs_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Isometric stairs - Escher-like recursive staircases.
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    cx, cy = width // 2, height // 2

    step_size = 20
    num_steps = 10
    time_2pi = t * 2 * math.pi

    # Create multiple staircase spirals
    num_spirals = 4

    for spiral_idx in range(num_spirals):
        angle_offset = spiral_idx * (math.pi / 2)

        for i in range(num_steps):
            # Spiral around center
            angle = i * 0.3 + time_2pi * 0.5 + angle_offset
            radius = 50 + i * 15

            x = math.cos(angle) * radius
            y = math.sin(angle) * radius
            z = i * step_size - num_steps * step_size / 2

            # Add rotation animation
            rotation = math.sin(time_2pi + i * 0.2) * 5
            z += rotation

            # Color cycles through palette
            color_idx = (i + spiral_idx * 3) % len(palette)

            draw_iso_cube(draw, x, y, z, step_size, palette, color_idx)

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
    'plotter_art': ('Plotter Art', generate_plotter_art_frame),
    'spiral': ('Spiral Patterns', generate_spiral_frame),
    'rings': ('Concentric Rings', generate_rings_frame),
    'grid_distortion': ('Grid Distortion', generate_grid_distortion_frame),
    'bezier_curves': ('Bezier Curves', generate_bezier_curves_frame),
    'physarum': ('Physarum Slime Mold', generate_physarum_frame),
    'penrose_tiling': ('Penrose Tiling', generate_penrose_tiling_frame),
    'pixel_sprites': ('Pixel Sprites', generate_pixel_sprites_frame),
    'chladni': ('Chladni Patterns', generate_chladni_frame),
    'domain_warp': ('Domain Warping', generate_domain_warp_frame),
    'superformula': ('Superformula', generate_superformula_frame),
    'strange_attractor': ('Strange Attractor', generate_strange_attractor_frame),
    'lsystem': ('L-System Plants', generate_lsystem_frame),
    'boids': ('Boids Flocking', generate_boids_frame),
    'reaction_diffusion': ('Reaction-Diffusion', generate_reaction_diffusion_frame),
    'differential_growth': ('Differential Growth', generate_differential_growth_frame),
    # Shader-based generators
    'shader_plasma': ('Shader: Plasma', generate_shader_plasma_frame),
    'shader_tunnel': ('Shader: Tunnel Effect', generate_shader_tunnel_frame),
    'shader_raymarching': ('Shader: Raymarching', generate_shader_raymarching_frame),
    'shader_mandelbrot': ('Shader: Mandelbrot Fractal', generate_shader_mandelbrot_frame),
    'shader_julia': ('Shader: Julia Set', generate_shader_julia_frame),
    'shader_metaballs': ('Shader: Metaballs', generate_shader_metaballs_frame),
    'shader_rotozoomer': ('Shader: Rotozoomer', generate_shader_rotozoomer_frame),
    'shader_voronoi': ('Shader: Voronoi Noise', generate_shader_voronoi_frame),
    'shader_kaleidoscope': ('Shader: Kaleidoscope', generate_shader_kaleidoscope_frame),
    'shader_fire': ('Shader: Fire Effect', generate_shader_fire_frame),
    'shader_starfield': ('Shader: Starfield', generate_shader_starfield_frame),
    'shader_hexagons': ('Shader: Hexagonal Tiling', generate_shader_hexagons_frame),
    'shader_dna': ('Shader: DNA Helix', generate_shader_dna_frame),
    'shader_matrix': ('Shader: Matrix Rain', generate_shader_matrix_frame),
    'shader_waves': ('Shader: Wave Interference', generate_shader_waves_frame),
    'shader_clock': ('Shader: Clockwork Gears', generate_shader_clock_frame),
    'shader_caustics': ('Shader: Caustics', generate_shader_caustics_frame),
    'shader_truchet': ('Shader: Truchet Tiles', generate_shader_truchet_frame),
    'shader_aurora': ('Shader: Aurora', generate_shader_aurora_frame),
    'shader_moire': ('Shader: Moire Patterns', generate_shader_moire_frame),
    'shader_perlin_flow': ('Shader: Perlin Flow', generate_shader_perlin_flow_frame),
    'shader_spirograph': ('Shader: Spirograph', generate_shader_spirograph_frame),
    'shader_electric': ('Shader: Electric', generate_shader_electric_frame),
    'shader_glitch': ('Shader: Glitch', generate_shader_glitch_frame),
    # Retro & Pixel Glitch shaders
    'shader_crt': ('Shader: CRT Screen', generate_shader_crt_frame),
    'shader_vhs': ('Shader: VHS Glitch', generate_shader_vhs_frame),
    'shader_pixelsort': ('Shader: Pixel Sort', generate_shader_pixelsort_frame),
    'shader_rgbsplit': ('Shader: RGB Split', generate_shader_rgbsplit_frame),
    'shader_dither': ('Shader: Dither', generate_shader_dither_frame),
    'shader_teletext': ('Shader: Teletext', generate_shader_teletext_frame),
    'shader_c64': ('Shader: C64', generate_shader_c64_frame),
    'shader_gameboy': ('Shader: Game Boy', generate_shader_gameboy_frame),
    # Additional shaders
    'shader_feedback': ('Shader: Feedback Loop', generate_shader_feedback_frame),
    'shader_lava': ('Shader: Lava Lamp', generate_shader_lava_frame),
    'shader_nebula': ('Shader: Nebula', generate_shader_nebula_frame),
    'shader_circuit': ('Shader: Circuit Board', generate_shader_circuit_frame),
    'shader_warp': ('Shader: Warp Speed', generate_shader_warp_frame),
    'shader_liquid_crystal': ('Shader: Liquid Crystal', generate_shader_liquid_crystal_frame),
    'shader_fractal_flame': ('Shader: Fractal Flame', generate_shader_fractal_flame_frame),
    'shader_oscilloscope': ('Shader: Oscilloscope', generate_shader_oscilloscope_frame),
    # Sprite Character generators
    'walking_character': ('Walking Character', generate_walking_character_frame),
    'jumping_character': ('Jumping Character', generate_jumping_character_frame),
    'dancing_character': ('Dancing Character', generate_dancing_character_frame),
    'flying_character': ('Flying Character', generate_flying_character_frame),
    'running_character': ('Running Character', generate_running_character_frame),
    # Isometric & Voxel generators
    'isometric_cubes': ('Isometric Cubes', generate_isometric_cubes_frame),
    'voxel_terrain': ('Voxel Terrain', generate_voxel_terrain_frame),
    'isometric_city': ('Isometric City', generate_isometric_city_frame),
    'voxel_waves': ('Voxel Waves', generate_voxel_waves_frame),
    'isometric_stairs': ('Isometric Stairs', generate_isometric_stairs_frame),
}

# ============================================================================
# VIDEO RENDERING
# ============================================================================

def render_video(generator_func, output_path, width=WIDTH, height=HEIGHT, fps=FPS, total_frames=TOTAL_FRAMES, random_seed=None):
    """
    Render a looping video using the given generator function.

    Args:
        random_seed: Seed for randomization. If None, uses current time for unique seed.
    """
    # Set random seed for this video
    if random_seed is None:
        import time
        random_seed = int(time.time() * 1000) % 1000000

    set_random_seed(random_seed)

    # Create randomized palette
    randomized_palette = randomize_palette(PALETTE, hue_shift_range=15)

    print(f"Rendering video: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    print(f"Random seed: {random_seed}")

    frames = []

    # Generate frames
    for frame_num in tqdm(range(total_frames), desc="Generating frames"):
        # Calculate time parameter (0 to 1, loops seamlessly)
        t = frame_num / total_frames

        # Generate frame with randomized palette
        img = generator_func(width, height, randomized_palette['colors'], t)

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

    # Shuffle the generator order for varied processing
    generator_items = list(GENERATORS.items())
    random.shuffle(generator_items)

    print("Generator order (shuffled):")
    for idx, (gen_id, (gen_name, _)) in enumerate(generator_items, 1):
        print(f"  {idx:2d}. {gen_name}")
    print()

    for idx, (gen_id, (gen_name, gen_func)) in enumerate(generator_items, 1):
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
