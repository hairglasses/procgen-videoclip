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

    num_spirals = 8
    random.seed(42)

    for spiral_idx in range(num_spirals):
        points = []
        num_points = 200
        max_radius = 300 * expansion

        phase_offset = (spiral_idx / num_spirals) * 2 * math.pi

        for i in range(num_points):
            progress = i / num_points
            radius = progress * max_radius
            angle = progress * 4 * math.pi + phase_offset + t * 2 * math.pi

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
    Retro pixel sprites with animated color cycling.
    t: time parameter from 0 to 1 (loops seamlessly)
    """
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate multiple pixel sprites
    sprite_size = 40
    num_sprites_x = gen_width // (sprite_size * 2)
    num_sprites_y = gen_height // (sprite_size * 2)

    # Animated color shift
    color_shift = t

    random.seed(42)  # Consistent sprite shapes

    for sy in range(num_sprites_y):
        for sx in range(num_sprites_x):
            # Random sprite template
            sprite_half_width = sprite_size // 2
            sprite_data = [[random.random() > 0.6 for _ in range(sprite_half_width)]
                          for _ in range(sprite_size)]

            # Position sprite
            base_x = sx * sprite_size * 2 + sprite_size // 2
            base_y = sy * sprite_size * 2 + sprite_size // 2

            # Animated sprite color
            value = (random.random() + color_shift) % 1.0
            sprite_color = get_color_from_palette(value, palette)

            # Draw sprite with horizontal mirroring (classic sprite style)
            for py in range(sprite_size):
                for px in range(sprite_half_width):
                    if sprite_data[py][px]:
                        # Left half
                        x1 = base_x + px
                        y1 = base_y + py
                        if 0 <= x1 < gen_width and 0 <= y1 < gen_height:
                            pixels[x1, y1] = sprite_color
                            if mirror_h:
                                pixels[width - 1 - x1, y1] = sprite_color
                            if mirror_v:
                                pixels[x1, height - 1 - y1] = sprite_color
                            if mirror_h and mirror_v:
                                pixels[width - 1 - x1, height - 1 - y1] = sprite_color

                        # Right half (mirror within sprite)
                        x2 = base_x + sprite_size - 1 - px
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

    for y in range(gen_height):
        for x in range(gen_width):
            # First level of warping
            q_x = perlin2D(x * scale + time_offset, y * scale, octaves=4, persistence=0.5)
            q_y = perlin2D(x * scale, y * scale + time_offset, octaves=4, persistence=0.5)

            # Second level - warp the coordinates with the first warp
            r_x = perlin2D((x + q_x * 50) * scale, (y + q_y * 50) * scale, octaves=4, persistence=0.5)
            r_y = perlin2D((x + q_x * 50) * scale + 5.2, (y + q_y * 50) * scale + 1.3, octaves=4, persistence=0.5)

            # Final value using double-warped coordinates
            value = perlin2D((x + r_x * 50) * scale, (y + r_y * 50) * scale, octaves=6, persistence=0.5)
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
