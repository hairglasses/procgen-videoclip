#!/usr/bin/env python3
"""
Generate multiple procedurally generated macOS wallpapers with various techniques.
"""

from PIL import Image, ImageDraw
from procgen.noise import perlin2D, simplex2D, combined
import random
import math
import os

# Wallpaper dimensions (macOS Retina display)
WIDTH = 2880
HEIGHT = 1800

# Custom color palette
def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

# Define the color scheme
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
# WALLPAPER GENERATION TECHNIQUES
# ============================================================================

def generate_layered_noise(width, height, palette, mirror_h=True, mirror_v=True):
    """Original layered Perlin/Simplex noise technique."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()
    scale = 0.002

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    for y in range(gen_height):
        for x in range(gen_width):
            noise1 = combined(perlin2D, x * scale, y * scale, octaves=6, persistence=0.5)
            noise2 = combined(simplex2D, x * scale * 1.5, y * scale * 1.5, octaves=4, persistence=0.6)
            noise3 = combined(perlin2D, x * scale * 0.3, y * scale * 0.3, octaves=3, persistence=0.7)

            combined_noise = (noise1 * 0.5 + noise2 * 0.3 + noise3 * 0.2)
            t = (combined_noise + 1) / 2
            color = get_color_from_palette(t, palette)

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

def generate_voronoi(width, height, palette, mirror_h=True, mirror_v=True):
    """Voronoi diagram / cellular pattern."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate random points in the generation region
    num_points = 50
    points = [(random.randint(0, gen_width), random.randint(0, gen_height)) for _ in range(num_points)]
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
            t = min(1.0, math.sqrt(min_dist) / 200 + (noise * 0.3))

            cell_color = point_colors[closest_idx]
            color = lerp_color(cell_color, PALETTE['background'], t * 0.5)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_flow_field(width, height, palette, mirror_h=True, mirror_v=True):
    """Perlin flow field with particle-like patterns."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    scale = 0.003
    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height
    num_lines = 3000
    line_length = 200

    for _ in range(num_lines):
        x = random.randint(0, gen_width)
        y = random.randint(0, gen_height)

        points = []
        for step in range(line_length):
            if 0 <= x < gen_width and 0 <= y < gen_height:
                points.append((x, y))

                # Get flow direction from noise
                angle = perlin2D(x * scale, y * scale) * math.pi * 2
                speed = (simplex2D(x * scale * 0.5, y * scale * 0.5) + 1) * 2

                x += math.cos(angle) * speed
                y += math.sin(angle) * speed
            else:
                break

        if len(points) > 1:
            # Color based on starting position
            t = (perlin2D(points[0][0] * 0.001, points[0][1] * 0.001) + 1) / 2
            color = get_color_from_palette(t, palette)
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

def generate_interference(width, height, palette, mirror_h=True, mirror_v=True):
    """Wave interference patterns."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate wave sources in the generation region
    num_sources = 8
    sources = []
    for _ in range(num_sources):
        sources.append({
            'x': random.randint(0, gen_width),
            'y': random.randint(0, gen_height),
            'frequency': random.uniform(0.002, 0.008),
            'phase': random.uniform(0, math.pi * 2)
        })

    for y in range(gen_height):
        for x in range(gen_width):
            wave_sum = 0

            for source in sources:
                dx = x - source['x']
                dy = y - source['y']
                dist = math.sqrt(dx * dx + dy * dy)

                wave = math.sin(dist * source['frequency'] + source['phase'])
                wave_sum += wave

            # Normalize and map to palette
            t = (wave_sum / len(sources) + 1) / 2

            # Add some noise for texture
            noise = perlin2D(x * 0.002, y * 0.002) * 0.2
            t = max(0, min(1, t + noise))

            color = get_color_from_palette(t, palette)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_fractal_noise(width, height, palette, mirror_h=True, mirror_v=True):
    """Fractal Brownian Motion with multiple octaves."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    scale = 0.0015

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    for y in range(gen_height):
        for x in range(gen_width):
            # High octave fractal for detailed patterns
            noise = combined(simplex2D, x * scale, y * scale, octaves=8, persistence=0.6, lacunarity=2.5)

            # Add ridged effect
            noise = abs(noise)
            noise = 1 - noise

            # Map to palette
            t = noise
            color = get_color_from_palette(t, palette)

            # Subtle darkening in corners
            center_x = width / 2
            center_y = height / 2
            dist_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = math.sqrt(center_x**2 + center_y**2)
            vignette = 1 - (dist_from_center / max_dist) * 0.3

            color = tuple(int(c * vignette) for c in color)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_cellular_automata(width, height, palette, mirror_h=True, mirror_v=True):
    """Cellular automata-inspired pattern."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Initialize grid with random noise
    grid = []
    for y in range(gen_height):
        row = []
        for x in range(gen_width):
            noise = perlin2D(x * 0.01, y * 0.01)
            row.append(1 if noise > 0 else 0)
        grid.append(row)

    # Apply simplified cellular automata rules (life-like)
    for iteration in range(3):
        new_grid = [[0] * gen_width for _ in range(gen_height)]
        for y in range(1, gen_height - 1):
            for x in range(1, gen_width - 1):
                # Count neighbors
                neighbors = sum([
                    grid[y-1][x-1], grid[y-1][x], grid[y-1][x+1],
                    grid[y][x-1], grid[y][x+1],
                    grid[y+1][x-1], grid[y+1][x], grid[y+1][x+1]
                ])

                # Simple rule: survive with 2-3 neighbors, born with 3
                if grid[y][x] == 1:
                    new_grid[y][x] = 1 if neighbors in [2, 3] else 0
                else:
                    new_grid[y][x] = 1 if neighbors == 3 else 0

        grid = new_grid

    # Render with smooth color transitions
    for y in range(gen_height):
        for x in range(gen_width):
            base_value = grid[y][x]
            noise = (perlin2D(x * 0.003, y * 0.003) + 1) / 2
            t = base_value * 0.7 + noise * 0.3

            color = get_color_from_palette(t, palette)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_plotter_art(width, height, palette, mirror_h=True, mirror_v=True):
    """Plotter-style geometric art inspired by vsketch."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Create concentric geometric shapes with rotation
    num_layers = 20
    center_x = gen_width // 2
    center_y = gen_height // 2

    for i in range(num_layers):
        radius = (i + 1) * 50 + random.randint(-10, 10)
        sides = random.choice([3, 4, 5, 6, 8])
        rotation = i * 0.1 + random.uniform(0, 0.3)

        # Calculate polygon points
        points = []
        for s in range(sides):
            angle = (2 * math.pi * s / sides) + rotation
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            points.append((x, y))

        # Close the polygon
        points.append(points[0])

        # Select color from palette
        t = i / num_layers
        color = get_color_from_palette(t, palette)
        alpha = random.randint(30, 80)

        # Draw the polygon
        draw.line(points, fill=color + (alpha,), width=2)

        # Mirror if needed
        if mirror_h or mirror_v:
            mirrored_points = []
            for x, y in points:
                mx = (width - 1 - x) if mirror_h else x
                my = (height - 1 - y) if mirror_v else y
                mirrored_points.append((mx, my))
            draw.line(mirrored_points, fill=color + (alpha,), width=2)

    # Add some random lines for variation
    for _ in range(100):
        x1 = random.randint(0, gen_width)
        y1 = random.randint(0, gen_height)
        length = random.randint(50, 200)
        angle = random.uniform(0, math.pi * 2)
        x2 = x1 + length * math.cos(angle)
        y2 = y1 + length * math.sin(angle)

        t = random.random()
        color = get_color_from_palette(t, palette)
        alpha = 20

        draw.line([(x1, y1), (x2, y2)], fill=color + (alpha,), width=1)

        if mirror_h:
            draw.line([(width - 1 - x1, y1), (width - 1 - x2, y2)], fill=color + (alpha,), width=1)
        if mirror_v:
            draw.line([(x1, height - 1 - y1), (x2, height - 1 - y2)], fill=color + (alpha,), width=1)
        if mirror_h and mirror_v:
            draw.line([(width - 1 - x1, height - 1 - y1), (width - 1 - x2, height - 1 - y2)], fill=color + (alpha,), width=1)

    return img

def generate_differential_growth(width, height, palette, mirror_h=True, mirror_v=True):
    """Organic growth patterns inspired by differential-line."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Initialize circular path of nodes
    num_initial_nodes = 30
    center_x = gen_width // 2
    center_y = gen_height // 2
    initial_radius = 100

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    nodes = []
    for i in range(num_initial_nodes):
        angle = 2 * math.pi * i / num_initial_nodes
        x = center_x + initial_radius * math.cos(angle)
        y = center_y + initial_radius * math.sin(angle)
        nodes.append(Node(x, y))

    # Simulate differential growth
    max_distance = 15
    min_distance = 5
    iterations = 50

    for iteration in range(iterations):
        # Move nodes based on neighbors
        for i, node in enumerate(nodes):
            prev_node = nodes[(i - 1) % len(nodes)]
            next_node = nodes[(i + 1) % len(nodes)]

            # Calculate forces
            dx_prev = node.x - prev_node.x
            dy_prev = node.y - prev_node.y
            dist_prev = math.sqrt(dx_prev**2 + dy_prev**2) or 1

            dx_next = node.x - next_node.x
            dy_next = node.y - next_node.y
            dist_next = math.sqrt(dx_next**2 + dy_next**2) or 1

            # Apply growth force
            growth = 0.5
            noise_val = perlin2D(node.x * 0.01, node.y * 0.01) * growth
            node.x += noise_val * math.cos(iteration * 0.1)
            node.y += noise_val * math.sin(iteration * 0.1)

            # Keep within bounds
            node.x = max(10, min(gen_width - 10, node.x))
            node.y = max(10, min(gen_height - 10, node.y))

        # Split edges that are too long
        new_nodes = []
        i = 0
        while i < len(nodes):
            current = nodes[i]
            next_node = nodes[(i + 1) % len(nodes)]

            dx = current.x - next_node.x
            dy = current.y - next_node.y
            dist = math.sqrt(dx**2 + dy**2)

            new_nodes.append(current)

            if dist > max_distance:
                # Insert new node in the middle
                mid_x = (current.x + next_node.x) / 2
                mid_y = (current.y + next_node.y) / 2
                new_nodes.append(Node(mid_x, mid_y))

            i += 1

        nodes = new_nodes

    # Draw the organic shape
    points = [(node.x, node.y) for node in nodes]
    if len(points) > 2:
        points.append(points[0])  # Close the loop

        # Draw filled polygon with gradient
        for i in range(len(points) - 1):
            t = i / len(points)
            color = get_color_from_palette(t, palette)
            draw.line([points[i], points[i + 1]], fill=color + (150,), width=3)

            # Mirror the growth pattern
            if mirror_h:
                p1_m = (width - 1 - points[i][0], points[i][1])
                p2_m = (width - 1 - points[i + 1][0], points[i + 1][1])
                draw.line([p1_m, p2_m], fill=color + (150,), width=3)

            if mirror_v:
                p1_m = (points[i][0], height - 1 - points[i][1])
                p2_m = (points[i + 1][0], height - 1 - points[i + 1][1])
                draw.line([p1_m, p2_m], fill=color + (150,), width=3)

            if mirror_h and mirror_v:
                p1_m = (width - 1 - points[i][0], height - 1 - points[i][1])
                p2_m = (width - 1 - points[i + 1][0], height - 1 - points[i + 1][1])
                draw.line([p1_m, p2_m], fill=color + (150,), width=3)

    return img

def generate_penrose_tiling(width, height, palette, mirror_h=True, mirror_v=True):
    """Penrose tiling with aperiodic patterns."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Simplified Penrose-inspired pattern using golden ratio
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    scale = 100

    for y in range(gen_height):
        for x in range(gen_width):
            # Create pseudo-Penrose pattern using overlapping grids at golden ratio angles
            sum_val = 0
            for i in range(5):
                angle = i * 2 * math.pi / 5
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)

                # Project point onto rotated grid
                proj = (x * cos_a + y * sin_a) / scale

                # Add wave pattern
                sum_val += math.sin(proj * phi) * math.cos(proj / phi)

            # Normalize and add noise
            t = (sum_val / 5 + 1) / 2
            noise = perlin2D(x * 0.002, y * 0.002) * 0.3
            t = max(0, min(1, t + noise))

            color = get_color_from_palette(t, palette)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_bezier_curves(width, height, palette, mirror_h=True, mirror_v=True):
    """Organic shapes with Bezier curves inspired by DeepSVG."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate smooth organic shapes using quadratic Bezier approximations
    num_shapes = 15

    for shape_idx in range(num_shapes):
        # Random start position in generation region
        start_x = random.randint(100, gen_width - 100)
        start_y = random.randint(100, gen_height - 100)

        # Create organic curved path
        points = []
        num_segments = 20
        angle = random.uniform(0, math.pi * 2)

        for i in range(num_segments):
            # Use noise to create smooth curves
            noise_x = perlin2D(i * 0.3 + shape_idx, 0) * 50
            noise_y = perlin2D(0, i * 0.3 + shape_idx) * 50

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
            t = shape_idx / num_shapes
            color = get_color_from_palette(t, palette)
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

def generate_physarum(width, height, palette, mirror_h=True, mirror_v=True):
    """Slime mold simulation creating organic network patterns."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Initialize particle agents
    num_particles = 5000
    particles = []

    class Particle:
        def __init__(self):
            self.x = random.randint(0, gen_width - 1)
            self.y = random.randint(0, gen_height - 1)
            self.angle = random.uniform(0, math.pi * 2)

    for _ in range(num_particles):
        particles.append(Particle())

    # Create trail map
    trail = [[0.0 for _ in range(gen_width)] for _ in range(gen_height)]

    # Simulate slime mold growth
    iterations = 100
    sensor_angle = 0.4
    sensor_distance = 9
    turn_angle = 0.4

    for iteration in range(iterations):
        for particle in particles:
            # Sense in three directions
            x, y, angle = particle.x, particle.y, particle.angle

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
            trail[py][px] = min(1.0, trail[py][px] + 0.1)

        # Diffuse and decay trails
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
            t = trail[y][x]
            color = get_color_from_palette(t, palette)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_pixel_sprites(width, height, palette, mirror_h=True, mirror_v=True):
    """Pixel sprite generator creating retro game-style sprites."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate multiple pixel sprites across the wallpaper
    sprite_size = 40
    num_sprites_x = gen_width // (sprite_size * 2)
    num_sprites_y = gen_height // (sprite_size * 2)

    for sy in range(num_sprites_y):
        for sx in range(num_sprites_x):
            # Random sprite template
            sprite_half_width = sprite_size // 2
            sprite_data = [[random.random() > 0.6 for _ in range(sprite_half_width)]
                          for _ in range(sprite_size)]

            # Position sprite
            base_x = sx * sprite_size * 2 + sprite_size // 2
            base_y = sy * sprite_size * 2 + sprite_size // 2

            # Choose sprite color
            t = random.random()
            sprite_color = get_color_from_palette(t, palette)

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

def generate_wave_function_collapse(width, height, palette, mirror_h=True, mirror_v=True):
    """Wave function collapse-inspired pattern generation."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Tile size for WFC-like generation
    tile_size = 20
    tiles_x = gen_width // tile_size
    tiles_y = gen_height // tile_size

    # Create simple tile templates (simplified WFC)
    num_tile_types = 8
    tile_patterns = []

    for t in range(num_tile_types):
        pattern = []
        for ty in range(tile_size):
            row = []
            for tx in range(tile_size):
                # Create pattern based on tile type
                value = (math.sin(tx * 0.5 + t) + math.cos(ty * 0.5 + t)) / 2
                row.append(value)
            pattern.append(row)
        tile_patterns.append(pattern)

    # Place tiles with constraints
    tile_grid = [[random.randint(0, num_tile_types - 1) for _ in range(tiles_x)]
                 for _ in range(tiles_y)]

    # Render tiles
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            tile_type = tile_grid[ty][tx]
            pattern = tile_patterns[tile_type]

            for py in range(tile_size):
                for px in range(tile_size):
                    x = tx * tile_size + px
                    y = ty * tile_size + py

                    if x < gen_width and y < gen_height:
                        value = (pattern[py][px] + 1) / 2
                        # Add noise for variation
                        noise = perlin2D(x * 0.01, y * 0.01) * 0.2
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

def generate_pixel_art_dithering(width, height, palette, mirror_h=True, mirror_v=True):
    """Pixel art style with dithering patterns."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Dithering patterns (Bayer matrix inspired)
    dither_matrix = [
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ]
    dither_size = 4

    # Reduced color palette for authentic pixel art feel
    pixel_palette = palette[::2]  # Use every other color for more distinct look

    for y in range(gen_height):
        for x in range(gen_width):
            # Generate base value from noise
            noise = combined(perlin2D, x * 0.003, y * 0.003, octaves=4, persistence=0.5)
            base_value = (noise + 1) / 2

            # Apply dithering
            threshold = dither_matrix[y % dither_size][x % dither_size] / 16.0
            dithered_value = base_value + (threshold - 0.5) * 0.3

            # Posterize to limited colors
            color_index = int(dithered_value * (len(pixel_palette) - 1))
            color_index = max(0, min(len(pixel_palette) - 1, color_index))
            color = pixel_palette[color_index]

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_sprite_characters(width, height, palette, mirror_h=True, mirror_v=True):
    """Procedural pixel character sprites."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate robot/character sprites
    sprite_size = 60
    num_sprites_x = gen_width // (sprite_size + 20)
    num_sprites_y = gen_height // (sprite_size + 20)

    for sy in range(num_sprites_y):
        for sx in range(num_sprites_x):
            base_x = sx * (sprite_size + 20) + 10
            base_y = sy * (sprite_size + 20) + 10

            # Choose character color
            t = random.random()
            char_color = get_color_from_palette(t, palette)
            dark_color = tuple(int(c * 0.7) for c in char_color)

            # Body (rectangle)
            body_width = sprite_size // 2
            body_height = sprite_size // 2
            body_x = base_x + sprite_size // 4
            body_y = base_y + sprite_size // 4

            # Draw body components
            draw.rectangle([body_x, body_y, body_x + body_width, body_y + body_height],
                          fill=char_color, outline=dark_color)

            # Head (smaller rectangle on top)
            head_size = sprite_size // 3
            head_x = body_x + (body_width - head_size) // 2
            head_y = body_y - head_size
            draw.rectangle([head_x, head_y, head_x + head_size, head_y + head_size],
                          fill=char_color, outline=dark_color)

            # Eyes
            eye_size = 4
            eye_color = PALETTE['colors'][0]
            draw.rectangle([head_x + 5, head_y + 8, head_x + 5 + eye_size, head_y + 8 + eye_size],
                          fill=eye_color)
            draw.rectangle([head_x + head_size - 5 - eye_size, head_y + 8,
                          head_x + head_size - 5, head_y + 8 + eye_size], fill=eye_color)

            # Limbs (simple lines)
            draw.rectangle([body_x - 10, body_y + 10, body_x, body_y + 20], fill=dark_color)
            draw.rectangle([body_x + body_width, body_y + 10, body_x + body_width + 10, body_y + 20],
                          fill=dark_color)

    # Mirror the entire sprite grid if needed
    if mirror_h or mirror_v:
        # This is already handled by drawing in gen_width/gen_height region
        # Create mirrored copy
        region = img.crop((0, 0, gen_width, gen_height))

        if mirror_h:
            flipped_h = region.transpose(Image.FLIP_LEFT_RIGHT)
            img.paste(flipped_h, (gen_width, 0))

        if mirror_v:
            flipped_v = region.transpose(Image.FLIP_TOP_BOTTOM)
            img.paste(flipped_v, (0, gen_height))

        if mirror_h and mirror_v:
            flipped_both = region.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
            img.paste(flipped_both, (gen_width, gen_height))

    return img

def generate_template_pixel_art(width, height, palette, mirror_h=True, mirror_v=True):
    """Template-based procedural pixel art generation."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Create a randomized pattern template
    pattern_size = 30
    num_patterns_x = gen_width // pattern_size
    num_patterns_y = gen_height // pattern_size

    for py in range(num_patterns_y):
        for px in range(num_patterns_x):
            # Choose pattern variation
            pattern_type = random.randint(0, 3)

            # Choose colors for this pattern
            t1 = random.random()
            t2 = random.random()
            color1 = get_color_from_palette(t1, palette)
            color2 = get_color_from_palette(t2, palette)

            base_x = px * pattern_size
            base_y = py * pattern_size

            for ty in range(pattern_size):
                for tx in range(pattern_size):
                    x = base_x + tx
                    y = base_y + ty

                    if x >= gen_width or y >= gen_height:
                        continue

                    # Generate pattern based on type
                    if pattern_type == 0:
                        # Checkerboard
                        use_color1 = (tx // 5 + ty // 5) % 2 == 0
                    elif pattern_type == 1:
                        # Diagonal stripes
                        use_color1 = (tx + ty) % 10 < 5
                    elif pattern_type == 2:
                        # Concentric
                        dist = math.sqrt((tx - pattern_size/2)**2 + (ty - pattern_size/2)**2)
                        use_color1 = int(dist) % 8 < 4
                    else:
                        # Random noise
                        use_color1 = random.random() > 0.5

                    color = color1 if use_color1 else color2

                    # Set pixel and apply mirroring
                    pixels[x, y] = color
                    if mirror_h:
                        pixels[width - 1 - x, y] = color
                    if mirror_v:
                        pixels[x, height - 1 - y] = color
                    if mirror_h and mirror_v:
                        pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_raytraced_sdf(width, height, palette, mirror_h=True, mirror_v=True):
    """Raytraced Signed Distance Field geometry inspired by retrace.gl."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Define multiple SDF primitives (spheres and boxes)
    num_shapes = 8
    shapes = []
    for _ in range(num_shapes):
        shapes.append({
            'x': random.randint(0, gen_width),
            'y': random.randint(0, gen_height),
            'radius': random.randint(100, 300),
            'type': random.choice(['sphere', 'box']),
            'color_offset': random.random()
        })

    for y in range(gen_height):
        for x in range(gen_width):
            # Calculate distance to nearest shape (SDF)
            min_dist = float('inf')
            nearest_shape = None

            for shape in shapes:
                dx = x - shape['x']
                dy = y - shape['y']

                if shape['type'] == 'sphere':
                    # Sphere SDF
                    dist = math.sqrt(dx * dx + dy * dy) - shape['radius']
                else:
                    # Box SDF (approximation)
                    abs_dx = abs(dx)
                    abs_dy = abs(dy)
                    box_size = shape['radius']
                    dist = math.sqrt(max(0, abs_dx - box_size)**2 + max(0, abs_dy - box_size)**2)

                if abs(dist) < abs(min_dist):
                    min_dist = dist
                    nearest_shape = shape

            # Create CSG-like effect with smooth blending
            t = 1 / (1 + math.exp(-min_dist / 50))  # Smooth step function

            # Add raytraced-style shading
            if nearest_shape:
                base_t = nearest_shape['color_offset']
                # Simulate lighting based on distance gradient
                shade = max(0, min(1, t * 0.7 + 0.3))
                color = get_color_from_palette(base_t, palette)
                color = tuple(int(c * shade) for c in color)
            else:
                noise = perlin2D(x * 0.002, y * 0.002)
                t_noise = (noise + 1) / 2
                color = get_color_from_palette(t_noise, palette)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_pathtraced_terrain(width, height, palette, mirror_h=True, mirror_v=True):
    """Path-traced procedural terrain inspired by THREE.js-PathTracing-Renderer."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate terrain heightmap
    scale = 0.002

    for y in range(gen_height):
        for x in range(gen_width):
            # Multi-octave terrain noise
            elevation = combined(perlin2D, x * scale, y * scale, octaves=6, persistence=0.5)

            # Add ridges for mountainous features
            ridge = abs(simplex2D(x * scale * 0.5, y * scale * 0.5))
            elevation = elevation * 0.7 + ridge * 0.3

            # Normalize to 0-1
            elevation = (elevation + 1) / 2

            # Simulate atmospheric perspective (retro 80s style)
            distance_from_bottom = y / gen_height
            atmosphere = 1 - (distance_from_bottom * 0.4)

            # Map elevation to colors with atmospheric fade
            t = elevation * atmosphere
            color = get_color_from_palette(t, palette)

            # Add subtle scan-line effect for retro aesthetic
            if y % 3 == 0:
                color = tuple(int(c * 0.95) for c in color)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_voxel_structures(width, height, palette, mirror_h=True, mirror_v=True):
    """Voxel-based L-system structures inspired by voxgen."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Voxel/block size for retro blocky aesthetic
    voxel_size = 20

    # Generate L-system-inspired growth pattern
    # Simple branching structure
    class Voxel:
        def __init__(self, x, y, generation):
            self.x = x
            self.y = y
            self.generation = generation

    voxels = []

    # Start from center
    center_x = gen_width // 2
    center_y = gen_height // 2

    # Seed voxels
    stack = [(center_x, center_y, 0, 0)]  # x, y, generation, angle
    visited = set()

    max_generations = 6
    branch_probability = 0.7

    while stack and len(voxels) < 500:
        x, y, gen, angle = stack.pop()

        if (x, y) in visited or gen > max_generations:
            continue
        if x < 0 or x >= gen_width or y < 0 or y >= gen_height:
            continue

        visited.add((x, y))
        voxels.append(Voxel(x, y, gen))

        # L-system style branching
        if random.random() < branch_probability:
            # Forward
            stack.append((x + voxel_size * int(math.cos(angle)),
                         y + voxel_size * int(math.sin(angle)),
                         gen + 1, angle))
            # Branch left
            stack.append((x + voxel_size * int(math.cos(angle - 0.5)),
                         y + voxel_size * int(math.sin(angle - 0.5)),
                         gen + 1, angle - 0.5))
            # Branch right
            stack.append((x + voxel_size * int(math.cos(angle + 0.5)),
                         y + voxel_size * int(math.sin(angle + 0.5)),
                         gen + 1, angle + 0.5))

    # Draw voxels in blocky Minecraft style
    for voxel in voxels:
        # Color based on generation (age)
        t = voxel.generation / max_generations
        voxel_color = get_color_from_palette(t, palette)

        # Add slight variation for each voxel
        variation = random.uniform(0.9, 1.1)
        voxel_color = tuple(int(min(255, c * variation)) for c in voxel_color)

        # Draw blocky voxel
        for dy in range(voxel_size):
            for dx in range(voxel_size):
                px = voxel.x + dx
                py = voxel.y + dy

                if 0 <= px < gen_width and 0 <= py < gen_height:
                    # Add edge darkening for 3D block effect
                    edge_darken = 1.0
                    if dx == 0 or dy == 0:
                        edge_darken = 0.7
                    elif dx == voxel_size - 1 or dy == voxel_size - 1:
                        edge_darken = 0.8

                    color = tuple(int(c * edge_darken) for c in voxel_color)

                    pixels[px, py] = color
                    if mirror_h:
                        pixels[width - 1 - px, py] = color
                    if mirror_v:
                        pixels[px, height - 1 - py] = color
                    if mirror_h and mirror_v:
                        pixels[width - 1 - px, height - 1 - py] = color

    return img

def generate_isometric_pixel(width, height, palette, mirror_h=True, mirror_v=True):
    """Isometric pixel art patterns inspired by ProceduralPixelArt."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Isometric grid parameters (dimetric projection)
    # In isometric view: x goes right-down, y goes left-down, z goes up
    tile_width = 40
    tile_height = 20

    # Generate isometric grid of blocks
    grid_size_x = gen_width // tile_width
    grid_size_y = gen_height // tile_height

    for grid_y in range(grid_size_y):
        for grid_x in range(grid_size_x):
            # Determine block height using noise
            noise = perlin2D(grid_x * 0.2, grid_y * 0.2)
            block_height = int((noise + 1) * 3)  # 0-6 blocks high

            if block_height <= 0:
                continue

            # Calculate isometric screen position
            iso_x = (grid_x - grid_y) * (tile_width // 2)
            iso_y = (grid_x + grid_y) * (tile_height // 2)

            # Center the grid
            iso_x += gen_width // 2
            iso_y += 100

            # Color based on height
            t = block_height / 6
            block_color = get_color_from_palette(t, palette)
            dark_color = tuple(int(c * 0.6) for c in block_color)
            light_color = tuple(int(min(255, c * 1.2)) for c in block_color)

            # Draw isometric block (simplified)
            # Top face (lighter)
            for ty in range(tile_height // 2):
                for tx in range(tile_width):
                    px = iso_x + tx - block_height * 2
                    py = iso_y + ty - block_height * (tile_height // 2)

                    if 0 <= px < gen_width and 0 <= py < gen_height:
                        # Diamond shape for top face
                        if abs(tx - tile_width // 2) + abs(ty - tile_height // 4) * 2 < tile_width // 2:
                            pixels[px, py] = light_color
                            if mirror_h:
                                pixels[width - 1 - px, py] = light_color
                            if mirror_v:
                                pixels[px, height - 1 - py] = light_color
                            if mirror_h and mirror_v:
                                pixels[width - 1 - px, height - 1 - py] = light_color

            # Left face (medium)
            for h in range(block_height * (tile_height // 2)):
                for tx in range(tile_width // 2):
                    px = iso_x + tx - block_height * 2
                    py = iso_y + tile_height // 2 + h

                    if 0 <= px < gen_width and 0 <= py < gen_height:
                        pixels[px, py] = block_color
                        if mirror_h:
                            pixels[width - 1 - px, py] = block_color
                        if mirror_v:
                            pixels[px, height - 1 - py] = block_color
                        if mirror_h and mirror_v:
                            pixels[width - 1 - px, height - 1 - py] = block_color

            # Right face (darker)
            for h in range(block_height * (tile_height // 2)):
                for tx in range(tile_width // 2):
                    px = iso_x + tile_width // 2 + tx - block_height * 2
                    py = iso_y + tile_height // 2 + h

                    if 0 <= px < gen_width and 0 <= py < gen_height:
                        pixels[px, py] = dark_color
                        if mirror_h:
                            pixels[width - 1 - px, py] = dark_color
                        if mirror_v:
                            pixels[px, height - 1 - py] = dark_color
                        if mirror_h and mirror_v:
                            pixels[width - 1 - px, height - 1 - py] = dark_color

    return img

def generate_lowpoly_terrain(width, height, palette, mirror_h=True, mirror_v=True):
    """Low-poly terrain with elevation-based coloring inspired by THREE.Terrain."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img)

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate terrain grid
    grid_size = 30
    cols = gen_width // grid_size
    rows = gen_height // grid_size

    # Create heightmap
    heightmap = []
    scale = 0.1

    for row in range(rows + 1):
        height_row = []
        for col in range(cols + 1):
            # Use Diamond-Square-like noise
            elevation = combined(perlin2D, col * scale, row * scale, octaves=4, persistence=0.6)
            # Add some Simplex for variation
            elevation += simplex2D(col * scale * 2, row * scale * 2) * 0.3
            height_row.append(elevation)
        heightmap.append(height_row)

    # Draw low-poly triangles
    for row in range(rows):
        for col in range(cols):
            x = col * grid_size
            y = row * grid_size

            # Get corner elevations
            h1 = heightmap[row][col]
            h2 = heightmap[row][col + 1]
            h3 = heightmap[row + 1][col]
            h4 = heightmap[row + 1][col + 1]

            # Average elevation for color
            avg_elevation_1 = (h1 + h2 + h3) / 3
            avg_elevation_2 = (h2 + h3 + h4) / 3

            # Map to palette (elevation-based coloring)
            t1 = (avg_elevation_1 + 1) / 2
            t2 = (avg_elevation_2 + 1) / 2

            color1 = get_color_from_palette(t1, palette)
            color2 = get_color_from_palette(t2, palette)

            # First triangle (top-left)
            triangle1 = [(x, y), (x + grid_size, y), (x, y + grid_size)]
            draw.polygon(triangle1, fill=color1, outline=PALETTE['background'])

            if mirror_h:
                tri1_m = [(width - 1 - px, py) for px, py in triangle1]
                draw.polygon(tri1_m, fill=color1, outline=PALETTE['background'])
            if mirror_v:
                tri1_m = [(px, height - 1 - py) for px, py in triangle1]
                draw.polygon(tri1_m, fill=color1, outline=PALETTE['background'])
            if mirror_h and mirror_v:
                tri1_m = [(width - 1 - px, height - 1 - py) for px, py in triangle1]
                draw.polygon(tri1_m, fill=color1, outline=PALETTE['background'])

            # Second triangle (bottom-right)
            triangle2 = [(x + grid_size, y), (x + grid_size, y + grid_size), (x, y + grid_size)]
            draw.polygon(triangle2, fill=color2, outline=PALETTE['background'])

            if mirror_h:
                tri2_m = [(width - 1 - px, py) for px, py in triangle2]
                draw.polygon(tri2_m, fill=color2, outline=PALETTE['background'])
            if mirror_v:
                tri2_m = [(px, height - 1 - py) for px, py in triangle2]
                draw.polygon(tri2_m, fill=color2, outline=PALETTE['background'])
            if mirror_h and mirror_v:
                tri2_m = [(width - 1 - px, height - 1 - py) for px, py in triangle2]
                draw.polygon(tri2_m, fill=color2, outline=PALETTE['background'])

    return img

def generate_reaction_diffusion(width, height, palette, mirror_h=True, mirror_v=True):
    """Reaction-diffusion system creating organic patterns inspired by Ready."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Initialize concentration grids (Gray-Scott model)
    # A and B are two chemical concentrations
    grid_a = [[1.0 for _ in range(gen_width)] for _ in range(gen_height)]
    grid_b = [[0.0 for _ in range(gen_width)] for _ in range(gen_height)]

    # Seed some initial B concentration in random spots
    for _ in range(20):
        seed_x = random.randint(gen_width // 4, 3 * gen_width // 4)
        seed_y = random.randint(gen_height // 4, 3 * gen_height // 4)
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                nx, ny = seed_x + dx, seed_y + dy
                if 0 <= nx < gen_width and 0 <= ny < gen_height:
                    grid_b[ny][nx] = 1.0

    # Gray-Scott parameters for coral-like patterns
    feed_rate = 0.055
    kill_rate = 0.062
    diffusion_a = 1.0
    diffusion_b = 0.5
    dt = 1.0

    # Run simulation (reduced for performance)
    iterations = 50

    for iteration in range(iterations):
        new_a = [[0.0 for _ in range(gen_width)] for _ in range(gen_height)]
        new_b = [[0.0 for _ in range(gen_width)] for _ in range(gen_height)]

        for y in range(1, gen_height - 1):
            for x in range(1, gen_width - 1):
                a = grid_a[y][x]
                b = grid_b[y][x]

                # Laplacian (diffusion)
                laplacian_a = (
                    grid_a[y-1][x] + grid_a[y+1][x] +
                    grid_a[y][x-1] + grid_a[y][x+1] -
                    4 * a
                )
                laplacian_b = (
                    grid_b[y-1][x] + grid_b[y+1][x] +
                    grid_b[y][x-1] + grid_b[y][x+1] -
                    4 * b
                )

                # Reaction
                reaction = a * b * b

                # Update
                new_a[y][x] = a + (diffusion_a * laplacian_a - reaction + feed_rate * (1 - a)) * dt
                new_b[y][x] = b + (diffusion_b * laplacian_b + reaction - (kill_rate + feed_rate) * b) * dt

                # Clamp
                new_a[y][x] = max(0, min(1, new_a[y][x]))
                new_b[y][x] = max(0, min(1, new_b[y][x]))

        grid_a = new_a
        grid_b = new_b

    # Render
    for y in range(gen_height):
        for x in range(gen_width):
            # Map concentration B to color
            t = grid_b[y][x]
            color = get_color_from_palette(t, palette)

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_strange_attractor(width, height, palette, mirror_h=True, mirror_v=True):
    """Chaotic dynamical systems creating strange attractors inspired by dysts."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Choose attractor type randomly
    attractor_type = random.choice(['lorenz', 'rossler', 'aizawa', 'halvorsen'])

    # Initialize starting point
    x, y, z = 0.1, 0.1, 0.1
    dt = 0.01
    iterations = 20000  # Reduced for performance

    # Store trajectory points
    points_2d = []

    for _ in range(iterations):
        # Compute derivatives based on attractor type
        if attractor_type == 'lorenz':
            # Lorenz system
            sigma, rho, beta = 10.0, 28.0, 8.0/3.0
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
        elif attractor_type == 'rossler':
            # Rössler system
            a, b, c = 0.2, 0.2, 5.7
            dx = -y - z
            dy = x + a * y
            dz = b + z * (x - c)
        elif attractor_type == 'aizawa':
            # Aizawa attractor
            a, b, c, d, e, f = 0.95, 0.7, 0.6, 3.5, 0.25, 0.1
            dx = (z - b) * x - d * y
            dy = d * x + (z - b) * y
            dz = c + a * z - (z**3)/3 - (x**2 + y**2) * (1 + e * z) + f * z * x**3
        else:  # halvorsen
            # Halvorsen attractor
            a = 1.4
            dx = -a * x - 4 * y - 4 * z - y * y
            dy = -a * y - 4 * z - 4 * x - z * z
            dz = -a * z - 4 * x - 4 * y - x * x

        # Update position
        x += dx * dt
        y += dy * dt
        z += dz * dt

        # Project 3D to 2D (isometric-ish)
        screen_x = int(gen_width / 2 + x * 15 + y * 10)
        screen_y = int(gen_height / 2 + z * 15 - y * 5)

        if 0 <= screen_x < gen_width and 0 <= screen_y < gen_height:
            points_2d.append((screen_x, screen_y))

    # Draw the attractor with color gradient
    for i in range(len(points_2d) - 1):
        t = i / len(points_2d)
        color = get_color_from_palette(t, palette)
        alpha = 20

        x1, y1 = points_2d[i]
        x2, y2 = points_2d[i + 1]

        draw.line([(x1, y1), (x2, y2)], fill=color + (alpha,), width=1)

        # Mirror
        if mirror_h:
            draw.line([(width - 1 - x1, y1), (width - 1 - x2, y2)], fill=color + (alpha,), width=1)
        if mirror_v:
            draw.line([(x1, height - 1 - y1), (x2, height - 1 - y2)], fill=color + (alpha,), width=1)
        if mirror_h and mirror_v:
            draw.line([(width - 1 - x1, height - 1 - y1), (width - 1 - x2, height - 1 - y2)],
                     fill=color + (alpha,), width=1)

    return img

def generate_dla(width, height, palette, mirror_h=True, mirror_v=True):
    """Diffusion Limited Aggregation creating coral-like structures inspired by dla-gpu."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Grid to track stuck particles
    stuck = [[False for _ in range(gen_width)] for _ in range(gen_height)]

    # Seed center
    center_x = gen_width // 2
    center_y = gen_height // 2
    stuck[center_y][center_x] = True

    # Track when each particle stuck (for coloring)
    stuck_time = [[0 for _ in range(gen_width)] for _ in range(gen_height)]
    stuck_time[center_y][center_x] = 1

    num_particles = 3000  # Reduced for performance
    current_time = 1

    for particle_idx in range(num_particles):
        # Start particle at random edge
        edge = random.randint(0, 3)
        if edge == 0:  # top
            x, y = random.randint(0, gen_width - 1), 0
        elif edge == 1:  # right
            x, y = gen_width - 1, random.randint(0, gen_height - 1)
        elif edge == 2:  # bottom
            x, y = random.randint(0, gen_width - 1), gen_height - 1
        else:  # left
            x, y = 0, random.randint(0, gen_height - 1)

        # Random walk until it touches a stuck particle
        max_steps = 10000
        for step in range(max_steps):
            # Check neighbors for stuck particles
            adjacent_stuck = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < gen_width and 0 <= ny < gen_height:
                        if stuck[ny][nx]:
                            adjacent_stuck = True
                            break
                if adjacent_stuck:
                    break

            if adjacent_stuck:
                # Stick here
                stuck[y][x] = True
                current_time += 1
                stuck_time[y][x] = current_time
                break

            # Random walk
            dx = random.choice([-1, 0, 1])
            dy = random.choice([-1, 0, 1])
            x = max(0, min(gen_width - 1, x + dx))
            y = max(0, min(gen_height - 1, y + dy))

    # Render with time-based coloring
    max_time = current_time

    for y in range(gen_height):
        for x in range(gen_width):
            if stuck[y][x]:
                # Color based on when it stuck (creates growth rings effect)
                t = stuck_time[y][x] / max_time
                color = get_color_from_palette(t, palette)
            else:
                color = PALETTE['background']

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_neural_ca(width, height, palette, mirror_h=True, mirror_v=True):
    """Neural cellular automata with self-organizing patterns inspired by Growing-Neural-Cellular-Automata."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Initialize grid with multiple channels (simplified neural CA)
    # Channels: 0=alive, 1-3=color info
    grid = [[[0.0 for _ in range(4)] for _ in range(gen_width)] for _ in range(gen_height)]

    # Seed center with initial pattern
    center_x = gen_width // 2
    center_y = gen_height // 2
    seed_size = 20

    for dy in range(-seed_size, seed_size):
        for dx in range(-seed_size, seed_size):
            if dx*dx + dy*dy < seed_size*seed_size:
                x, y = center_x + dx, center_y + dy
                if 0 <= x < gen_width and 0 <= y < gen_height:
                    grid[y][x][0] = 1.0  # alive
                    grid[y][x][1] = random.random()
                    grid[y][x][2] = random.random()
                    grid[y][x][3] = random.random()

    # Simulate growth (simplified neural rules)
    iterations = 100

    for iteration in range(iterations):
        new_grid = [[[0.0 for _ in range(4)] for _ in range(gen_width)] for _ in range(gen_height)]

        for y in range(1, gen_height - 1):
            for x in range(1, gen_width - 1):
                # Count alive neighbors
                alive_neighbors = 0
                avg_color = [0, 0, 0]

                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if grid[ny][nx][0] > 0.5:
                            alive_neighbors += 1
                            for c in range(3):
                                avg_color[c] += grid[ny][nx][c + 1]

                if alive_neighbors > 0:
                    for c in range(3):
                        avg_color[c] /= alive_neighbors

                # Neural-like growth rule
                current_alive = grid[y][x][0]

                # Grow if 2-3 alive neighbors
                if alive_neighbors >= 2 and alive_neighbors <= 3:
                    new_grid[y][x][0] = min(1.0, current_alive + 0.1)
                    # Inherit color with mutation
                    for c in range(3):
                        mutation = (perlin2D(x * 0.01 + iteration * 0.1, y * 0.01) + 1) / 2
                        new_grid[y][x][c + 1] = avg_color[c] * 0.9 + mutation * 0.1
                elif current_alive > 0.5:
                    # Decay
                    new_grid[y][x][0] = max(0.0, current_alive - 0.05)
                    for c in range(3):
                        new_grid[y][x][c + 1] = grid[y][x][c + 1]

        grid = new_grid

    # Render
    for y in range(gen_height):
        for x in range(gen_width):
            if grid[y][x][0] > 0.1:
                # Use the evolved color values
                t = (grid[y][x][1] + grid[y][x][2] + grid[y][x][3]) / 3
                color = get_color_from_palette(t, palette)

                # Modulate by aliveness
                brightness = grid[y][x][0]
                color = tuple(int(c * brightness) for c in color)
            else:
                color = PALETTE['background']

            # Set pixel and apply mirroring
            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_space_colonization(width, height, palette, mirror_h=True, mirror_v=True):
    """Space colonization algorithm creating venation patterns inspired by morphogenesis-resources."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Attraction points (where the tree wants to grow)
    num_attractors = 500
    attractors = []
    for _ in range(num_attractors):
        # Cluster attractors in interesting regions
        cluster_x = random.randint(gen_width // 4, 3 * gen_width // 4)
        cluster_y = random.randint(gen_height // 4, 3 * gen_height // 4)
        x = cluster_x + random.randint(-100, 100)
        y = cluster_y + random.randint(-100, 100)
        if 0 <= x < gen_width and 0 <= y < gen_height:
            attractors.append([x, y, True])  # x, y, active

    # Tree nodes
    class Node:
        def __init__(self, x, y, parent=None):
            self.x = x
            self.y = y
            self.parent = parent
            self.children = []

    # Start from bottom center
    root = Node(gen_width // 2, gen_height - 100)
    nodes = [root]

    # Parameters
    influence_distance = 150
    kill_distance = 10
    segment_length = 5
    max_iterations = 100

    for iteration in range(max_iterations):
        # For each node, find influencing attractors
        influences = {}

        for attractor in attractors:
            if not attractor[2]:  # not active
                continue

            ax, ay = attractor[0], attractor[1]
            closest_node = None
            closest_dist = float('inf')

            for node in nodes:
                dx = ax - node.x
                dy = ay - node.y
                dist = math.sqrt(dx*dx + dy*dy)

                if dist < kill_distance:
                    attractor[2] = False  # deactivate
                    break

                if dist < influence_distance and dist < closest_dist:
                    closest_dist = dist
                    closest_node = node

            if closest_node and attractor[2]:
                if closest_node not in influences:
                    influences[closest_node] = []
                influences[closest_node].append((ax, ay))

        if not influences:
            break

        # Grow new nodes
        new_nodes = []
        for node, attractors_list in influences.items():
            # Average direction to attractors
            avg_dx = 0
            avg_dy = 0
            for ax, ay in attractors_list:
                dx = ax - node.x
                dy = ay - node.y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    avg_dx += dx / dist
                    avg_dy += dy / dist

            if len(attractors_list) > 0:
                avg_dx /= len(attractors_list)
                avg_dy /= len(attractors_list)

                # Normalize and scale
                mag = math.sqrt(avg_dx*avg_dx + avg_dy*avg_dy)
                if mag > 0:
                    avg_dx = (avg_dx / mag) * segment_length
                    avg_dy = (avg_dy / mag) * segment_length

                    new_x = node.x + avg_dx
                    new_y = node.y + avg_dy

                    if 0 <= new_x < gen_width and 0 <= new_y < gen_height:
                        new_node = Node(new_x, new_y, node)
                        node.children.append(new_node)
                        new_nodes.append(new_node)

        nodes.extend(new_nodes)

    # Draw the tree structure
    def draw_node(node, depth=0):
        if node.parent:
            t = depth / 50  # color based on depth
            color = get_color_from_palette(t % 1.0, palette)
            alpha = 150

            x1, y1 = int(node.parent.x), int(node.parent.y)
            x2, y2 = int(node.x), int(node.y)

            draw.line([(x1, y1), (x2, y2)], fill=color + (alpha,), width=2)

            # Mirror
            if mirror_h:
                draw.line([(width - 1 - x1, y1), (width - 1 - x2, y2)], fill=color + (alpha,), width=2)
            if mirror_v:
                draw.line([(x1, height - 1 - y1), (x2, height - 1 - y2)], fill=color + (alpha,), width=2)
            if mirror_h and mirror_v:
                draw.line([(width - 1 - x1, height - 1 - y1), (width - 1 - x2, height - 1 - y2)],
                         fill=color + (alpha,), width=2)

        for child in node.children:
            draw_node(child, depth + 1)

    draw_node(root)

    return img

def generate_isometric_voxel_art(width, height, palette, mirror_h=True, mirror_v=True):
    """Isometric voxel rendering with multiple angles inspired by IsoVoxel."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Define voxel grid (3D)
    grid_size = 20
    voxel_grid = {}

    # Procedurally generate voxels
    for z in range(grid_size):
        for y in range(grid_size):
            for x in range(grid_size):
                # Use 3D noise to determine if voxel exists
                noise_val = perlin2D(x * 0.2, y * 0.2) + simplex2D(z * 0.2, (x + y) * 0.1)
                if noise_val > 0.3:
                    voxel_grid[(x, y, z)] = True

    # Isometric projection parameters
    # Standard isometric angle: 30 degrees
    iso_scale = 10

    def iso_project(x, y, z):
        """Project 3D coordinates to 2D isometric view."""
        screen_x = (x - y) * iso_scale
        screen_y = (x + y) * iso_scale * 0.5 - z * iso_scale
        return screen_x, screen_y

    # Center offset
    center_x = gen_width // 2
    center_y = gen_height // 2

    # Draw voxels in back-to-front order for proper occlusion
    # Sort by distance from camera
    sorted_coords = sorted(voxel_grid.keys(), key=lambda coord: coord[0] + coord[1] - coord[2])

    for (vx, vy, vz) in sorted_coords:
        sx, sy = iso_project(vx, vy, vz)
        sx += center_x
        sy += center_y

        # Determine color based on height
        t = vz / grid_size
        voxel_color = get_color_from_palette(t, palette)

        # Draw the isometric cube faces (top, left, right)
        # Top face
        top_points = [
            (sx, sy),
            (sx + iso_scale, sy + iso_scale * 0.5),
            (sx, sy + iso_scale),
            (sx - iso_scale, sy + iso_scale * 0.5)
        ]

        # Light color for top
        light_color = tuple(int(min(255, c * 1.2)) for c in voxel_color)

        for point in top_points:
            px, py = int(point[0]), int(point[1])
            # Fill with simple scan
            for dy in range(-iso_scale, iso_scale):
                for dx in range(-iso_scale, iso_scale):
                    test_x, test_y = px + dx, py + dy
                    if 0 <= test_x < gen_width and 0 <= test_y < gen_height:
                        # Check if inside diamond
                        if abs(dx) + abs(dy * 2) < iso_scale:
                            pixels[test_x, test_y] = light_color
                            if mirror_h:
                                pixels[width - 1 - test_x, test_y] = light_color
                            if mirror_v:
                                pixels[test_x, height - 1 - test_y] = light_color
                            if mirror_h and mirror_v:
                                pixels[width - 1 - test_x, height - 1 - test_y] = light_color

    return img

def generate_svg_isometric(width, height, palette, mirror_h=True, mirror_v=True):
    """SVG-style isometric voxel art inspired by isovoxel library."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Create equilateral triangle grid pattern
    tri_size = 30
    cols = gen_width // tri_size
    rows = gen_height // int(tri_size * 0.866)  # Height of equilateral triangle

    for row in range(rows):
        for col in range(cols):
            # Determine if this triangle should be filled
            noise = perlin2D(col * 0.3, row * 0.3)
            if noise > 0:
                # Calculate triangle vertices
                offset_x = (tri_size // 2) if (row % 2 == 1) else 0
                x = col * tri_size + offset_x
                y = row * int(tri_size * 0.866)

                # Pointing up or down based on position
                pointing_up = (col + row) % 2 == 0

                if pointing_up:
                    vertices = [
                        (x, y + int(tri_size * 0.866)),
                        (x + tri_size // 2, y),
                        (x + tri_size, y + int(tri_size * 0.866))
                    ]
                else:
                    vertices = [
                        (x, y),
                        (x + tri_size, y),
                        (x + tri_size // 2, y + int(tri_size * 0.866))
                    ]

                # Color based on position
                t = (noise + 1) / 2
                color = get_color_from_palette(t, palette)
                alpha = 200

                draw.polygon(vertices, fill=color + (alpha,), outline=PALETTE['cursor'])

                # Mirror
                if mirror_h or mirror_v:
                    mirrored_verts = []
                    for vx, vy in vertices:
                        mx = (width - 1 - vx) if mirror_h else vx
                        my = (height - 1 - vy) if mirror_v else vy
                        mirrored_verts.append((mx, my))
                    draw.polygon(mirrored_verts, fill=color + (alpha,), outline=PALETTE['cursor'])

    return img

def generate_voxel_world(width, height, palette, mirror_h=True, mirror_v=True):
    """2.5D isometric world with Perlin noise terrain inspired by IsoEngine."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate heightmap using Perlin noise
    world_size = 40
    heightmap = {}

    for wx in range(world_size):
        for wy in range(world_size):
            height_val = combined(perlin2D, wx * 0.1, wy * 0.1, octaves=3, persistence=0.5)
            # Normalize to 0-10 blocks high
            block_height = int((height_val + 1) * 5)
            heightmap[(wx, wy)] = max(0, block_height)

    # Isometric tile size
    tile_w = 16
    tile_h = 8

    def world_to_screen(wx, wy, wz=0):
        """Convert world coordinates to screen isometric coordinates."""
        screen_x = (wx - wy) * (tile_w // 2)
        screen_y = (wx + wy) * (tile_h // 2) - wz * tile_h
        return screen_x, screen_y

    # Center offset
    center_x = gen_width // 2
    center_y = gen_height // 2 + 200

    # Draw tiles back to front
    for wy in range(world_size - 1, -1, -1):
        for wx in range(world_size):
            height = heightmap.get((wx, wy), 0)

            # Draw stack of blocks from 0 to height
            for wz in range(height + 1):
                sx, sy = world_to_screen(wx, wy, wz)
                sx += center_x
                sy += center_y

                if not (0 <= sx < gen_width and 0 <= sy < gen_height):
                    continue

                # Color based on height
                t = wz / 10
                block_color = get_color_from_palette(t, palette)

                # Draw isometric tile (diamond shape)
                # Top face
                for dy in range(-tile_h // 2, tile_h // 2):
                    for dx in range(-tile_w // 2, tile_w // 2):
                        px = int(sx + dx)
                        py = int(sy + dy)

                        # Diamond shape check
                        if abs(dx) * tile_h + abs(dy) * tile_w < tile_w * tile_h // 2:
                            if 0 <= px < gen_width and 0 <= py < gen_height:
                                pixels[px, py] = block_color
                                if mirror_h:
                                    pixels[width - 1 - px, py] = block_color
                                if mirror_v:
                                    pixels[px, height - 1 - py] = block_color
                                if mirror_h and mirror_v:
                                    pixels[width - 1 - px, height - 1 - py] = block_color

    return img

def generate_multiangle_voxels(width, height, palette, mirror_h=True, mirror_v=True):
    """Multi-angle voxel rendering with sloped edges inspired by spotvox."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Create procedural voxel structure
    structure_size = 15

    # Build a random voxel structure
    voxels = []
    for x in range(structure_size):
        for y in range(structure_size):
            for z in range(structure_size):
                # Create interesting shape using multiple noise functions
                noise1 = perlin2D(x * 0.2, y * 0.2)
                noise2 = simplex2D(y * 0.2, z * 0.2)
                combined_noise = (noise1 + noise2) / 2

                # Create hollow shell effect
                dist_from_center = math.sqrt((x - structure_size/2)**2 +
                                            (y - structure_size/2)**2 +
                                            (z - structure_size/2)**2)
                if 3 < dist_from_center < 7 and combined_noise > 0.2:
                    voxels.append((x, y, z))

    # Isometric projection with rotation
    angle = math.pi / 6  # 30 degrees
    scale = 15

    def project_voxel(vx, vy, vz):
        """Project voxel with rotation."""
        # Rotate around Y axis
        rotated_x = vx * math.cos(angle) - vz * math.sin(angle)
        rotated_z = vx * math.sin(angle) + vz * math.cos(angle)

        # Isometric projection
        screen_x = (rotated_x - vy) * scale * 0.866
        screen_y = (rotated_x + vy) * scale * 0.5 - rotated_z * scale
        return screen_x, screen_y

    center_x = gen_width // 2
    center_y = gen_height // 2

    # Sort for painter's algorithm
    sorted_voxels = sorted(voxels, key=lambda v: v[0] + v[1] - v[2])

    for vx, vy, vz in sorted_voxels:
        sx, sy = project_voxel(vx, vy, vz)
        sx += center_x
        sy += center_y

        # Color with emissive-like effect based on position
        t = (vx + vy + vz) / (structure_size * 3)
        base_color = get_color_from_palette(t, palette)

        # Add glow effect
        glow_radius = 20
        for dy in range(-glow_radius, glow_radius):
            for dx in range(-glow_radius, glow_radius):
                px = int(sx + dx)
                py = int(sy + dy)

                dist = math.sqrt(dx*dx + dy*dy)
                if dist < glow_radius and 0 <= px < gen_width and 0 <= py < gen_height:
                    # Glow falloff
                    intensity = 1 - (dist / glow_radius)
                    intensity = max(0, intensity) ** 2
                    alpha = int(intensity * 100)

                    if alpha > 0:
                        color_with_alpha = base_color + (alpha,)
                        # Draw pixel
                        draw.point((px, py), fill=color_with_alpha)

                        # Mirror
                        if mirror_h:
                            draw.point((width - 1 - px, py), fill=color_with_alpha)
                        if mirror_v:
                            draw.point((px, height - 1 - py), fill=color_with_alpha)
                        if mirror_h and mirror_v:
                            draw.point((width - 1 - px, height - 1 - py), fill=color_with_alpha)

    return img

def generate_procedural_voxel_mesh(width, height, palette, mirror_h=True, mirror_v=True):
    """Procedural voxel mesh generation inspired by voxgen L-systems."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    draw = ImageDraw.Draw(img, 'RGBA')

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate fractal voxel structure using L-system-like rules
    # Simple axiom: F (forward)
    # Rules: F -> F[+F][-F]F
    axiom = "F"
    rules = {"F": "F[+F][-F]F"}
    iterations = 3

    # Apply L-system
    current = axiom
    for _ in range(iterations):
        next_gen = ""
        for char in current:
            next_gen += rules.get(char, char)
        current = next_gen

    # Turtle graphics in 3D
    voxels = set()
    position = [0, 0, 0]
    angle = [0, 0, 0]  # pitch, yaw, roll
    stack = []
    step_size = 2

    for char in current:
        if char == 'F':
            # Move forward in current direction
            direction = [
                math.cos(angle[1]) * math.cos(angle[0]),
                math.sin(angle[0]),
                math.sin(angle[1]) * math.cos(angle[0])
            ]
            position[0] += direction[0] * step_size
            position[1] += direction[1] * step_size
            position[2] += direction[2] * step_size

            voxels.add((int(position[0]), int(position[1]), int(position[2])))
        elif char == '+':
            angle[0] += math.pi / 6  # 30 degrees
        elif char == '-':
            angle[0] -= math.pi / 6
        elif char == '[':
            stack.append((position[:], angle[:]))
        elif char == ']':
            if stack:
                position, angle = stack.pop()

    # Render voxels isometrically
    scale = 8

    def iso_project(x, y, z):
        screen_x = (x - y) * scale
        screen_y = (x + y) * scale * 0.5 - z * scale
        return screen_x, screen_y

    center_x = gen_width // 2
    center_y = gen_height // 2

    # Sort voxels for rendering order
    sorted_voxels = sorted(voxels, key=lambda v: v[0] + v[1] - v[2])

    for vx, vy, vz in sorted_voxels:
        sx, sy = iso_project(vx, vy, vz)
        sx += center_x
        sy += center_y

        # Color based on branch depth (approximate)
        t = (vz + 20) / 40  # Normalize around expected range
        t = max(0, min(1, t))
        voxel_color = get_color_from_palette(t, palette)

        # Draw simple cube representation
        cube_size = scale
        cube_points = [
            (sx, sy),
            (sx + cube_size, sy + cube_size // 2),
            (sx, sy + cube_size),
            (sx - cube_size, sy + cube_size // 2)
        ]

        alpha = 180
        draw.polygon(cube_points, fill=voxel_color + (alpha,), outline=PALETTE['cursor'])

        # Mirror
        if mirror_h or mirror_v:
            mirrored_points = []
            for px, py in cube_points:
                mx = (width - 1 - px) if mirror_h else px
                my = (height - 1 - py) if mirror_v else py
                mirrored_points.append((mx, my))
            draw.polygon(mirrored_points, fill=voxel_color + (alpha,), outline=PALETTE['cursor'])

    return img

def generate_seamless_texture(width, height, palette, mirror_h=True, mirror_v=True):
    """Seamless tileable texture using alpha-gradient blending inspired by img2texture."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate base texture pattern
    tile_size = 256
    overlap = 0.15  # 15% overlap for blending

    # Create base texture with noise
    base_texture = {}
    for y in range(tile_size):
        for x in range(tile_size):
            noise = combined(perlin2D, x * 0.02, y * 0.02, octaves=4, persistence=0.6)
            t = (noise + 1) / 2
            base_texture[(x, y)] = get_color_from_palette(t, palette)

    # Tile with alpha blending at edges
    for ty in range(0, gen_height, tile_size):
        for tx in range(0, gen_width, tile_size):
            for py in range(tile_size):
                for px in range(tile_size):
                    dest_x = tx + px
                    dest_y = ty + py

                    if dest_x >= gen_width or dest_y >= gen_height:
                        continue

                    # Calculate blend factors for seamless tiling
                    blend_x = 1.0
                    blend_y = 1.0

                    overlap_pixels = int(tile_size * overlap)
                    if px < overlap_pixels:
                        blend_x = px / overlap_pixels
                    elif px > tile_size - overlap_pixels:
                        blend_x = (tile_size - px) / overlap_pixels

                    if py < overlap_pixels:
                        blend_y = py / overlap_pixels
                    elif py > tile_size - overlap_pixels:
                        blend_y = (tile_size - py) / overlap_pixels

                    blend_factor = blend_x * blend_y
                    current_color = base_texture[(px, py)]

                    # Blend with neighbors for seamless effect
                    if blend_factor < 1.0:
                        neighbor_x = (px + tile_size // 2) % tile_size
                        neighbor_y = (py + tile_size // 2) % tile_size
                        neighbor_color = base_texture[(neighbor_x, neighbor_y)]
                        current_color = lerp_color(neighbor_color, current_color, blend_factor)

                    pixels[dest_x, dest_y] = current_color
                    if mirror_h:
                        pixels[width - 1 - dest_x, dest_y] = current_color
                    if mirror_v:
                        pixels[dest_x, height - 1 - dest_y] = current_color
                    if mirror_h and mirror_v:
                        pixels[width - 1 - dest_x, height - 1 - dest_y] = current_color

    return img

def generate_example_synthesis(width, height, palette, mirror_h=True, mirror_v=True):
    """Example-based texture synthesis inspired by texture-synthesis."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Create example patch
    patch_size = 64
    example_patches = []

    # Generate 4 example patches with different patterns
    for p in range(4):
        patch = {}
        offset = p * 100
        for py in range(patch_size):
            for px in range(patch_size):
                noise = combined(perlin2D, (px + offset) * 0.05, (py + offset) * 0.05, octaves=3)
                # Add some structure
                structure = math.sin(px * 0.2 + p) * math.cos(py * 0.2 + p)
                combined_val = noise * 0.7 + structure * 0.3
                t = (combined_val + 1) / 2
                patch[(px, py)] = get_color_from_palette(t, palette)
        example_patches.append(patch)

    # Synthesize texture using patch matching
    for y in range(gen_height):
        for x in range(gen_width):
            # Choose patch based on position to create variation
            patch_idx = ((x // patch_size) + (y // patch_size)) % len(example_patches)
            patch = example_patches[patch_idx]

            px = x % patch_size
            py = y % patch_size

            color = patch[(px, py)]

            # Add slight variation to reduce obvious repetition
            variation = (perlin2D(x * 0.003, y * 0.003) + 1) / 2
            variation_amount = 0.1
            varied_color = tuple(int(c * (1 - variation_amount + variation * variation_amount)) for c in color)

            pixels[x, y] = varied_color
            if mirror_h:
                pixels[width - 1 - x, y] = varied_color
            if mirror_v:
                pixels[x, height - 1 - y] = varied_color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = varied_color

    return img

def generate_hyperbolic_tiling(width, height, palette, mirror_h=True, mirror_v=True):
    """Hyperbolic tiling in Poincaré disk inspired by Escher."""
    img = Image.new('RGB', (width, height), PALETTE['background'])
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Poincaré disk center and radius
    center_x = gen_width / 2
    center_y = gen_height / 2
    disk_radius = min(gen_width, gen_height) / 2 * 0.9

    # Hyperbolic triangle vertices (in Poincaré model)
    # Using {7,3} tiling (7 triangles meeting at each vertex)
    num_triangles = 21
    triangles = []

    # Generate triangles radiating from center
    for i in range(num_triangles):
        angle = (2 * math.pi * i) / num_triangles
        r1 = random.uniform(0.3, 0.7)
        r2 = random.uniform(0.7, 0.95)

        # Triangle vertices in hyperbolic space
        v1 = (0, 0)  # Center
        v2 = (r1 * math.cos(angle), r1 * math.sin(angle))
        v3 = (r2 * math.cos(angle + math.pi / num_triangles),
              r2 * math.sin(angle + math.pi / num_triangles))

        triangles.append((v1, v2, v3, i % len(palette)))

    # Render hyperbolic triangles
    for y in range(gen_height):
        for x in range(gen_width):
            # Convert to Poincaré disk coordinates
            dx = (x - center_x) / disk_radius
            dy = (y - center_y) / disk_radius
            r = math.sqrt(dx*dx + dy*dy)

            if r >= 1.0:  # Outside Poincaré disk
                continue

            # Find which triangle contains this point
            for v1, v2, v3, color_idx in triangles:
                # Simple point-in-triangle test (approximate for hyperbolic)
                # Using barycentric coordinates
                def sign(p1, p2, p3):
                    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

                d1 = sign((dx, dy), v1, v2)
                d2 = sign((dx, dy), v2, v3)
                d3 = sign((dx, dy), v3, v1)

                has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
                has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

                if not (has_neg and has_pos):
                    # Inside this triangle
                    color = palette[color_idx]

                    # Add hyperbolic distortion effect
                    distortion = 1 - r * 0.5
                    color = tuple(int(c * distortion) for c in color)

                    pixels[x, y] = color
                    if mirror_h:
                        pixels[width - 1 - x, y] = color
                    if mirror_v:
                        pixels[x, height - 1 - y] = color
                    if mirror_h and mirror_v:
                        pixels[width - 1 - x, height - 1 - y] = color
                    break

    return img

def generate_wang_tiles(width, height, palette, mirror_h=True, mirror_v=True):
    """Wang tiles creating aperiodic patterns inspired by WangTile."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Wang tile parameters - tiles have colored edges (N, E, S, W)
    tile_size = 32
    num_colors = 4  # Number of edge colors

    # Create a set of Wang tiles with edge colors
    wang_tiles = []
    for _ in range(16):  # Generate 16 different tiles
        # Random edge colors (N, E, S, W)
        edges = tuple(random.randint(0, num_colors - 1) for _ in range(4))

        # Generate tile pattern
        tile_pattern = {}
        for ty in range(tile_size):
            for tx in range(tile_size):
                # Use edge colors to influence pattern
                influence = (
                    edges[0] * (tile_size - ty) / tile_size +  # North
                    edges[1] * tx / tile_size +                 # East
                    edges[2] * ty / tile_size +                 # South
                    edges[3] * (tile_size - tx) / tile_size     # West
                ) / (num_colors * 4)

                noise = perlin2D(tx * 0.1, ty * 0.1)
                combined_val = influence + noise * 0.3
                t = combined_val % 1.0

                tile_pattern[(tx, ty)] = get_color_from_palette(t, palette)

        wang_tiles.append((edges, tile_pattern))

    # Place tiles ensuring edge matching
    tile_grid = {}
    for ty in range((gen_height // tile_size) + 1):
        for tx in range((gen_width // tile_size) + 1):
            # Find compatible tile
            required_west = tile_grid.get((tx - 1, ty), (None, None))[0][1] if (tx - 1, ty) in tile_grid else None
            required_north = tile_grid.get((tx, ty - 1), (None, None))[0][2] if (tx, ty - 1) in tile_grid else None

            # Find matching tile
            compatible_tiles = []
            for tile_edges, pattern in wang_tiles:
                match = True
                if required_west is not None and tile_edges[3] != required_west:
                    match = False
                if required_north is not None and tile_edges[0] != required_north:
                    match = False
                if match:
                    compatible_tiles.append((tile_edges, pattern))

            if compatible_tiles:
                tile_grid[(tx, ty)] = random.choice(compatible_tiles)
            else:
                tile_grid[(tx, ty)] = random.choice(wang_tiles)

    # Render tiles
    for y in range(gen_height):
        for x in range(gen_width):
            tx = x // tile_size
            ty = y // tile_size

            if (tx, ty) in tile_grid:
                edges, pattern = tile_grid[(tx, ty)]
                px = x % tile_size
                py = y % tile_size

                color = pattern[(px, py)]

                pixels[x, y] = color
                if mirror_h:
                    pixels[width - 1 - x, y] = color
                if mirror_v:
                    pixels[x, height - 1 - y] = color
                if mirror_h and mirror_v:
                    pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_graph_cut_synthesis(width, height, palette, mirror_h=True, mirror_v=True):
    """Graph-cut based texture synthesis inspired by TileableTextureSynthesis."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate source texture patches
    patch_size = 48
    num_patches = 20

    source_patches = []
    for p in range(num_patches):
        patch = {}
        seed_x = random.randint(0, 1000)
        seed_y = random.randint(0, 1000)

        for py in range(patch_size):
            for px in range(patch_size):
                # Multi-scale noise for detail
                noise = combined(perlin2D, (px + seed_x) * 0.03, (py + seed_y) * 0.03,
                               octaves=5, persistence=0.5)

                # Add structure
                structure = math.sin(px * 0.1) * math.cos(py * 0.1)
                value = noise * 0.8 + structure * 0.2

                t = (value + 1) / 2
                patch[(px, py)] = get_color_from_palette(t, palette)

        source_patches.append(patch)

    # Quilt patches using graph-cut-like approach
    overlap = patch_size // 4

    for ty in range(0, gen_height, patch_size - overlap):
        for tx in range(0, gen_width, patch_size - overlap):
            # Select patch
            patch = random.choice(source_patches)

            # Place patch with blending in overlap regions
            for py in range(patch_size):
                for px in range(patch_size):
                    dest_x = tx + px
                    dest_y = ty + py

                    if dest_x >= gen_width or dest_y >= gen_height:
                        continue

                    new_color = patch[(px, py)]

                    # Blend in overlap regions
                    blend = 1.0
                    if px < overlap and tx > 0:
                        blend = px / overlap
                    if py < overlap and ty > 0:
                        blend *= py / overlap

                    if blend < 1.0 and 0 <= dest_x < gen_width and 0 <= dest_y < gen_height:
                        existing = pixels[dest_x, dest_y]
                        if existing != (0, 0, 0):  # If pixel already set
                            new_color = lerp_color(existing, new_color, blend)

                    pixels[dest_x, dest_y] = new_color
                    if mirror_h:
                        pixels[width - 1 - dest_x, dest_y] = new_color
                    if mirror_v:
                        pixels[dest_x, height - 1 - dest_y] = new_color
                    if mirror_h and mirror_v:
                        pixels[width - 1 - dest_x, height - 1 - dest_y] = new_color

    return img

def generate_gaussian_tiling(width, height, palette, mirror_h=True, mirror_v=True):
    """Gaussian-masked overlapping for seamless tiling inspired by TileMaker."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    # Calculate generation region based on mirroring
    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Generate base tile
    tile_size = 128
    base_tile = {}

    for ty in range(tile_size):
        for tx in range(tile_size):
            # Generate pattern with structure
            noise = combined(perlin2D, tx * 0.02, ty * 0.02, octaves=4, persistence=0.6)

            # Add directional structure
            angle_noise = simplex2D(tx * 0.01, ty * 0.01) * math.pi
            directional = math.sin(tx * 0.05 + angle_noise) * math.cos(ty * 0.05 + angle_noise)

            value = noise * 0.7 + directional * 0.3
            t = (value + 1) / 2

            base_tile[(tx, ty)] = get_color_from_palette(t, palette)

    # Gaussian mask for smooth blending
    def gaussian_weight(x, y, sigma):
        return math.exp(-(x*x + y*y) / (2 * sigma * sigma))

    # Tile with Gaussian blending
    sigma = tile_size / 6

    for y in range(gen_height):
        for x in range(gen_width):
            # Sample from multiple overlapping tiles
            accumulated_color = [0, 0, 0]
            total_weight = 0

            # Check 4 nearest tile centers
            for ty_offset in [-1, 0]:
                for tx_offset in [-1, 0]:
                    tile_center_x = ((x // tile_size) + tx_offset) * tile_size + tile_size // 2
                    tile_center_y = ((y // tile_size) + ty_offset) * tile_size + tile_size // 2

                    # Distance from tile center
                    dx = x - tile_center_x
                    dy = y - tile_center_y

                    # Gaussian weight
                    weight = gaussian_weight(dx, dy, sigma)

                    # Sample from tile (wrap coordinates)
                    sample_x = x % tile_size
                    sample_y = y % tile_size

                    tile_color = base_tile[(sample_x, sample_y)]

                    accumulated_color[0] += tile_color[0] * weight
                    accumulated_color[1] += tile_color[1] * weight
                    accumulated_color[2] += tile_color[2] * weight
                    total_weight += weight

            # Normalize
            if total_weight > 0:
                final_color = tuple(int(c / total_weight) for c in accumulated_color)
            else:
                final_color = base_tile[(x % tile_size, y % tile_size)]

            pixels[x, y] = final_color
            if mirror_h:
                pixels[width - 1 - x, y] = final_color
            if mirror_v:
                pixels[x, height - 1 - y] = final_color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = final_color

    return img

def generate_liquid_distortion(width, height, palette, mirror_h=True, mirror_v=True):
    """Liquid shape distortions with psychedelic motion inspired by liquid-shape-distortions."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    scale = 0.003
    time = random.random() * 100
    layers = 6  # Fractal Brownian Motion layers

    # Fractal Brownian motion for liquid distortion
    def fbm(x, y, t, octaves=6):
        value = 0
        amplitude = 1.0
        frequency = 1.0
        max_value = 0

        for _ in range(octaves):
            # Use time offset to simulate 3D noise
            value += amplitude * perlin2D(x * frequency + t * 10, y * frequency + t * 10)
            max_value += amplitude
            amplitude *= 0.5
            frequency *= 2.0

        return value / max_value

    for y in range(gen_height):
        for x in range(gen_width):
            # Multi-layered FBM for liquid effect
            distort_x = fbm(x * scale, y * scale, time, layers)
            distort_y = fbm(x * scale + 100, y * scale + 100, time + 50, layers)

            # Apply distortion
            dx = x + distort_x * 50
            dy = y + distort_y * 50

            # Secondary distortion for psychedelic ripples
            ripple = math.sin(dx * 0.05) * math.cos(dy * 0.05)
            combined = (distort_x + distort_y + ripple) / 3

            # Map to palette
            t = (combined + 1) / 2  # Normalize to 0-1
            color = get_color_from_palette(t, palette)

            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_fractal_sets(width, height, palette, mirror_h=True, mirror_v=True):
    """Classic fractals (Mandelbrot, Julia, Burning Ship) inspired by shader-fractals."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Randomly choose fractal type
    fractal_type = random.choice(['mandelbrot', 'julia', 'burning_ship'])

    # Julia set constant
    c_real = random.uniform(-0.8, 0.8)
    c_imag = random.uniform(-0.8, 0.8)

    # Zoom and center
    zoom = random.uniform(0.5, 2.0)
    center_x = random.uniform(-0.5, 0.5)
    center_y = random.uniform(-0.5, 0.5)

    max_iter = 100

    for y in range(gen_height):
        for x in range(gen_width):
            # Map pixel to complex plane
            zx = ((x / gen_width) - 0.5) * 4 / zoom + center_x
            zy = ((y / gen_height) - 0.5) * 4 / zoom + center_y

            if fractal_type == 'mandelbrot':
                cx, cy = zx, zy
            elif fractal_type == 'julia':
                cx, cy = c_real, c_imag
            else:  # burning_ship
                cx, cy = zx, zy

            iteration = 0
            while zx * zx + zy * zy < 4 and iteration < max_iter:
                if fractal_type == 'burning_ship':
                    # Burning Ship uses absolute values
                    zx, zy = abs(zx), abs(zy)

                tmp = zx * zx - zy * zy + cx
                zy = 2 * zx * zy + cy
                zx = tmp
                iteration += 1

            # Smooth coloring
            if iteration < max_iter:
                # Smooth iteration count
                log_zn = math.log(zx * zx + zy * zy) / 2
                nu = math.log(log_zn / math.log(2)) / math.log(2)
                iteration = iteration + 1 - nu

            # Map to palette
            t = (iteration / max_iter) % 1.0
            color = get_color_from_palette(t, palette)

            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_glitch_effects(width, height, palette, mirror_h=True, mirror_v=True):
    """Analog and digital glitch effects inspired by KinoGlitch."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Base pattern using noise
    scale = 0.005

    # Generate base image
    base_pixels = {}
    for y in range(gen_height):
        for x in range(gen_width):
            n = perlin2D(x * scale, y * scale)
            t = (n + 1) / 2
            base_pixels[(x, y)] = get_color_from_palette(t, palette)

    # Apply glitch effects
    glitch_type = random.choice(['scan_line', 'rgb_drift', 'block_corruption', 'vertical_jump'])

    for y in range(gen_height):
        for x in range(gen_width):
            if glitch_type == 'scan_line':
                # Analog scan line jitter
                jitter = int(perlin2D(0, y * 0.1) * 20)
                source_x = (x + jitter) % gen_width
                color = base_pixels[(source_x, y)]

            elif glitch_type == 'rgb_drift':
                # RGB channel separation
                r = base_pixels[(x, y)][0]
                g = base_pixels[((x + 5) % gen_width, y)][1]
                b = base_pixels[((x - 5) % gen_width, y)][2]
                color = (r, g, b)

            elif glitch_type == 'block_corruption':
                # Digital block corruption
                block_size = 20
                if random.random() < 0.1:
                    block_x = (x // block_size) * block_size
                    block_y = (y // block_size) * block_size
                    color = base_pixels[(block_x % gen_width, block_y % gen_height)]
                else:
                    color = base_pixels[(x, y)]

            else:  # vertical_jump
                # Vertical displacement
                jump = int(perlin2D(x * 0.05, 0) * 50)
                source_y = (y + jump) % gen_height
                color = base_pixels[(x, source_y)]

            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_raymarched_tunnel(width, height, palette, mirror_h=True, mirror_v=True):
    """Raymarched infinite tunnel effects inspired by shaderbox."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    time = random.random() * 10

    for y in range(gen_height):
        for x in range(gen_width):
            # Center coordinates
            cx = (x - gen_width / 2) / gen_height
            cy = (y - gen_height / 2) / gen_height

            # Convert to polar coordinates for tunnel
            angle = math.atan2(cy, cx)
            radius = math.sqrt(cx * cx + cy * cy)

            if radius < 0.01:
                radius = 0.01  # Avoid division by zero

            # Tunnel depth
            depth = 1.0 / radius + time

            # Tunnel texture (checkerboard pattern)
            u = angle / math.pi  # -1 to 1
            v = depth

            checker_u = int(u * 10) % 2
            checker_v = int(v * 10) % 2
            checker = (checker_u + checker_v) % 2

            # Distance fog
            fog = 1.0 / (1.0 + depth * 0.1)

            # Map to palette with checker pattern
            t = (checker * 0.5 + fog * 0.5)
            color = get_color_from_palette(t, palette)

            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

def generate_kaleidoscope_effect(width, height, palette, mirror_h=True, mirror_v=True):
    """Kaleidoscope and symmetry effects inspired by MusicVisualizer."""
    img = Image.new('RGB', (width, height))
    pixels = img.load()

    gen_width = (width // 2) if mirror_h else width
    gen_height = (height // 2) if mirror_v else height

    # Kaleidoscope parameters
    segments = random.choice([4, 6, 8, 12])  # N-fold symmetry
    scale = 0.003

    for y in range(gen_height):
        for x in range(gen_width):
            # Center coordinates
            cx = x - gen_width / 2
            cy = y - gen_height / 2

            # Polar coordinates
            angle = math.atan2(cy, cx)
            radius = math.sqrt(cx * cx + cy * cy)

            # Kaleidoscope reflection
            segment_angle = (2 * math.pi) / segments
            folded_angle = angle % segment_angle
            if (int(angle / segment_angle) % 2) == 1:
                folded_angle = segment_angle - folded_angle

            # Spiral pattern
            spiral = folded_angle + radius * 0.05

            # Generate pattern using noise
            pattern_x = radius * math.cos(folded_angle)
            pattern_y = radius * math.sin(folded_angle)

            n1 = perlin2D(pattern_x * scale, pattern_y * scale)
            n2 = perlin2D(pattern_x * scale + 100, pattern_y * scale + 100)

            # Combine with spiral
            combined = (n1 + n2 + math.sin(spiral)) / 3

            # Map to palette
            t = (combined + 1) / 2
            color = get_color_from_palette(t, palette)

            pixels[x, y] = color
            if mirror_h:
                pixels[width - 1 - x, y] = color
            if mirror_v:
                pixels[x, height - 1 - y] = color
            if mirror_h and mirror_v:
                pixels[width - 1 - x, height - 1 - y] = color

    return img

# ============================================================================
# MAIN GENERATION
# ============================================================================

GENERATORS = {
    'layered_noise': ('Layered Noise', generate_layered_noise),
    'voronoi': ('Voronoi Cells', generate_voronoi),
    'flow_field': ('Flow Field', generate_flow_field),
    'interference': ('Wave Interference', generate_interference),
    'fractal_noise': ('Fractal Noise', generate_fractal_noise),
    'cellular': ('Cellular Automata', generate_cellular_automata),
    'plotter_art': ('Plotter Art', generate_plotter_art),
    'differential_growth': ('Differential Growth', generate_differential_growth),
    'penrose_tiling': ('Penrose Tiling', generate_penrose_tiling),
    'bezier_curves': ('Bezier Curves', generate_bezier_curves),
    'physarum': ('Physarum Slime Mold', generate_physarum),
    'pixel_sprites': ('Pixel Sprites', generate_pixel_sprites),
    'wave_collapse': ('Wave Function Collapse', generate_wave_function_collapse),
    'pixel_dithering': ('Pixel Art Dithering', generate_pixel_art_dithering),
    'sprite_characters': ('Sprite Characters', generate_sprite_characters),
    'template_pixels': ('Template Pixel Art', generate_template_pixel_art),
    'raytraced_sdf': ('Raytraced SDF', generate_raytraced_sdf),
    'pathtraced_terrain': ('Path-Traced Terrain', generate_pathtraced_terrain),
    'voxel_structures': ('Voxel L-Systems', generate_voxel_structures),
    'isometric_pixel': ('Isometric Pixel Art', generate_isometric_pixel),
    'lowpoly_terrain': ('Low-Poly Terrain', generate_lowpoly_terrain),
    'reaction_diffusion': ('Reaction-Diffusion', generate_reaction_diffusion),
    'strange_attractor': ('Strange Attractors', generate_strange_attractor),
    'dla': ('DLA Aggregation', generate_dla),
    'neural_ca': ('Neural Cellular Automata', generate_neural_ca),
    'space_colonization': ('Space Colonization', generate_space_colonization),
    'isometric_voxel_art': ('Isometric Voxel Art', generate_isometric_voxel_art),
    'svg_isometric': ('SVG Isometric', generate_svg_isometric),
    'voxel_world': ('Voxel World Engine', generate_voxel_world),
    'multiangle_voxels': ('Multi-Angle Voxels', generate_multiangle_voxels),
    'procedural_voxel_mesh': ('Procedural Voxel Mesh', generate_procedural_voxel_mesh),
    'seamless_texture': ('Seamless Texture Tiling', generate_seamless_texture),
    'example_synthesis': ('Example-Based Synthesis', generate_example_synthesis),
    'hyperbolic_tiling': ('Hyperbolic Tiling', generate_hyperbolic_tiling),
    'wang_tiles': ('Wang Tiles', generate_wang_tiles),
    'graph_cut_synthesis': ('Graph-Cut Synthesis', generate_graph_cut_synthesis),
    'gaussian_tiling': ('Gaussian Tiling', generate_gaussian_tiling),
    'liquid_distortion': ('Liquid Distortion', generate_liquid_distortion),
    'fractal_sets': ('Fractal Sets', generate_fractal_sets),
    'glitch_effects': ('Glitch Effects', generate_glitch_effects),
    'raymarched_tunnel': ('Raymarched Tunnel', generate_raymarched_tunnel),
    'kaleidoscope_effect': ('Kaleidoscope Effect', generate_kaleidoscope_effect),
}

def add_logo(img, logo_path):
    """Add logo to center of image."""
    logo = Image.open(logo_path)

    # Resize logo to reasonable size
    max_logo_width = 650
    if logo.width > max_logo_width:
        ratio = max_logo_width / logo.width
        new_size = (int(logo.width * ratio), int(logo.height * ratio))
        logo = logo.resize(new_size, Image.Resampling.LANCZOS)

    # Calculate center position
    logo_x = (WIDTH - logo.width) // 2
    logo_y = (HEIGHT - logo.height) // 2

    # Paste logo with transparency
    if logo.mode in ('RGBA', 'LA'):
        img.paste(logo, (logo_x, logo_y), logo)
    else:
        img.paste(logo, (logo_x, logo_y))

    return img

def main():
    # Get available logos
    logo_dir = 'logos'
    logos = [os.path.join(logo_dir, f) for f in os.listdir(logo_dir) if f.endswith('.png')]

    if not logos:
        print("Error: No logos found in logos/ directory")
        return

    print(f"Found {len(logos)} logos:")
    for logo in logos:
        print(f"  - {os.path.basename(logo)}")
    print()

    # Create output directory
    output_dir = 'wallpapers'
    os.makedirs(output_dir, exist_ok=True)

    # Configuration
    mirror_h = True  # Horizontal symmetry enabled by default
    mirror_v = True  # Vertical symmetry enabled by default

    # Generate one wallpaper from each technique
    # This ensures variety and makes it easy to add more generators in the future
    generators_list = list(GENERATORS.items())
    batch_size = len(generators_list)

    print(f"Generating {batch_size} wallpapers (one from each technique) with h+v symmetry...\n")

    # Generate one wallpaper from each technique
    for i, (gen_id, (gen_name, gen_func)) in enumerate(generators_list):
        print(f"[{i+1}/{batch_size}] Generating {gen_name}...")

        # Generate base wallpaper with mirroring
        img = gen_func(WIDTH, HEIGHT, PALETTE['colors'], mirror_h=mirror_h, mirror_v=mirror_v)

        # Randomly select a logo
        selected_logo = random.choice(logos)
        logo_name = os.path.basename(selected_logo).replace('.png', '')

        print(f"  Adding logo: {os.path.basename(selected_logo)}")
        img = add_logo(img, selected_logo)

        # Save with number prefix for curation
        import time
        timestamp = int(time.time() * 1000) % 1000000  # Use milliseconds modulo to keep it short
        output_file = os.path.join(output_dir, f'{i+1:02d}_wallpaper_{gen_id}_{logo_name}_{timestamp}.png')
        img.save(output_file, 'PNG')
        print(f"  ✓ Saved: {output_file}\n")

    print(f"\n✓ Generated {batch_size} wallpapers in {output_dir}/")
    print(f"   Used all {batch_size} available techniques")
    print("\nTo set as wallpaper:")
    print("  1. Open System Settings > Wallpaper")
    print("  2. Click '+' and select a wallpaper from the wallpapers/ folder")

if __name__ == '__main__':
    main()
