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

        # Save with timestamp to ensure unique filenames
        import time
        timestamp = int(time.time() * 1000) % 1000000  # Use milliseconds modulo to keep it short
        output_file = os.path.join(output_dir, f'wallpaper_{gen_id}_{logo_name}_{timestamp}.png')
        img.save(output_file, 'PNG')
        print(f"  ✓ Saved: {output_file}\n")

    print(f"\n✓ Generated {batch_size} wallpapers in {output_dir}/")
    print(f"   Used all {batch_size} available techniques")
    print("\nTo set as wallpaper:")
    print("  1. Open System Settings > Wallpaper")
    print("  2. Click '+' and select a wallpaper from the wallpapers/ folder")

if __name__ == '__main__':
    main()
