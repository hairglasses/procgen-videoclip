#!/usr/bin/env python3
"""
Procedurally generate a macOS wallpaper using the procgen library.
"""

import argparse
from PIL import Image
from procgen.noise import perlin2D, simplex2D, combined

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

def get_color_from_palette(noise_value, palette_colors):
    """Map a noise value (-1 to 1) to a color in the palette with smooth blending."""
    # Normalize noise to 0-1
    t = (noise_value + 1) / 2

    # Map to palette index with blending
    palette_size = len(palette_colors)
    scaled = t * (palette_size - 1)
    idx1 = int(scaled)
    idx2 = min(idx1 + 1, palette_size - 1)
    blend = scaled - idx1

    return lerp_color(palette_colors[idx1], palette_colors[idx2], blend)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Generate procedural wallpaper with optional mirroring')
parser.add_argument('--mirror-h', action='store_true', help='Mirror pattern horizontally')
parser.add_argument('--mirror-v', action='store_true', help='Mirror pattern vertically')
parser.add_argument('--mirror-both', action='store_true', default=True, help='Mirror pattern both horizontally and vertically (default: True)')
parser.add_argument('--no-mirror', action='store_true', help='Disable all mirroring')
args = parser.parse_args()

# Set mirror flags - default to both unless --no-mirror is specified
if args.no_mirror:
    mirror_horizontal = False
    mirror_vertical = False
else:
    mirror_horizontal = args.mirror_h or args.mirror_both
    mirror_vertical = args.mirror_v or args.mirror_both

# Create a new image
img = Image.new('RGB', (WIDTH, HEIGHT))
pixels = img.load()

# Scale for noise (smaller = more zoomed in, larger = more zoomed out)
scale = 0.002

mirror_info = []
if mirror_horizontal:
    mirror_info.append("horizontal")
if mirror_vertical:
    mirror_info.append("vertical")
mirror_text = " + ".join(mirror_info) if mirror_info else "none"

print("Generating procedural wallpaper with custom color scheme...")
print(f"Resolution: {WIDTH}x{HEIGHT}")
print(f"Color palette: {len(PALETTE['colors'])} colors")
print(f"Mirroring: {mirror_text}")

# Calculate the actual generation region based on mirroring
# If mirroring, we only need to generate half (or quarter) of the image
gen_width = (WIDTH // 2) if mirror_horizontal else WIDTH
gen_height = (HEIGHT // 2) if mirror_vertical else HEIGHT

# Generate the wallpaper using layered noise
for y in range(gen_height):
    if y % 200 == 0:
        print(f"Progress: {int(y/gen_height*100)}%")

    for x in range(gen_width):
        # Apply mirroring to noise coordinates
        noise_x = x
        noise_y = y

        # Use combined noise for more interesting patterns
        # Layer 1: Base color selection using perlin noise
        noise1 = combined(perlin2D, noise_x * scale, noise_y * scale, octaves=6, persistence=0.5)

        # Layer 2: Color variation using simplex noise
        noise2 = combined(simplex2D, noise_x * scale * 1.5, noise_y * scale * 1.5, octaves=4, persistence=0.6)

        # Layer 3: Large-scale regions
        noise3 = combined(perlin2D, noise_x * scale * 0.3, noise_y * scale * 0.3, octaves=3, persistence=0.7)

        # Blend noises for more complex patterns
        combined_noise = (noise1 * 0.5 + noise2 * 0.3 + noise3 * 0.2)

        # Get color from palette based on noise
        color = get_color_from_palette(combined_noise, PALETTE['colors'])

        # Apply subtle blending with background color for depth
        background_blend = (noise2 + 1) / 8  # 0 to 0.25
        color = lerp_color(color, PALETTE['background'], background_blend)

        # Set the pixel and mirror if needed
        pixels[x, y] = color

        # Mirror horizontally
        if mirror_horizontal:
            pixels[WIDTH - 1 - x, y] = color

        # Mirror vertically
        if mirror_vertical:
            pixels[x, HEIGHT - 1 - y] = color

        # Mirror both (corner quadrant)
        if mirror_horizontal and mirror_vertical:
            pixels[WIDTH - 1 - x, HEIGHT - 1 - y] = color

print("Progress: 100%")
print("Adding logo to center...")

# Load and add logo
logo_path = 'logos/Logomark-Mercury-Light.png'
logo = Image.open(logo_path)

# Resize logo to reasonable size (max width 650px while maintaining aspect ratio)
max_logo_width = 650
if logo.width > max_logo_width:
    ratio = max_logo_width / logo.width
    new_size = (int(logo.width * ratio), int(logo.height * ratio))
    logo = logo.resize(new_size, Image.Resampling.LANCZOS)

# Calculate center position
logo_x = (WIDTH - logo.width) // 2
logo_y = (HEIGHT - logo.height) // 2

# Paste logo (using alpha channel if available for transparency)
if logo.mode in ('RGBA', 'LA'):
    img.paste(logo, (logo_x, logo_y), logo)
else:
    img.paste(logo, (logo_x, logo_y))

print("Saving wallpaper...")

# Save the image
output_file = 'wallpaper.png'
img.save(output_file, 'PNG')

print(f"✓ Wallpaper saved as: {output_file}")
print(f"  Logo: {logo_path} ({logo.width}x{logo.height})")
print(f"  To set as wallpaper:")
print(f"  1. Open System Settings > Wallpaper")
print(f"  2. Click '+' and select {output_file}")
