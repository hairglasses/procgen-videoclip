# Procedural Video Generator - Category Scripts

This project includes 72 unique procedural video generators organized into categories. Use these scripts to generate videos by category.

## Quick Start

Generate all videos:
```bash
python3 generate_videos.py
```

Or generate by category:
```bash
python3 generate_cpu.py          # CPU-based generators (22 videos)
python3 generate_shaders.py      # All GPU shaders (40 videos)
python3 generate_retro.py        # Retro/glitch shaders (9 videos)
python3 generate_isometric.py    # Isometric/voxel generators (5 videos)
python3 generate_characters.py   # Sprite characters (5 videos)
python3 generate_new_shaders.py  # Newly added shaders (8 videos)
```

## Categories

### CPU-Based Generators (22 total)
**Script:** `generate_cpu.py`

Procedural algorithms running on CPU:
- Cellular Automata
- Wave Interference
- Grid Distortion
- Layered Noise
- Voronoi Cells
- Spiral Patterns
- Concentric Rings
- Fractal Noise
- Flow Field
- Chladni Patterns
- Domain Warping
- Superformula
- Strange Attractor
- L-System Plants
- Boids Flocking
- Reaction-Diffusion
- Differential Growth
- Bezier Curves
- Penrose Tiling
- Physarum Slime Mold
- Plotter Art
- Pixel Sprites

### GPU Shaders (40 total)
**Script:** `generate_shaders.py`

All GLSL shader-based effects:
- Plasma
- Tunnel Effect
- Raymarching
- Mandelbrot Fractal
- Julia Set
- Metaballs
- Rotozoomer
- Voronoi Noise
- Kaleidoscope
- Fire Effect
- Starfield
- Hexagonal Tiling
- DNA Helix
- Matrix Rain
- Wave Interference
- Clockwork Gears
- Caustics
- Truchet Tiles
- Aurora
- Moire Patterns
- Perlin Flow
- Spirograph
- Electric
- Glitch
- CRT Screen
- VHS Glitch
- Pixel Sort
- RGB Split
- Dither
- Teletext
- C64
- Game Boy
- Feedback Loop
- Lava Lamp
- Nebula
- Circuit Board
- Warp Speed
- Liquid Crystal
- Fractal Flame
- Oscilloscope

### Retro/Glitch Shaders (9 total)
**Script:** `generate_retro.py`

Vintage computer and glitch effects:
- CRT Screen (scanlines, phosphor glow, curvature)
- VHS Glitch (tracking errors, tape noise)
- Pixel Sort (datamoshing effect)
- RGB Split (chromatic aberration)
- Dither (Bayer matrix dithering)
- Teletext (blocky videotext)
- C64 (Commodore 64 aesthetic)
- Game Boy (4-shade LCD)
- Glitch (digital corruption)

### Isometric/Voxel Generators (5 total)
**Script:** `generate_isometric.py`

3D isometric and voxel-based animations:
- Isometric Cubes (floating animated grid)
- Voxel Terrain (procedural heightmap)
- Isometric City (pulsing buildings)
- Voxel Waves (rippling voxel patterns)
- Isometric Stairs (Escher-like spirals)

### Sprite Characters (5 total)
**Script:** `generate_characters.py`

Animated pixel art characters:
- Walking Character (side-scrolling walk cycle)
- Jumping Character (physics-based jumping)
- Dancing Character (rhythmic 8-frame dance)
- Flying Character (wing flapping, circular motion)
- Running Character (fast run cycle with perspective)

### New Shaders (8 total)
**Script:** `generate_new_shaders.py`

Recently added shader effects:
- Feedback Loop (psychedelic video feedback)
- Lava Lamp (organic morphing blobs)
- Nebula (cosmic clouds with stars)
- Circuit Board (flowing circuit signals)
- Warp Speed (hyperspace star streaking)
- Liquid Crystal (LCD interference patterns)
- Fractal Flame (iterated function systems)
- Oscilloscope (Lissajous curve visualizer)

## Video Specifications

- **Resolution:** 1280x720 (HD)
- **Frame Rate:** 30 fps
- **Duration:** 10 seconds
- **Loop:** Seamless (first frame = last frame)
- **Color Palette:** Dark red theme
- **Format:** MP4 (H.264)
- **Output Directory:** `videos/`

## Generate Individual Videos

To generate specific videos by name, use the `generate_single.py` script:
```bash
python3 generate_single.py isometric_cubes
python3 generate_single.py shader_plasma shader_nebula walking_character
python3 generate_single.py --list    # Show all available generators
```

Or call the main script directly:
```bash
python3 generate_videos.py shader_plasma shader_nebula walking_character
```

## Total Generator Count

**72 unique procedural video generators**
- 22 CPU-based generators
- 40 GPU shaders
- 5 Sprite character generators
- 5 Isometric/voxel generators

All categories have some overlap (e.g., retro shaders are a subset of GPU shaders).
