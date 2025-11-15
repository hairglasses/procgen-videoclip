# Procedural Wallpaper Generator

Generate beautiful, unique macOS wallpapers using procedural generation techniques with your custom color scheme.

## Features

- **37 Distinct Generation Techniques**:
  - Layered Noise (Perlin/Simplex blend)
  - Voronoi Cells (organic cellular patterns)
  - Flow Field (particle-like flowing patterns)
  - Wave Interference (ripple patterns)
  - Fractal Noise (high-detail ridged terrain)
  - Cellular Automata (Conway's Game of Life inspired)
  - Plotter Art (geometric pen-plotter style patterns - [vsketch](https://github.com/abey79/vsketch))
  - Differential Growth (organic coral-like structures - [differential-line](https://github.com/inconvergent/differential-line))
  - Penrose Tiling (aperiodic mathematical patterns - [penrose](https://github.com/xnx/penrose))
  - Bezier Curves (smooth organic shapes - [DeepSVG](https://github.com/alexandre01/deepsvg))
  - Physarum Slime Mold (nature-inspired network patterns - [physarum](https://github.com/fogleman/physarum))
  - Pixel Sprites (retro game-style sprites - [pixel-sprite-generator](https://github.com/zfedoran/pixel-sprite-generator))
  - Wave Function Collapse (algorithmic pattern generation - [WaveFunctionCollapse](https://github.com/mxgmn/WaveFunctionCollapse))
  - Pixel Art Dithering (authentic 8-bit style - [pyxelate](https://github.com/sedthh/pyxelate))
  - Sprite Characters (procedural robot sprites - [Sprite-Generator](https://github.com/MaartenGr/Sprite-Generator))
  - Template Pixel Art (rule-based pixel patterns - [Procedural-Pixel-Art-Generator](https://github.com/Darkhax-Forked/Procedural-Pixel-Art-Generator))
  - Raytraced SDF (signed distance field geometry with CSG operations - [retrace.gl](https://github.com/stasilo/retrace.gl))
  - Path-Traced Terrain (photorealistic terrain with retro 80s aesthetics - [THREE.js-PathTracing-Renderer](https://github.com/erichlof/THREE.js-PathTracing-Renderer))
  - Voxel L-Systems (blocky organic structures using turtle graphics - [voxgen](https://github.com/wodend/voxgen))
  - Isometric Pixel Art (dimetric projection with pixel-perfect rendering - [ProceduralPixelArt](https://github.com/jlcarr/ProceduralPixelArt))
  - Low-Poly Terrain (elevation-based landscapes with triangulated mesh - [THREE.Terrain](https://github.com/IceCreamYou/THREE.Terrain))
  - Reaction-Diffusion (Gray-Scott model creating organic biological patterns - [Ready](https://github.com/GollyGang/ready))
  - Strange Attractors (chaotic dynamical systems like Lorenz and Rössler - [dysts](https://github.com/GilpinLab/dysts))
  - DLA Aggregation (diffusion limited aggregation forming coral structures - [dla-gpu](https://github.com/zentralwerkstatt/dla-gpu))
  - Neural Cellular Automata (self-organizing patterns that grow and heal - [Growing-Neural-Cellular-Automata](https://github.com/PWhiddy/Growing-Neural-Cellular-Automata-Pytorch))
  - Space Colonization (venation patterns and leaf-vein structures - [morphogenesis-resources](https://github.com/jasonwebb/morphogenesis-resources))
  - Isometric Voxel Art (multi-angle isometric voxel rendering - [IsoVoxel](https://github.com/tommyettinger/IsoVoxel))
  - SVG Isometric (triangle grid-based isometric generative art - [isovoxel](https://github.com/rsimmons/isovoxel))
  - Voxel World Engine (2.5D isometric terrain generation - [IsoEngine](https://github.com/7hebel/IsoEngine))
  - Multi-Angle Voxels (voxel rendering with emissive glow effects - [spotvox](https://github.com/tommyettinger/spotvox))
  - Procedural Voxel Mesh (L-system fractal voxel generation - [voxgen](https://github.com/wodend/voxgen))
  - Seamless Texture Tiling (alpha-gradient blending for infinite tiling - [img2texture](https://github.com/rtmigo/img2texture))
  - Example-Based Synthesis (multi-example texture synthesis - [texture-synthesis](https://github.com/EmbarkStudios/texture-synthesis))
  - Hyperbolic Tiling (Escher-like Poincaré disk tesselations - [Escher](https://github.com/b5strbal/Escher))
  - Wang Tiles (aperiodic coded-edge tiling patterns - [WangTile](https://github.com/sashaouellet/WangTile))
  - Graph-Cut Synthesis (patch quilting with seam minimization - [TileableTextureSynthesis](https://github.com/lzqsd/TileableTextureSynthesis))
  - Gaussian Tiling (Gaussian-masked seamless overlapping - [TileMaker](https://github.com/mdushkoff/TileMaker))

- **Horizontal + Vertical Symmetry**: All wallpapers default to beautiful symmetric patterns
- **Custom Color Palette**: All wallpapers use your dark red/brown color scheme
- **Random Logo Placement**: Automatically centers either Mercury Light or Vanguard logo
- **Growing Art Gallery**: Add wallpapers to a Photos album that grows over time
- **Automated Generation**: Set up daily/weekly generation to continuously expand your collection

## Quick Start

### Method 1: Growing Photos Album (Recommended)

Create a Photos album that grows as your personal procgen art gallery:

```bash
# Add 37 new wallpapers to your Photos album (one from each technique)
./generate_to_photos_album.py
```

Then set up the wallpaper slideshow:
1. Open **System Settings** > **Wallpaper**
2. Select the Photos album: **"Procgen Wallpapers"**
3. Enable **Change picture** (every 30 minutes recommended)
4. Enable **Random order**

**Set up automatic daily generation:**
```bash
./setup_auto_generate.sh
```

This adds 37 new wallpapers to your album every day at 9 AM. Your collection keeps growing!

### Method 2: Local Directory

Generate wallpapers to a local directory:

```bash
# Generate 37 wallpapers (one from each technique) with h+v symmetry (default)
python3 generate_wallpapers_multi.py

# Generate single wallpaper with symmetry
./generate_wallpaper.py

# Generate without symmetry
./generate_wallpaper.py --no-mirror
```

### Legacy: Login-Based Generation

```bash
# Set up generation at every login
./install_login_script.sh

# Uninstall
./uninstall_login_script.sh
```

## Files

**Main Scripts:**
- `generate_to_photos_album.py` - Generate and add to Photos album (recommended)
- `generate_wallpapers_multi.py` - Generate batch of 37 wallpapers to directory (one from each technique)
- `generate_wallpaper.py` - Generate single wallpaper to directory

**Setup Scripts:**
- `setup_auto_generate.sh` - Setup daily automatic generation (9 AM)
- `install_login_script.sh` - Setup generation at login (legacy)
- `uninstall_login_script.sh` - Remove login generation

**Utilities:**
- `setup_wallpaper_slideshow.py` - Generate + configure slideshow manually

**Directories:**
- `wallpapers/` - Output directory for local wallpapers
- `temp_wallpapers/` - Temporary storage before Photos import
- `logos/` - Logo assets (Mercury Light, Vanguard)

## Requirements

- Python 3.x
- Pillow (PIL)
- procgen library

Install dependencies:
```bash
pip3 install Pillow git+https://github.com/jcarlosroldan/procgen
```

## Customization

### Add More Generation Techniques

To add a new wallpaper generation technique, simply add it to the `GENERATORS` dictionary in `generate_wallpapers_multi.py`:

```python
GENERATORS = {
    'layered_noise': ('Layered Noise', generate_layered_noise),
    'voronoi': ('Voronoi Cells', generate_voronoi),
    # ... existing generators ...
    'your_new_technique': ('Your New Technique', generate_your_new_technique),
}
```

The script will automatically generate one wallpaper from each technique, so adding a new generator increases the batch size by 1.

### Change Automatic Generation Time

Edit `setup_auto_generate.sh` to change from 9 AM:
```xml
<key>Hour</key>
<integer>9</integer>  <!-- Change hour (0-23) -->
```

### Disable Symmetry

For asymmetric wallpapers, edit the generator functions:
```python
img = gen_func(WIDTH, HEIGHT, PALETTE['colors'], mirror_h=False, mirror_v=False)
```

### Modify Color Palette

Edit the `PALETTE` dictionary in `generate_wallpapers_multi.py` (lines 20-44).

### Adjust Pattern Scale

Modify the `scale` variable in each generator function to zoom in/out on patterns.

## Troubleshooting

### Photos App Permission

First time running `generate_to_photos_album.py`, you may need to:
1. Grant **Photos** permission when prompted
2. Check **System Settings** > **Privacy & Security** > **Automation**
3. Enable Python/Terminal to control Photos

### Photos Album Not Appearing in Wallpaper Settings

- Open Photos app to verify the album exists
- Try restarting System Settings
- The album appears under "Photos" section in wallpaper picker

### Check Auto-Generation Status

```bash
# Check if launchd agent is running
launchctl list | grep com.procgen.wallpaper

# View logs
tail -f ~/Library/Logs/ProcgenWallpapers/wallpaper_generator.log

# Test the service manually
launchctl start com.procgen.wallpaper
```

### Manual Slideshow Setup

If the automated setup fails:
1. Open **System Settings** > **Wallpaper**
2. For Photos album: Select "Procgen Wallpapers" from Photos section
3. For folder: Click **+** and select the `wallpapers/` folder
4. Enable **Change picture** and set interval to 30 minutes

## Logs

**Automatic generation logs:**
- `~/Library/Logs/ProcgenWallpapers/wallpaper_generator.log` - Standard output
- `~/Library/Logs/ProcgenWallpapers/wallpaper_generator_error.log` - Error messages

**Login script logs:**
- `wallpaper_generator.log` - Standard output
- `wallpaper_generator_error.log` - Error messages

## How It Works

1. **Generation**: Each wallpaper uses one of 37 procedural techniques with random seeds
2. **Symmetry**: Patterns are mirrored horizontally and vertically for aesthetic balance
3. **Photos Import**: Images are imported to Photos and added to "Procgen Wallpapers" album
4. **Slideshow**: macOS cycles through the Photos album, changing wallpaper periodically
5. **Growth**: Each run adds more wallpapers, creating an ever-growing art collection
