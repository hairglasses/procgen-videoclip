# Procedural Video Clip Generator

Generate beautiful, looping video animations using procedural generation techniques with your custom color scheme.

## Features

- **Looping Video Animations**: All videos loop seamlessly with 10-second duration
- **Multiple Generation Techniques**:
  - Layered Noise (Perlin/Simplex blend with time evolution)
  - Flow Field (animated particle-like flowing patterns)
  - Wave Interference (moving ripple patterns)
  - Voronoi Cells (animated cellular patterns)
  - Fractal Noise (evolving ridged terrain)
  - Cellular Automata (evolving pattern states)
  - *More generators to be added...*

- **High Quality Output**: 1920x1080 @ 30fps MP4 videos
- **Horizontal + Vertical Symmetry**: All videos feature beautiful symmetric patterns
- **Custom Color Palette**: Uses dark red/brown color scheme matching the wallpaper generator
- **MP4 Format**: H.264 codec for universal compatibility

## Quick Start

### Installation

Install dependencies:
```bash
pip3 install -r requirements.txt
```

Dependencies:
- Pillow (image processing)
- numpy (array operations)
- imageio (video rendering)
- imageio-ffmpeg (video codec support)
- tqdm (progress bars)
- procgen (procedural generation library)

### Generate Videos

Generate all available looping video animations:
```bash
python3 generate_videos.py
```

This will create MP4 files in the `videos/` directory:
- `01_video_layered_noise.mp4`
- `02_video_flow_field.mp4`
- `03_video_interference.mp4`
- `04_video_voronoi.mp4`
- `05_video_fractal_noise.mp4`
- `06_video_cellular.mp4`

Each video is a seamless 10-second loop at 1920x1080 resolution.

## Video Settings

You can customize the video output by editing `generate_videos.py`:

```python
# Video dimensions
WIDTH = 1920
HEIGHT = 1080

# Video settings
FPS = 30
DURATION = 10  # seconds
```

## Technical Details

### Looping Animation

All generators use a time parameter `t` (0 to 1) that represents the position in the loop:
```python
def generate_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    # t goes from 0 to 1 and loops seamlessly
    time_offset = math.sin(t * 2 * math.pi) * amplitude
    # ... use time_offset to animate noise/patterns
```

The sine wave ensures smooth looping without jumps at the start/end.

### Video Rendering

Videos are rendered using `imageio` with H.264 codec:
- Codec: libx264
- Pixel Format: yuv420p (universal compatibility)
- Quality: 8 (high quality)

### Symmetry

Like the wallpaper generator, videos maintain horizontal and vertical symmetry by:
1. Generating only one quadrant (top-left)
2. Mirroring to create 4-way symmetry

## Adding New Generators

To add a new video generator:

1. Create a frame generation function:
```python
def generate_my_effect_frame(width, height, palette, t, mirror_h=True, mirror_v=True):
    """
    Generate a single frame at time t (0 to 1).
    """
    img = Image.new('RGB', (width, height))
    # ... your animation logic using time parameter t
    return img
```

2. Add it to the `GENERATORS` dictionary:
```python
GENERATORS = {
    'layered_noise': ('Layered Noise', generate_layered_noise_frame),
    # ... existing generators ...
    'my_effect': ('My Effect', generate_my_effect_frame),
}
```

The script will automatically render your new generator.

## Performance Notes

- **Rendering Time**: Each 10-second video (300 frames) takes approximately 1-5 minutes depending on generator complexity
- **Memory Usage**: Peak memory usage is ~500MB-2GB depending on frame complexity
- **CPU Usage**: Rendering is CPU-intensive (100% of available cores during frame generation)

## Output Directory

All videos are saved to the `videos/` directory:
```
videos/
├── 01_video_layered_noise.mp4
├── 02_video_flow_field.mp4
├── 03_video_interference.mp4
└── ...
```

## Customization

### Color Palette

Edit the `PALETTE` dictionary in `generate_videos.py` to change colors:
```python
PALETTE = {
    'background': hex_to_rgb('#1a1a1a'),
    'colors': [
        hex_to_rgb('#8B0000'),  # Red
        # ... more colors
    ]
}
```

### Animation Speed

Adjust the time offset multiplier to speed up/slow down animations:
```python
# Slower animation
time_offset = math.sin(t * 2 * math.pi) * 50

# Faster animation
time_offset = math.sin(t * 2 * math.pi) * 200
```

### Video Duration

Change `DURATION` in `generate_videos.py`:
```python
DURATION = 5   # 5-second loops
DURATION = 15  # 15-second loops
```

## Differences from Wallpaper Generator

This is an alternative version of the wallpaper-procgen repository that:
- **Generates videos instead of static images**
- **All patterns animate over time**
- **Outputs MP4 files instead of PNG images**
- **Optimized for 1920x1080 video resolution**
- **Adds time parameter to all generators**

## Troubleshooting

### FFmpeg Not Found

If you get an error about FFmpeg:
```bash
# Install FFmpeg via Homebrew (macOS)
brew install ffmpeg

# Or let imageio download it
pip3 install imageio-ffmpeg
```

### Out of Memory

If rendering fails due to memory:
1. Reduce `WIDTH` and `HEIGHT`
2. Reduce `DURATION`
3. Render one video at a time instead of all

### Slow Rendering

To speed up rendering:
1. Reduce `FPS` (e.g., 24fps instead of 30fps)
2. Reduce `DURATION` (e.g., 5 seconds instead of 10)
3. Simplify generator complexity (fewer particles, lower octaves)

## License

Same as wallpaper-procgen - feel free to use and modify!

## Credits

Based on the wallpaper-procgen repository by the same author.
Uses the `procgen` library for noise generation.
