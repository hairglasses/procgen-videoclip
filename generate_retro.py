#!/usr/bin/env python3
"""
Generate videos for retro/glitch shader generators.
Includes: CRT, VHS, C64, Game Boy, pixel effects, and more.
"""

import subprocess
import sys

RETRO_GENERATORS = [
    'shader_crt',
    'shader_vhs',
    'shader_pixelsort',
    'shader_rgbsplit',
    'shader_dither',
    'shader_teletext',
    'shader_c64',
    'shader_gameboy',
    'shader_glitch',
]

if __name__ == '__main__':
    print(f"Generating {len(RETRO_GENERATORS)} retro/glitch shader videos...")
    cmd = ['python3', 'generate_videos.py'] + RETRO_GENERATORS
    subprocess.run(cmd)
