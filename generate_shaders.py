#!/usr/bin/env python3
"""
Generate videos for all GPU shader-based generators.
Includes: plasma, tunnel, raymarching, fractals, and more.
"""

import subprocess
import sys

SHADER_GENERATORS = [
    'shader_plasma',
    'shader_tunnel',
    'shader_raymarching',
    'shader_mandelbrot',
    'shader_julia',
    'shader_metaballs',
    'shader_rotozoomer',
    'shader_voronoi',
    'shader_kaleidoscope',
    'shader_fire',
    'shader_starfield',
    'shader_hexagons',
    'shader_dna',
    'shader_matrix',
    'shader_waves',
    'shader_clock',
    'shader_caustics',
    'shader_truchet',
    'shader_aurora',
    'shader_moire',
    'shader_perlin_flow',
    'shader_spirograph',
    'shader_electric',
    'shader_glitch',
    'shader_crt',
    'shader_vhs',
    'shader_pixelsort',
    'shader_rgbsplit',
    'shader_dither',
    'shader_teletext',
    'shader_c64',
    'shader_gameboy',
    'shader_feedback',
    'shader_lava',
    'shader_nebula',
    'shader_circuit',
    'shader_warp',
    'shader_liquid_crystal',
    'shader_fractal_flame',
    'shader_oscilloscope',
]

if __name__ == '__main__':
    print(f"Generating {len(SHADER_GENERATORS)} GPU shader videos...")
    cmd = ['python3', 'generate_videos.py'] + SHADER_GENERATORS
    subprocess.run(cmd)
