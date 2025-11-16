#!/usr/bin/env python3
"""
Generate videos for CPU-based procedural generators.
Includes: fractals, cellular automata, L-systems, boids, and more.
"""

import subprocess
import sys

CPU_GENERATORS = [
    'cellular',
    'interference',
    'grid_distortion',
    'layered_noise',
    'voronoi',
    'spiral',
    'rings',
    'fractal_noise',
    'flow_field',
    'chladni',
    'domain_warp',
    'superformula',
    'strange_attractor',
    'lsystem',
    'boids',
    'reaction_diffusion',
    'differential_growth',
    'bezier_curves',
    'penrose_tiling',
    'physarum',
    'plotter_art',
    'pixel_sprites',
]

if __name__ == '__main__':
    print(f"Generating {len(CPU_GENERATORS)} CPU-based videos...")
    cmd = ['python3', 'generate_videos.py'] + CPU_GENERATORS
    subprocess.run(cmd)
