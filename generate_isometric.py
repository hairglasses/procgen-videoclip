#!/usr/bin/env python3
"""
Generate videos for isometric and voxel generators.
Includes: isometric cubes, voxel terrain, isometric city, and more.
"""

import subprocess
import sys

ISOMETRIC_GENERATORS = [
    'isometric_cubes',
    'voxel_terrain',
    'isometric_city',
    'voxel_waves',
    'isometric_stairs',
]

if __name__ == '__main__':
    print(f"Generating {len(ISOMETRIC_GENERATORS)} isometric/voxel videos...")
    cmd = ['python3', 'generate_videos.py'] + ISOMETRIC_GENERATORS
    subprocess.run(cmd)
