#!/usr/bin/env python3
"""
Generate videos for the newly added shader effects.
Includes: feedback, lava lamp, nebula, circuit, warp speed, liquid crystal, fractal flame, oscilloscope.
"""

import subprocess
import sys

NEW_SHADER_GENERATORS = [
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
    print(f"Generating {len(NEW_SHADER_GENERATORS)} new shader videos...")
    cmd = ['python3', 'generate_videos.py'] + NEW_SHADER_GENERATORS
    subprocess.run(cmd)
