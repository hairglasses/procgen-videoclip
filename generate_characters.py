#!/usr/bin/env python3
"""
Generate videos for sprite character generators.
Includes: walking, jumping, dancing, flying, and running characters.
"""

import subprocess
import sys

CHARACTER_GENERATORS = [
    'walking_character',
    'jumping_character',
    'dancing_character',
    'flying_character',
    'running_character',
]

if __name__ == '__main__':
    print(f"Generating {len(CHARACTER_GENERATORS)} sprite character videos...")
    cmd = ['python3', 'generate_videos.py'] + CHARACTER_GENERATORS
    subprocess.run(cmd)
