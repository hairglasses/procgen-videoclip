#!/usr/bin/env python3
"""
Generate individual procedural videos by name.
Usage: python3 generate_single.py <generator_name> [generator_name2] ...

Examples:
    python3 generate_single.py isometric_cubes
    python3 generate_single.py shader_plasma voxel_terrain walking_character
    python3 generate_single.py --list    # Show all available generators
"""

import subprocess
import sys

# All available generators
AVAILABLE_GENERATORS = {
    # CPU-based generators
    'cellular': 'Cellular Automata',
    'interference': 'Wave Interference',
    'grid_distortion': 'Grid Distortion',
    'layered_noise': 'Layered Noise',
    'voronoi': 'Voronoi Cells',
    'spiral': 'Spiral Patterns',
    'rings': 'Concentric Rings',
    'fractal_noise': 'Fractal Noise',
    'flow_field': 'Flow Field',
    'chladni': 'Chladni Patterns',
    'domain_warp': 'Domain Warping',
    'superformula': 'Superformula',
    'strange_attractor': 'Strange Attractor',
    'lsystem': 'L-System Plants',
    'boids': 'Boids Flocking',
    'reaction_diffusion': 'Reaction-Diffusion',
    'differential_growth': 'Differential Growth',
    'bezier_curves': 'Bezier Curves',
    'penrose_tiling': 'Penrose Tiling',
    'physarum': 'Physarum Slime Mold',
    'plotter_art': 'Plotter Art',
    'pixel_sprites': 'Pixel Sprites',

    # GPU Shaders
    'shader_plasma': 'Shader: Plasma',
    'shader_tunnel': 'Shader: Tunnel Effect',
    'shader_raymarching': 'Shader: Raymarching',
    'shader_mandelbrot': 'Shader: Mandelbrot Fractal',
    'shader_julia': 'Shader: Julia Set',
    'shader_metaballs': 'Shader: Metaballs',
    'shader_rotozoomer': 'Shader: Rotozoomer',
    'shader_voronoi': 'Shader: Voronoi Noise',
    'shader_kaleidoscope': 'Shader: Kaleidoscope',
    'shader_fire': 'Shader: Fire Effect',
    'shader_starfield': 'Shader: Starfield',
    'shader_hexagons': 'Shader: Hexagonal Tiling',
    'shader_dna': 'Shader: DNA Helix',
    'shader_matrix': 'Shader: Matrix Rain',
    'shader_waves': 'Shader: Wave Interference',
    'shader_clock': 'Shader: Clockwork Gears',
    'shader_caustics': 'Shader: Caustics',
    'shader_truchet': 'Shader: Truchet Tiles',
    'shader_aurora': 'Shader: Aurora',
    'shader_moire': 'Shader: Moire Patterns',
    'shader_perlin_flow': 'Shader: Perlin Flow',
    'shader_spirograph': 'Shader: Spirograph',
    'shader_electric': 'Shader: Electric',
    'shader_glitch': 'Shader: Glitch',
    'shader_crt': 'Shader: CRT Screen',
    'shader_vhs': 'Shader: VHS Glitch',
    'shader_pixelsort': 'Shader: Pixel Sort',
    'shader_rgbsplit': 'Shader: RGB Split',
    'shader_dither': 'Shader: Dither',
    'shader_teletext': 'Shader: Teletext',
    'shader_c64': 'Shader: C64',
    'shader_gameboy': 'Shader: Game Boy',
    'shader_feedback': 'Shader: Feedback Loop',
    'shader_lava': 'Shader: Lava Lamp',
    'shader_nebula': 'Shader: Nebula',
    'shader_circuit': 'Shader: Circuit Board',
    'shader_warp': 'Shader: Warp Speed',
    'shader_liquid_crystal': 'Shader: Liquid Crystal',
    'shader_fractal_flame': 'Shader: Fractal Flame',
    'shader_oscilloscope': 'Shader: Oscilloscope',

    # Isometric/Voxel generators
    'isometric_cubes': 'Isometric Cubes',
    'voxel_terrain': 'Voxel Terrain',
    'isometric_city': 'Isometric City',
    'voxel_waves': 'Voxel Waves',
    'isometric_stairs': 'Isometric Stairs',

    # Character generators
    'walking_character': 'Walking Character',
    'jumping_character': 'Jumping Character',
    'dancing_character': 'Dancing Character',
    'flying_character': 'Flying Character',
    'running_character': 'Running Character',
}

def list_generators():
    """List all available generators organized by category."""
    print("\n" + "="*60)
    print("AVAILABLE GENERATORS")
    print("="*60 + "\n")

    categories = {
        'CPU-based (22 generators)': [
            'cellular', 'interference', 'grid_distortion', 'layered_noise',
            'voronoi', 'spiral', 'rings', 'fractal_noise', 'flow_field',
            'chladni', 'domain_warp', 'superformula', 'strange_attractor',
            'lsystem', 'boids', 'reaction_diffusion', 'differential_growth',
            'bezier_curves', 'penrose_tiling', 'physarum', 'plotter_art',
            'pixel_sprites'
        ],
        'GPU Shaders (40 generators)': [
            'shader_plasma', 'shader_tunnel', 'shader_raymarching',
            'shader_mandelbrot', 'shader_julia', 'shader_metaballs',
            'shader_rotozoomer', 'shader_voronoi', 'shader_kaleidoscope',
            'shader_fire', 'shader_starfield', 'shader_hexagons',
            'shader_dna', 'shader_matrix', 'shader_waves', 'shader_clock',
            'shader_caustics', 'shader_truchet', 'shader_aurora',
            'shader_moire', 'shader_perlin_flow', 'shader_spirograph',
            'shader_electric', 'shader_glitch', 'shader_crt', 'shader_vhs',
            'shader_pixelsort', 'shader_rgbsplit', 'shader_dither',
            'shader_teletext', 'shader_c64', 'shader_gameboy',
            'shader_feedback', 'shader_lava', 'shader_nebula',
            'shader_circuit', 'shader_warp', 'shader_liquid_crystal',
            'shader_fractal_flame', 'shader_oscilloscope'
        ],
        'Isometric/Voxel (5 generators)': [
            'isometric_cubes', 'voxel_terrain', 'isometric_city',
            'voxel_waves', 'isometric_stairs'
        ],
        'Characters (5 generators)': [
            'walking_character', 'jumping_character', 'dancing_character',
            'flying_character', 'running_character'
        ]
    }

    for category, generators in categories.items():
        print(f"{category}:")
        for gen in generators:
            print(f"  • {gen:25s} - {AVAILABLE_GENERATORS[gen]}")
        print()

    print(f"Total: {len(AVAILABLE_GENERATORS)} generators\n")

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nUse --list to see all available generators")
        sys.exit(1)

    if sys.argv[1] == '--list' or sys.argv[1] == '-l':
        list_generators()
        sys.exit(0)

    # Validate all requested generators
    requested = sys.argv[1:]
    invalid = [g for g in requested if g not in AVAILABLE_GENERATORS]

    if invalid:
        print(f"\nError: Unknown generator(s): {', '.join(invalid)}")
        print("\nUse --list to see all available generators")
        sys.exit(1)

    # Show what we're generating
    print(f"\nGenerating {len(requested)} video(s):")
    for gen in requested:
        print(f"  • {AVAILABLE_GENERATORS[gen]}")
    print()

    # Run the generator
    cmd = ['python3', 'generate_videos.py'] + requested
    subprocess.run(cmd)

if __name__ == '__main__':
    main()
