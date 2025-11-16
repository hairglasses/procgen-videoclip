#!/usr/bin/env python3
"""
Generate videos only for the newly added generators.
"""

from generate_videos import render_video, GENERATORS, OUTPUT_DIR
import os

# List of new generators to generate
NEW_GENERATORS = [
    # Sprite generators (12)
    'swimming_character',
    'climbing_character',
    'crawling_character',
    'combat_character',
    'idle_breathing',
    'sprite_particles',
    'transforming_sprite',
    'multi_sprite_interaction',
    'vehicle_sprites',
    'procedural_sprite',
    'sprite_shadows',
    'emoting_sprites',
    # Voxel generators (15)
    'voxel_trees',
    'voxel_clouds',
    'voxel_water',
    'voxel_fire',
    'voxel_smoke',
    'voxel_maze',
    'voxel_pillars',
    'voxel_spiral',
    'voxel_dna',
    'voxel_bridges',
    'voxel_explosion',
    'voxel_rain',
    'voxel_planets',
    'voxel_characters',
    'smooth_voxels',
    # Shader effects (20)
    'shader_ripple',
    'shader_liquid_metal',
    'shader_oil_slick',
    'shader_ink_diffusion',
    'shader_watercolor',
    'shader_refraction',
    'shader_hologram',
    'shader_godrays',
    'shader_prism',
    'shader_magnetic_field',
    'shader_cel_shading',
    'shader_charcoal',
    'shader_mosaic',
    'shader_stained_glass',
    'shader_neon',
    'shader_snow',
    'shader_rain',
    'shader_sandstorm',
    'shader_fireflies',
    'shader_dandelion',
]

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("GENERATING NEW PROCEDURAL VIDEOS")
    print(f"{'='*60}\n")
    print(f"Generating {len(NEW_GENERATORS)} new videos")
    print(f"Output directory: {OUTPUT_DIR}/\n")

    for idx, gen_id in enumerate(NEW_GENERATORS, 1):
        if gen_id not in GENERATORS:
            print(f"[{idx}/{len(NEW_GENERATORS)}] WARNING: {gen_id} not found in GENERATORS!")
            continue

        gen_name, gen_func = GENERATORS[gen_id]

        print(f"\n[{idx}/{len(NEW_GENERATORS)}] {gen_name}")
        print("-" * 60)

        output_path = os.path.join(OUTPUT_DIR, f"new_{idx:02d}_{gen_id}.mp4")
        render_video(gen_func, output_path)

    print(f"\n{'='*60}")
    print("ALL NEW VIDEOS GENERATED SUCCESSFULLY!")
    print(f"{'='*60}\n")
    print(f"Videos saved to: {OUTPUT_DIR}/")
    print(f"Total new videos: {len(NEW_GENERATORS)}")
