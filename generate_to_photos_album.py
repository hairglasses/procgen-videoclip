#!/usr/bin/env python3
"""
Generate procedural wallpapers and add them to a macOS Photos album.
Creates a growing gallery of procgen art that can be used as wallpaper slideshow.
"""

import subprocess
import os
import sys
from pathlib import Path
from generate_wallpapers_multi import (
    WIDTH, HEIGHT, PALETTE, GENERATORS,
    add_logo, random
)

# Configuration
# BATCH_SIZE is now automatically set to number of available generators
PHOTOS_ALBUM_NAME = "Procgen Wallpapers"
TEMP_DIR = Path(__file__).parent / 'temp_wallpapers'

def generate_wallpapers():
    """Generate one wallpaper from each technique and return list of file paths."""
    # Generate one from each technique to ensure variety
    generators_list = list(GENERATORS.items())
    count = len(generators_list)

    print(f"Generating {count} new wallpapers (one from each technique) with h+v symmetry...\n")

    # Get available logos
    logo_dir = Path(__file__).parent / 'logos'
    logos = list(logo_dir.glob('*.png'))

    if not logos:
        print("Error: No logos found in logos/ directory")
        return []

    # Create temp directory
    TEMP_DIR.mkdir(exist_ok=True)

    generated_files = []

    # Generate one wallpaper from each technique
    for i, (gen_id, (gen_name, gen_func)) in enumerate(generators_list):
        print(f"[{i+1}/{count}] Generating {gen_name}...")

        # Generate base wallpaper with mirroring (h+v symmetry enabled by default)
        img = gen_func(WIDTH, HEIGHT, PALETTE['colors'], mirror_h=True, mirror_v=True)

        # Randomly select a logo
        selected_logo = random.choice(logos)
        logo_name = selected_logo.stem

        print(f"  Adding logo: {selected_logo.name}")
        img = add_logo(img, str(selected_logo))

        # Save with timestamp
        import time
        timestamp = int(time.time() * 1000)
        output_file = TEMP_DIR / f'wallpaper_{gen_id}_{logo_name}_{timestamp}.png'
        img.save(str(output_file), 'PNG')
        print(f"  ✓ Saved: {output_file.name}\n")

        generated_files.append(str(output_file))

    return generated_files

def add_to_photos_album(image_paths, album_name=PHOTOS_ALBUM_NAME):
    """Add images to Photos album using AppleScript."""
    if not image_paths:
        print("No images to add to Photos")
        return False

    print(f"\nAdding {len(image_paths)} wallpapers to Photos album '{album_name}'...")

    # Convert paths to POSIX format for AppleScript
    posix_paths = [str(Path(p).resolve()) for p in image_paths]
    paths_list = ', '.join(f'POSIX file "{p}"' for p in posix_paths)

    # AppleScript to import photos and add to album
    applescript = f'''
    tell application "Photos"
        -- Import the photos
        set importedPhotos to {{}}
        repeat with imagePath in {{{paths_list}}}
            set importedPhoto to import imagePath
            if importedPhoto is not missing value then
                set end of importedPhotos to item 1 of importedPhoto
            end if
        end repeat

        -- Create album if it doesn't exist, or get existing album
        set albumExists to false
        repeat with anAlbum in albums
            if name of anAlbum is "{album_name}" then
                set targetAlbum to anAlbum
                set albumExists to true
                exit repeat
            end if
        end repeat

        if not albumExists then
            set targetAlbum to make new album named "{album_name}"
        end if

        -- Add photos to album
        repeat with aPhoto in importedPhotos
            add {{aPhoto}} to targetAlbum
        end repeat

        return (count of importedPhotos)
    end tell
    '''

    try:
        result = subprocess.run(
            ['osascript', '-e', applescript],
            capture_output=True,
            text=True,
            check=True
        )
        count = result.stdout.strip()
        print(f"✓ Successfully added {count} wallpapers to Photos album '{album_name}'")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error adding to Photos: {e}")
        print(f"stderr: {e.stderr}")
        return False

def cleanup_temp_files(image_paths):
    """Remove temporary image files after import."""
    print("\nCleaning up temporary files...")
    for path in image_paths:
        try:
            Path(path).unlink()
        except Exception as e:
            print(f"Warning: Could not delete {path}: {e}")

    # Remove temp directory if empty
    try:
        TEMP_DIR.rmdir()
    except:
        pass

def show_setup_instructions(album_name=PHOTOS_ALBUM_NAME):
    """Display instructions for setting up wallpaper slideshow."""
    print("\n" + "=" * 70)
    print("WALLPAPER SLIDESHOW SETUP")
    print("=" * 70)
    print(f"\nTo use '{album_name}' as your wallpaper slideshow:")
    print("\n1. Open System Settings > Wallpaper")
    print("2. Click the dropdown under 'Dynamic' or current wallpaper")
    print(f"3. Scroll down and select the Photos album: '{album_name}'")
    print("4. Enable 'Change picture' and set your preferred interval")
    print("5. (Optional) Enable 'Random order'")
    print("\n" + "=" * 70)
    batch_count = len(GENERATORS)
    print(f"\n💡 Tip: Run this script anytime to add {batch_count} more wallpapers!")
    print("    Your collection will keep growing as a procgen art gallery.")
    print(f"    (Currently generating one from each of {batch_count} techniques)")
    print("=" * 70)

def main():
    print("=" * 70)
    print("PROCGEN WALLPAPER GALLERY")
    print("=" * 70)
    print()

    # Generate wallpapers (one from each technique)
    wallpaper_paths = generate_wallpapers()

    if not wallpaper_paths:
        print("Failed to generate wallpapers. Exiting.")
        sys.exit(1)

    # Add to Photos album
    success = add_to_photos_album(wallpaper_paths, PHOTOS_ALBUM_NAME)

    if success:
        # Cleanup temp files
        cleanup_temp_files(wallpaper_paths)

        # Show setup instructions
        show_setup_instructions(PHOTOS_ALBUM_NAME)

        print("\n✓ Done! Your procgen art gallery is growing.")
    else:
        print("\n✗ Failed to add to Photos. Temporary files saved in:", TEMP_DIR)
        sys.exit(1)

if __name__ == '__main__':
    main()
