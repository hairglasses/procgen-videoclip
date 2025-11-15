#!/usr/bin/env python3
"""
Generate new wallpapers and configure macOS to use them as a slideshow.
This script can be run at login to create fresh wallpapers daily.
"""

import subprocess
import os
import sys
from pathlib import Path

# Get the directory where this script is located
SCRIPT_DIR = Path(__file__).parent.absolute()
WALLPAPER_DIR = SCRIPT_DIR / 'wallpapers'
GENERATOR_SCRIPT = SCRIPT_DIR / 'generate_wallpapers_multi.py'

def generate_wallpapers():
    """Generate new wallpapers using the multi-generator script."""
    print("Generating new wallpapers...")
    try:
        result = subprocess.run(
            [sys.executable, str(GENERATOR_SCRIPT)],
            cwd=str(SCRIPT_DIR),
            capture_output=True,
            text=True,
            check=True
        )
        print(result.stdout)
        print("✓ Wallpapers generated successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error generating wallpapers: {e}")
        print(e.stderr)
        return False

def set_wallpaper_slideshow():
    """Configure macOS to use the wallpapers directory as a slideshow."""
    print("\nConfiguring wallpaper slideshow...")
    print("Note: Due to macOS security, automatic slideshow configuration may require permissions.")
    print("\nAttempting to open System Settings to wallpaper preferences...")

    # Try to open System Settings to wallpaper preferences
    try:
        # Open System Settings to Desktop & Screen Saver
        subprocess.run(['open', 'x-apple.systempreferences:com.apple.Wallpaper-Settings.extension'], check=False)
        print("\n✓ Opened System Settings > Wallpaper")
    except:
        print("Could not automatically open System Settings")

    print("\n" + "─" * 60)
    print("MANUAL SETUP REQUIRED:")
    print("─" * 60)
    print("\nTo set up the slideshow in System Settings (now open):")
    print(f"\n  1. Click the '+' button (Add Folder)")
    print(f"  2. Navigate to and select:")
    print(f"     {WALLPAPER_DIR}")
    print(f"\n  3. After selecting the folder, configure:")
    print(f"     • Enable: 'Change picture'")
    print(f"     • Set interval: Every 30 minutes")
    print(f"     • Enable: 'Random order' (optional)")
    print("\n  4. Close System Settings")
    print("─" * 60)

    # Wait for user confirmation
    input("\nPress Enter once you've completed the setup...")
    return True

def main():
    print("=" * 60)
    print("Wallpaper Slideshow Setup")
    print("=" * 60)
    print()

    # Generate wallpapers
    if not generate_wallpapers():
        print("\nFailed to generate wallpapers. Exiting.")
        sys.exit(1)

    # Set up slideshow
    set_wallpaper_slideshow()

    print("\n" + "=" * 60)
    print("✓ Setup complete! Your wallpapers will change every 30 minutes.")
    print("=" * 60)

if __name__ == '__main__':
    main()
