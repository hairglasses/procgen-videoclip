#!/bin/bash
#
# Uninstall the wallpaper generator login item
#

PLIST_NAME="com.procgen.wallpaper.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$LAUNCH_AGENTS_DIR/$PLIST_NAME"

echo "========================================"
echo "Uninstalling Wallpaper Generator"
echo "========================================"
echo ""

if [ -f "$PLIST_PATH" ]; then
    # Unload the launch agent
    launchctl unload "$PLIST_PATH" 2>/dev/null

    # Remove the plist file
    rm "$PLIST_PATH"

    echo "✓ LaunchAgent uninstalled successfully"
    echo ""
    echo "The wallpaper generator will no longer run at login."
    echo "Your existing wallpapers will remain in the wallpapers/ directory."
else
    echo "✗ LaunchAgent plist not found"
    echo "It may have already been uninstalled."
fi
