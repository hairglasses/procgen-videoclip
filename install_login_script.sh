#!/bin/bash
#
# Install the wallpaper generator as a login item
# This will run the wallpaper generator every time you log in
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLIST_NAME="com.procgen.wallpaper.plist"
LAUNCH_AGENTS_DIR="$HOME/Library/LaunchAgents"
PLIST_PATH="$LAUNCH_AGENTS_DIR/$PLIST_NAME"
SETUP_SCRIPT="$SCRIPT_DIR/setup_wallpaper_slideshow.py"
PYTHON_PATH=$(which python3)

echo "========================================"
echo "Installing Wallpaper Generator at Login"
echo "========================================"
echo ""

# Create LaunchAgents directory if it doesn't exist
mkdir -p "$LAUNCH_AGENTS_DIR"

# Create the plist file
cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$PLIST_NAME</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_PATH</string>
        <string>$SETUP_SCRIPT</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>StandardOutPath</key>
    <string>$SCRIPT_DIR/wallpaper_generator.log</string>

    <key>StandardErrorPath</key>
    <string>$SCRIPT_DIR/wallpaper_generator_error.log</string>

    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>
</dict>
</plist>
EOF

echo "✓ Created LaunchAgent plist at:"
echo "  $PLIST_PATH"
echo ""

# Load the launch agent
launchctl unload "$PLIST_PATH" 2>/dev/null
launchctl load "$PLIST_PATH"

if [ $? -eq 0 ]; then
    echo "✓ LaunchAgent loaded successfully"
    echo ""
    echo "The wallpaper generator will now run:"
    echo "  - At every login"
    echo "  - Generating 6 new wallpapers"
    echo "  - Setting them as a 30-minute slideshow"
    echo ""
    echo "Logs will be saved to:"
    echo "  $SCRIPT_DIR/wallpaper_generator.log"
    echo "  $SCRIPT_DIR/wallpaper_generator_error.log"
    echo ""
    echo "To test now, run:"
    echo "  python3 $SETUP_SCRIPT"
    echo ""
    echo "To uninstall, run:"
    echo "  $SCRIPT_DIR/uninstall_login_script.sh"
else
    echo "✗ Failed to load LaunchAgent"
    echo "Please check the plist file and try again"
    exit 1
fi
