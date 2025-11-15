#!/bin/bash
# Setup automatic wallpaper generation using launchd
# This will run the generator script daily at 9 AM

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PLIST_PATH="$HOME/Library/LaunchAgents/com.procgen.wallpaper.plist"
GENERATOR_SCRIPT="$SCRIPT_DIR/generate_to_photos_album.py"
LOG_DIR="$HOME/Library/Logs/ProcgenWallpapers"

# Create log directory
mkdir -p "$LOG_DIR"

echo "Setting up automatic wallpaper generation..."
echo ""
echo "Configuration:"
echo "  Script: $GENERATOR_SCRIPT"
echo "  Schedule: Daily at 9:00 AM"
echo "  Logs: $LOG_DIR"
echo ""

# Create launchd plist
cat > "$PLIST_PATH" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.procgen.wallpaper</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>$GENERATOR_SCRIPT</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$SCRIPT_DIR</string>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>9</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>$LOG_DIR/wallpaper_generator.log</string>

    <key>StandardErrorPath</key>
    <string>$LOG_DIR/wallpaper_generator_error.log</string>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

# Load the launch agent
launchctl unload "$PLIST_PATH" 2>/dev/null
launchctl load "$PLIST_PATH"

echo "✓ Automatic wallpaper generation configured!"
echo ""
echo "Your procgen art gallery will grow daily at 9:00 AM."
echo ""
echo "Commands:"
echo "  • Run now:      $GENERATOR_SCRIPT"
echo "  • Stop auto:    launchctl unload $PLIST_PATH"
echo "  • Start auto:   launchctl load $PLIST_PATH"
echo "  • View logs:    tail -f $LOG_DIR/wallpaper_generator.log"
echo "  • Test service: launchctl start com.procgen.wallpaper"
echo ""
