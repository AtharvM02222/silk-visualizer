# Running in GitHub Codespaces

## Quick Start

1. Go to https://github.com/AtharvM02222/silk-visualizer
2. Click the green "Code" button → "Codespaces" tab → "Create codespace on main"
3. Wait for the container to build (~1-2 minutes)
4. Run the visualizer:

```bash
# Use audio2.mp3 that's already in the repo
python3 silk_visualizer.py audio2.mp3 purple --resolution 1080p

# For audio5 from 40s to end:
# First upload audio5.mp3 to the Codespaces workspace, then:
python3 silk_visualizer.py audio5.mp3 purple --resolution 1080p --start 40
```

## Benefits of Codespaces
- Cloud compute (won't slow down your local machine)
- Pre-installed FFmpeg
- Can close the browser and it keeps running
- Download results when complete

## Machine Type
- Default: 2-core (free tier) - slower but free
- Upgrade to 4-core or 8-core for faster rendering (uses paid compute)
  - Go to Codespace settings → Change machine type

## Monitoring Progress
```bash
# FFmpeg will show real-time progress bars
# Look for: frame= fps= time= speed=
```
