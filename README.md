This repository provides MoCap C3D file loading and visualization on Linux, implemented in Python. See the installation instructions below to set up and run.
# C3D Viewer (ezc3d + PyQt5 + pyqtgraph)

## Install
```bash
pip install ezc3d pyqtgraph PyQt5 numpy
```

## Run
```bash
python c3d_viewer.py /path/to/your_file.c3d
```

## Features
- Play/Pause
- Frame slider + time display
- Playback speed (0.1x ~ 4x)
- Marker size
- Trail length (recent history)
- Toggle axes & grid
- Open file from menu
