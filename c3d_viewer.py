#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight C3D Viewer for Linux (and cross-platform)
Dependencies:
  pip install ezc3d pyqtgraph PyQt5 numpy

Usage:
  python c3d_viewer.py /path/to/file.c3d

Features:
  - Play/Pause animation
  - Frame slider with time display
  - Playback speed control (0.25x ~ 4x)
  - Marker size control
  - Trail length control (draws short lines of recent history)
  - Toggle axes and grid
  - Open another C3D from menu
"""

import sys, os, math, argparse
import numpy as np

try:
    import ezc3d
except Exception as e:
    print("Error: ezc3d not found. Please install with: pip install ezc3d")
    raise

from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl


def safe_get_frame_rate(c3d):
    # Try POINT:RATE first, then default to 100.0 Hz if missing
    try:
        params = c3d['parameters']
        if 'POINT' in params and 'RATE' in params['POINT']:
            rate = float(params['POINT']['RATE']['value'])
            if isinstance(rate, (list, tuple, np.ndarray)):
                rate = float(rate[0])
            if rate > 0:
                return rate
    except Exception:
        pass
    return 100.0


class C3DData:
    def __init__(self, path):
        self.path = path
        self.c3d = ezc3d.c3d(path)
        self.points = self.c3d['data']['points']  # shape: (4, nPoints, nFrames)
        # ezc3d returns NaNs for invalid points. We will keep them as NaNs.
        self.nFrames = self.points.shape[2]
        self.nMarkers = self.points.shape[1]
        self.rate = safe_get_frame_rate(self.c3d)

        # Marker labels
        self.labels = []
        try:
            params = self.c3d['parameters']
            if 'POINT' in params and 'LABELS' in params['POINT']:
                labs = params['POINT']['LABELS']['value']
                self.labels = [str(x) for x in labs]
        except Exception:
            pass
        if not self.labels or len(self.labels) != self.nMarkers:
            self.labels = [f"M{i}" for i in range(self.nMarkers)]

        # Normalize coordinates to meters if necessary: ezc3d usually gives in mm or in meters?
        # ezc3d documentation: points are in mm if SCALE != -1? To be robust, we check 'SCALE' param.
        self.scale = 1.0
        try:
            params = self.c3d['parameters']
            if 'POINT' in params and 'SCALE' in params['POINT']:
                sc = float(params['POINT']['SCALE']['value'])
                # If SCALE is -1, coordinates are in meters (floating). If >0, integer scale is used.
                # Many modern C3D store in mm with SCALE = 1.0 -> we convert to meters.
                # Heuristic: if values look big (thousands), convert mm->m.
                # We'll inspect a few points:
                sample = self.points[:3, :min(10,self.nMarkers), :min(10,self.nFrames)].copy()
                sample = sample[np.isfinite(sample)]
                if sample.size > 0:
                    median_abs = np.nanmedian(np.abs(sample))
                    if median_abs > 10.0:  # likely mm
                        self.scale = 0.001
        except Exception:
            pass

    def frame_xyz(self, idx):
        """Return Nx3 array of xyz at frame idx, in meters (with NaNs preserved)."""
        # points[:3, :, idx] is (3, nMarkers), shape -> (nMarkers, 3)
        xyz = self.points[:3, :, idx].T * self.scale
        return xyz


class GLTrail(gl.GLLinePlotItem):
    """Simple line object for drawing marker trails."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class C3DViewer(QtWidgets.QMainWindow):
    def __init__(self, path=None):
        super().__init__()
        self.setWindowTitle("C3D Viewer (ezc3d + pyqtgraph)")
        self.resize(1200, 800)
        pg.setConfigOptions(antialias=True)

        # State
        self.data = None
        self.curFrame = 0
        self.playing = False
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.on_tick)

        # UI
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        hbox = QtWidgets.QHBoxLayout(central)

        # 3D view
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 2.0
        self.view.opts['elevation'] = 20
        self.view.opts['azimuth'] = 45
        hbox.addWidget(self.view, 1)

        # Right panel controls
        panel = QtWidgets.QFrame()
        panel.setFixedWidth(320)
        v = QtWidgets.QVBoxLayout(panel)
        hbox.addWidget(panel, 0)

        # File info
        self.pathLabel = QtWidgets.QLabel("No file loaded")
        self.pathLabel.setWordWrap(True)
        v.addWidget(self.pathLabel)

        # Time + slider
        self.timeLabel = QtWidgets.QLabel("Frame 0 / 0 | 0.000 s")
        v.addWidget(self.timeLabel)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self.on_slider)
        v.addWidget(self.slider)

        # Play controls
        controls = QtWidgets.QHBoxLayout()
        self.btnPlay = QtWidgets.QPushButton("Play")
        self.btnPlay.clicked.connect(self.toggle_play)
        controls.addWidget(self.btnPlay)

        self.btnPrev = QtWidgets.QPushButton("◀")
        self.btnPrev.clicked.connect(self.step_prev)
        controls.addWidget(self.btnPrev)

        self.btnNext = QtWidgets.QPushButton("▶")
        self.btnNext.clicked.connect(self.step_next)
        controls.addWidget(self.btnNext)
        v.addLayout(controls)

        # Speed
        spdBox = QtWidgets.QHBoxLayout()
        spdBox.addWidget(QtWidgets.QLabel("Speed:"))
        self.speed = QtWidgets.QDoubleSpinBox()
        self.speed.setRange(0.1, 4.0)
        self.speed.setSingleStep(0.1)
        self.speed.setValue(1.0)
        spdBox.addWidget(self.speed)
        v.addLayout(spdBox)

        # Marker size
        msBox = QtWidgets.QHBoxLayout()
        msBox.addWidget(QtWidgets.QLabel("Marker size:"))
        self.markerSize = QtWidgets.QDoubleSpinBox()
        self.markerSize.setRange(1.0, 20.0)
        self.markerSize.setSingleStep(0.5)
        self.markerSize.setValue(6.0)
        self.markerSize.valueChanged.connect(self.update_marker_size)
        msBox.addWidget(self.markerSize)
        v.addLayout(msBox)

        # Trail length
        tlBox = QtWidgets.QHBoxLayout()
        tlBox.addWidget(QtWidgets.QLabel("Trail length (frames):"))
        self.trailLen = QtWidgets.QSpinBox()
        self.trailLen.setRange(0, 200)
        self.trailLen.setValue(20)
        tlBox.addWidget(self.trailLen)
        v.addLayout(tlBox)

        # Checkboxes
        self.chkAxes = QtWidgets.QCheckBox("Show axes")
        self.chkAxes.setChecked(True)
        self.chkAxes.toggled.connect(self.toggle_axes)
        v.addWidget(self.chkAxes)

        self.chkGrid = QtWidgets.QCheckBox("Show floor grid")
        self.chkGrid.setChecked(True)
        self.chkGrid.toggled.connect(self.toggle_grid)
        v.addWidget(self.chkGrid)

        v.addStretch(1)

        # Menu
        menubar = self.menuBar()
        mFile = menubar.addMenu("&File")
        actOpen = QtWidgets.QAction("Open C3D...", self)
        actOpen.triggered.connect(self.open_file_dialog)
        mFile.addAction(actOpen)

        actExit = QtWidgets.QAction("Exit", self)
        actExit.triggered.connect(self.close)
        mFile.addAction(actExit)

        # 3D primitives
        self.scatter = gl.GLScatterPlotItem()
        self.scatter.setGLOptions('opaque')
        self.view.addItem(self.scatter)

        # axes & grid
        self.axes = self.make_axes_item(size=0.5)
        self.view.addItem(self.axes)
        self.grid = gl.GLGridItem()
        self.grid.scale(0.2, 0.2, 1.0)
        self.view.addItem(self.grid)

        # trails (one per marker)
        self.trails = []

        # Load initial file if provided
        if path is not None and os.path.exists(path):
            self.load_file(path)

        # Keyboard shortcuts
        QtWidgets.QShortcut(QtGui.QKeySequence("Space"), self, activated=self.toggle_play)
        QtWidgets.QShortcut(QtGui.QKeySequence("Left"), self, activated=self.step_prev)
        QtWidgets.QShortcut(QtGui.QKeySequence("Right"), self, activated=self.step_next)

    def make_axes_item(self, size=1.0):
        # Simple XYZ axes using 3 lines
        pts = np.array([
            [0,0,0],[size,0,0],  # X (red-ish)
            [0,0,0],[0,size,0],  # Y (green-ish)
            [0,0,0],[0,0,size],  # Z (blue-ish)
        ], dtype=float)
        colors = np.array([
            [1,0,0,1],[1,0,0,1],
            [0,1,0,1],[0,1,0,1],
            [0,0,1,1],[0,0,1,1],
        ], dtype=float)
        item = gl.GLLinePlotItem(pos=pts, color=colors, width=2, mode='lines')
        return item

    def toggle_axes(self, on):
        self.axes.setVisible(on)

    def toggle_grid(self, on):
        self.grid.setVisible(on)

    def open_file_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open C3D", "", "C3D Files (*.c3d)")
        if path:
            self.load_file(path)

    def load_file(self, path):
        try:
            self.data = C3DData(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load {path}\n\n{e}")
            return

        self.pathLabel.setText(os.path.basename(path))
        self.curFrame = 0
        self.slider.setRange(0, self.data.nFrames - 1)
        self.slider.setValue(0)
        self.update_time_label()

        # Reset view
        self.view.setCameraPosition(distance=2.0, elevation=20, azimuth=45)
        self.update_scatter()
        self.clear_trails()
        self.ensure_timer()

    def ensure_timer(self):
        if self.data is None:
            return
        # target GUI refresh ~60Hz, but animation advances based on C3D rate * playback speed
        self.timer.setInterval(int(1000/60))

    def on_slider(self, val):
        self.curFrame = int(val)
        self.update_time_label()
        self.update_scatter()

    def update_time_label(self):
        if self.data is None:
            self.timeLabel.setText("Frame 0 / 0 | 0.000 s")
            return
        t = self.curFrame / max(self.data.rate, 1e-6)
        self.timeLabel.setText(f"Frame {self.curFrame+1} / {self.data.nFrames} | {t:.3f} s @ {self.data.rate:.2f} Hz")

    def toggle_play(self):
        if self.data is None:
            return
        self.playing = not self.playing
        self.btnPlay.setText("Pause" if self.playing else "Play")
        if self.playing and not self.timer.isActive():
            self.timer.start()
        elif not self.playing and self.timer.isActive():
            self.timer.stop()

    def step_prev(self):
        if self.data is None:
            return
        self.curFrame = max(0, self.curFrame - 1)
        self.slider.blockSignals(True)
        self.slider.setValue(self.curFrame)
        self.slider.blockSignals(False)
        self.update_time_label()
        self.update_scatter()

    def step_next(self):
        if self.data is None:
            return
        self.curFrame = min(self.data.nFrames - 1, self.curFrame + 1)
        self.slider.blockSignals(True)
        self.slider.setValue(self.curFrame)
        self.slider.blockSignals(False)
        self.update_time_label()
        self.update_scatter()

    def update_marker_size(self):
        size = float(self.markerSize.value())
        self.scatter.setData(size=size)

    def clear_trails(self):
        for tr in self.trails:
            self.view.removeItem(tr)
        self.trails = []

    def update_scatter(self):
        if self.data is None:
            self.scatter.setData(pos=np.zeros((1,3), dtype=float), size=float(self.markerSize.value()))
            return
        xyz = self.data.frame_xyz(self.curFrame)  # (N,3) with NaNs
        # Replace NaNs with giant values? Better: filter out NaNs by setting size 0 at those points.
        valid = np.all(np.isfinite(xyz), axis=1)
        pos = xyz.copy()
        pos[~valid] = np.nan

        # Update scatter
        self.scatter.setData(pos=pos, size=float(self.markerSize.value()))

        # Update trails if enabled
        L = int(self.trailLen.value())
        # first clear and rebuild to keep implementation simple
        self.clear_trails()
        if L > 0 and self.curFrame > 0:
            start = max(0, self.curFrame - L)
            # We draw per-marker polylines over the last L frames
            for mi in range(self.data.nMarkers):
                seg = []
                for f in range(start, self.curFrame+1):
                    p = self.data.frame_xyz(f)[mi]
                    if np.all(np.isfinite(p)):
                        seg.append(p)
                if len(seg) >= 2:
                    seg = np.array(seg, dtype=float)
                    tr = GLTrail(pos=seg, color=(1,1,1,0.6), width=1.5, mode='line_strip')
                    self.view.addItem(tr)
                    self.trails.append(tr)

    def on_tick(self):
        if self.data is None or not self.playing:
            return
        # advance frames according to playback speed
        # At 60Hz GUI tick, we step ~ rate/60 frames per tick * speed
        step = self.data.rate / 60.0 * float(self.speed.value())
        # accumulate fractional step
        if not hasattr(self, "_facc"):
            self._facc = 0.0
        self._facc += step
        adv = int(self._facc)
        if adv >= 1:
            self._facc -= adv
            self.curFrame += adv
            if self.curFrame >= self.data.nFrames:
                self.curFrame = self.data.nFrames - 1
                self.playing = False
                self.btnPlay.setText("Play")
                self.timer.stop()
            self.slider.blockSignals(True)
            self.slider.setValue(self.curFrame)
            self.slider.blockSignals(False)
            self.update_time_label()
            self.update_scatter()


def main():
    ap = argparse.ArgumentParser(description="Simple C3D Viewer (ezc3d + pyqtgraph)")
    ap.add_argument("c3d", nargs="?", help="Path to .c3d file")
    args = ap.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    # optional: better OpenGL
    pg.setConfigOption('useOpenGL', True)
    pg.setConfigOption('enableExperimental', True)

    path = args.c3d if args.c3d and os.path.exists(args.c3d) else None
    win = C3DViewer(path)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
