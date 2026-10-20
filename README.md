# Stereo Vision for Robot Obstacle Detection

One file (`stereo_vision.py`), three ways to run it.

## Install
```
pip install opencv-python numpy matplotlib
pip install pyrealsense2        # only for the Intel RealSense D415
```

## Run
| Goal | Command |
|---|---|
| Demo, no camera needed | `python stereo_vision.py` |
| Your own image pair | `python stereo_vision.py --left L.png --right R.png --focal 700 --baseline 0.10` |
| **Intel RealSense D415** (left/right IR only) | `python stereo_vision.py --realsense` |
| Live robot cameras | `python stereo_vision.py --live --cams 0 1 --focal 700 --baseline 0.10` |

Phones as cameras: `--cams http://PHONE1_IP:8080/video http://PHONE2_IP:8080/video` (e.g. IP Webcam app; DroidCam uses port 4747).

RealSense extras: `--rs-size 640x480|848x480|1280x720` IR dot projector is ON by default (`--no-emitter` to start without it; key **L** toggles) `--min-depth 0.4` (closest distance, sets the disparity range). Focal length and baseline are read from the camera's factory calibration, so `--focal` / `--baseline` are not needed. The camera's depth stream is never used.

3D viewer (opens after the report): drag = rotate, mouse wheel or +/- = zoom, **1 / 2 / 3** = front / side / top, **R** = reset, **Q** = close. The default front view is the left camera's own view, so it matches the calibration image. `output/reconstruction3d.png` saves the front / side / top views.

Live keys: **C** calibrate and open the full report (F-matrix, epipoles, epipolar lines, rectified pair, disparity, depth, 3D plot; close the windows to resume live) | **R** report again | **E** epipolar lines | **L** RealSense IR projector | **S** save 3D cloud | **Q** quit.
Live sliders: danger distance, warn distance, window size, disparity range.

## Options
`--focal` (px) `--baseline` (m) `--disp-scale` (1 = sharpest, 0.5 = fast, default for live) `--max-points` (3D viewer) `--danger` (m, red STOP) `--warn` (m, yellow) `--max-depth` `--num-disp` (0 = auto) `--block` (default 5) `--width` `--save-dir` `--no-show`

## How it works
1. **Match** SIFT features between left and right.
2. **F-matrix**: normalized 8-point algorithm (written from scratch with SVD, rank-2 fix, RANSAC to drop bad matches). Rule: `x_right^T F x_left = 0`.
3. **Epipolar geometry**: epipoles = null space of `F` and `F^T`; epipolar line in the right image = `F x_left`. For parallel cameras the epipoles are at infinity and the lines are horizontal.
4. **Rectify** with `stereoRectifyUncalibrated` so matching points share a row.
5. **Disparity** with StereoSGBM (search range chosen automatically from the matches).
6. **Depth**: `Z = focal * baseline / disparity`.
7. **3D**: `X = (u-cx) Z / f`, `Y = (v-cy) Z / f`. Saved as `output/cloud.ply` (open in MeshLab/CloudCompare).
8. **Obstacles**: pixels closer than `--warn` / `--danger` are boxed with their median distance.

## Tips for accurate distances
- Mount both cameras rigidly, same resolution, as parallel as possible, and use hardware-synced or at least fast cameras.
- `--focal` and `--baseline` must be right; they scale every distance. Focal in pixels ~ `width / (2 * tan(FOV/2))`.
- Larger baseline = better far range, but more blind zone up close.
- Because F is estimated without calibration, depth is approximate. For exact metric depth, calibrate the cameras (chessboard) and use `cv2.stereoRectify`.
- Plain walls give no features or matches; the scene needs texture.
