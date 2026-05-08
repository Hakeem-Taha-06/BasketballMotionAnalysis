# Basketball Free Throw — Motion Tracking Analysis

Kinematic analysis of a basketball free throw using markerless pose estimation. The script uses [Sports2D](https://github.com/davidpagnon/Sports2D) (RTMLib-based) to extract body landmarks, then computes linear and angular kinematics and identifies the optimal release window via projectile physics.

## What It Does

1. **Pose estimation** — Sports2D detects 2D body landmarks and converts pixel coordinates to real-world metres using the player's known height.
2. **Linear kinematics** — Computes wrist displacement, velocity, and acceleration from the tracked landmark positions using Savitzky–Golay smoothing and central differences.
3. **Angular kinematics** — Reads joint angles (elbow, knee, trunk) from Sports2D's `.mot` output and derives angular velocity and acceleration.
4. **Release window analysis** — Filters frames by biomechanical constraints (forward/upward motion, hand above shoulder, valid launch angle), then for each candidate frame calculates the projectile launch speed required to reach the hoop and compares it to the actual wrist speed.
5. **Projectile path visualisation** — Overlays the predicted parabolic trajectory onto each candidate frame image using a per-frame affine transform for accurate metre-to-pixel mapping.

## Output Files

All outputs are saved to the `results/` directory:

| File | Description |
|------|-------------|
| `wrist_position_s2d.png` | X/Y wrist position over time |
| `wrist_velocity_s2d.png` | Wrist velocity over time |
| `wrist_acceleration_s2d.png` | Wrist acceleration over time |
| `elbow_angle_s2d.png` | Elbow angle over time |
| `elbow_angular_velocity_s2d.png` | Elbow angular velocity |
| `knee_angle_s2d.png` | Knee angle over time |
| `all_joint_angles_s2d.png` | All joint angles on one plot |
| `output_annotated_s2d.mp4` | Video with skeleton overlay and HUD |
| `positions_s2d.csv` | Raw landmark positions |
| `angles_s2d.csv` | Raw joint angles |
| `release_error_s2d.png` | Relative error per candidate frame |
| `release_ratio_s2d.png` | Feasibility ratio per candidate frame |
| `release_speed_comparison_s2d.png` | Actual vs required wrist speed |
| `release_analysis_s2d.csv` | Full release window data |
| `debug/` | Candidate frames with projected trajectory arc |

## Setup

**Python 3.12** is recommended (Sports2D works best with it).
1. Create a virtual environment
```bash
python -m venv .venv_s2d
```
2. Activate it

Windows:
```bash
.venv_s2d\Scripts\activate
```
macOS / Linux:
```bash
source .venv_s2d/bin/activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Usage

```bash
python motion_analysis_s2d.py <video_path> <player_height_m>
```

**Arguments:**

| Argument | Description | Default |
|----------|-------------|---------|
| `video_path` | Path to the video file | `free_throw.mp4` |
| `player_height_m` | Player's height in metres (used for pixel→metre calibration) | `1.75` |

**Example:**

```bash
python motion_analysis_s2d.py free_throw.mp4 1.83
```

During execution the script will open a window showing the first frame — **click on the centre of the hoop** to set the target for the release window analysis, then wait for it to process.

## Configuration

Key constants at the top of `motion_analysis_s2d.py`:

| Constant | Description |
|----------|-------------|
| `PERSON_ID` | Which detected person to track (`"person00"`, `"person01"`, …) |
| `MIN_LAUNCH_ANGLE_DEG` / `MAX_LAUNCH_ANGLE_DEG` | Launch angle range for candidate frame filtering |
| `OPTIMAL_WINDOW_THRESHOLD_PCT` | Relative error threshold for the optimal release window |

## Dependencies

| Package | Role |
|---------|------|
| [Sports2D](https://github.com/davidpagnon/Sports2D) | Markerless pose estimation and angle computation |
| OpenCV | Video I/O, frame drawing, interactive hoop selection |
| NumPy | Array operations and linear algebra |
| Pandas | CSV reading/writing and data handling |
| SciPy | Savitzky–Golay smoothing filter |
| Matplotlib | Plot generation |
