#!/usr/bin/env python3
"""
Kinematic Motion Analysis of a Basketball Free Throw  (Sports2D edition)
========================================================================
Replicates the functionality of motion_analysis.py but uses **Sports2D**
(RTMLib-based pose estimation) instead of MediaPipe.

Sports2D handles:
    • Pose estimation (via RTMLib)
    • Pixel → metre conversion  (via player height)
    • Joint & segment angle computation
    • Butterworth filtering
    • Annotated video + TRC/MOT file output

This script orchestrates the Sports2D pipeline and then post-processes
the output files to:
    1) Compute linear kinematics  (displacement, velocity, acceleration)
    2) Print a kinematic summary table
    3) Generate 7 matplotlib plots
    4) Write a custom annotated output video with a HUD overlay

Run with the Sports2D venv:
    .venv_s2d\\Scripts\\python.exe motion_analysis_s2d.py [video_path] [player_height_m]

Required:  pip install sports2d opencv-python numpy scipy matplotlib
"""

import sys
import os
import glob
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from Sports2D import Sports2D


# ── Configuration ──────────────────────────────────────────────────────────
VIDEO_PATH   = "free_throw.mp4"
RESULTS_DIR  = "results"
OUTPUT_VIDEO = os.path.join(RESULTS_DIR, "output_annotated_s2d.mp4")
PERSON_ID    = "person00"   # "person00", "person01", etc.

# Default player height for pixel → metre conversion
DEFAULT_HEIGHT = 1.75  # metres — override via CLI arg

# Landmarks of interest (Sports2D HALPE_26 marker names in the TRC file)
TRC_MARKERS = {
    "RIGHT_HIP":      "RHip",
    "RIGHT_KNEE":     "RKnee",
    "RIGHT_ANKLE":    "RAnkle",
    "RIGHT_SHOULDER": "RShoulder",
    "RIGHT_ELBOW":    "RElbow",
    "RIGHT_WRIST":    "RWrist",
}

LANDMARK_NAMES = list(TRC_MARKERS.keys())

# Angles of interest from the MOT file
MOT_ANGLES = {
    "knee":  "right knee",
    "elbow": "right elbow",
    "trunk": "trunk",
}

# Savitzky-Golay parameters (used for linear kinematics only —
# Sports2D already filters the pose/angle data with Butterworth)
SG_WINDOW = 11
SG_POLY   = 3

# Pose connection pairs for manual skeleton drawing using HALPE_26 IDs
POSE_CONNECTIONS = [
    ("LHip", "LKnee"), ("LKnee", "LAnkle"),
    ("LHip", "LShoulder"), ("LShoulder", "LElbow"), ("LElbow", "LWrist"),
    ("LAnkle", "LBigToe"), ("LAnkle", "LHeel"),
    ("Hip", "LHip"), ("Hip", "RHip"),
    ("RHip", "RKnee"), ("RKnee", "RAnkle"),
    ("Neck", "RShoulder"), ("Neck", "LShoulder"),
    ("RShoulder", "RElbow"), ("RElbow", "RWrist"),
    ("Neck", "Head"), ("Head", "Nose"),
    ("RAnkle", "RBigToe"), ("RAnkle", "RHeel"),
]


# ═══════════════════════════════════════════════════════════════════════════
# Step 1 — Run Sports2D
# ═══════════════════════════════════════════════════════════════════════════

def run_sports2d(video_path: str, player_height: float, result_dir: str):
    """Run the Sports2D pipeline on the video.

    Returns the path to the result directory where TRC/MOT files are saved.
    """
    video_path = os.path.abspath(video_path)
    video_dir  = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)

    config_dict = {
        "base": {
            "video_input":           video_name,
            "video_dir":             video_dir,
            "result_dir":            result_dir,
            "nb_persons_to_detect":  2,
            "person_ordering_method": "greatest_displacement",
            "first_person_height":   player_height,
            "show_realtime_results": False,
            "save_vid":              True,
            "save_img":              False,
            "save_pose":             True,
            "calculate_angles":      True,
            "save_angles":           True,
        },
        "pose": {
            "mode":          "balanced",
            "det_frequency": 4,
        },
        "angles": {
            "joint_angles":   ["Right knee", "Right elbow"],
            "segment_angles": ["Trunk"],
            "display_angle_values_on": "none",
            "flip_left_right": False,
        },
        "px_to_meters_conversion": {
            "to_meters":  True,
            "make_c3d":   False,
            "save_calib": False,
        },
        "post-processing": {
            "interpolate":    True,
            "filter":         True,
            "filter_type":    "butterworth",
            "show_graphs":    False,
            "save_graphs":    False,
            "butterworth": {
                "cut_off_frequency": 6,
                "order": 4,
            },
        },
        "logging": {
            "use_custom_logging": False,
        },
    }

    print(f"[INFO] Running Sports2D on: {video_path}")
    print(f"[INFO] Player height: {player_height:.2f} m")
    try:
        Sports2D.process(config_dict)
    except Exception as e:
        print(f"\n[WARN] Sports2D encountered an error during post-processing: {e}")
        print("[INFO] Attempting to continue with the already generated TRC/MOT files...\n")
    print("[INFO] Sports2D processing step complete.")
    return result_dir


# ═══════════════════════════════════════════════════════════════════════════
# Step 2 — Parse TRC file  (landmark positions in metres)
# ═══════════════════════════════════════════════════════════════════════════

def parse_trc(trc_path: str):
    """Parse a Sports2D TRC file.

    Returns
    -------
    landmarks : dict[str, np.ndarray]   {our_name: (N, 2)}  in metres
    time_arr  : np.ndarray              (N,) in seconds
    all_markers_px : dict[str, np.ndarray]  all markers in pixel coords (for drawing)
    """
    # TRC has 5 header lines then data rows
    # Line 4 has marker names (every 3rd column starting at col 2)
    with open(trc_path, "r") as f:
        lines = f.readlines()

    # Extract marker names from header line 4 (0-indexed line 3)
    marker_line = lines[3].strip().split("\t")
    # marker_line: ['Frame#', 'Time', 'Marker1', '', '', 'Marker2', '', '', ...]
    marker_names = [m for m in marker_line[2:] if m != ""]

    # Build column names
    cols = ["Frame", "Time"]
    for m in marker_names:
        cols.extend([f"{m}_X", f"{m}_Y", f"{m}_Z"])

    # Read data (skip 5 header lines)
    df = pd.read_csv(trc_path, sep="\t", skiprows=5, header=None,
                     names=cols, index_col=False)

    time_arr = df["Time"].values

    landmarks = {}
    for our_name, trc_name in TRC_MARKERS.items():
        x_col = f"{trc_name}_X"
        y_col = f"{trc_name}_Y"
        if x_col in df.columns and y_col in df.columns:
            landmarks[our_name] = np.column_stack([
                df[x_col].values, df[y_col].values
            ])
        else:
            warnings.warn(f"Marker '{trc_name}' not found in TRC — using zeros.")
            landmarks[our_name] = np.zeros((len(time_arr), 2))

    # Also collect ALL markers for skeleton drawing
    all_markers = {}
    for m in marker_names:
        x_col = f"{m}_X"
        y_col = f"{m}_Y"
        if x_col in df.columns and y_col in df.columns:
            all_markers[m] = np.column_stack([
                df[x_col].values, df[y_col].values
            ])

    print(f"[INFO] Parsed TRC: {len(time_arr)} frames, "
          f"{len(marker_names)} markers.")
    return landmarks, time_arr, all_markers


# ═══════════════════════════════════════════════════════════════════════════
# Step 3 — Parse MOT file  (joint & segment angles)
# ═══════════════════════════════════════════════════════════════════════════

def parse_mot(mot_path: str):
    """Parse a Sports2D MOT file.

    Returns
    -------
    angles : dict[str, np.ndarray]  {our_name: (N,) in degrees}
    """
    # MOT file: header lines until 'endheader', then column names, then data
    with open(mot_path, "r") as f:
        lines = f.readlines()

    # Find the 'endheader' line
    header_end = 0
    for i, line in enumerate(lines):
        if line.strip() == "endheader":
            header_end = i
            break

    # Column names are on the line after 'endheader'
    col_line = lines[header_end + 1].strip().split("\t")

    # Data starts after that
    df = pd.read_csv(mot_path, sep="\t", skiprows=header_end + 2,
                     header=None, names=col_line, index_col=False)

    angles = {}
    for our_name, mot_name in MOT_ANGLES.items():
        if mot_name in df.columns:
            angles[our_name] = df[mot_name].values
        else:
            warnings.warn(f"Angle '{mot_name}' not found in MOT — using zeros.")
            angles[our_name] = np.zeros(len(df))

    print(f"[INFO] Parsed MOT: {len(df)} frames, "
          f"columns = {list(df.columns)}")
    return angles


# ═══════════════════════════════════════════════════════════════════════════
# Step 3b — Export parsed data to CSV
# ═══════════════════════════════════════════════════════════════════════════

def save_positions_csv(landmarks: dict, time_arr: np.ndarray,
                       out_path: str = os.path.join(RESULTS_DIR, "positions_s2d.csv")) -> None:
    """Save marker positions (metres) to a tidy CSV file."""
    data = {"time": time_arr}
    for name, pos in landmarks.items():
        marker = TRC_MARKERS[name]          # e.g. "LHip"
        data[f"{marker}_X"] = pos[:, 0]
        data[f"{marker}_Y"] = pos[:, 1]
    pd.DataFrame(data).to_csv(out_path, index=False)
    print(f"[INFO] Positions CSV saved → {out_path}")


def save_angles_csv(angles: dict, time_arr: np.ndarray,
                    out_path: str = os.path.join(RESULTS_DIR, "angles_s2d.csv")) -> None:
    """Save joint/segment angles (degrees) to a tidy CSV file."""
    data = {"time": time_arr}
    for name, ang in angles.items():
        data[MOT_ANGLES[name]] = ang        # original MOT column name
    pd.DataFrame(data).to_csv(out_path, index=False)
    print(f"[INFO] Angles CSV saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Step 4 — Linear Kinematics  (computed from TRC positions)
# ═══════════════════════════════════════════════════════════════════════════

def _central_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """Central finite differences with forward/backward at boundaries."""
    deriv = np.empty_like(arr, dtype=float)
    if arr.ndim == 1:
        deriv[0]    = (arr[1]  - arr[0])    / dt
        deriv[-1]   = (arr[-1] - arr[-2])   / dt
        deriv[1:-1] = (arr[2:] - arr[:-2])  / (2.0 * dt)
    else:
        deriv[0]    = (arr[1]  - arr[0])    / dt
        deriv[-1]   = (arr[-1] - arr[-2])   / dt
        deriv[1:-1] = (arr[2:] - arr[:-2])  / (2.0 * dt)
    return deriv


def _smooth(arr: np.ndarray) -> np.ndarray:
    """Apply Savitzky-Golay smoothing, handling short/multidim arrays."""
    win = min(SG_WINDOW, len(arr) if arr.ndim == 1 else arr.shape[0])
    if win % 2 == 0:
        win -= 1
    if win < SG_POLY + 2:
        return arr
    if arr.ndim == 1:
        return savgol_filter(arr, win, SG_POLY)
    # Smooth each column independently for 2-D arrays
    out = arr.copy()
    for c in range(arr.shape[1]):
        out[:, c] = savgol_filter(arr[:, c], win, SG_POLY)
    return out


def compute_linear_kinematics(landmarks: dict, time: np.ndarray) -> dict:
    """Compute displacement, velocity, and acceleration for every landmark.

    Returns nested dict:
        {landmark: {disp, vel, acc, disp_mag, vel_mag, acc_mag}}
    """
    dt = float(time[1] - time[0]) if len(time) > 1 else 1.0 / 30.0
    kin = {}

    for name, pos in landmarks.items():
        disp     = pos - pos[0]
        disp_mag = np.linalg.norm(disp, axis=1)
        vel      = _smooth(_central_diff(pos, dt))
        vel_mag  = np.linalg.norm(vel, axis=1)
        acc      = _smooth(_central_diff(vel, dt))
        acc_mag  = np.linalg.norm(acc, axis=1)

        kin[name] = dict(
            disp=disp, vel=vel, acc=acc,
            disp_mag=disp_mag, vel_mag=vel_mag, acc_mag=acc_mag,
        )

    print("[INFO] Linear kinematics computed for all landmarks.")
    return kin


# ═══════════════════════════════════════════════════════════════════════════
# Step 5 — Angular Kinematics  (from Sports2D MOT angles)
# ═══════════════════════════════════════════════════════════════════════════

def compute_angular_kinematics(angles: dict, time: np.ndarray) -> dict:
    """Compute angular velocity and acceleration from Sports2D angles.

    Sports2D already provides joint/segment angles in degrees; we just
    need to differentiate for velocity and acceleration.

    Returns
    -------
    angular_kin : dict
        {name: {angle, angular_vel, angular_acc}}
    """
    dt = float(time[1] - time[0]) if len(time) > 1 else 1.0 / 30.0
    angular_kin = {}

    for name, angle_deg in angles.items():
        ang_vel = _smooth(_central_diff(angle_deg, dt))   # deg/s
        ang_acc = _smooth(_central_diff(ang_vel,   dt))   # deg/s²

        angular_kin[name] = dict(
            angle=angle_deg,
            angular_vel=ang_vel,
            angular_acc=ang_acc,
        )

    print("[INFO] Angular kinematics computed (knee, elbow, trunk).")
    return angular_kin


# ═══════════════════════════════════════════════════════════════════════════
# Step 6 — Summary Table
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(linear_kin: dict, angular_kin: dict) -> None:
    """Print a formatted summary table of key kinematic variables."""

    def _row(label, arr):
        return (f"  {label:<40s}  "
                f"{np.nanmin(arr):>10.3f}  {np.nanmax(arr):>10.3f}  "
                f"{np.nanmean(arr):>10.3f}")

    sep = "  " + "─" * 76
    print("\n" + "=" * 80)
    print("  KINEMATIC SUMMARY  (Sports2D)")
    print("=" * 80)
    print(f"  {'Variable':<40s}  {'Min':>10s}  {'Max':>10s}  {'Mean':>10s}")
    print(sep)

    wk = linear_kin["RIGHT_WRIST"]
    print("  ── RIGHT WRIST (linear) ──")
    print(_row("Displacement magnitude (m)", wk["disp_mag"]))
    print(_row("Velocity magnitude (m/s)",   wk["vel_mag"]))
    print(_row("Acceleration magnitude (m/s²)", wk["acc_mag"]))
    print(sep)

    ea = angular_kin["elbow"]
    print("  ── ELBOW ANGLE (angular, Sports2D) ──")
    print(_row("Angle (°)",               ea["angle"]))
    print(_row("Angular velocity (°/s)",  ea["angular_vel"]))
    print(_row("Angular acceleration (°/s²)", ea["angular_acc"]))
    print(sep)

    tr = angular_kin["trunk"]
    print("  ── TRUNK (segment angle, Sports2D) ──")
    print(_row("Trunk angle (°)",                 tr["angle"]))
    print(_row("Trunk angular velocity (°/s)",    tr["angular_vel"]))
    print(_row("Trunk angular acceleration (°/s²)", tr["angular_acc"]))

    print("=" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# Step 7 — Plots
# ═══════════════════════════════════════════════════════════════════════════

def plot_all(time: np.ndarray, linear_kin: dict, angular_kin: dict) -> None:
    """Generate and save 7 matplotlib figures as PNGs."""

    def _save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS_DIR, name), dpi=150)
        plt.close(fig)
        print(f"  → saved {name}")

    wk = linear_kin["RIGHT_WRIST"]

    # 1. Wrist position (x, y) vs time
    fig, ax = plt.subplots()
    ax.plot(time, wk["disp"][:, 0], label="x displacement (m)")
    ax.plot(time, wk["disp"][:, 1], label="y displacement (m)")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Displacement (m)")
    ax.set_title("Wrist Position (displacement from frame 0)")
    ax.legend()
    _save(fig, "wrist_position_s2d.png")

    # 2. Wrist linear velocity magnitude
    fig, ax = plt.subplots()
    ax.plot(time, wk["vel_mag"], color="tab:orange")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Wrist Linear Velocity Magnitude")
    _save(fig, "wrist_velocity_s2d.png")

    # 3. Wrist linear acceleration magnitude
    fig, ax = plt.subplots()
    ax.plot(time, wk["acc_mag"], color="tab:red")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Acceleration (m/s²)")
    ax.set_title("Wrist Linear Acceleration Magnitude")
    _save(fig, "wrist_acceleration_s2d.png")

    # 4. Elbow joint angle
    fig, ax = plt.subplots()
    ax.plot(time, angular_kin["elbow"]["angle"], color="tab:green")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (°)")
    ax.set_title("Elbow Joint Angle (Sports2D)")
    _save(fig, "elbow_angle_s2d.png")

    # 5. Elbow angular velocity
    fig, ax = plt.subplots()
    ax.plot(time, angular_kin["elbow"]["angular_vel"], color="tab:purple")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angular Velocity (°/s)")
    ax.set_title("Elbow Angular Velocity")
    _save(fig, "elbow_angular_velocity_s2d.png")

    # 6. Knee joint angle
    fig, ax = plt.subplots()
    ax.plot(time, angular_kin["knee"]["angle"], color="tab:blue")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (°)")
    ax.set_title("Knee Joint Angle (Sports2D)")
    _save(fig, "knee_angle_s2d.png")

    # 7. All joint/segment angles overlaid
    fig, ax = plt.subplots()
    ax.plot(time, angular_kin["knee"]["angle"],  label="Knee")
    ax.plot(time, angular_kin["elbow"]["angle"], label="Elbow")
    ax.plot(time, angular_kin["trunk"]["angle"], label="Trunk")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (°)")
    ax.set_title("Joint Angles – Knee, Elbow, Trunk (Sports2D)")
    ax.legend()
    _save(fig, "all_joint_angles_s2d.png")

    print("[INFO] All plots saved.")


# ═══════════════════════════════════════════════════════════════════════════
# Step 8 — Annotated Output Video  (custom HUD using TRC pixel coords)
# ═══════════════════════════════════════════════════════════════════════════

def parse_trc_px(trc_px_path: str):
    """Parse the *pixel-coordinate* TRC file to get drawing positions.

    Returns dict {marker_name: (N, 2) array of pixel coords}.
    """
    with open(trc_px_path, "r") as f:
        lines = f.readlines()

    marker_line = lines[3].strip().split("\t")
    marker_names = [m for m in marker_line[2:] if m != ""]

    cols = ["Frame", "Time"]
    for m in marker_names:
        cols.extend([f"{m}_X", f"{m}_Y", f"{m}_Z"])

    df = pd.read_csv(trc_px_path, sep="\t", skiprows=5, header=None,
                     names=cols, index_col=False)

    markers_px = {}
    for m in marker_names:
        x_col = f"{m}_X"
        y_col = f"{m}_Y"
        if x_col in df.columns and y_col in df.columns:
            markers_px[m] = np.column_stack([
                df[x_col].values, df[y_col].values
            ])

    return markers_px


def write_output_video(video_path: str, fps: float, width: int, height: int,
                       markers_px: dict,
                       linear_kin: dict, angular_kin: dict,
                       time_arr: np.ndarray) -> None:
    """Write annotated video with skeleton, coloured landmarks, and HUD."""

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    COLORS = {
        "RHip":      (255,   0,   0),
        "RKnee":     (  0, 255,   0),
        "RAnkle":    (  0,   0, 255),
        "RShoulder": (255, 255,   0),
        "RElbow":    (255,   0, 255),
        "RWrist":    (  0, 255, 255),
    }

    cap = cv2.VideoCapture(video_path)
    n = len(time_arr)

    for i in range(n):
        ret, frame = cap.read()
        if not ret:
            break

        # Draw skeleton connections
        for (a, b) in POSE_CONNECTIONS:
            if a in markers_px and b in markers_px:
                pt_a = markers_px[a][i]
                pt_b = markers_px[b][i]
                if not (np.isnan(pt_a).any() or np.isnan(pt_b).any()):
                    cv2.line(frame,
                             (int(pt_a[0]), int(pt_a[1])),
                             (int(pt_b[0]), int(pt_b[1])),
                             (0, 255, 0), 2)

        # Draw all landmark dots
        for m_name, coords in markers_px.items():
            pt = coords[i]
            if not np.isnan(pt).any():
                color = COLORS.get(m_name, (200, 200, 200))
                radius = 7 if m_name in COLORS else 3
                cv2.circle(frame, (int(pt[0]), int(pt[1])),
                           radius, color, -1)

        # ── HUD overlay ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (380, 110), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        t_str = f"Time: {time_arr[i]:.2f} s"
        v_str = f"Wrist vel: {linear_kin['RIGHT_WRIST']['vel_mag'][i]:.2f} m/s"
        a_str = f"Elbow angle: {angular_kin['elbow']['angle'][i]:.1f} deg"

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, t_str, (20, 38),  font, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, v_str, (20, 65),  font, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, a_str, (20, 92),  font, 0.65, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    cap.release()
    print(f"[INFO] Custom annotated video saved → {OUTPUT_VIDEO}")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers — locate Sports2D output files
# ═══════════════════════════════════════════════════════════════════════════

def _find_file(result_dir: str, pattern: str, label: str) -> str:
    """Find a file matching *pattern* under *result_dir* (recursive)."""
    matches = glob.glob(os.path.join(result_dir, "**", pattern),
                        recursive=True)
    if not matches:
        sys.exit(f"[ERROR] Could not find {label} in {result_dir}")
    # Prefer the metres TRC if multiple exist
    path = matches[0]
    print(f"[INFO] Found {label}: {path}")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Step 9 — Release Window Analysis  (projectile-based optimality)
# ═══════════════════════════════════════════════════════════════════════════

GRAVITY = 9.81  # m/s²

# Biomechanical plausibility constraints
MIN_LAUNCH_ANGLE_DEG = 35.0
MAX_LAUNCH_ANGLE_DEG = 60.0
OPTIMAL_WINDOW_THRESHOLD_PCT = 10.0  # % relative error for window


def _click_handler(event, x, y, flags, param):
    """OpenCV mouse callback — stores clicked point."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param["point"] = (x, y)
        param["done"] = True


def select_hoop_position(video_path: str,
                         markers_px: dict,
                         markers_m: dict) -> tuple:
    """Show the first frame and let the user click the hoop centre.

    Converts the pixel click to real-world metres using the ratio between
    pixel and metre TRC data on frame 0.

    Returns:
        hoop_pos  – (hoop_x_m, hoop_y_m) in physics coords (Y-up)
        coord_cvt – dict with keys 'ref_px', 'ref_m', 'px_per_m' for
                    converting between metres and pixels later.
    """
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("[ERROR] Cannot read first frame for hoop selection.")

    # ── Compute px → m scale factor from TRC data ──
    m_a = "RHip"
    m_b = "RShoulder"
    if m_a in markers_px and m_b in markers_px and \
       m_a in markers_m  and m_b in markers_m:
        px_dist = np.linalg.norm(markers_px[m_a][0] - markers_px[m_b][0])
        m_dist  = np.linalg.norm(markers_m[m_a][0]  - markers_m[m_b][0])
        if px_dist > 0:
            px_per_m = px_dist / m_dist
        else:
            px_per_m = 100.0
    else:
        px_per_m = 100.0

    # ── Reference point (frame-0) in both coordinate systems ──
    ref_px = markers_px["RHip"][0].copy()
    ref_m  = markers_m["RHip"][0].copy()

    # ── Interactive click ──
    display = frame.copy()
    cv2.putText(display,
                "Click on the HOOP CENTER, then press any key",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 255, 255), 2)

    state = {"point": None, "done": False}
    cv2.namedWindow("Select Hoop", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Hoop", _click_handler, state)
    cv2.imshow("Select Hoop", display)

    while not state["done"]:
        key = cv2.waitKey(50)
        if key == 27:
            cv2.destroyAllWindows()
            sys.exit("[INFO] Hoop selection cancelled.")

    click_px = np.array(state["point"], dtype=float)

    cv2.circle(display, state["point"], 10, (0, 0, 255), -1)
    cv2.putText(display, "Hoop", (state["point"][0]+15, state["point"][1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Select Hoop", display)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    # ── Convert pixel click to metres ──
    delta_px = click_px - ref_px
    delta_m_x =  delta_px[0] / px_per_m
    delta_m_y = -delta_px[1] / px_per_m   # image-Y-down → TRC-Y-up

    hoop_x = float(ref_m[0] + delta_m_x)
    hoop_y = float(ref_m[1] + delta_m_y)

    print(f"[INFO] Hoop position: ({hoop_x:.3f}, {hoop_y:.3f}) m  "
          f"[physics coords, Y-up]")

    coord_cvt = {"ref_px": ref_px, "ref_m": ref_m, "px_per_m": px_per_m}
    return (hoop_x, hoop_y), coord_cvt


def find_candidate_frames(landmarks_m: dict, linear_kin: dict,
                          hoop_pos: tuple, time_arr: np.ndarray) -> np.ndarray:
    """Filter frames by biomechanical plausibility constraints.

    Coordinates from Sports2D TRC are already Y-up natively in metres.
    """
    hand = landmarks_m["RIGHT_WRIST"]         # (N, 2) in m, Y-up
    shoulder = landmarks_m["RIGHT_SHOULDER"]
    vel = linear_kin["RIGHT_WRIST"]["vel"]     # (N, 2) in m/s, Y-up

    hx, hy = hoop_pos  # physics coords, Y-up

    n = len(time_arr)
    valid = []

    # Diagnostic counters
    fail_forward = 0
    fail_upward  = 0
    fail_above   = 0
    fail_angle   = 0

    for i in range(n):
        # Coordinates are already physics Y-up
        hand_x, hand_y = hand[i, 0], hand[i, 1]
        sh_y = shoulder[i, 1]
        vx, vy = vel[i, 0], vel[i, 1]

        # 1) Forward motion toward hoop
        if hx > hand_x:
            if vx <= 0:
                fail_forward += 1
                continue
        else:
            if vx >= 0:
                fail_forward += 1
                continue

        # 2) Upward motion
        if vy <= 0:
            fail_upward += 1
            continue

        # 3) Hand above shoulder
        if hand_y <= sh_y:
            fail_above += 1
            continue

        # 4) Launch angle between 35° and 65°
        speed = np.sqrt(vx**2 + vy**2)
        if speed < 0.01:
            fail_angle += 1
            continue
        angle_deg = np.degrees(np.arctan2(vy, abs(vx)))
        if angle_deg < MIN_LAUNCH_ANGLE_DEG or angle_deg > MAX_LAUNCH_ANGLE_DEG:
            fail_angle += 1
            continue

        valid.append(i)

    candidates = np.array(valid, dtype=int)
    print(f"[INFO] Candidate release frames: {len(candidates)} / {n}")
    print(f"       Rejected — forward: {fail_forward}, upward: {fail_upward}, "
          f"above shoulder: {fail_above}, angle: {fail_angle}")
    return candidates


def compute_required_speed(x0: float, y0: float,
                           xh: float, yh: float,
                           theta_rad: float, g: float = GRAVITY) -> float:
    """Compute minimum launch speed for a projectile to reach the hoop.

    Uses:  v² = g·Δx² / (2·cos²θ · (Δx·tanθ − Δy))

    Returns v_required, or NaN if no valid trajectory exists.
    """
    dx = xh - x0
    dy = yh - y0
    cos_t = np.cos(theta_rad)
    tan_t = np.tan(theta_rad)

    denom = 2.0 * cos_t**2 * (abs(dx) * tan_t - dy)
    if denom <= 0:
        return float('nan')

    v2 = g * dx**2 / denom
    if v2 < 0:
        return float('nan')

    return np.sqrt(v2)


def analyze_release_window(candidates: np.ndarray,
                           landmarks_m: dict, linear_kin: dict,
                           hoop_pos: tuple, time_arr: np.ndarray) -> dict:
    """Compute optimality metrics for each candidate frame.

    Returns dict with arrays:
        frame_idx, time, actual_speed, required_speed, launch_angle_deg,
        speed_error, relative_error_pct, feasibility_ratio,
        optimal_idx, optimal_window
    """
    hand = landmarks_m["RIGHT_WRIST"]
    vel  = linear_kin["RIGHT_WRIST"]["vel"]
    hx, hy = hoop_pos

    frame_idx      = []
    times          = []
    actual_speeds  = []
    required_speeds = []
    angles_deg     = []
    speed_errors   = []
    rel_errors     = []
    feas_ratios    = []

    for i in candidates:
        # Coordinates are already physics Y-up
        hand_x, hand_y = hand[i, 0], hand[i, 1]
        vx, vy = vel[i, 0], vel[i, 1]

        speed  = np.sqrt(vx**2 + vy**2)
        angle  = np.arctan2(vy, abs(vx))
        angle_d = np.degrees(angle)

        v_req = compute_required_speed(hand_x, hand_y, hx, hy, angle)

        if np.isnan(v_req):
            continue

        err     = abs(speed - v_req)
        rel_err = (err / v_req) * 100.0  if v_req > 0 else float('nan')
        ratio   = speed / v_req          if v_req > 0 else float('nan')

        frame_idx.append(i)
        times.append(time_arr[i])
        actual_speeds.append(speed)
        required_speeds.append(v_req)
        angles_deg.append(angle_d)
        speed_errors.append(err)
        rel_errors.append(rel_err)
        feas_ratios.append(ratio)

    frame_idx      = np.array(frame_idx)
    times          = np.array(times)
    actual_speeds  = np.array(actual_speeds)
    required_speeds = np.array(required_speeds)
    angles_deg     = np.array(angles_deg)
    speed_errors   = np.array(speed_errors)
    rel_errors     = np.array(rel_errors)
    feas_ratios    = np.array(feas_ratios)

    # Optimal frame
    if len(rel_errors) > 0:
        opt_local = int(np.nanargmin(rel_errors))
        opt_frame = int(frame_idx[opt_local])
        # Optimal window: frames within threshold
        window_mask = rel_errors <= OPTIMAL_WINDOW_THRESHOLD_PCT
        window_frames = frame_idx[window_mask]
    else:
        opt_frame = -1
        window_frames = np.array([], dtype=int)

    results = dict(
        frame_idx=frame_idx, time=times,
        actual_speed=actual_speeds, required_speed=required_speeds,
        launch_angle_deg=angles_deg,
        speed_error=speed_errors, relative_error_pct=rel_errors,
        feasibility_ratio=feas_ratios,
        optimal_frame=opt_frame, optimal_window=window_frames,
    )

    print(f"[INFO] Valid projectile solutions: {len(frame_idx)}")
    if opt_frame >= 0:
        print(f"[INFO] Optimal release frame: {opt_frame} "
              f"(t={time_arr[opt_frame]:.3f}s, "
              f"error={rel_errors[opt_local]:.1f}%)")
        print(f"[INFO] Optimal window ({OPTIMAL_WINDOW_THRESHOLD_PCT:.0f}% "
              f"threshold): {len(window_frames)} frames")

    return results


def print_release_summary(results: dict) -> None:
    """Print formatted release window analysis table."""
    if len(results["frame_idx"]) == 0:
        print("\n  [WARN] No valid candidate frames found.\n")
        return

    sep = "  " + "─" * 92
    print("\n" + "=" * 96)
    print("  RELEASE WINDOW ANALYSIS")
    print("=" * 96)
    print(f"  {'Frame':>6s}  {'Time':>7s}  {'Angle':>7s}  "
          f"{'Actual':>8s}  {'Required':>9s}  {'Error':>7s}  "
          f"{'Rel.Err':>8s}  {'Ratio':>7s}  {'Status':>8s}")
    print(f"  {'':>6s}  {'(s)':>7s}  {'(°)':>7s}  "
          f"{'(m/s)':>8s}  {'(m/s)':>9s}  {'(m/s)':>7s}  "
          f"{'(%)':>8s}  {'':>7s}  {'':>8s}")
    print(sep)

    opt = results["optimal_frame"]
    win = set(results["optimal_window"].tolist())

    for j in range(len(results["frame_idx"])):
        fi = results["frame_idx"][j]
        status = ""
        if fi == opt:
            status = "★ BEST"
        elif fi in win:
            status = "● WINDOW"

        print(f"  {fi:6d}  {results['time'][j]:7.3f}  "
              f"{results['launch_angle_deg'][j]:7.1f}  "
              f"{results['actual_speed'][j]:8.2f}  "
              f"{results['required_speed'][j]:9.2f}  "
              f"{results['speed_error'][j]:7.2f}  "
              f"{results['relative_error_pct'][j]:7.1f}%  "
              f"{results['feasibility_ratio'][j]:7.3f}  "
              f"{status:>8s}")

    print(sep)

    re = results["relative_error_pct"]
    print(f"  Min relative error: {np.nanmin(re):.1f}%  |  "
          f"Mean: {np.nanmean(re):.1f}%  |  "
          f"Optimal window frames: {len(results['optimal_window'])}")
    print("=" * 96 + "\n")


def plot_release_analysis(time_arr: np.ndarray,
                          results: dict,
                          linear_kin: dict) -> None:
    """Generate 3 release analysis plots."""
    if len(results["frame_idx"]) == 0:
        print("  [WARN] No release data to plot.")
        return

    t = results["time"]
    win = results["optimal_window"]
    opt = results["optimal_frame"]

    # --- Plot 1: Relative error vs time ---
    fig, ax = plt.subplots()
    ax.plot(t, results["relative_error_pct"], "o-", ms=3, color="tab:red",
            label="Relative error (%)")
    if len(win) > 0:
        win_mask = np.isin(results["frame_idx"], win)
        ax.fill_between(t, 0, results["relative_error_pct"],
                         where=win_mask, alpha=0.25, color="green",
                         label=f"Optimal window (≤{OPTIMAL_WINDOW_THRESHOLD_PCT:.0f}%)")
    if opt >= 0:
        j_opt = np.where(results["frame_idx"] == opt)[0][0]
        ax.axvline(t[j_opt], color="gold", ls="--", lw=1.5, label="Best frame")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Relative Error (%)")
    ax.set_title("Release Optimality — Relative Speed Error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "release_error_s2d.png"), dpi=150)
    plt.close(fig)
    print("  → saved release_error_s2d.png")

    # --- Plot 2: Feasibility ratio vs time ---
    fig, ax = plt.subplots()
    ax.plot(t, results["feasibility_ratio"], "o-", ms=3, color="tab:blue")
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="Ratio = 1 (optimal)")
    if opt >= 0:
        j_opt = np.where(results["frame_idx"] == opt)[0][0]
        ax.axvline(t[j_opt], color="gold", ls="--", lw=1.5, label="Best frame")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Feasibility Ratio (actual / required)")
    ax.set_title("Release Optimality — Feasibility Ratio")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "release_ratio_s2d.png"), dpi=150)
    plt.close(fig)
    print("  → saved release_ratio_s2d.png")

    # --- Plot 3: Actual vs Required speed comparison ---
    fig, ax = plt.subplots()
    # Full wrist speed for context (all frames)
    full_vel = linear_kin["RIGHT_WRIST"]["vel_mag"]
    ax.plot(time_arr, full_vel, color="lightgray", lw=1,
            label="Wrist speed (all frames)")
    # Actual speed at candidate frames
    ax.plot(t, results["actual_speed"], "o-", ms=3, color="tab:orange",
            label="Actual speed (candidates)")
    # Required speed at candidate frames
    ax.plot(t, results["required_speed"], "s-", ms=3, color="tab:green",
            label="Required speed (projectile)")
    if opt >= 0:
        j_opt = np.where(results["frame_idx"] == opt)[0][0]
        ax.axvline(t[j_opt], color="gold", ls="--", lw=1.5, label="Best frame")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Actual vs Required Wrist Speed for Hoop")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "release_speed_comparison_s2d.png"),
                dpi=150)
    plt.close(fig)
    print("  → saved release_speed_comparison_s2d.png")


def _compute_affine_m_to_px(markers_m: dict, markers_px: dict,
                            frame_idx: int) -> tuple:
    """Compute an affine transform from TRC metres to image pixels at a frame.

    Uses all shared markers to fit:  px = A · [m_x, m_y, 1]^T
    via least-squares.  Returns (sol_x, sol_y) coefficient vectors.
    """
    shared = [m for m in markers_m if m in markers_px]
    pts_m, pts_px = [], []
    for m in shared:
        if frame_idx < len(markers_m[m]) and frame_idx < len(markers_px[m]):
            pts_m.append(markers_m[m][frame_idx])
            pts_px.append(markers_px[m][frame_idx])
    pts_m  = np.array(pts_m)
    pts_px = np.array(pts_px)
    M = np.hstack([pts_m, np.ones((len(pts_m), 1))])  # (K, 3)
    sol_x, _, _, _ = np.linalg.lstsq(M, pts_px[:, 0], rcond=None)
    sol_y, _, _, _ = np.linalg.lstsq(M, pts_px[:, 1], rcond=None)
    return sol_x, sol_y


def _affine_m_to_px(x_m: float, y_m: float,
                    sol_x: np.ndarray, sol_y: np.ndarray) -> tuple:
    """Apply a precomputed affine transform to convert metres → pixels."""
    v = np.array([x_m, y_m, 1.0])
    return int(round(sol_x @ v)), int(round(sol_y @ v))


def compute_projectile_path(x0: float, y0: float,
                            vx: float, vy: float,
                            g: float = GRAVITY,
                            dt: float = 0.005,
                            t_max: float = 2.0) -> np.ndarray:
    """Simulate a projectile from (x0, y0) with velocity (vx, vy).

    Returns an (M, 2) array of (x, y) positions in metres (Y-up).
    Stops when the ball drops below y0 - 0.5 m or t_max is reached.
    """
    pts = []
    t = 0.0
    while t <= t_max:
        x = x0 + vx * t
        y = y0 + vy * t - 0.5 * g * t * t
        pts.append((x, y))
        if y < y0 - 0.5:      # fell well below release height
            break
        t += dt
    return np.array(pts)


def save_debug_frames(video_path: str, candidates: np.ndarray,
                      landmarks_m: dict, linear_kin: dict,
                      hoop_pos: tuple,
                      all_markers_m: dict, markers_px: dict,
                      out_dir: str = os.path.join(RESULTS_DIR, "debug")) -> None:
    """Export candidate frames with the predicted projectile arc overlaid.

    Uses a per-frame affine transform (fit from all TRC marker pairs)
    to accurately map metre-space projectile points into image pixels.
    """
    if len(candidates) == 0:
        return

    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Could not open {video_path} for debug frames.")
        return

    hand = landmarks_m["RIGHT_WRIST"]
    vel  = linear_kin["RIGHT_WRIST"]["vel"]
    hx, hy = hoop_pos

    saved_count = 0
    for frame_idx in candidates:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if not success:
            continue

        # Per-frame affine from metre → pixel
        sol_x, sol_y = _compute_affine_m_to_px(all_markers_m, markers_px,
                                                frame_idx)

        # Wrist state at this frame (metres, Y-up)
        wx, wy = hand[frame_idx, 0], hand[frame_idx, 1]
        vx_f, vy_f = vel[frame_idx, 0], vel[frame_idx, 1]

        # Compute projectile arc in metres then convert to pixels
        arc_m  = compute_projectile_path(wx, wy, vx_f, vy_f)
        arc_px = np.array([_affine_m_to_px(p[0], p[1], sol_x, sol_y)
                           for p in arc_m], dtype=np.int32)

        # Draw green arc
        if len(arc_px) > 1:
            cv2.polylines(frame, [arc_px], isClosed=False,
                          color=(0, 255, 0), thickness=2)

        # Draw yellow wrist release dot (use pixel TRC for accuracy)
        wrist_px_pos = markers_px["RWrist"][frame_idx]
        cv2.circle(frame, (int(wrist_px_pos[0]), int(wrist_px_pos[1])),
                   8, (0, 255, 255), -1)

        # Draw red hoop circle (convert via this frame's affine)
        hoop_px = _affine_m_to_px(hx, hy, sol_x, sol_y)
        cv2.circle(frame, hoop_px, 12, (0, 0, 255), 3)

        # Label
        cv2.putText(frame, f"Frame {frame_idx}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        out_file = os.path.join(out_dir, f"candidate_{frame_idx:04d}.jpg")
        cv2.imwrite(out_file, frame)
        saved_count += 1

    cap.release()
    print(f"[INFO] Saved {saved_count} annotated candidate frames to {out_dir}/")


def save_release_csv(results: dict,
                     out_path: str = os.path.join(RESULTS_DIR,
                                                  "release_analysis_s2d.csv")
                     ) -> None:
    """Save release window analysis data to CSV."""
    if len(results["frame_idx"]) == 0:
        return
    df = pd.DataFrame({
        "frame": results["frame_idx"],
        "time": results["time"],
        "launch_angle_deg": results["launch_angle_deg"],
        "actual_speed_m_s": results["actual_speed"],
        "required_speed_m_s": results["required_speed"],
        "speed_error_m_s": results["speed_error"],
        "relative_error_pct": results["relative_error_pct"],
        "feasibility_ratio": results["feasibility_ratio"],
    })
    df.to_csv(out_path, index=False)
    print(f"[INFO] Release analysis CSV saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── CLI args ──
    video_path    = sys.argv[1] if len(sys.argv) > 1 else VIDEO_PATH
    player_height = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_HEIGHT

    if not os.path.isfile(video_path):
        sys.exit(f"[ERROR] Video file not found: {video_path}")

    # Read video metadata
    cap = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    result_dir = os.path.join(os.path.dirname(os.path.abspath(video_path)),
                              "Sports2D_Results")
    os.makedirs(result_dir, exist_ok=True)

    # ── Step 1: Run Sports2D ──
    run_sports2d(video_path, player_height, result_dir)

    # ── Step 2: Parse TRC (metre coordinates) ──
    trc_m_path = _find_file(result_dir, f"*_m_{PERSON_ID}.trc", "TRC (metres)")
    landmarks, time_arr, all_markers_m = parse_trc(trc_m_path)

    # ── Step 3: Parse MOT (angles) ──
    mot_path = _find_file(result_dir, f"*_{PERSON_ID}.mot", "MOT (angles)")
    angles = parse_mot(mot_path)

    # Ensure time arrays match (use the shorter one)
    n = min(len(time_arr), min(len(v) for v in angles.values()))
    time_arr = time_arr[:n]
    landmarks = {k: v[:n] for k, v in landmarks.items()}
    angles    = {k: v[:n] for k, v in angles.items()}
    all_markers_m = {k: v[:n] for k, v in all_markers_m.items()}

    # ── Step 3b: Export to CSV ──
    save_positions_csv(landmarks, time_arr)
    save_angles_csv(angles, time_arr)

    # ── Step 4: Linear kinematics ──
    linear_kin = compute_linear_kinematics(landmarks, time_arr)

    # ── Step 5: Angular kinematics ──
    angular_kin = compute_angular_kinematics(angles, time_arr)

    # ── Step 6: Summary table ──
    print_summary(linear_kin, angular_kin)

    # ── Step 7: Plots ──
    plot_all(time_arr, linear_kin, angular_kin)

    # ── Step 8: Custom annotated video ──
    trc_px_path = _find_file(result_dir, f"*_px_{PERSON_ID}.trc", "TRC (pixels)")
    markers_px  = parse_trc_px(trc_px_path)
    # Trim to same length
    markers_px  = {k: v[:n] for k, v in markers_px.items()}
    write_output_video(video_path, fps, width, height,
                       markers_px, linear_kin, angular_kin, time_arr)

    # ── Step 9: Release window analysis ──
    hoop_pos, coord_cvt = select_hoop_position(video_path, markers_px,
                                                all_markers_m)
    candidates = find_candidate_frames(landmarks, linear_kin,
                                       hoop_pos, time_arr)
    
    # Save debug frames with projectile overlay
    save_debug_frames(video_path, candidates, landmarks, linear_kin,
                      hoop_pos, all_markers_m, markers_px)

    release_results = analyze_release_window(candidates, landmarks,
                                             linear_kin, hoop_pos, time_arr)
    print_release_summary(release_results)
    plot_release_analysis(time_arr, release_results, linear_kin)
    save_release_csv(release_results)

    print("\n[DONE] Analysis complete. Files produced:")
    print("  • wrist_position_s2d.png")
    print("  • wrist_velocity_s2d.png")
    print("  • wrist_acceleration_s2d.png")
    print("  • elbow_angle_s2d.png")
    print("  • elbow_angular_velocity_s2d.png")
    print("  • knee_angle_s2d.png")
    print("  • all_joint_angles_s2d.png")
    print(f"  • {OUTPUT_VIDEO}")
    print("  • positions_s2d.csv")
    print("  • angles_s2d.csv")
    print("  • release_error_s2d.png")
    print("  • release_ratio_s2d.png")
    print("  • release_speed_comparison_s2d.png")
    print("  • release_analysis_s2d.csv")
    print("  • debug/ (valid candidate frames)")
    print(f"  • Sports2D native outputs in: {result_dir}")


if __name__ == "__main__":
    main()
