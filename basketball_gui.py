#!/usr/bin/env python3
"""
SHOT LAB  —  Basketball Free-Throw Kinematic Analyzer
======================================================
Professional sports analytics GUI wrapping motion_analysis_s2d.py.

Requirements (same venv as motion_analysis_s2d.py):
    pip install pillow

Place this file in the same folder as motion_analysis_s2d.py and run:
    python basketball_gui.py
"""

import os, sys, threading, queue, importlib.util, traceback

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ── locate analysis module ─────────────────────────────────────────────────
ANALYSIS_MODULE_PATH = os.path.join(os.path.dirname(__file__),
                                    "motion_analysis_s2d.py")
_ana = None

def _load_ana():
    global _ana
    if _ana:
        return _ana
    if not os.path.isfile(ANALYSIS_MODULE_PATH):
        raise FileNotFoundError(
            f"motion_analysis_s2d.py not found at:\n{ANALYSIS_MODULE_PATH}\n"
            "Place basketball_gui.py in the same folder.")
    spec = importlib.util.spec_from_file_location("m", ANALYSIS_MODULE_PATH)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _ana = mod
    return _ana


# ══════════════════════════════════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════════════════════════════════
BG    = "#080808"
SURF  = "#111111"
CARD  = "#161616"
LINE  = "#252525"
AMBER = "#F5A623"
AMB2  = "#FFCA6B"
DIM   = "#404040"
MUTED = "#666666"
TEXT  = "#E8E3D8"
GREEN = "#2ECC71"
RED   = "#E74C3C"
BLUE  = "#4A9EFF"
GOLD  = "#FFD700"

MPL_STYLE = {
    "figure.facecolor":  CARD,
    "axes.facecolor":    CARD,
    "axes.edgecolor":    LINE,
    "axes.labelcolor":   MUTED,
    "axes.titlecolor":   TEXT,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "grid.color":        LINE,
    "grid.linewidth":    0.5,
    "text.color":        TEXT,
    "lines.linewidth":   1.8,
    "legend.facecolor":  SURF,
    "legend.edgecolor":  LINE,
    "legend.fontsize":   8,
    "axes.titlesize":    10,
    "axes.labelsize":    8,
    "figure.autolayout": True,
}

def _mpl():
    for k, v in MPL_STYLE.items():
        try: matplotlib.rcParams[k] = v
        except: pass

_mpl()


# ══════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════
def _lbl(parent, text, size=9, weight="normal", color=TEXT, **kw):
    bg = kw.pop("bg", None) or _get_bg(parent)
    return tk.Label(parent, text=text,
                    font=("Helvetica Neue", size, weight),
                    fg=color, bg=bg, **kw)

def _get_bg(w):
    try: return w["bg"]
    except: return BG

def _btn(parent, text, cmd, accent=False, danger=False, **kw):
    bg_ = AMBER if accent else (RED if danger else CARD)
    fg_ = "#000" if accent else TEXT
    hbg = AMB2  if accent else (LINE if not danger else "#9B1C1C")
    b = tk.Button(parent, text=text, command=cmd,
                  font=("Helvetica Neue", 9, "bold"),
                  bg=bg_, fg=fg_, relief="flat", cursor="hand2",
                  activebackground=hbg, activeforeground=fg_,
                  padx=10, pady=5, highlightthickness=0, bd=0, **kw)
    b.bind("<Enter>", lambda e, b=b, h=hbg: b.config(bg=h))
    b.bind("<Leave>", lambda e, b=b, c=bg_: b.config(bg=c))
    return b

def _entry(parent, var, **kw):
    return tk.Entry(parent, textvariable=var,
                    font=("Helvetica Neue", 10),
                    bg=CARD, fg=TEXT, insertbackground=AMBER,
                    relief="flat", highlightthickness=1,
                    highlightbackground=LINE, highlightcolor=AMBER,
                    bd=0, **kw)

def _hsep(parent, pady=6):
    tk.Frame(parent, bg=LINE, height=1).pack(fill="x", pady=pady)


# ══════════════════════════════════════════════════════════════════════════
# VIDEO PLAYER
# ══════════════════════════════════════════════════════════════════════════
class VideoPlayer(tk.Frame):
    def __init__(self, master, on_click=None, on_box=None, **kw):
        super().__init__(master, bg="#050505", **kw)
        self._cap       = None
        self._fps       = 30.0
        self._total     = 0
        self._playing   = False
        self._idx       = 0
        self._pil       = None
        self._tkimg     = None
        self._draw      = False
        self._hoop      = False
        self._dstart    = None
        self._on_click  = on_click
        self._on_box    = on_box
        self._overlays  = []
        self._build()

    def _build(self):
        self.canvas = tk.Canvas(self, bg="#050505",
                                highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True)

        # slim control bar
        bar = tk.Frame(self, bg=SURF)
        bar.pack(fill="x", side="bottom")

        # thin scrub strip
        self._scrub = tk.Canvas(bar, bg=DIM, height=3,
                                highlightthickness=0, cursor="hand2")
        self._scrub.pack(fill="x", pady=0)
        self._scrub_fill = None
        self._scrub.bind("<ButtonPress-1>",   self._scrub_press)
        self._scrub.bind("<B1-Motion>",        self._scrub_drag)

        ctrl = tk.Frame(bar, bg=SURF)
        ctrl.pack(fill="x", padx=10, pady=5)

        self._pbtn = tk.Button(ctrl, text="▶",
                               font=("Helvetica Neue", 10),
                               bg=SURF, fg=TEXT, relief="flat",
                               bd=0, cursor="hand2", padx=4,
                               command=self.toggle,
                               activebackground=SURF,
                               activeforeground=AMBER)
        self._pbtn.pack(side="left")

        self._tlbl = _lbl(ctrl, "0:00 / 0:00", 8, color=MUTED, bg=SURF)
        self._tlbl.pack(side="left", padx=8)

        self._flbl = _lbl(ctrl, "Frame 0", 8, color=DIM, bg=SURF)
        self._flbl.pack(side="right")

        self.canvas.bind("<ButtonPress-1>",   self._md)
        self.canvas.bind("<B1-Motion>",       self._mm)
        self.canvas.bind("<ButtonRelease-1>", self._mu)

    # public
    def load(self, path):
        if self._cap: self._cap.release()
        self._cap   = cv2.VideoCapture(path)
        self._fps   = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._idx   = 0
        self._show(0)

    def toggle(self):
        if not self._cap: return
        self._playing = not self._playing
        self._pbtn.config(text="⏸" if self._playing else "▶")
        if self._playing: self._loop()

    def pause(self):
        self._playing = False
        self._pbtn.config(text="▶")

    def seek(self, i):
        self._idx = max(0, min(i, self._total - 1))
        self._show(self._idx)

    def idx(self): return self._idx
    def pil(self): return self._pil

    def draw_mode(self, on):
        self._draw   = on
        self._dstart = None
        self.canvas.delete("rubber")

    def hoop_mode(self, on):
        self._hoop = on

    def add_circ(self, fx, fy, r=15, col=RED, tag="ov"):
        self._overlays.append({"t":"c","fx":fx,"fy":fy,"r":r,"col":col,"tag":tag})
        self._ov()

    def add_rect(self, fx0, fy0, fx1, fy1, col=GREEN, tag="ov"):
        self._overlays.append({"t":"r","fx0":fx0,"fy0":fy0,
                                "fx1":fx1,"fy1":fy1,"col":col,"tag":tag})
        self._ov()

    def clear_ov(self, tag=None):
        if tag: self._overlays = [o for o in self._overlays if o["tag"]!=tag]
        else:   self._overlays.clear()
        self.canvas.delete("ov")

    def show_tracked_frame(self, frame_idx, bbox):
        """Advance video to frame_idx and draw tracker bbox on top atomically.
        Called from the tracker thread via .after() so it runs on the main thread.
        bbox = (x, y, w, h) in frame pixel coordinates.
        """
        # update the overlay list first, then show the frame (which calls _ov)
        self._overlays = [o for o in self._overlays if o.get("tag") != "ball"]
        x, y, w, h = bbox
        self._overlays.append({"t":"r",
                                "fx0": float(x),   "fy0": float(y),
                                "fx1": float(x+w), "fy1": float(y+h),
                                "col": GREEN, "tag": "ball"})
        self._show(frame_idx)

    def c2f(self, cx, cy):
        s, xo, yo = self._si()
        return (cx-xo)/s, (cy-yo)/s

    def f2c(self, fx, fy):
        s, xo, yo = self._si()
        return fx*s+xo, fy*s+yo

    # internal
    def _loop(self):
        if not self._playing or not self._cap: return
        self._idx = (self._idx + 1) % self._total
        self._show(self._idx)
        self.after(int(1000/self._fps), self._loop)

    def _show(self, i):
        if not self._cap: return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, fr = self._cap.read()
        if not ret: return
        self._pil = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
        cw = self.canvas.winfo_width()  or 720
        ch = self.canvas.winfo_height() or 400
        iw, ih = self._pil.size
        s = min(cw/iw, ch/ih)
        nw, nh = int(iw*s), int(ih*s)
        self._tkimg = ImageTk.PhotoImage(
            self._pil.resize((nw, nh), Image.LANCZOS))
        self.canvas.delete("frame")
        xo, yo = (cw-nw)//2, (ch-nh)//2
        self.canvas.create_image(xo, yo, anchor="nw",
                                 image=self._tkimg, tags="frame")
        self._ov()
        pct = i / max(1, self._total-1)
        sw = self._scrub.winfo_width() or 1
        self._scrub.delete("fill")
        self._scrub.create_rectangle(0, 0, int(pct*sw), 3,
                                     fill=AMBER, outline="", tags="fill")
        t = i/self._fps; tot = self._total/self._fps
        self._tlbl.config(text=f"{self._tf(t)} / {self._tf(tot)}")
        self._flbl.config(text=f"Frame {i}")
        self._idx = i

    def _tf(self, s): return f"{int(s//60)}:{int(s%60):02d}"

    def _si(self):
        if not self._pil: return 1.0, 0, 0
        cw = self.canvas.winfo_width()  or 720
        ch = self.canvas.winfo_height() or 400
        iw, ih = self._pil.size
        s = min(cw/iw, ch/ih)
        return s, (cw-iw*s)/2, (ch-ih*s)/2

    def _ov(self):
        self.canvas.delete("ov")
        for o in self._overlays:
            if o["t"] == "c":
                cx, cy = self.f2c(o["fx"], o["fy"])
                r = o["r"]
                self.canvas.create_oval(cx-r, cy-r, cx+r, cy+r,
                    outline=o["col"], width=2, tags="ov")
                self.canvas.create_line(cx-r*1.6, cy, cx+r*1.6, cy,
                    fill=o["col"], width=1, tags="ov")
                self.canvas.create_line(cx, cy-r*1.6, cx, cy+r*1.6,
                    fill=o["col"], width=1, tags="ov")
            elif o["t"] == "r":
                x0,y0 = self.f2c(o["fx0"], o["fy0"])
                x1,y1 = self.f2c(o["fx1"], o["fy1"])
                self.canvas.create_rectangle(x0,y0,x1,y1,
                    outline=o["col"], width=2, tags="ov")

    def _scrub_press(self, e):
        w = max(1, self._scrub.winfo_width())
        self.seek(int(e.x/w * (self._total-1)))

    def _scrub_drag(self, e):
        self._scrub_press(e)

    def _md(self, e):
        if self._hoop:
            fx, fy = self.c2f(e.x, e.y)
            if self._on_click: self._on_click(fx, fy)
            return
        if self._draw:
            self._dstart = (e.x, e.y)

    def _mm(self, e):
        if self._draw and self._dstart:
            self.canvas.delete("rubber")
            x0,y0 = self._dstart
            self.canvas.create_rectangle(x0,y0,e.x,e.y,
                outline=AMBER, width=1, dash=(5,3), tags="rubber")

    def _mu(self, e):
        if self._draw and self._dstart:
            self.canvas.delete("rubber")
            x0,y0 = self._dstart
            bx0,by0 = min(x0,e.x), min(y0,e.y)
            bx1,by1 = max(x0,e.x), max(y0,e.y)
            self._dstart = None
            if self._on_box: self._on_box(bx0,by0,bx1,by1)


# ══════════════════════════════════════════════════════════════════════════
# BALL TRACKER  —  exact original track_ball_speed logic, OpenCV window
# ══════════════════════════════════════════════════════════════════════════
class BallFlow:
    """Launches the EXACT original track_ball_speed() flow in an OpenCV
    window.  After tracking finishes the user sees an Accept / Redo dialog
    so they can re-run if the box drifted."""

    IDLE="idle"; RUN="run"; CONFIRM="confirm"; DONE="done"

    def __init__(self, log_q, on_done):
        self.q        = log_q
        self.done_cb  = on_done
        self.state    = self.IDLE
        self._video   = ""
        self._px2m    = 0.005
        self._times   = []
        self._speeds  = []
        self._fps     = 30.0
        # sidebar widgets wired externally
        self.btn_start = None
        self.slbl      = None

    # ── public ──────────────────────────────────────────────────────────
    def start(self):
        """Spawn the OpenCV tracking window in a background thread."""
        if not self._video:
            self.q.put(("log","[TRACKER] No video loaded.")); return
        self.state = self.RUN
        self._sync()
        self._st("OpenCV window opening…  press Q to stop early")
        threading.Thread(target=self._opencv_flow, daemon=True).start()

    # ── internal ─────────────────────────────────────────────────────────
    def _st(self, msg):
        if self.slbl:
            c = GREEN  if "✓" in msg \
                else (AMBER if "opening" in msg or "Redo" in msg
                      else (RED if "Error" in msg else MUTED))
            self.slbl.config(text=msg, fg=c)

    def _sync(self):
        if self.btn_start:
            self.btn_start.config(
                state="disabled" if self.state==self.RUN else "normal")

    def _opencv_flow(self):
        """Replicates track_ball_speed() exactly, then shows accept/redo."""
        try:
            ana = _load_ana()
            video_path    = self._video
            pixels_to_m   = self._px2m

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                self.q.put(("log","[TRACKER] Cannot open video.")); return

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            dt  = 1.0 / fps

            # ── Instructions (exact from original) ──────────────────────
            print("\n🎯 BALL TRACKER INSTRUCTIONS:")
            print("  1. The video will play.")
            print("  2. Press SPACE to pause right before the player releases the ball.")
            print("  3. Draw a tight box around the ball (avoid fingers), then press ENTER.")
            print("  4. Press Q at any time to stop tracking.\n")

            tracker         = None
            tracking_active = False
            times           = []
            ball_speeds     = []
            frame_count     = 0
            prev_center     = None

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if not tracking_active:
                    cv2.putText(frame,
                                "Press SPACE to pause & select ball",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)
                    cv2.imshow("Ball Tracker — SHOT LAB", frame)

                    key = cv2.waitKey(int(dt * 1000)) & 0xFF
                    if key == ord(' '):
                        bbox = cv2.selectROI("Ball Tracker — SHOT LAB",
                                             frame,
                                             fromCenter=False,
                                             showCrosshair=True)
                        tracker = cv2.TrackerCSRT_create()
                        tracker.init(frame, bbox)
                        prev_center = (bbox[0] + bbox[2] / 2.0,
                                       bbox[1] + bbox[3] / 2.0)
                        tracking_active = True
                        self.q.put(("log","[TRACKER] Ball tracking active."))
                    elif key == ord('q'):
                        break
                else:
                    success, bbox = tracker.update(frame)
                    current_speed = 0.0

                    if success:
                        x, y, w, h = [int(v) for v in bbox]
                        current_center = (x + w / 2.0, y + h / 2.0)
                        cv2.rectangle(frame, (x, y), (x+w, y+h),
                                      (0, 255, 0), 2)
                        dx = current_center[0] - prev_center[0]
                        dy = current_center[1] - prev_center[1]
                        dist_px = np.sqrt(dx**2 + dy**2)
                        raw_speed = (dist_px * pixels_to_m) / dt
                        current_speed = min(raw_speed, ana.BALL_SPEED_CAP)
                        prev_center = current_center
                    else:
                        cv2.putText(frame, "LOST TRACKING", (20, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 3)

                    cv2.imshow("Ball Tracker — SHOT LAB", frame)
                    key = cv2.waitKey(30) & 0xFF
                    if key == ord('q'):
                        break

                    # store the actual video timestamp so times are in
                    # the same coordinate system as the pose/TRC data
                    video_ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    times.append(video_ts)
                    ball_speeds.append(current_speed)

                frame_count += 1

            cap.release()
            cv2.destroyAllWindows()

            if len(ball_speeds) > 5:
                from scipy.signal import savgol_filter as _sg
                smoothed = _sg(np.array(ball_speeds),
                               window_length=5, polyorder=2)
                ball_speeds = np.maximum(smoothed, 0.0).tolist()

            self.q.put(("log",
                f"[TRACKER] Recorded {len(ball_speeds)} frames."))

            self._times  = times
            self._speeds = ball_speeds
            self._fps    = fps

            # ── Ask Accept / Redo on the main thread ────────────────────
            if len(ball_speeds) > 0:
                self.slbl.after(0, self._ask_accept_redo)
            else:
                self.state = self.IDLE
                self.slbl.after(0, self._sync)
                self.slbl.after(0, lambda: self._st("No frames tracked — try again"))

        except Exception as e:
            self.q.put(("log", f"[TRACKER ERROR] {e}\n{traceback.format_exc()}"))
            self.state = self.IDLE
            if self.slbl:
                self.slbl.after(0, self._sync)
                self.slbl.after(0, lambda: self._st("Error — see log"))

    def _ask_accept_redo(self):
        """Show a small Toplevel with Accept / Redo buttons."""
        win = tk.Toplevel()
        win.title("Ball Tracking — Review")
        win.configure(bg=SURF)
        win.resizable(False, False)
        win.grab_set()          # modal

        tk.Frame(win, bg=AMBER, height=2).pack(fill="x")

        _lbl(win, f"  Tracked {len(self._speeds)} frames.",
             10, "bold", TEXT, bg=SURF).pack(
             anchor="w", padx=20, pady=(14,2))
        _lbl(win,
             f"  Peak ball speed: {max(self._speeds):.2f} m/s\n"
             f"  Was the tracking accurate?",
             9, color=MUTED, bg=SURF).pack(anchor="w", padx=20, pady=(0,14))

        tk.Frame(win, bg=LINE, height=1).pack(fill="x")

        btn_row = tk.Frame(win, bg=SURF)
        btn_row.pack(fill="x", padx=20, pady=14)

        def _accept():
            win.destroy()
            self.state = self.DONE
            self._sync()
            self._st(f"✓  {len(self._speeds)} frames accepted")
            self.done_cb(self._times, self._speeds, self._fps)

        def _redo():
            win.destroy()
            self.state = self.IDLE
            self._sync()
            self._st("Redo — click Start Tracking again")
            self._times=[]; self._speeds=[]

        _btn(btn_row, "✔  Accept", _accept, accent=True).pack(
            side="left", fill="x", expand=True, padx=(0,6))
        _btn(btn_row, "↺  Redo", _redo, danger=True).pack(
            side="left", fill="x", expand=True)


# ══════════════════════════════════════════════════════════════════════════
# SCROLLABLE WRAPPER
# ══════════════════════════════════════════════════════════════════════════
class Scrollable(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=BG, **kw)
        c = tk.Canvas(self, bg=BG, highlightthickness=0)
        sb= tk.Scrollbar(self, orient="vertical", command=c.yview,
                         bg=SURF, troughcolor=BG, width=5,
                         relief="flat", bd=0)
        c.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        c.pack(side="left", fill="both", expand=True)
        self.inner=tk.Frame(c,bg=BG)
        _id=c.create_window((0,0),window=self.inner,anchor="nw")
        self.inner.bind("<Configure>",
            lambda e: c.configure(scrollregion=c.bbox("all")))
        c.bind("<Configure>",
            lambda e: c.itemconfig(_id,width=e.width))
        c.bind_all("<MouseWheel>",
            lambda e: c.yview_scroll(-1*(e.delta//120),"units"))


# ══════════════════════════════════════════════════════════════════════════
# CLEAN TREEVIEW HELPER
# ══════════════════════════════════════════════════════════════════════════
def _make_tree(parent, cols, style_name="T.Treeview", col_widths=None):
    style=ttk.Style()
    try: style.theme_use("clam")
    except: pass
    style.configure(style_name,
        background=CARD, foreground=TEXT,
        fieldbackground=CARD, rowheight=24,
        font=("Helvetica Neue",9),
        borderwidth=0, relief="flat")
    style.configure(f"{style_name}.Heading",
        background=SURF, foreground=MUTED,
        font=("Helvetica Neue",8,"bold"),
        borderwidth=0, relief="flat")
    style.map(style_name,
        background=[("selected","#1E1E1E")],
        foreground=[("selected",AMBER)])

    tv=ttk.Treeview(parent,columns=cols,show="headings",
                     style=style_name,selectmode="browse")
    for i,col in enumerate(cols):
        w=col_widths[i] if col_widths else max(80,len(col)*9)
        tv.heading(col,text=col)
        tv.column(col,width=w,anchor="e",minwidth=50)
    tv.tag_configure("alt",background="#121212")
    return tv


# ══════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════
class ShotLab(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SHOT LAB  ·  Basketball Kinematic Analyzer")
        self.geometry("1440x860")
        self.minsize(1100,680)
        self.configure(bg=BG)

        self._video=None; self._fps=30.0
        self._kin={}; self._ang={}; self._time=None
        self._lm={}; self._am={}; self._px={}; self._angles={}
        self._hoop_px=None; self._bt_data=None
        self._q=queue.Queue()

        self._build()
        self._poll()

    # ══════════════════════════════════════════════════════════════════════
    # BUILD UI
    # ══════════════════════════════════════════════════════════════════════
    def _build(self):
        # ── header ──────────────────────────────────────────────────────
        hdr=tk.Frame(self,bg=BG)
        hdr.pack(fill="x",side="top")
        tk.Frame(hdr,bg=AMBER,height=2).pack(fill="x")
        hi=tk.Frame(hdr,bg=BG)
        hi.pack(fill="x",padx=20,pady=10)
        _lbl(hi,"SHOT LAB",16,"bold",AMBER,bg=BG).pack(side="left")
        _lbl(hi,"  Basketball Kinematic Analyzer",9,color=MUTED,bg=BG).pack(
            side="left",pady=3)

        # ── body ─────────────────────────────────────────────────────────
        body=tk.Frame(self,bg=BG)
        body.pack(fill="both",expand=True)

        # sidebar
        self._side=tk.Frame(body,bg=SURF,width=230)
        self._side.pack(side="left",fill="y")
        self._side.pack_propagate(False)
        tk.Frame(body,bg=LINE,width=1).pack(side="left",fill="y")
        self._build_side()

        # main
        main=tk.Frame(body,bg=BG)
        main.pack(side="left",fill="both",expand=True)
        self._build_main(main)

        # ── bottom bar ───────────────────────────────────────────────────
        bot=tk.Frame(self,bg=SURF)
        bot.pack(fill="x",side="bottom")
        tk.Frame(bot,bg=LINE,height=1).pack(fill="x")

        self._stat_frame=tk.Frame(bot,bg=SURF)
        self._stat_frame.pack(fill="x")
        self._stat_placeholders()

        log_row=tk.Frame(bot,bg=SURF)
        log_row.pack(fill="x",padx=14,pady=4)
        self._dot=_lbl(log_row,"●",8,color=GREEN,bg=SURF)
        self._dot.pack(side="left")
        self._log_lbl=_lbl(log_row,"Ready.",8,color=MUTED,bg=SURF)
        self._log_lbl.pack(side="left",padx=6)
        self._spin_lbl=_lbl(log_row,"",8,color=AMBER,bg=SURF)
        self._spin_lbl.pack(side="left")
        self._spinning=False; self._spin_i=0

    # ── SIDEBAR ──────────────────────────────────────────────────────────
    def _build_side(self):
        s=self._side
        # top padding
        tk.Frame(s,bg=SURF,height=10).pack()

        def sec(title):
            tk.Frame(s,bg=LINE,height=1).pack(fill="x",padx=0)
            _lbl(s,title,7,"bold",MUTED,bg=SURF).pack(
                anchor="w",padx=14,pady=(8,4))
            f=tk.Frame(s,bg=SURF)
            f.pack(fill="x",padx=14)
            return f

        def field(parent,label,var):
            _lbl(parent,label,8,color=MUTED,bg=SURF).pack(anchor="w",pady=(4,2))
            e=_entry(parent,var)
            e.pack(fill="x",ipady=5)

        # File
        f0=sec("FILE")
        self._vid_lbl=_lbl(f0,"No file selected",8,color=DIM,bg=SURF,
                            wraplength=195,justify="left")
        self._vid_lbl.pack(anchor="w",pady=(0,6))
        _btn(f0,"Open Video…",self._open,accent=True).pack(fill="x")

        # Parameters
        f1=sec("PARAMETERS")
        self._hv=tk.StringVar(value="1.75")
        self._pv=tk.StringVar(value="0.005")
        self._idv=tk.StringVar(value="person00")
        field(f1,"Player Height (m)",self._hv)
        field(f1,"Pixels → Metres",  self._pv)
        _lbl(f1,"Person ID",8,color=MUTED,bg=SURF).pack(anchor="w",pady=(8,2))
        om=tk.OptionMenu(f1,self._idv,"person00","person01","person02")
        om.config(bg=CARD,fg=TEXT,font=("Helvetica Neue",9),
                  relief="flat",highlightthickness=1,
                  highlightbackground=LINE,bd=0,activebackground=LINE)
        om["menu"].config(bg=CARD,fg=TEXT,font=("Helvetica Neue",9))
        om.pack(fill="x")

        # Hoop
        f2=sec("HOOP")
        self._hoop_lbl=_lbl(f2,"Not set",8,color=DIM,bg=SURF)
        self._hoop_lbl.pack(anchor="w",pady=(0,6))
        self._btn_hoop=_btn(f2,"Mark Hoop in Player",self._hoop_start)
        self._btn_hoop.config(state="disabled")
        self._btn_hoop.pack(fill="x")

        # Ball tracking
        f3=sec("BALL TRACKING")
        self._bt_lbl=_lbl(f3,"Not tracked",8,color=DIM,bg=SURF,
                           wraplength=195,justify="left")
        self._bt_lbl.pack(anchor="w",pady=(0,6))
        self._btn_bt=_btn(f3,"▶  Start Tracking",self._bt_start)
        self._btn_bt.config(state="disabled")
        self._btn_bt.pack(fill="x",pady=(0,2))
        _lbl(f3,"Opens a video window. Press SPACE to pause, draw box, then ENTER.",7,color=DIM,bg=SURF).pack(anchor="w",pady=(0,4))

        # Run
        f4=sec("RUN ANALYSIS")
        self._btn_run=_btn(f4,"Run Analysis",self._run,accent=True)
        self._btn_run.config(state="disabled",
                              font=("Helvetica Neue",11,"bold"))
        self._btn_run.pack(fill="x",ipady=5)

        # spacer
        tk.Frame(s,bg=SURF).pack(fill="both",expand=True)

    def _stat_placeholders(self):
        for w in self._stat_frame.winfo_children(): w.destroy()
        items=[("—","m/s","Peak Wrist Speed"),
               ("—","m/s²","Peak Acceleration"),
               ("—","°","Min Elbow Angle"),
               ("—","°","Min Knee Angle"),
               ("—","°","Avg Trunk Angle")]
        for i,(v,u,l) in enumerate(items):
            c=tk.Frame(self._stat_frame,bg=SURF,
                       highlightthickness=1 if i>0 else 0,
                       highlightbackground=LINE)
            c.pack(side="left",fill="both",expand=True)
            _lbl(c,l,7,color=MUTED,bg=SURF).pack(anchor="w",padx=12,pady=(6,0))
            r=tk.Frame(c,bg=SURF); r.pack(anchor="w",padx=12,pady=(0,6))
            _lbl(r,v,17,"bold",TEXT,bg=SURF).pack(side="left",anchor="s")
            _lbl(r,f" {u}",8,color=MUTED,bg=SURF).pack(side="left",
                                                         anchor="s",pady=2)

    def _set_stats(self,items):
        for w in self._stat_frame.winfo_children(): w.destroy()
        for i,(v,u,l) in enumerate(items):
            c=tk.Frame(self._stat_frame,bg=SURF,
                       highlightthickness=1 if i>0 else 0,
                       highlightbackground=LINE)
            c.pack(side="left",fill="both",expand=True)
            _lbl(c,l,7,color=MUTED,bg=SURF).pack(anchor="w",padx=12,pady=(6,0))
            r=tk.Frame(c,bg=SURF); r.pack(anchor="w",padx=12,pady=(0,6))
            _lbl(r,v,17,"bold",TEXT,bg=SURF).pack(side="left",anchor="s")
            _lbl(r,f" {u}",8,color=MUTED,bg=SURF).pack(side="left",
                                                         anchor="s",pady=2)

    # ── MAIN ─────────────────────────────────────────────────────────────
    def _build_main(self, parent):
        # tab bar
        tab_bar=tk.Frame(parent,bg=SURF)
        tab_bar.pack(fill="x",side="top")
        tk.Frame(tab_bar,bg=LINE,height=1).pack(fill="x",side="bottom")

        self._tab_btns=[]
        self._tab_frames={}
        self._active_tab=0

        content=tk.Frame(parent,bg=BG)
        content.pack(fill="both",expand=True)
        self._content=content

        tabs=["Video","Kinematics","Release","Data","Overlay"]
        for i,t in enumerate(tabs):
            b=tk.Button(tab_bar,text=t,
                        font=("Helvetica Neue",9),
                        bg=SURF,fg=MUTED,
                        relief="flat",bd=0,padx=18,pady=10,
                        cursor="hand2",highlightthickness=0,
                        command=lambda i=i:self._tab(i))
            b.pack(side="left")
            self._tab_btns.append(b)

        # ── TAB 0: Video ──
        v=tk.Frame(content,bg=BG)
        self._player=VideoPlayer(v,
                                  on_click=self._hoop_click)
        self._player.pack(fill="both",expand=True,padx=6,pady=6)
        self._tab_frames[0]=v

        # wire ball flow
        self._flow=BallFlow(self._q,self._bt_done)
        self._flow.btn_start=self._btn_bt
        self._flow.slbl=self._bt_lbl

        # ── TAB 1: Kinematics ──
        k_scroll=Scrollable(content)
        k_inner=k_scroll.inner
        self._k_inner=k_inner
        self._tab_frames[1]=k_scroll

        # ── TAB 2: Release ──
        r=tk.Frame(content,bg=BG)
        # top half: plots grid
        self._r_plot_frame=tk.Frame(r,bg=BG)
        self._r_plot_frame.pack(fill="both",expand=True,side="top",pady=(6,0))
        # bottom half: table
        tk.Frame(r,bg=LINE,height=1).pack(fill="x")
        self._r_table_frame=tk.Frame(r,bg=BG)
        self._r_table_frame.pack(fill="both",expand=True,side="bottom")
        _lbl(self._r_table_frame,
             "Run analysis to see release window data.",
             9,color=MUTED).pack(pady=30)
        self._tab_frames[2]=r

        # ── TAB 3: Data ──
        d=tk.Frame(content,bg=BG)
        dtab_bar=tk.Frame(d,bg=SURF)
        dtab_bar.pack(fill="x")
        tk.Frame(dtab_bar,bg=LINE,height=1).pack(fill="x",side="bottom")
        self._d_tab_btns=[]
        self._d_tab_frames={}
        self._d_active=0
        d_content=tk.Frame(d,bg=BG)
        d_content.pack(fill="both",expand=True)
        for i,t in enumerate(["Positions","Angles","Kinematics"]):
            b=tk.Button(dtab_bar,text=t,
                        font=("Helvetica Neue",8),
                        bg=SURF,fg=MUTED,
                        relief="flat",bd=0,padx=14,pady=8,
                        cursor="hand2",highlightthickness=0,
                        command=lambda i=i:self._d_tab(i,d_content))
            b.pack(side="left")
            self._d_tab_btns.append(b)
            f=tk.Frame(d_content,bg=BG)
            self._d_tab_frames[i]=f
        self._tab_frames[3]=d

        # ── TAB 4: Overlay video ──
        ov=tk.Frame(content,bg=BG)
        self._ov_placeholder=_lbl(ov,
            "Run analysis to load the annotated overlay video.",
            10,color=MUTED)
        self._ov_placeholder.pack(expand=True)
        self._ov_player=None   # VideoPlayer created lazily after analysis
        self._tab_frames[4]=ov

        self._tab(0)

    def _tab(self, i):
        for f in self._tab_frames.values(): f.pack_forget()
        self._tab_frames[i].pack(fill="both",expand=True)
        self._active_tab=i
        for j,b in enumerate(self._tab_btns):
            if j==i:
                b.config(fg=TEXT,font=("Helvetica Neue",9,"bold"))
                for c in b.winfo_children(): c.destroy()
                tk.Frame(b,bg=AMBER,height=2).place(
                    x=0,rely=1.0,anchor="sw",relwidth=1)
            else:
                b.config(fg=MUTED,font=("Helvetica Neue",9))
                for c in b.winfo_children(): c.destroy()
        if i==3: self._d_tab(self._d_active,
                              self._tab_frames[3].winfo_children()[-1])

    def _d_tab(self, i, parent):
        for f in self._d_tab_frames.values(): f.pack_forget()
        self._d_tab_frames[i].pack(fill="both",expand=True)
        self._d_active=i
        for j,b in enumerate(self._d_tab_btns):
            if j==i:
                b.config(fg=TEXT,font=("Helvetica Neue",8,"bold"))
            else:
                b.config(fg=MUTED,font=("Helvetica Neue",8))

    # ── FILE ─────────────────────────────────────────────────────────────
    def _open(self):
        p=filedialog.askopenfilename(
            title="Open Video",
            filetypes=[("Video","*.mp4 *.avi *.mov *.mkv"),("All","*.*")])
        if not p: return
        self._video=p
        name=os.path.basename(p)
        self._vid_lbl.config(text=name,fg=TEXT)
        cap=cv2.VideoCapture(p)
        self._fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()
        self._player.load(p)
        self._flow._video=p
        self._btn_hoop.config(state="normal")
        self._btn_bt.config(state="normal")
        self._btn_run.config(state="normal")
        self._tab(0)
        self._log(f"Loaded  {name}  ({self._fps:.0f} fps)")

    # ── HOOP ─────────────────────────────────────────────────────────────
    def _hoop_start(self):
        self._player.hoop_mode(True)
        self._player.draw_mode(False)
        self._hoop_lbl.config(text="Click the hoop ↑",fg=AMBER)
        self._log("Click on the hoop centre in the video player above")

    def _hoop_click(self, fx, fy):
        if not self._player._hoop: return
        self._player.hoop_mode(False)
        self._hoop_px=(fx,fy)
        self._player.clear_ov("hoop")
        self._player.add_circ(fx,fy,16,RED,"hoop")
        self._hoop_lbl.config(
            text=f"Hoop @ ({int(fx)}, {int(fy)}) px",fg=GREEN)
        self._log(f"Hoop marked at ({int(fx)}, {int(fy)}) px")

    # ── BALL TRACKING ────────────────────────────────────────────────────
    def _bt_start(self):
        try: self._flow._px2m=float(self._pv.get())
        except: pass
        self._flow._video=self._video
        self._flow.start()

    def _bt_done(self, times, speeds, fps):
        self._bt_data={"times":times,"speeds":speeds}
        self._log(f"Ball tracking complete — {len(speeds)} frames")

    # ── RUN ANALYSIS ─────────────────────────────────────────────────────
    def _run(self):
        if not self._video:
            messagebox.showwarning("No Video","Please open a video first.")
            return
        self._btn_run.config(state="disabled",text="Running…")
        self._start_spin("Analyzing")
        threading.Thread(target=self._thread,daemon=True).start()

    def _thread(self):
        import builtins
        orig=builtins.print
        def gp(*a,**k):
            builtins.print=orig
            self._q.put(("log"," ".join(str(x) for x in a)))
            builtins.print=gp
        builtins.print=gp
        try:
            ana=_load_ana()
            vid=self._video
            ph=float(self._hv.get())
            pid=self._idv.get()
            rdir=os.path.join(os.path.dirname(os.path.abspath(vid)),
                              "Sports2D_Results")
            os.makedirs(rdir,exist_ok=True)

            self._q.put(("log","[1/5] Running Sports2D pose estimation…"))
            ana.run_sports2d(vid,ph,rdir)

            self._q.put(("log","[2/5] Parsing pose TRC (metres)…"))
            tm=ana._find_file(rdir,f"*_m_{pid}.trc","TRC metres")
            lm,ta,am=ana.parse_trc(tm)

            self._q.put(("log","[3/5] Parsing joint angles (MOT)…"))
            mp=ana._find_file(rdir,f"*_{pid}.mot","MOT")
            ang=ana.parse_mot(mp)

            n=min(len(ta),min(len(v) for v in ang.values()))
            ta=ta[:n]; lm={k:v[:n] for k,v in lm.items()}
            ang={k:v[:n] for k,v in ang.items()}
            am={k:v[:n] for k,v in am.items()}

            self._q.put(("log","[4/5] Computing kinematics…"))
            lk=ana.compute_linear_kinematics(lm,ta)
            ak=ana.compute_angular_kinematics(ang,ta)

            self._q.put(("log","[5/5] Parsing pixel TRC…"))
            tp=ana._find_file(rdir,f"*_px_{pid}.trc","TRC pixels")
            px=ana.parse_trc_px(tp)
            px={k:v[:n] for k,v in px.items()}

            hoop=None
            if self._hoop_px:
                hoop=self._px2m(self._hoop_px,am,px)

            self._q.put(("done",{
                "lk":lk,"ak":ak,"ta":ta,"lm":lm,
                "am":am,"px":px,"ang":ang,
                "hoop":hoop,"rdir":rdir,
                "vid":vid,
            }))
        except Exception as e:
            self._q.put(("error",f"{e}\n\n{traceback.format_exc()}"))
        finally:
            builtins.print=orig

    def _px2m(self, hpx, am, markers_px):
        a,b="RHip","RShoulder"
        if a in markers_px and b in markers_px and a in am and b in am:
            pd_=np.linalg.norm(markers_px[a][0]-markers_px[b][0])
            md_=np.linalg.norm(am[a][0]-am[b][0])
            ppm=pd_/md_ if pd_>0 else 100.0
        else: ppm=100.0
        rp=markers_px["RHip"][0].copy()
        rm=am["RHip"][0].copy()
        dp=np.array(hpx,float)-rp
        return (float(rm[0]+dp[0]/ppm), float(rm[1]-dp[1]/ppm))

    # ── RENDER RESULTS ───────────────────────────────────────────────────
    def _render(self, d):
        ana=_load_ana()
        lk=d["lk"]; ak=d["ak"]; ta=d["ta"]
        lm=d["lm"]; am=d["am"]; px=d["px"]
        ang=d["ang"]; hoop=d["hoop"]
        self._kin=lk; self._ang=ak; self._time=ta
        self._lm=lm; self._am=am; self._px=px; self._angles=ang

        wk=lk["RIGHT_WRIST"]

        # stats bar
        self._set_stats([
            (f"{float(np.nanmax(wk['vel_mag'])):.2f}","m/s","Peak Wrist Speed"),
            (f"{float(np.nanmax(wk['acc_mag'])):.1f}","m/s²","Peak Acceleration"),
            (f"{float(np.nanmin(ak['elbow']['angle'])):.0f}","°","Min Elbow Angle"),
            (f"{float(np.nanmin(ak['knee']['angle'])):.0f}","°","Min Knee Angle"),
            (f"{float(np.nanmean(ak['trunk']['angle'])):.0f}","°","Avg Trunk Angle"),
        ])

        # ── Kinematic plots ──
        for w in self._k_inner.winfo_children(): w.destroy()
        _mpl()

        plot_defs=[
            ("Wrist Position",
             [(ta,wk["disp"][:,0],AMBER,"X"),(ta,wk["disp"][:,1],BLUE,"Y")],
             "m"),
            ("Wrist Velocity",
             [(ta,wk["vel_mag"],AMBER,"Speed")],"m/s"),
            ("Wrist Acceleration",
             [(ta,wk["acc_mag"],RED,"Accel.")],"m/s²"),
            ("Elbow Angle",
             [(ta,ak["elbow"]["angle"],GREEN,"Elbow")],"°"),
            ("Elbow Angular Velocity",
             [(ta,ak["elbow"]["angular_vel"],BLUE,"°/s")],"°/s"),
            ("Knee Angle",
             [(ta,ak["knee"]["angle"],GOLD,"Knee")],"°"),
            ("All Joint Angles",
             [(ta,ak["knee"]["angle"],GOLD,"Knee"),
              (ta,ak["elbow"]["angle"],GREEN,"Elbow"),
              (ta,ak["trunk"]["angle"],BLUE,"Trunk")],"°"),
        ]

        cols=2
        for i,(title,series,ylabel) in enumerate(plot_defs):
            row_i=i//cols; col_i=i%cols
            cell=tk.Frame(self._k_inner,bg=CARD,
                          highlightthickness=1,highlightbackground=LINE)
            cell.grid(row=row_i,column=col_i,
                      sticky="nsew",padx=5,pady=5)
            self._k_inner.grid_columnconfigure(col_i,weight=1)

            fig,ax=plt.subplots(figsize=(5.8,2.6))
            for x,y,color,label in series:
                ax.plot(x,y,color=color,label=label,lw=1.8)
            ax.set_xlabel("Time (s)"); ax.set_ylabel(ylabel)
            ax.set_title(title,fontweight="bold",pad=5)
            ax.grid(True,alpha=0.35)
            if len(series)>1: ax.legend(frameon=True)
            fig.tight_layout(pad=0.7)

            cv=FigureCanvasTkAgg(fig,master=cell)
            cv.draw()
            cv.get_tk_widget().config(highlightthickness=0,bd=0)
            cv.get_tk_widget().pack(fill="both",expand=True,padx=4,pady=4)

        # ── Release analysis ──
        for w in self._r_plot_frame.winfo_children(): w.destroy()
        for w in self._r_table_frame.winfo_children(): w.destroy()

        if hoop:
            cands=ana.find_candidate_frames(lm,lk,hoop,ta)
            res=ana.analyze_release_window(cands,lm,lk,hoop,ta)

            # ── Replace wrist speed with tracker speed only for candidate frames
            # that fall WITHIN the tracker's recorded time range. Frames before
            # the box was drawn keep their original wrist-based speed.
            if self._bt_data and len(self._bt_data.get("times",[])):
                btt_all = np.array(self._bt_data["times"], dtype=float)
                bts_all = np.array(self._bt_data["speeds"], dtype=float)
                if len(btt_all) > 1 and np.nanmax(bts_all) > 0:
                    t_min = float(btt_all.min())
                    t_max = float(btt_all.max())
                    dt_tol = (t_max - t_min) / max(1, len(btt_all) - 1)

                    candidate_times = np.array(res["time"], dtype=float)
                    bt_actual = res["actual_speed"].copy()  # start from wrist

                    covered = np.zeros(len(candidate_times), dtype=bool)
                    for ci, ct in enumerate(candidate_times):
                        if (ct >= t_min - dt_tol) and (ct <= t_max + dt_tol):
                            closest_idx = int(np.argmin(np.abs(btt_all - ct)))
                            bt_actual[ci] = bts_all[closest_idx]
                            covered[ci] = True
                        # else: keep original wrist speed

                    res = dict(res)
                    res["actual_speed"] = bt_actual
                    req = res["required_speed"]
                    err   = np.abs(bt_actual - req)
                    rel   = np.where(req > 0, err / req * 100.0, np.nan)
                    ratio = np.where(req > 0, bt_actual / req,   np.nan)
                    res["speed_error"]        = err
                    res["relative_error_pct"] = rel
                    res["feasibility_ratio"]  = ratio
                    # best frame = lowest error among covered frames only
                    rel_masked = rel.copy()
                    if covered.any():
                        rel_masked[~covered] = np.inf
                    best_local = int(np.nanargmin(rel_masked))
                    res["optimal_frame"] = int(res["frame_idx"][best_local])
                    thresh = ana.OPTIMAL_WINDOW_THRESHOLD_PCT
                    res["optimal_window"] = res["frame_idx"][rel <= thresh]

            # table
            if len(res.get("frame_idx",[])):
                cols_r=("Frame","Time","Angle","Ball (m/s)","Required",
                         "Error","Rel %","Ratio","")
                tv=_make_tree(self._r_table_frame,cols_r,"R.Treeview",
                              [55,70,65,82,85,68,62,62,80])
                vsb=tk.Scrollbar(self._r_table_frame,orient="vertical",
                                 command=tv.yview,bg=SURF,
                                 troughcolor=SURF,width=5,
                                 relief="flat",bd=0)
                tv.configure(yscrollcommand=vsb.set)
                tv.tag_configure("best",foreground=GOLD,
                                  background="#1A1400")
                tv.tag_configure("win", foreground=GREEN,
                                  background="#0D1A0D")
                opt=res["optimal_frame"]
                win=set(res["optimal_window"].tolist())
                for j in range(len(res["frame_idx"])):
                    fi=res["frame_idx"][j]
                    status="★ BEST" if fi==opt else ("● WIN" if fi in win else "")
                    tag="best" if fi==opt else ("win" if fi in win else "")
                    tv.insert("","end",tags=(tag,),values=(
                        fi,
                        f"{res['time'][j]:.3f}",
                        f"{res['launch_angle_deg'][j]:.1f}°",
                        f"{res['actual_speed'][j]:.2f}",
                        f"{res['required_speed'][j]:.2f}",
                        f"{res['speed_error'][j]:.2f}",
                        f"{res['relative_error_pct'][j]:.1f}",
                        f"{res['feasibility_ratio'][j]:.3f}",
                        status,
                    ))
                tv.grid(row=0,column=0,sticky="nsew")
                vsb.grid(row=0,column=1,sticky="ns")
                self._r_table_frame.grid_rowconfigure(0,weight=1)
                self._r_table_frame.grid_columnconfigure(0,weight=1)

                # release plots — re-read opt after possible tracker override
                t=res["time"]
                opt_idx=res["optimal_frame"]
                j_opt_arr=np.where(res["frame_idx"]==opt_idx)[0]
                j_opt=int(j_opt_arr[0]) if opt_idx>=0 and len(j_opt_arr) else None
                opt_time=float(t[j_opt]) if j_opt is not None else None

                figs_r=[]

                f1,a1=plt.subplots(figsize=(5.5,2.5))
                a1.plot(t,res["relative_error_pct"],color=RED,lw=1.8)
                wm_=np.isin(res["frame_idx"],list(win))
                if wm_.any():
                    a1.fill_between(t,0,res["relative_error_pct"],
                                     where=wm_,alpha=0.15,color=GREEN)
                if opt_time is not None:
                    a1.axvline(opt_time,color=GOLD,ls="--",lw=1,
                               alpha=0.8,label="Best frame")
                a1.grid(True,alpha=0.3)
                a1.set_xlabel("Time (s)"); a1.set_ylabel("Error (%)")
                a1.set_title("Release Error",fontweight="bold")
                f1.tight_layout(pad=0.7)
                figs_r.append(("Release Error",f1))

                f2,a2=plt.subplots(figsize=(5.5,2.5))
                a2.plot(t,res["feasibility_ratio"],color=BLUE,lw=1.8)
                a2.axhline(1.0,color=MUTED,ls="--",lw=0.8,label="Optimal")
                if opt_time is not None:
                    a2.axvline(opt_time,color=GOLD,ls="--",lw=1,
                               alpha=0.8,label="Best frame")
                a2.grid(True,alpha=0.3)
                a2.set_xlabel("Time (s)"); a2.set_ylabel("Actual / Required")
                a2.set_title("Feasibility Ratio",fontweight="bold")
                f2.tight_layout(pad=0.7)
                figs_r.append(("Feasibility Ratio",f2))

                f3,a3=plt.subplots(figsize=(5.5,2.5))
                if self._bt_data and len(self._bt_data.get("times",[])):
                    btt=np.array(self._bt_data["times"])
                    bts=np.array(self._bt_data["speeds"])
                    # btt is already real video-time — plot directly
                    a3.plot(btt,bts,color=GREEN,lw=1.8,label="Ball (tracker)")
                    # red dot marks the BEST CANDIDATE FRAME (same as gold line)
                    if opt_time is not None:
                        dot_idx = int(np.argmin(np.abs(btt - opt_time)))
                        a3.scatter([btt[dot_idx]],[bts[dot_idx]],
                                   color=RED,zorder=5,s=40)
                else:
                    a3.plot(ta,lk["RIGHT_WRIST"]["vel_mag"],
                             color=DIM,lw=1,label="Wrist speed")
                    a3.plot(t,res["actual_speed"],
                             color=AMBER,lw=1.8,label="Candidates")
                a3.plot(t,res["required_speed"],
                         color=BLUE,ls="--",lw=1.5,label="Required")
                if opt_time is not None:
                    a3.axvline(opt_time,color=GOLD,ls="--",lw=1,
                               alpha=0.8,label="Best frame")
                a3.legend(frameon=True)
                a3.grid(True,alpha=0.3)
                a3.set_xlabel("Time (s)"); a3.set_ylabel("m/s")
                a3.set_title("Speed Comparison",fontweight="bold")
                f3.tight_layout(pad=0.7)
                figs_r.append(("Speed Comparison",f3))

                # 2-col grid
                for i,(title,fig) in enumerate(figs_r):
                    r_c=i%3; r_r=i//3
                    cell=tk.Frame(self._r_plot_frame,bg=CARD,
                                  highlightthickness=1,
                                  highlightbackground=LINE)
                    cell.grid(row=r_r,column=r_c,
                               sticky="nsew",padx=5,pady=5)
                    self._r_plot_frame.grid_columnconfigure(r_c,weight=1)
                    self._r_plot_frame.grid_rowconfigure(r_r,weight=1)
                    cw2=FigureCanvasTkAgg(fig,master=cell)
                    cw2.draw()
                    cw2.get_tk_widget().config(highlightthickness=0,bd=0)
                    cw2.get_tk_widget().pack(fill="both",expand=True,
                                             padx=4,pady=4)
            else:
                _lbl(self._r_table_frame,"No valid candidate frames.",
                     9,color=MUTED).pack(pady=30)
        else:
            _lbl(self._r_table_frame,
                 "Mark the hoop to enable release analysis.",
                 9,color=MUTED).pack(pady=30)

        # ── Data tables ──
        import pandas as pd

        def _load_table(frame_idx, df):
            f,_=list(self._d_tab_frames.items())[frame_idx]
            for w in self._d_tab_frames[frame_idx].winfo_children():
                w.destroy()
            cols=list(df.columns)
            tv=_make_tree(self._d_tab_frames[frame_idx],cols)
            vsb=tk.Scrollbar(self._d_tab_frames[frame_idx],orient="vertical",
                             command=tv.yview,bg=SURF,troughcolor=SURF,
                             width=5,relief="flat",bd=0)
            hsb=tk.Scrollbar(self._d_tab_frames[frame_idx],orient="horizontal",
                             command=tv.xview,bg=SURF,troughcolor=SURF,
                             width=5,relief="flat",bd=0)
            tv.configure(yscrollcommand=vsb.set,xscrollcommand=hsb.set)
            for i,(_,row) in enumerate(df.iterrows()):
                vals=[f"{v:.4f}" if isinstance(v,float) else str(v)
                       for v in row]
                tv.insert("","end",values=vals,tags=("alt",) if i%2 else ())
            tv.grid(row=0,column=0,sticky="nsew")
            vsb.grid(row=0,column=1,sticky="ns")
            hsb.grid(row=1,column=0,sticky="ew")
            self._d_tab_frames[frame_idx].grid_rowconfigure(0,weight=1)
            self._d_tab_frames[frame_idx].grid_columnconfigure(0,weight=1)

        pos={"time":ta}
        for n_,pos_ in lm.items():
            mk=ana.TRC_MARKERS[n_]
            pos[f"{mk}_X"]=pos_[:,0]; pos[f"{mk}_Y"]=pos_[:,1]
        _load_table(0,pd.DataFrame(pos))

        ad={"time":ta}
        for n_,a_ in ang.items(): ad[ana.MOT_ANGLES[n_]]=a_
        _load_table(1,pd.DataFrame(ad))

        kd={"time":ta,"wrist_vel_mag":wk["vel_mag"],
            "wrist_acc_mag":wk["acc_mag"],
            "disp_x":wk["disp"][:,0],"disp_y":wk["disp"][:,1]}
        _load_table(2,pd.DataFrame(kd))

        # ── Overlay tab ──
        self._load_overlay(d["rdir"], d.get("vid"))

        self._tab(1)

    # ── OVERLAY TAB ──────────────────────────────────────────────────────
    def _load_overlay(self, rdir, vid_path=None):
        """Find Sports2D annotated output video and load it into the Overlay tab."""
        import glob

        # Sports2D names output as  {video_stem}_Sports2D.mp4
        # Search for that exact stem first so we never pick up a stale old file.
        candidates = []
        if vid_path:
            stem = os.path.splitext(os.path.basename(vid_path))[0]
            for pat in [
                os.path.join(rdir, f"{stem}_Sports2D.mp4"),
                os.path.join(rdir, f"{stem}*.mp4"),
                os.path.join(rdir, "**", f"{stem}*.mp4"),
            ]:
                candidates += glob.glob(pat, recursive=True)

        # Fallback: any mp4 with Sports2D in the name, sorted newest-first
        if not candidates:
            for pat in [
                os.path.join(rdir, "*Sports2D*.mp4"),
                os.path.join(rdir, "**", "*Sports2D*.mp4"),
            ]:
                candidates += glob.glob(pat, recursive=True)
            candidates = sorted(set(candidates),
                                key=os.path.getmtime, reverse=True)

        if not candidates:
            self._log("Overlay: no annotated video found in results folder.")
            return

        overlay_path = candidates[0]
        self._log(f"Overlay: loaded {os.path.basename(overlay_path)}")

        # Clear placeholder label
        for w in self._tab_frames[4].winfo_children():
            w.destroy()

        # Reuse VideoPlayer — no hoop/box callbacks needed here
        self._ov_player = VideoPlayer(self._tab_frames[4])
        self._ov_player.pack(fill="both", expand=True, padx=6, pady=6)
        self._ov_player.load(overlay_path)

        # Update the Overlay tab button to show it's ready
        if len(self._tab_btns) > 4:
            self._tab_btns[4].config(fg=AMBER)

    # ── LOG / SPINNER ────────────────────────────────────────────────────
    def _log(self, msg):
        clean=msg.replace("[INFO] ","").replace("[INFO]","").strip()
        if not clean: return
        self._log_lbl.config(text=clean[:120]+("…" if len(clean)>120 else ""))

    def _start_spin(self, text=""):
        self._spinning=True; self._spin_i=0; self._spin_text=text
        self._do_spin()

    def _stop_spin(self):
        self._spinning=False; self._spin_lbl.config(text="")

    def _do_spin(self):
        if not self._spinning: return
        ch=["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        self._spin_lbl.config(
            text=f"  {ch[self._spin_i%len(ch)]}  {self._spin_text}")
        self._spin_i+=1
        self.after(80,self._do_spin)

    # ── QUEUE ────────────────────────────────────────────────────────────
    def _poll(self):
        try:
            while True:
                kind,payload=self._q.get_nowait()
                if kind=="log":
                    self._log(payload)
                elif kind=="done":
                    self._stop_spin()
                    self._btn_run.config(state="normal",text="Run Analysis")
                    self._render(payload)
                elif kind=="error":
                    self._stop_spin()
                    self._btn_run.config(state="normal",text="Run Analysis")
                    self._log(f"Error — check console")
                    messagebox.showerror("Analysis Error",str(payload)[:500])
        except queue.Empty:
            pass
        self.after(80,self._poll)


if __name__=="__main__":
    ShotLab().mainloop()
