import sys
import os
import traceback
import tempfile
import uuid
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QPlainTextEdit, QFileDialog,
    QSizePolicy, QScrollArea, QSplitter, QCheckBox
)
from PySide6.QtCore import QTimer, Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Wedge


# ===================== SETTINGS =====================
FPS = 30
DEFAULT_ANIMATION_DURATION = 10
DEFAULT_PAUSE_DURATION = 3

PREVIEW_DPI = 95.0
AUTO_APPLY_DEBOUNCE_MS = 350

DEFAULT_VALUES = "20,20,20,20,20"
DEFAULT_COLORS = [
    "#0B7A2E",  # dark green
    "#8EE889",  # light green
    "#F3F2BF",  # pale yellow
    "#FF780C",  # orange
    "#00B0F0",  # blue
]

# Outer ring harsh gradient behavior
OUTER_BLEND_START = 0.85   # blend only in last 15% of a segment
OUTER_HARSH_POWER = 4.0    # larger => harsher

# Ring resolution
PREVIEW_RING_RES = 700     # preview quality
EXPORT_RING_RES = 900      # export quality
# ====================================================


def parse_values_csv(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",")]
    vals = []
    for p in parts:
        if p == "" or p.lower() == "nan":
            vals.append(np.nan)
        else:
            vals.append(float(p))
    arr = np.array(vals, dtype=float)
    if np.any(np.isnan(arr)):
        arr = np.nan_to_num(arr, nan=0.0)
    return arr


def parse_positive_float(text: str, field_name: str) -> float:
    v = float(text.strip())
    if v <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return v


def parse_colors_block(text: str) -> list[str]:
    colors = [c.strip() for c in text.splitlines() if c.strip()]
    if not colors:
        raise ValueError("Provide at least one color (one per line).")
    return colors


def hex_to_rgb01(h: str) -> np.ndarray:
    s = h.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"Bad hex color: {h}")
    r = int(s[0:2], 16) / 255.0
    g = int(s[2:4], 16) / 255.0
    b = int(s[4:6], 16) / 255.0
    return np.array([r, g, b], dtype=float)


def lerp(c1, c2, t):
    return c1 * (1 - t) + c2 * t


def build_gradient_ring_rgba(values_pct: np.ndarray, colors_rgb: np.ndarray,
                             outer_radius=1.35, outer_width=0.07, N=900):
    """
    Outer ring:
      - segments follow values_pct (sum=100)
      - each segment stays solid until near end, then blends to next quickly
      - last segment ends on its own color (no blend back to first)
    Returns: rgba, theta_deg_clockwise_from_12, alpha_ring, extent
    """
    values_pct = np.array(values_pct, dtype=float)

    ys, xs = np.ogrid[-1:1:N*1j, -1:1:N*1j]
    r = np.sqrt(xs**2 + ys**2)

    ang = (np.degrees(np.arctan2(ys, xs)) + 360) % 360
    theta = (90 - ang) % 360          # 0..360 clockwise from 12
    p = theta / 360.0                 # 0..1 around ring

    bounds = np.cumsum(np.insert(values_pct, 0, 0.0)) / 100.0
    rgb = np.zeros((N, N, 3), dtype=float)

    m = len(values_pct)
    for j in range(m):
        a = bounds[j]
        b = bounds[j + 1]
        mask = (p >= a) & (p <= b)
        if not np.any(mask):
            continue

        u = (p[mask] - a) / (b - a + 1e-12)  # 0..1 within segment

        # hold solid until OUTER_BLEND_START, then harsh blend
        t = np.clip((u - OUTER_BLEND_START) / (1.0 - OUTER_BLEND_START + 1e-12), 0.0, 1.0)
        t = t ** OUTER_HARSH_POWER

        c1 = colors_rgb[j]
        c2 = c1 if j == m - 1 else colors_rgb[j + 1]
        rgb[mask] = lerp(c1, c2, t[..., None])

    r_min = (outer_radius - outer_width) / outer_radius
    r_max = 1.0
    alpha_ring = ((r >= r_min) & (r <= r_max)).astype(float)

    rgba = np.dstack([rgb, alpha_ring])
    extent = (-outer_radius, outer_radius, -outer_radius, outer_radius)
    return rgba, theta, alpha_ring, extent


class AspectRatioWidget(QWidget):
    def __init__(self, child: QWidget, aspect: float, parent=None):
        super().__init__(parent)
        self._child = child
        self._aspect = max(1e-6, float(aspect))
        self._child.setParent(self)

    def set_aspect(self, aspect: float):
        self._aspect = max(1e-6, float(aspect))
        self.updateGeometry()
        self.update()

    def resizeEvent(self, event):
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0:
            return

        target_w = w
        target_h = int(round(target_w / self._aspect))
        if target_h > h:
            target_h = h
            target_w = int(round(target_h * self._aspect))

        x = (w - target_w) // 2
        y = (h - target_h) // 2
        self._child.setGeometry(x, y, target_w, target_h)


class MplCanvas(FigureCanvas):
    def __init__(self):
        fig = Figure(figsize=(10, 6))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()


class PlotFullscreenWindow(QMainWindow):
    def __init__(self, parent, plot_widget: QWidget, on_close_restore):
        super().__init__(parent)
        self._on_close_restore = on_close_restore
        self.setWindowTitle("Plot Fullscreen")

        cw = QWidget()
        lay = QVBoxLayout(cw)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(plot_widget)
        self.setCentralWidget(cw)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        self._on_close_restore()
        super().closeEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Donut + Gradient Outer Ring (Optimized + Stable Export)")

        # inputs
        self.values_in = QLineEdit(DEFAULT_VALUES)
        self.figw_in = QLineEdit("10")
        self.figh_in = QLineEdit("6")

        self.colors_in = QPlainTextEdit("\n".join(DEFAULT_COLORS))
        self.colors_in.setFixedHeight(110)

                # animation timing
        self.anim_sec_in = QLineEdit("10")
        self.pause_sec_in = QLineEdit("3")

        # option: toggle ONLY the % sign
        self.show_pct_cb = QCheckBox("Add % sign to labels")
        self.show_pct_cb.setChecked(True)

        # buttons
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.export_btn = QPushButton("Export MP4")
        self.plot_full_btn = QPushButton("Plot Fullscreen")

        self.start_btn.clicked.connect(self.start_animation)
        self.stop_btn.clicked.connect(self.stop_animation)
        self.export_btn.clicked.connect(self.export_mp4)
        self.plot_full_btn.clicked.connect(self.open_plot_fullscreen)

        self.status = QLabel("")

        # plot
        self.canvas = MplCanvas()
        self.ax = self.canvas.ax
        self.aspect_wrap = AspectRatioWidget(self.canvas, aspect=10 / 6)

        self.plot_scroll = QScrollArea()
        self.plot_scroll.setFrameShape(QScrollArea.NoFrame)
        self.plot_scroll.setWidgetResizable(False)
        self.plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.plot_scroll.setWidget(self.aspect_wrap)

        # controls panel
        controls = QWidget()
        cl = QVBoxLayout(controls)

        r1 = QWidget(); l1 = QHBoxLayout(r1)
        l1.addWidget(QLabel("Values (must sum to 100):"))
        l1.addWidget(self.values_in)
        cl.addWidget(r1)

        r2 = QWidget(); l2 = QHBoxLayout(r2)
        l2.addWidget(QLabel("Fig W"))
        l2.addWidget(self.figw_in)
        l2.addWidget(QLabel("Fig H"))
        l2.addWidget(self.figh_in)
        cl.addWidget(r2)

        r3 = QWidget(); l3 = QHBoxLayout(r3)
        l3.addWidget(QLabel("Anim sec"))
        l3.addWidget(self.anim_sec_in)
        l3.addWidget(QLabel("Pause sec"))
        l3.addWidget(self.pause_sec_in)
        cl.addWidget(r3)

        cl.addWidget(QLabel("Colors (one per value):"))
        cl.addWidget(self.colors_in)
        cl.addWidget(self.show_pct_cb)

        btnrow = QWidget(); bl = QHBoxLayout(btnrow)
        bl.addWidget(self.start_btn)
        bl.addWidget(self.stop_btn)
        bl.addWidget(self.export_btn)
        bl.addWidget(self.plot_full_btn)
        cl.addWidget(btnrow)

        cl.addWidget(self.status)
        cl.addStretch(1)

        # splitter
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(controls)
        self.splitter.addWidget(self.plot_scroll)
        self.splitter.setSizes([280, 800])

        central = QWidget()
        main = QVBoxLayout(central)
        main.addWidget(self.splitter)
        self.setCentralWidget(central)

        # animation
        self.frame = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step)

        self._apply_debounce = QTimer(self)
        self._apply_debounce.setSingleShot(True)
        self._apply_debounce.timeout.connect(self.apply_all)

        def schedule_heavy_rebuild():
            self.status.setText("⏳ Pending changes...")
            self.status.setStyleSheet("color: #444;")
            self._apply_debounce.start(AUTO_APPLY_DEBOUNCE_MS)

        # Heavy rebuild triggers
        self.values_in.textChanged.connect(schedule_heavy_rebuild)
        self.figw_in.textChanged.connect(schedule_heavy_rebuild)
        self.figh_in.textChanged.connect(schedule_heavy_rebuild)
        self.colors_in.textChanged.connect(schedule_heavy_rebuild)

        self.anim_sec_in.textChanged.connect(schedule_heavy_rebuild)
        self.pause_sec_in.textChanged.connect(schedule_heavy_rebuild)


        # Checkbox does NOT rebuild; it just updates label text
        self.show_pct_cb.toggled.connect(self.on_toggle_percent_sign)

        # fullscreen
        self._plot_full_win = None
        self._placeholder = None

        # look
        self.outer_radius = 1.35
        self.outer_width = 0.07

        self.donut_width = 0.42
        self.wedge_edge = 2.0
        self.label_threshold = 0.6
        self.label_font = 11


        # ring cache (preview)
        self._ring_rgba = None
        self._ring_theta = None
        self._ring_alpha = None
        self._ring_extent = None
        self._ring_frame = None

        # artists
        self._donut_wedges: list[Wedge] = []
        self._pct_texts = []
        self._ring_im = None

        self.values = None
        self.colors_rgb = None
        self._starts = None

        self._init_axes()
        self.apply_all()
        self.start_animation()
        self.resize(1200, 800)

    def _init_axes(self):
        ax = self.ax
        ax.clear()

        # reset cached artist references (they were detached by clear)
        self._ring_im = None
        self._donut_wedges = []
        self._pct_texts = []

        ax.set_aspect("equal")
        ax.axis("off")
        self.canvas.figure.patch.set_facecolor("white")
        ax.set_xlim(-1.55, 1.55)
        ax.set_ylim(-1.35, 1.35)



    def update_plot_widget_size(self):
        w_px = int(self.figw * PREVIEW_DPI)
        h_px = int(self.figh * PREVIEW_DPI)
        w_px = max(w_px, 500)
        h_px = max(h_px, 300)
        self.aspect_wrap.setFixedSize(w_px, h_px)
        self.aspect_wrap.updateGeometry()
        self.plot_scroll.viewport().update()
        QApplication.processEvents()

    def open_plot_fullscreen(self):
        if self._plot_full_win is not None:
            self._plot_full_win.activateWindow()
            return

        self._placeholder = QWidget()
        self.splitter.replaceWidget(1, self._placeholder)

        def restore():
            self.splitter.replaceWidget(1, self.plot_scroll)
            self.splitter.setSizes([280, 800])
            self._plot_full_win = None
            if self._placeholder is not None:
                self._placeholder.deleteLater()
                self._placeholder = None

        self._plot_full_win = PlotFullscreenWindow(self, self.plot_scroll, restore)
        self._plot_full_win.showFullScreen()

    def start_animation(self):
        self.timer.start(int(1000 / FPS))

    def stop_animation(self):
        self.timer.stop()

    def on_toggle_percent_sign(self, checked: bool):
        self.update_frame(self.frame)
        self.canvas.draw_idle()

    def _build_artists(self):
        ax = self.ax
        self._init_axes()

        # ALWAYS recreate ring image after clear
        self._ring_im = ax.imshow(
            np.zeros((2, 2, 4), dtype=float),
            extent=(-1, 1, -1, 1),
            origin="lower",
            interpolation="nearest",
            zorder=0
        )

        self._donut_wedges = []
        self._pct_texts = []

        for i in range(len(self.values)):
            c = tuple(self.colors_rgb[i])
            w = Wedge(
                center=(0, 0),
                r=1.0,
                theta1=0,
                theta2=0,
                width=self.donut_width,   # <-- donut hole comes from this
                facecolor=c,
                edgecolor="white",
                linewidth=self.wedge_edge,
                zorder=5
            )
            ax.add_patch(w)
            self._donut_wedges.append(w)

            txt = ax.text(
                0, 0, "",
                ha="center", va="center",
                fontsize=self.label_font,
                fontweight="normal",
                color="black",
                zorder=10
            )
            self._pct_texts.append(txt)


    def apply_all(self):
        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        try:
            values = parse_values_csv(self.values_in.text())
            if len(values) < 2:
                raise ValueError("Provide at least 2 values.")

            s = float(np.sum(values))
            if abs(s - 100.0) > 1e-6:
                raise ValueError(f"Values must sum to 100. Current sum = {s:g}")

            colors_hex = parse_colors_block(self.colors_in.toPlainText())

            # auto-extend colors if needed
            if len(colors_hex) < len(values):
                ext = list(colors_hex)
                while len(ext) < len(values):
                    ext.append(ext[-1])
                self.colors_in.blockSignals(True)
                try:
                    self.colors_in.setPlainText("\n".join(ext))
                finally:
                    self.colors_in.blockSignals(False)
                colors_hex = ext

            colors_hex = colors_hex[:len(values)]
            colors_rgb = np.array([hex_to_rgb01(c) for c in colors_hex], dtype=float)

            self.figw = parse_positive_float(self.figw_in.text(), "Figure width")
            self.figh = parse_positive_float(self.figh_in.text(), "Figure height")

            self.values = values
            self.colors_rgb = colors_rgb

            self.aspect_wrap.set_aspect(self.figw / self.figh)
            self.update_plot_widget_size()

            anim_sec = parse_positive_float(self.anim_sec_in.text(), "Animation seconds")
            pause_sec = parse_positive_float(self.pause_sec_in.text(), "Pause seconds")

            self.total_frames = int(FPS * anim_sec)
            self.pause_frames = int(FPS * pause_sec)
            self.total_frames_with_pause = self.total_frames + self.pause_frames

            self.frame = 0

            # preview ring cache
            self._ring_rgba, self._ring_theta, self._ring_alpha, self._ring_extent = build_gradient_ring_rgba(
                self.values,
                self.colors_rgb,
                outer_radius=self.outer_radius,
                outer_width=self.outer_width,
                N=PREVIEW_RING_RES
            )
            self._ring_frame = self._ring_rgba.copy()

            # start angles
            spans_full = (self.values / 100.0) * 360.0
            starts = [90.0]
            for i in range(1, len(spans_full)):
                starts.append(starts[-1] - spans_full[i - 1])
            self._starts = np.array(starts, dtype=float)

            self._build_artists()
            self.update_frame(self.frame)
            self.canvas.draw_idle()

            self.status.setText("✅ Applied.")
            self.status.setStyleSheet("color: green;")

        except ValueError as e:
            self.status.setText(f"❌ {e}")
            self.status.setStyleSheet("color: red;")

        finally:
            if was_running:
                self.timer.start(int(1000 / FPS))

    def step(self):
        self.frame = (self.frame + 1) % self.total_frames_with_pause
        self.update_frame(self.frame)
        self.canvas.draw_idle()

    def update_frame(self, frame: int):
        self.canvas.figure.set_size_inches(self.figw, self.figh, forward=True)

        progress = 1.0 if frame >= self.total_frames else (frame / max(1, self.total_frames))
        painted = 100.0 * progress

        # sequential reveal
        visible = np.zeros_like(self.values, dtype=float)
        remaining = painted
        for i, v in enumerate(self.values):
            vis = float(np.clip(remaining, 0.0, v))
            visible[i] = vis
            remaining -= v

        # donut
        for i, w in enumerate(self._donut_wedges):
            span_vis = (visible[i] / 100.0) * 360.0
            start = float(self._starts[i])
            end = start - span_vis
            w.set_theta1(end)
            w.set_theta2(start)

        # labels: checkbox only toggles the % sign
        add_percent_sign = self.show_pct_cb.isChecked()
        for i, txt in enumerate(self._pct_texts):
            v = float(visible[i])
            if v <= self.label_threshold:
                txt.set_visible(False)
                continue

            start_pct = float(np.sum(visible[:i]))
            mid_pct = start_pct + v / 2.0
            angle = 90.0 - (mid_pct / 100.0) * 360.0
            x = np.cos(np.deg2rad(angle)) * 0.77
            y = np.sin(np.deg2rad(angle)) * 0.77

            s = f"{v:.1f}"
            if add_percent_sign:
                s += "%"

            txt.set_text(s)
            txt.set_position((x, y))
            txt.set_visible(True)

        # ring reveal
        reveal_theta = 360.0 * (painted / 100.0)
        alpha_reveal = ((self._ring_theta <= reveal_theta).astype(float)) * self._ring_alpha
        self._ring_frame[..., :3] = self._ring_rgba[..., :3]
        self._ring_frame[..., 3] = alpha_reveal
        self._ring_im.set_data(self._ring_frame)
        self._ring_im.set_extent(self._ring_extent)

    def export_mp4(self):
        was_running = self.timer.isActive()
        self.timer.stop()

        final_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save animation as MP4",
            "donut_gradient_ring.mp4",
            "MP4 Video (*.mp4)"
        )
        if not final_path:
            if was_running:
                self.timer.start(int(1000 / FPS))
            return

        if not final_path.lower().endswith(".mp4"):
            final_path += ".mp4"

        tmp_path = os.path.join(tempfile.gettempdir(), f"anim_{uuid.uuid4().hex}.mp4")

        try:
            self.status.setText("⏳ Exporting... (final file appears only when done)")
            self.status.setStyleSheet("color: #444;")
            QApplication.processEvents()

            # Build HIGH-RES ring for export (once)
            ring_rgba, ring_theta, ring_alpha, ring_extent = build_gradient_ring_rgba(
                self.values,
                self.colors_rgb,
                outer_radius=self.outer_radius,
                outer_width=self.outer_width,
                N=EXPORT_RING_RES
            )
            ring_frame = ring_rgba.copy()

            # OFFSCREEN figure (Agg) for stable export
            fig = Figure(figsize=(self.figw, self.figh))
            FigureCanvasAgg(fig)  # attach Agg canvas
            ax = fig.add_subplot(111)
            ax.set_aspect("equal")
            ax.axis("off")
            fig.patch.set_facecolor("white")
            ax.set_xlim(-1.55, 1.55)
            ax.set_ylim(-1.35, 1.35)

            ring_im = ax.imshow(
                ring_frame,
                extent=ring_extent,
                origin="lower",
                interpolation="nearest",
                zorder=0
            )

            # Donut wedges/texts in offscreen figure
            wedges = []
            texts = []
            for i in range(len(self.values)):
                c = tuple(self.colors_rgb[i])
                w = Wedge(
                    center=(0, 0),
                    r=1.0,
                    theta1=0,
                    theta2=0,
                    width=self.donut_width,
                    facecolor=c,
                    edgecolor="white",
                    linewidth=self.wedge_edge,
                    zorder=5
                )
                ax.add_patch(w)
                wedges.append(w)

                t = ax.text(
                    0, 0, "",
                    ha="center", va="center",
                    fontsize=self.label_font,
                    fontweight="normal",
                    color="black",
                    zorder=10
                )
                texts.append(t)

            # start angles
            spans_full = (self.values / 100.0) * 360.0
            starts = [90.0]
            for i in range(1, len(spans_full)):
                starts.append(starts[-1] - spans_full[i - 1])
            starts = np.array(starts, dtype=float)

            add_percent_sign = self.show_pct_cb.isChecked()

            writer = FFMpegWriter(
                fps=FPS,
                codec="libx264",
                bitrate=3000,
                extra_args=[
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"
                ]
            )

            with writer.saving(fig, tmp_path, dpi=300):
                for f in range(self.total_frames_with_pause):
                    progress = 1.0 if f >= self.total_frames else (f / max(1, self.total_frames))
                    painted = 100.0 * progress

                    visible = np.zeros_like(self.values, dtype=float)
                    remaining = painted
                    for i, v in enumerate(self.values):
                        vis = float(np.clip(remaining, 0.0, v))
                        visible[i] = vis
                        remaining -= v

                    for i, w in enumerate(wedges):
                        span_vis = (visible[i] / 100.0) * 360.0
                        start = float(starts[i])
                        end = start - span_vis
                        w.set_theta1(end)
                        w.set_theta2(start)

                    for i, txt in enumerate(texts):
                        v = float(visible[i])
                        if v <= self.label_threshold:
                            txt.set_visible(False)
                            continue

                        start_pct = float(np.sum(visible[:i]))
                        mid_pct = start_pct + v / 2.0
                        angle = 90.0 - (mid_pct / 100.0) * 360.0
                        x = np.cos(np.deg2rad(angle)) * 0.77
                        y = np.sin(np.deg2rad(angle)) * 0.77

                        s = f"{v:.1f}"
                        if add_percent_sign:
                            s += "%"

                        txt.set_text(s)
                        txt.set_position((x, y))
                        txt.set_visible(True)

                    reveal_theta = 360.0 * (painted / 100.0)
                    alpha_reveal = ((ring_theta <= reveal_theta).astype(float)) * ring_alpha
                    ring_frame[..., :3] = ring_rgba[..., :3]
                    ring_frame[..., 3] = alpha_reveal
                    ring_im.set_data(ring_frame)

                    writer.grab_frame()

            size = os.path.getsize(tmp_path)
            if size < 50_000:
                raise RuntimeError(f"Export produced very small file ({size} bytes).")

            os.replace(tmp_path, final_path)

            self.status.setText(f"✅ Saved MP4: {final_path}")
            self.status.setStyleSheet("color: green;")

        except Exception as e:
            print("\n=== EXPORT ERROR ===")
            traceback.print_exc()

            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

            msg = str(e)
            if "ffmpeg" in msg.lower() or "FileNotFoundError" in msg:
                msg += " | Tip: install FFmpeg and ensure it's on PATH."

            self.status.setText(f"❌ Export failed: {msg} (see console)")
            self.status.setStyleSheet("color: red;")

        finally:
            if was_running:
                self.timer.start(int(1000 / FPS))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

