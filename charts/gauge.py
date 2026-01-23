import sys
import os
import tempfile
import uuid
import traceback
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QFileDialog,
    QSizePolicy, QScrollArea, QSplitter, QCheckBox
)
from PySide6.QtCore import QTimer, Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Circle, Arc
from matplotlib.ticker import FuncFormatter
from PySide6.QtGui import QDoubleValidator  
from PySide6.QtWidgets import QProgressDialog
import shutil

def _add_bundled_ffmpeg_to_path():
    if shutil.which("ffmpeg"):
        return

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = sys._MEIPASS
        ffmpeg_dir = os.path.join(base, "bin")
        ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
        if os.path.exists(ffmpeg_exe):
            os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
            return

    here = os.path.dirname(os.path.abspath(__file__))
    ffmpeg_dir = os.path.join(here, "bin")
    ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
    if os.path.exists(ffmpeg_exe):
        os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")
        return

# ===================== SETTINGS =====================
FPS = 30
PREVIEW_DPI = 95.0
AUTO_APPLY_DEBOUNCE_MS = 350

DEFAULT_FIGW = "7"
DEFAULT_FIGH = "4.5"

DEFAULT_TITLE = ""
DEFAULT_VALUE = "83.8"

DEFAULT_FILL = "#B56400"
DEFAULT_BGARC = "#E3E3E3"
DEFAULT_TICK = "#D8D8D8"
DEFAULT_PANEL_BG = "#F2F2F2"

DEFAULT_ARC_LW = "14"
DEFAULT_TICK_COUNT = "70"
DEFAULT_TICK_LEN = "0.10"
DEFAULT_TICK_LW = "2.0"
DEFAULT_TICK_RADIUS = "0.82"   # relative to R=1.0

DEFAULT_ANIM_SEC = "2.5"
DEFAULT_PAUSE_SEC = "1.5"
# ====================================================


def parse_float(text: str, name: str) -> float:
    try:
        return float(text.strip())
    except Exception:
        raise ValueError(f"{name} must be a number.")


def parse_positive_float(text: str, name: str) -> float:
    v = parse_float(text, name)
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")
    return v


def parse_int_pos(text: str, name: str) -> int:
    try:
        v = int(text.strip())
    except Exception:
        raise ValueError(f"{name} must be an integer.")
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")
    return v


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def validate_hex_color(s: str, name: str) -> str:
    s = s.strip()
    if not s.startswith("#") or len(s) != 7:
        raise ValueError(f"{name} must be hex like #B56400")
    int(s[1:], 16)
    return s

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
        fig = Figure(figsize=(7, 4.5))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
    def wheelEvent(self, event):
        # Let the parent scroll area handle the wheel (so page scroll works on hover)
        event.ignore()


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
        _add_bundled_ffmpeg_to_path()
        self.setWindowTitle("Gauge (Semi-circle) - Stable Export")

        # inputs
        self.figw_in = QLineEdit(DEFAULT_FIGW)
        self.figh_in = QLineEdit(DEFAULT_FIGH)

        self.title_in = QLineEdit(DEFAULT_TITLE)
        self.value_in = QLineEdit(DEFAULT_VALUE)

        self.fill_in = QLineEdit(DEFAULT_FILL)
        self.bgarc_in = QLineEdit(DEFAULT_BGARC)
        self.tick_in = QLineEdit(DEFAULT_TICK)
        self.panelbg_in = QLineEdit(DEFAULT_PANEL_BG)

        self.arc_lw_in = QLineEdit(DEFAULT_ARC_LW)
        self.tick_count_in = QLineEdit(DEFAULT_TICK_COUNT)
        self.tick_len_in = QLineEdit(DEFAULT_TICK_LEN)
        self.tick_lw_in = QLineEdit(DEFAULT_TICK_LW)
        self.tick_radius_in = QLineEdit(DEFAULT_TICK_RADIUS)

        self.anim_sec_in = QLineEdit(DEFAULT_ANIM_SEC)
        self.pause_sec_in = QLineEdit(DEFAULT_PAUSE_SEC)

        self.show_ticks_cb = QCheckBox("Show ticks")
        self.show_ticks_cb.setChecked(True)

        self.show_ends_cb = QCheckBox("Show 0 / 100")
        self.show_ends_cb.setChecked(True)

        self.comma_decimal_cb = QCheckBox("Use comma decimal (83,8)")
        self.comma_decimal_cb.setChecked(True)

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
        self.aspect_wrap = AspectRatioWidget(self.canvas, aspect=float(DEFAULT_FIGW) / float(DEFAULT_FIGH))
        # Inner wrapper that will center the plot
        self.plot_inner = QWidget()
        inner_lay = QHBoxLayout(self.plot_inner)
        inner_lay.setContentsMargins(0, 0, 0, 0)
        inner_lay.addStretch(1)
        inner_lay.addWidget(self.aspect_wrap)
        inner_lay.addStretch(1)

        # Scroll area: full width, horizontal scroll when needed
        self.plot_hscroll = QScrollArea()
        self.plot_hscroll.setFrameShape(QScrollArea.NoFrame)
        self.plot_hscroll.setWidgetResizable(True)  # needed for centering
        self.plot_hscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.plot_hscroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plot_hscroll.setWidget(self.plot_inner)

        # Plot container
        self.plot_container = QWidget()
        pl = QVBoxLayout(self.plot_container)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.plot_hscroll)

        # Make it expand to full width (prevents "ultra narrow")
        self.plot_hscroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.plot_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # controls layout
        controls = QWidget()
        cl = QVBoxLayout(controls)

        r0 = QWidget(); l0 = QHBoxLayout(r0)
        l0.addWidget(QLabel("Fig W")); l0.addWidget(self.figw_in)
        l0.addWidget(QLabel("Fig H")); l0.addWidget(self.figh_in)
        cl.addWidget(r0)

        r1 = QWidget(); l1 = QHBoxLayout(r1)
        l1.addWidget(QLabel("Title")); l1.addWidget(self.title_in)
        cl.addWidget(r1)

        r2 = QWidget(); l2 = QHBoxLayout(r2)
        l2.addWidget(QLabel("Value (0-100)")); l2.addWidget(self.value_in)
        cl.addWidget(r2)

        r3 = QWidget(); l3 = QHBoxLayout(r3)
        l3.addWidget(QLabel("Fill")); l3.addWidget(self.fill_in)
        l3.addWidget(QLabel("BG arc")); l3.addWidget(self.bgarc_in)
        cl.addWidget(r3)

        r4 = QWidget(); l4 = QHBoxLayout(r4)
        l4.addWidget(QLabel("Tick")); l4.addWidget(self.tick_in)
        l4.addWidget(QLabel("Panel BG")); l4.addWidget(self.panelbg_in)
        cl.addWidget(r4)

        r5 = QWidget(); l5 = QHBoxLayout(r5)
        l5.addWidget(QLabel("Arc thickness")); l5.addWidget(self.arc_lw_in)
        l5.addWidget(QLabel("Tick count")); l5.addWidget(self.tick_count_in)
        cl.addWidget(r5)

        r6 = QWidget(); l6 = QHBoxLayout(r6)
        l6.addWidget(QLabel("Tick len")); l6.addWidget(self.tick_len_in)
        l6.addWidget(QLabel("Tick lw")); l6.addWidget(self.tick_lw_in)
        l6.addWidget(QLabel("Tick radius")); l6.addWidget(self.tick_radius_in)
        cl.addWidget(r6)

        r7 = QWidget(); l7 = QHBoxLayout(r7)
        l7.addWidget(QLabel("Anim sec")); l7.addWidget(self.anim_sec_in)
        l7.addWidget(QLabel("Pause sec")); l7.addWidget(self.pause_sec_in)
        cl.addWidget(r7)

        cl.addWidget(self.show_ticks_cb)
        cl.addWidget(self.show_ends_cb)
        cl.addWidget(self.comma_decimal_cb)

        btnrow = QWidget(); bl = QHBoxLayout(btnrow)
        bl.addWidget(self.start_btn)
        bl.addWidget(self.stop_btn)
        bl.addWidget(self.export_btn)
        bl.addWidget(self.plot_full_btn)
        cl.addWidget(btnrow)

        cl.addWidget(self.status)
        cl.addStretch(1)

        # -------- One-page scroll: controls + plot together --------
        page = QWidget()
        self.page_lay = QVBoxLayout(page)
        self.page_lay.addWidget(controls)
        self.page_lay.addWidget(self.plot_container)  # no stretch spacer here

        self.page_scroll = QScrollArea()
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QScrollArea.NoFrame)
        self.page_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.page_scroll.setWidget(page)
        self.setCentralWidget(self.page_scroll)


        # animation
        self.frame = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step)

        self._apply_debounce = QTimer(self)
        self._apply_debounce.setSingleShot(True)
        self._apply_debounce.timeout.connect(self.apply_all)

        def schedule_rebuild():
            self.status.setText("⏳ Pending changes...")
            self.status.setStyleSheet("color: #444;")
            self._apply_debounce.start(AUTO_APPLY_DEBOUNCE_MS)

        for w in [
            self.figw_in, self.figh_in, self.title_in, self.value_in,
            self.fill_in, self.bgarc_in, self.tick_in, self.panelbg_in,
            self.arc_lw_in, self.tick_count_in, self.tick_len_in, self.tick_lw_in, self.tick_radius_in,
            self.anim_sec_in, self.pause_sec_in,
        ]:
            w.textChanged.connect(schedule_rebuild)

        self.show_ticks_cb.toggled.connect(lambda _: self.canvas.draw_idle())
        self.show_ends_cb.toggled.connect(lambda _: self.canvas.draw_idle())
        self.comma_decimal_cb.toggled.connect(lambda _: self.canvas.draw_idle())

        # fullscreen
        self._plot_full_win = None
        self._placeholder = None

        # state
        self.figw = float(DEFAULT_FIGW)
        self.figh = float(DEFAULT_FIGH)
        self.title = DEFAULT_TITLE
        self.value = float(DEFAULT_VALUE)

        self.fill_color = DEFAULT_FILL
        self.bg_color = DEFAULT_BGARC
        self.tick_color = DEFAULT_TICK
        self.panel_bg = DEFAULT_PANEL_BG

        self.R = 1.0
        self.arc_lw = float(DEFAULT_ARC_LW)
        self.tick_count = int(DEFAULT_TICK_COUNT)
        self.tick_len = float(DEFAULT_TICK_LEN)
        self.tick_lw = float(DEFAULT_TICK_LW)
        self.tick_radius = float(DEFAULT_TICK_RADIUS)

        self.anim_frames = 2
        self.pause_frames = 0
        self.total_frames = 2

        self.apply_all()
        self.start_animation()
        self.resize(1200, 800)

    def update_plot_widget_size(self):
        w_px = int(self.figw * PREVIEW_DPI)
        h_px = int(self.figh * PREVIEW_DPI)
        w_px = max(w_px, 600)
        h_px = max(h_px, 360)
        self.aspect_wrap.setFixedSize(w_px, h_px)
        self.aspect_wrap.updateGeometry()
        # This makes scrollbars appear when needed, but keeps centering when not needed
        if hasattr(self, "plot_inner"):
            self.plot_inner.setMinimumWidth(w_px)
            self.plot_inner.setMinimumHeight(h_px)

        # Keep plot area visible (not “ultra short”)
        if hasattr(self, "plot_hscroll"):
            sb_h = self.plot_hscroll.horizontalScrollBar().sizeHint().height()
            self.plot_hscroll.setFixedHeight(h_px + sb_h + 4)

        if hasattr(self, "page_scroll"):
            self.page_scroll.viewport().update()
        QApplication.processEvents()

    def open_plot_fullscreen(self):
        if self._plot_full_win is not None:
            self._plot_full_win.activateWindow()
            return

        # remove from page layout + detach
        self._plot_index = self.page_lay.indexOf(self.plot_container)
        self.page_lay.removeWidget(self.plot_container)
        self.plot_container.setParent(None)

        def restore():
            insert_at = self._plot_index if self._plot_index >= 0 else self.page_lay.count()
            self.page_lay.insertWidget(insert_at, self.plot_container)
            self._plot_full_win = None

        self._plot_full_win = PlotFullscreenWindow(self, self.plot_container, restore)
        self._plot_full_win.showFullScreen()

    def start_animation(self):
        self.timer.start(int(1000 / FPS))

    def stop_animation(self):
        self.timer.stop()

    def _draw_ticks(self, ax):
        if not self.show_ticks_cb.isChecked():
            return
        angles = np.linspace(180, 0, self.tick_count)
        for a in angles:
            th = np.deg2rad(a)
            x0, y0 = self.tick_radius * np.cos(th), self.tick_radius * np.sin(th)
            x1, y1 = (self.tick_radius - self.tick_len) * np.cos(th), (self.tick_radius - self.tick_len) * np.sin(th)
            ax.plot([x0, x1], [y0, y1], color=self.tick_color, lw=self.tick_lw,
                    solid_capstyle="round", zorder=1)

    def _format_center_value(self, pct: float) -> str:
        s = f"{pct:.1f}"
        if self.comma_decimal_cb.isChecked():
            s = s.replace(".", ",")
        return s

    def draw_gauge(self, pct: float):
        ax = self.ax
        ax.clear()
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("white")
        self.canvas.figure.patch.set_facecolor(self.panel_bg)

        # title
        ax.text(0, 1.12, self.title, ha="center", va="center",
                fontsize=14, fontweight="bold", color="#7A7A7A")

        # ticks
        self._draw_ticks(ax)

        # background arc (0..180)
        bg_arc = Arc((0, 0), 2*self.R, 2*self.R, angle=0, theta1=0, theta2=180,
                     lw=self.arc_lw, color=self.bg_color, capstyle="round", zorder=2)
        ax.add_patch(bg_arc)

        # fill arc portion: 0..100 -> 180..0
        pct = clamp(pct, 0.0, 100.0)
        end_angle = 180 - 180 * (pct / 100.0)  # 0% -> 180, 100% -> 0
        if pct > 0:
            fill_arc = Arc((0, 0), 2*self.R, 2*self.R, angle=0, theta1=end_angle, theta2=180,
                           lw=self.arc_lw, color=self.fill_color, capstyle="round", zorder=3)
            ax.add_patch(fill_arc)

        # center circle + shadow
        shadow = Circle((0.02, -0.02), 0.40, facecolor="black", edgecolor="none", alpha=0.08, zorder=3)
        ax.add_patch(shadow)
        center = Circle((0, 0), 0.40, facecolor="white", edgecolor="#F0F0F0", lw=1.0, zorder=4)
        ax.add_patch(center)

        # center value
        ax.text(0, -0.02, self._format_center_value(pct), ha="center", va="center",
                fontsize=24, fontweight="bold", color=self.fill_color, zorder=5)

        # end labels
        if self.show_ends_cb.isChecked():
            ax.text(-1, -0.2, "0", ha="center", va="center", fontsize=12, color="#333333")
            ax.text( 1, -0.2, "100", ha="center", va="center", fontsize=12, color="#333333")

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-0.65, 1.25)

    def apply_all(self):
        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        try:
            self.figw = parse_positive_float(self.figw_in.text(), "Figure width")
            self.figh = parse_positive_float(self.figh_in.text(), "Figure height")
            self.aspect_wrap.set_aspect(self.figw / self.figh)
            self.update_plot_widget_size()

            self.title = self.title_in.text()
            self.value = clamp(parse_float(self.value_in.text(), "Value"), 0.0, 100.0)

            self.fill_color = validate_hex_color(self.fill_in.text(), "Fill color")
            self.bg_color = validate_hex_color(self.bgarc_in.text(), "BG arc color")
            self.tick_color = validate_hex_color(self.tick_in.text(), "Tick color")
            self.panel_bg = validate_hex_color(self.panelbg_in.text(), "Panel BG")

            self.arc_lw = parse_positive_float(self.arc_lw_in.text(), "Arc thickness")
            self.tick_count = parse_int_pos(self.tick_count_in.text(), "Tick count")
            self.tick_len = parse_positive_float(self.tick_len_in.text(), "Tick length")
            self.tick_lw = parse_positive_float(self.tick_lw_in.text(), "Tick thickness")
            self.tick_radius = parse_positive_float(self.tick_radius_in.text(), "Tick radius")

            anim_sec = parse_positive_float(self.anim_sec_in.text(), "Animation seconds")
            pause_sec = parse_positive_float(self.pause_sec_in.text(), "Pause seconds")
            self.anim_frames = max(2, int(FPS * anim_sec))
            self.pause_frames = int(FPS * pause_sec)
            self.total_frames = self.anim_frames + self.pause_frames

            self.frame = 0
            self.draw_gauge(0.0)
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
        self.frame = (self.frame + 1) % self.total_frames
        self.update_frame(self.frame)
        self.canvas.draw_idle()

    def update_frame(self, frame: int):
        if frame >= self.anim_frames:
            pct = self.value
        else:
            pct = self.value * (frame / max(1, self.anim_frames))
        self.draw_gauge(pct)

    def export_mp4(self):
        was_running = self.timer.isActive()
        _add_bundled_ffmpeg_to_path()
        self.timer.stop()

        final_path, _ = QFileDialog.getSaveFileName(
            self, "Save animation as MP4",
            "gauge.mp4",
            "MP4 Video (*.mp4)"
        )
        if not final_path:
            if was_running:
                self.timer.start(int(1000 / FPS))
            return

        if not final_path.lower().endswith(".mp4"):
            final_path += ".mp4"

        tmp_path = os.path.join(tempfile.gettempdir(), f"anim_{uuid.uuid4().hex}.mp4")

        # --- Progress dialog ---
        total = int(self.total_frames)
        progress_dlg = QProgressDialog("Exporting video...\nPlease wait.", "Cancel", 0, total, self)
        progress_dlg.setWindowTitle("Exporting")
        progress_dlg.setWindowModality(Qt.ApplicationModal)
        progress_dlg.setMinimumDuration(0)  # show immediately
        progress_dlg.setAutoClose(True)
        progress_dlg.setAutoReset(True)
        progress_dlg.setValue(0)
        progress_dlg.show()
        QApplication.processEvents()

        try:
            self.status.setText("⏳ Exporting... (final file appears only when done)")
            self.status.setStyleSheet("color: #444;")
            QApplication.processEvents()

            fig = Figure(figsize=(self.figw, self.figh))
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            writer = FFMpegWriter(
                fps=FPS,
                codec="libx264",
                bitrate=2500,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
            )

            def draw_off(pct: float):
                ax.clear()
                ax.set_aspect("equal")
                ax.axis("off")
                ax.set_facecolor("white")
                fig.patch.set_facecolor(self.panel_bg)

                ax.text(0, 1.12, self.title, ha="center", va="center",
                        fontsize=14, fontweight="bold", color="#7A7A7A")

                if self.show_ticks_cb.isChecked():
                    angles = np.linspace(180, 0, self.tick_count)
                    for a in angles:
                        th = np.deg2rad(a)
                        x0, y0 = self.tick_radius * np.cos(th), self.tick_radius * np.sin(th)
                        x1, y1 = (self.tick_radius - self.tick_len) * np.cos(th), (self.tick_radius - self.tick_len) * np.sin(th)
                        ax.plot([x0, x1], [y0, y1], color=self.tick_color, lw=self.tick_lw,
                                solid_capstyle="round", zorder=1)

                bg_arc = Arc((0, 0), 2*self.R, 2*self.R, angle=0, theta1=0, theta2=180,
                            lw=self.arc_lw, color=self.bg_color, capstyle="round", zorder=2)
                ax.add_patch(bg_arc)

                pct = clamp(pct, 0.0, 100.0)
                end_angle = 180 - 180 * (pct / 100.0)
                if pct > 0:
                    fill_arc = Arc((0, 0), 2*self.R, 2*self.R, angle=0, theta1=end_angle, theta2=180,
                                lw=self.arc_lw, color=self.fill_color, capstyle="round", zorder=3)
                    ax.add_patch(fill_arc)

                shadow = Circle((0.02, -0.02), 0.40, facecolor="black", edgecolor="none", alpha=0.08, zorder=3)
                ax.add_patch(shadow)
                center = Circle((0, 0), 0.40, facecolor="white", edgecolor="#F0F0F0", lw=1.0, zorder=4)
                ax.add_patch(center)

                s = f"{pct:.1f}"
                if self.comma_decimal_cb.isChecked():
                    s = s.replace(".", ",")
                ax.text(0, -0.02, s, ha="center", va="center",
                        fontsize=24, fontweight="bold", color=self.fill_color, zorder=5)

                if self.show_ends_cb.isChecked():
                    ax.text(-1, -0.2, "0", ha="center", va="center", fontsize=12, color="#333333")
                    ax.text( 1, -0.2, "100", ha="center", va="center", fontsize=12, color="#333333")

                ax.set_xlim(-1.25, 1.25)
                ax.set_ylim(-0.65, 1.25)

            with writer.saving(fig, tmp_path, dpi=300):
                for frame in range(total):
                    if progress_dlg.wasCanceled():
                        raise RuntimeError("Export canceled by user.")

                    if frame >= self.anim_frames:
                        pct = self.value
                    else:
                        pct = self.value * (frame / max(1, self.anim_frames))

                    draw_off(pct)
                    writer.grab_frame()

                    # update progress (throttle optional)
                    if frame % 2 == 0 or frame == total - 1:
                        progress_dlg.setValue(frame + 1)
                        QApplication.processEvents()

            if os.path.getsize(tmp_path) < 30_000:
                raise RuntimeError("Export produced very small file; check ffmpeg setup.")

            os.replace(tmp_path, final_path)
            self.status.setText(f"✅ Saved MP4: {final_path}")
            self.status.setStyleSheet("color: green;")

        except Exception as e:
            traceback.print_exc()
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            msg = str(e)
            if "ffmpeg" in msg.lower() or "FileNotFoundError" in msg:
                msg += " | Tip: install FFmpeg and ensure it's on PATH."
            self.status.setText(f"❌ Export failed: {msg}")
            self.status.setStyleSheet("color: red;")

        finally:
            try:
                progress_dlg.close()
            except Exception:
                pass

            if was_running:
                self.timer.start(int(1000 / FPS))



if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
