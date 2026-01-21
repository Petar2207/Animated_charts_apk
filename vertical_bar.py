import sys
import os
import tempfile
import uuid
import traceback
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
from matplotlib.ticker import MultipleLocator
from PySide6.QtGui import QDoubleValidator  

# ===================== SETTINGS =====================
FPS = 30
PREVIEW_DPI = 95.0
AUTO_APPLY_DEBOUNCE_MS = 350

DEFAULT_FIGW = "12"
DEFAULT_FIGH = "5"
DEFAULT_ANIM_SEC = "10"
DEFAULT_PAUSE_SEC = "2"

DEFAULT_BG = "#EAEAEF"
DEFAULT_DARK = "#5c1f2b"
DEFAULT_LIGHT = "#B2B2B2"
DEFAULT_GRID = "gray"
DEFAULT_TEXT = "gray"

DEFAULT_XMIN = "-20"
DEFAULT_XMAX = "20.5"
DEFAULT_XTICK = "10"

DEFAULT_LABEL_A = "X"
DEFAULT_LABEL_B = "Y"

DEFAULT_CATEGORIES = """Mitarbeiterorientierte Führung
Mitarbeiterorientierte Organisation
Kund:innen
Leistung
Vision & Werte
Nachhaltigkeit & Resilienz
Work Life Flexibility
Führung
Team
Agilität
STARKE KULTUR
"""

DEFAULT_A = "-3.9,3.2,-5.3,-1.6,3.0,0.2,-1.1,-0.6,0.7,-0.9,-2.1"
DEFAULT_B = "-0.8,-0.5,-0.4,-0.3,-1.1,-0.2,-0.1,0.2,2.0,-0.7,-0.6"
# ====================================================


def parse_categories(text: str) -> list[str]:
    cats = [c.strip() for c in text.splitlines() if c.strip()]
    if not cats:
        raise ValueError("Provide at least 1 category.")
    return cats


def parse_csv_floats(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    return np.array([float(p) for p in parts], dtype=float)


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


def validate_hex_color(s: str, name: str) -> str:
    s = s.strip()
    if not s.startswith("#") or len(s) != 7:
        raise ValueError(f"{name} must be hex like #5c1f2b")
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
        fig = Figure(figsize=(12, 5))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
    def wheelEvent(self, event):
        # Let the parent scroll area handle the wheel
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
        self.setWindowTitle("Diverging Barh Animation (2 Series)")

        # ----- inputs -----
        self.figw_in = QLineEdit(DEFAULT_FIGW)
        self.figh_in = QLineEdit(DEFAULT_FIGH)
        self.anim_sec_in = QLineEdit(DEFAULT_ANIM_SEC)
        self.pause_sec_in = QLineEdit(DEFAULT_PAUSE_SEC)

        self.cats_in = QPlainTextEdit(DEFAULT_CATEGORIES)
        self.cats_in.setFixedHeight(150)

        self.a_in = QLineEdit(DEFAULT_A)  # series A (dark)
        self.b_in = QLineEdit(DEFAULT_B)  # series B (light)

        self.color_a_in = QLineEdit(DEFAULT_DARK)
        self.color_b_in = QLineEdit(DEFAULT_LIGHT)

        self.xmin_in = QLineEdit(DEFAULT_XMIN)
        self.xmax_in = QLineEdit(DEFAULT_XMAX)
        self.xtick_in = QLineEdit(DEFAULT_XTICK)

        self.label_a_in = QLineEdit(DEFAULT_LABEL_A)
        self.label_b_in = QLineEdit(DEFAULT_LABEL_B)

        self.show_a_cb = QCheckBox("Show Bars A")
        self.show_a_cb.setChecked(True)
        self.show_b_cb = QCheckBox("Show Bars B")
        self.show_b_cb.setChecked(True)

        self.show_values_cb = QCheckBox("Show value labels")
        self.show_values_cb.setChecked(True)

        self.show_plus_cb = QCheckBox("Show + sign on positives")
        self.show_plus_cb.setChecked(True)

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

        # ----- plot -----
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

        # ----- controls layout -----
        controls = QWidget()
        cl = QVBoxLayout(controls)

        r1 = QWidget(); l1 = QHBoxLayout(r1)
        l1.addWidget(QLabel("Fig W")); l1.addWidget(self.figw_in)
        l1.addWidget(QLabel("Fig H")); l1.addWidget(self.figh_in)
        cl.addWidget(r1)

        r2 = QWidget(); l2 = QHBoxLayout(r2)
        l2.addWidget(QLabel("Anim sec")); l2.addWidget(self.anim_sec_in)
        l2.addWidget(QLabel("Pause sec")); l2.addWidget(self.pause_sec_in)
        cl.addWidget(r2)

        cl.addWidget(QLabel("Categories (one per line):"))
        cl.addWidget(self.cats_in)

        cl.addWidget(QLabel("Bars A values (CSV):"))
        cl.addWidget(self.a_in)
        cl.addWidget(QLabel("Bars B values (CSV):"))
        cl.addWidget(self.b_in)

        rcol = QWidget(); lcol = QHBoxLayout(rcol)
        lcol.addWidget(QLabel("Color A")); lcol.addWidget(self.color_a_in)
        lcol.addWidget(QLabel("Color B")); lcol.addWidget(self.color_b_in)
        cl.addWidget(rcol)

        rx = QWidget(); lx = QHBoxLayout(rx)
        lx.addWidget(QLabel("X min")); lx.addWidget(self.xmin_in)
        lx.addWidget(QLabel("X max")); lx.addWidget(self.xmax_in)
        lx.addWidget(QLabel("X tick")); lx.addWidget(self.xtick_in)
        cl.addWidget(rx)

        rlab = QWidget(); llab = QHBoxLayout(rlab)
        llab.addWidget(QLabel("Label A")); llab.addWidget(self.label_a_in)
        llab.addWidget(QLabel("Label B")); llab.addWidget(self.label_b_in)
        cl.addWidget(rlab)

        cl.addWidget(self.show_a_cb)
        cl.addWidget(self.show_b_cb)
        cl.addWidget(self.show_values_cb)
        cl.addWidget(self.show_plus_cb)

        btnrow = QWidget(); bl = QHBoxLayout(btnrow)
        bl.addWidget(self.start_btn)
        bl.addWidget(self.stop_btn)
        bl.addWidget(self.export_btn)
        bl.addWidget(self.plot_full_btn)
        cl.addWidget(btnrow)

        cl.addWidget(self.status)
        cl.addStretch(1)

        # splitter
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

        # ----- animation -----
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

        for w in [
            self.figw_in, self.figh_in, self.anim_sec_in, self.pause_sec_in,
            self.a_in, self.b_in,
            self.color_a_in, self.color_b_in,
            self.xmin_in, self.xmax_in, self.xtick_in,
            self.label_a_in, self.label_b_in,
        ]:
            w.textChanged.connect(schedule_heavy_rebuild)

        self.cats_in.textChanged.connect(schedule_heavy_rebuild)
        self.show_a_cb.toggled.connect(schedule_heavy_rebuild)
        self.show_b_cb.toggled.connect(schedule_heavy_rebuild)
        self.show_values_cb.toggled.connect(lambda _: self.canvas.draw_idle())
        self.show_plus_cb.toggled.connect(lambda _: self.canvas.draw_idle())

        # fullscreen
        self._plot_full_win = None
        self._placeholder = None

        # ----- state -----
        self.categories = []
        self.a_vals = None
        self.b_vals = None
        self.y = None

        self.figw = float(DEFAULT_FIGW)
        self.figh = float(DEFAULT_FIGH)

        self.anim_frames = 2
        self.pause_frames = 0
        self.total_frames = 2

        self.bg = DEFAULT_BG
        self.grid_color = DEFAULT_GRID
        self.text_color = DEFAULT_TEXT

        self.color_a = DEFAULT_DARK
        self.color_b = DEFAULT_LIGHT
        self.label_a = DEFAULT_LABEL_A
        self.label_b = DEFAULT_LABEL_B

        self.show_a = True
        self.show_b = True

        self.xmin = float(DEFAULT_XMIN)
        self.xmax = float(DEFAULT_XMAX)
        self.xtick = float(DEFAULT_XTICK)

        # artists
        self.bars_a = []
        self.bars_b = []
        self.value_texts_a = []
        self.value_texts_b = []
        self.cat_texts = []
        self._zero_line = None
        self._legend = None

        self.apply_all()
        self.start_animation()
        self.resize(1300, 900)

    def update_plot_widget_size(self):
        w_px = int(self.figw * PREVIEW_DPI)
        h_px = int(self.figh * PREVIEW_DPI)
        w_px = max(w_px, 700)
        h_px = max(h_px, 350)
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

    def _init_axes(self, ax):
        ax.clear()
        ax.set_facecolor(self.bg)
        ax.figure.patch.set_facecolor(self.bg)

        for side in ["right", "left", "top", "bottom"]:
            ax.spines[side].set_visible(False)

        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='y', which='major', length=0)

    def _build_artists(self):
        ax = self.ax
        self._init_axes(ax)

        n = len(self.categories)
        self.y = np.arange(n, dtype=float)

        # Layout similar to your notebook code
        self.canvas.figure.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25)

        ax.set_xlim(self.xmin, self.xmax)
        ax.set_ylim(-0.5, n - 0.5)

        # X ticks + vertical grid
        ax.xaxis.set_major_locator(MultipleLocator(self.xtick))
        ax.grid(True, axis='x', linestyle='-', color=self.grid_color, alpha=0.8, linewidth=0.2, zorder=1)

        # Horizontal grid between rows (dashed)
        y_grid_lines = np.arange(0.5, n - 0.5, 1)
        ax.set_yticks(y_grid_lines)
        ax.set_yticklabels([])
        ax.grid(True, axis='y', linestyle=(0, (5, 5)), color=self.grid_color, alpha=0.8, linewidth=0.2, zorder=1)

        # Center (zero) line
        self._zero_line = ax.axvline(x=0, color=self.grid_color, linewidth=1.2, zorder=2)

        # Category labels on left (static)
        self.cat_texts = []
        x_text = self.xmin + (abs(self.xmin) * 0.025)  # slightly inside left bound
        for yy, label in zip(self.y, self.categories):
            t = ax.text(x_text, yy, label, va='center', ha='left', fontsize=9,
                        color=self.text_color, zorder=4)
            self.cat_texts.append(t)

        # Bars (start at 0 width)
        self.bars_a = []
        self.bars_b = []

        if self.show_a:
            bars_a = ax.barh(self.y + 0.15, np.zeros(n), color=self.color_a, height=0.3, zorder=3, label=self.label_a)
            self.bars_a = list(bars_a)

        if self.show_b:
            bars_b = ax.barh(self.y - 0.15, np.zeros(n), color=self.color_b, height=0.3, zorder=3, label=self.label_b)
            self.bars_b = list(bars_b)

        # Value texts (two rows)
        self.value_texts_a = []
        self.value_texts_b = []
        for yy in self.y:
            if self.show_a:
                ta = ax.text(0, yy + 0.13, "", va='center', ha='left', fontsize=8, zorder=4, color="#707070")
                self.value_texts_a.append(ta)
            if self.show_b:
                tb = ax.text(0, yy - 0.18, "", va='center', ha='left', fontsize=6, zorder=4, color="#707070")
                self.value_texts_b.append(tb)

        # X tick labels color
        for tick in ax.get_xticklabels():
            tick.set_fontsize(8)
            tick.set_color(self.text_color)

        # Legend (only if at least one is visible)
        handles, labs = ax.get_legend_handles_labels()
        if handles:
            self._legend = ax.legend(
                loc='upper center',
                bbox_to_anchor=(0.5, -0.25),
                ncol=len(handles),
                frameon=False,
                fontsize=10,
                handlelength=1.8,
                columnspacing=2.0
            )
        else:
            self._legend = None

    def apply_all(self):
        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        try:
            # size
            self.figw = parse_positive_float(self.figw_in.text(), "Figure width")
            self.figh = parse_positive_float(self.figh_in.text(), "Figure height")
            self.aspect_wrap.set_aspect(self.figw / self.figh)
            self.update_plot_widget_size()

            # timing
            anim_sec = parse_positive_float(self.anim_sec_in.text(), "Animation seconds")
            pause_sec = parse_positive_float(self.pause_sec_in.text(), "Pause seconds")
            self.anim_frames = max(2, int(FPS * anim_sec))
            self.pause_frames = int(FPS * pause_sec)
            self.total_frames = self.anim_frames + self.pause_frames

            # labels + toggles
            self.label_a = self.label_a_in.text().strip() or DEFAULT_LABEL_A
            self.label_b = self.label_b_in.text().strip() or DEFAULT_LABEL_B
            self.show_a = self.show_a_cb.isChecked()
            self.show_b = self.show_b_cb.isChecked()

            # colors
            self.color_a = validate_hex_color(self.color_a_in.text(), "Color A")
            self.color_b = validate_hex_color(self.color_b_in.text(), "Color B")

            # axis range
            xmin = parse_float(self.xmin_in.text(), "X min")
            xmax = parse_float(self.xmax_in.text(), "X max")
            if xmax <= xmin:
                raise ValueError("X max must be > X min.")
            self.xmin, self.xmax = xmin, xmax
            self.xtick = parse_positive_float(self.xtick_in.text(), "X tick step")

            # data
            cats = parse_categories(self.cats_in.toPlainText())
            a = parse_csv_floats(self.a_in.text())
            b = parse_csv_floats(self.b_in.text())
            if len(a) != len(cats) or len(b) != len(cats):
                raise ValueError(f"Lengths must match categories ({len(cats)}). A={len(a)}, B={len(b)}")

            # reverse order so last category is at top (like your notebook)
            self.categories = cats[::-1]
            self.a_vals = a[::-1]
            self.b_vals = b[::-1]

            self.frame = 0
            self._build_artists()
            self.update_frame(0)
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

    def _format_value(self, v: float) -> str:
        if abs(v) <= 0.05:
            return ""
        if self.show_plus_cb.isChecked():
            return f"{v:+.1f}"
        return f"{v:.1f}"

    def update_frame(self, frame: int):
        if frame >= self.anim_frames:
            progress = 1.0
        else:
            progress = frame / max(1, (self.anim_frames - 1))

        show_values = self.show_values_cb.isChecked()
        gap = 0.2

        # Update A bars + texts
        if self.show_a:
            for i, bar in enumerate(self.bars_a):
                v = float(self.a_vals[i] * progress)
                bar.set_x(min(0.0, v))
                bar.set_width(abs(v))

                if self.value_texts_a:
                    t = self.value_texts_a[i]
                    if show_values:
                        t.set_text(self._format_value(v))
                        if v >= 0:
                            t.set_x(bar.get_x() + bar.get_width() + gap)
                            t.set_ha("left")
                        else:
                            t.set_x(bar.get_x() - gap)
                            t.set_ha("right")
                    else:
                        t.set_text("")

        # Update B bars + texts
        if self.show_b:
            for i, bar in enumerate(self.bars_b):
                v = float(self.b_vals[i] * progress)
                bar.set_x(min(0.0, v))
                bar.set_width(abs(v))

                if self.value_texts_b:
                    t = self.value_texts_b[i]
                    if show_values:
                        t.set_text(self._format_value(v))
                        if v >= 0:
                            t.set_x(bar.get_x() + bar.get_width() + gap)
                            t.set_ha("left")
                        else:
                            t.set_x(bar.get_x() - gap)
                            t.set_ha("right")
                    else:
                        t.set_text("")

    def export_mp4(self):
        was_running = self.timer.isActive()
        self.timer.stop()

        final_path, _ = QFileDialog.getSaveFileName(
            self, "Save animation as MP4",
            "diverging_barh.mp4",
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

            fig = Figure(figsize=(self.figw, self.figh))
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            # build the same plot offscreen
            def init_axes_off():
                ax.clear()
                ax.set_facecolor(self.bg)
                fig.patch.set_facecolor(self.bg)
                for side in ["right", "left", "top", "bottom"]:
                    ax.spines[side].set_visible(False)
                ax.tick_params(axis='x', which='major', length=0)
                ax.tick_params(axis='y', which='major', length=0)

            init_axes_off()

            n = len(self.categories)
            y = np.arange(n, dtype=float)

            fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.25)

            ax.set_xlim(self.xmin, self.xmax)
            ax.set_ylim(-0.5, n - 0.5)

            ax.xaxis.set_major_locator(MultipleLocator(self.xtick))
            ax.grid(True, axis='x', linestyle='-', color=self.grid_color, alpha=0.8, linewidth=0.2, zorder=1)

            y_grid_lines = np.arange(0.5, n - 0.5, 1)
            ax.set_yticks(y_grid_lines)
            ax.set_yticklabels([])
            ax.grid(True, axis='y', linestyle=(0, (5, 5)), color=self.grid_color, alpha=0.8, linewidth=0.2, zorder=1)

            ax.axvline(x=0, color=self.grid_color, linewidth=1.2, zorder=2)

            for tick in ax.get_xticklabels():
                tick.set_fontsize(8)
                tick.set_color(self.text_color)

            x_text = self.xmin + (abs(self.xmin) * 0.025)
            for yy, label in zip(y, self.categories):
                ax.text(x_text, yy, label, va='center', ha='left', fontsize=9,
                        color=self.text_color, zorder=4)

            bars_a = []
            bars_b = []
            if self.show_a:
                bars_a = list(ax.barh(y + 0.15, np.zeros(n), color=self.color_a, height=0.3, zorder=3, label=self.label_a))
            if self.show_b:
                bars_b = list(ax.barh(y - 0.15, np.zeros(n), color=self.color_b, height=0.3, zorder=3, label=self.label_b))

            texts_a = []
            texts_b = []
            if self.show_a:
                texts_a = [ax.text(0, yy + 0.13, "", va='center', ha='left', fontsize=8, zorder=4, color="#707070") for yy in y]
            if self.show_b:
                texts_b = [ax.text(0, yy - 0.18, "", va='center', ha='left', fontsize=6, zorder=4, color="#707070") for yy in y]

            handles, labs = ax.get_legend_handles_labels()
            if handles:
                ax.legend(
                    loc='upper center',
                    bbox_to_anchor=(0.5, -0.25),
                    ncol=len(handles),
                    frameon=False,
                    fontsize=10,
                    handlelength=1.8,
                    columnspacing=2.0
                )

            writer = FFMpegWriter(
                fps=FPS,
                codec="libx264",
                bitrate=2500,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
            )

            gap = 0.2
            show_plus = self.show_plus_cb.isChecked()
            show_values = self.show_values_cb.isChecked()

            def fmt(v: float) -> str:
                if abs(v) <= 0.05:
                    return ""
                return f"{v:+.1f}" if show_plus else f"{v:.1f}"

            with writer.saving(fig, tmp_path, dpi=300):
                for frame in range(self.total_frames):
                    if frame >= self.anim_frames:
                        progress = 1.0
                    else:
                        progress = frame / max(1, (self.anim_frames - 1))

                    if self.show_a:
                        for i, bar in enumerate(bars_a):
                            v = float(self.a_vals[i] * progress)
                            bar.set_x(min(0.0, v))
                            bar.set_width(abs(v))

                            t = texts_a[i]
                            if show_values:
                                t.set_text(fmt(v))
                                if v >= 0:
                                    t.set_x(bar.get_x() + bar.get_width() + gap)
                                    t.set_ha("left")
                                else:
                                    t.set_x(bar.get_x() - gap)
                                    t.set_ha("right")
                            else:
                                t.set_text("")

                    if self.show_b:
                        for i, bar in enumerate(bars_b):
                            v = float(self.b_vals[i] * progress)
                            bar.set_x(min(0.0, v))
                            bar.set_width(abs(v))

                            t = texts_b[i]
                            if show_values:
                                t.set_text(fmt(v))
                                if v >= 0:
                                    t.set_x(bar.get_x() + bar.get_width() + gap)
                                    t.set_ha("left")
                                else:
                                    t.set_x(bar.get_x() - gap)
                                    t.set_ha("right")
                            else:
                                t.set_text("")

                    writer.grab_frame()

            if os.path.getsize(tmp_path) < 50_000:
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
            if was_running:
                self.timer.start(int(1000 / FPS))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
