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
from matplotlib.ticker import FuncFormatter, MultipleLocator
from PySide6.QtGui import QDoubleValidator  

# ===================== SETTINGS =====================
FPS = 30
PREVIEW_DPI = 95.0
AUTO_APPLY_DEBOUNCE_MS = 350

DEFAULT_FIGW = "13"
DEFAULT_FIGH = "6"
DEFAULT_ANIM_SEC = "10"
DEFAULT_PAUSE_SEC = "3"

DEFAULT_CATEGORIES = """Führung
Personalentwicklung
Mitarbeiterfokus
Effizienz
Offenheit
Kundenzentrierung
Wertezentrierung
Digitalisierung
Innovation
Stabilität
Kundenerlebnis
Sonstiges
Mitglieder
Kooperation
Wachstum
Vision
Nachhaltigkeit
"""

DEFAULT_BARS = "52.9,41.2,35.3,23.5,23.5,11.8,11.8,11.8,11.8,11.8,5.9,5.9,5.9,5.9,5.9,5.9,0.0"
DEFAULT_LINE2 = "14.0,32.9,46.3,37.5,24.5,17.6,7.8,27.7,12.8,11.4,22.9,3.4,5.1,8.6,4.9,6.4,8.2"
DEFAULT_LINE1 = "17.4,33.9,39.7,37.1,27.8,12.7,9.3,28.3,13.9,16.1,13.7,4.1,3.5,8.6,6.0,7.8,10.6"

DEFAULT_BAR_COLOR = "#7F3C57"
DEFAULT_LINE_COLOR = "gray"

DEFAULT_YMAX = "0"       # 0 = auto
DEFAULT_YTICK = "10"     # tick step
# ====================================================


def parse_csv_floats(text: str) -> np.ndarray:
    parts = [p.strip() for p in text.split(",") if p.strip() != ""]
    return np.array([float(p) for p in parts], dtype=float)


def parse_positive_float_allow_zero(text: str, name: str) -> float:
    v = float(text.strip())
    if v < 0:
        raise ValueError(f"{name} must be >= 0.")
    return v


def parse_positive_float(text: str, name: str) -> float:
    v = float(text.strip())
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")
    return v


def parse_categories(text: str) -> list[str]:
    cats = [c.strip() for c in text.splitlines() if c.strip()]
    if not cats:
        raise ValueError("Provide at least 1 category.")
    return cats


def validate_hex_color(s: str) -> str:
    s = s.strip()
    if not s.startswith("#") or len(s) != 7:
        raise ValueError("Bars color must be hex like #7F3C57")
    # validate hex digits
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
        fig = Figure(figsize=(13, 6))
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
        self.setWindowTitle("Animated Bars + Optional Lines (Stable Export)")

        # inputs
        self.figw_in = QLineEdit(DEFAULT_FIGW)
        self.figh_in = QLineEdit(DEFAULT_FIGH)
        self.anim_sec_in = QLineEdit(DEFAULT_ANIM_SEC)
        self.pause_sec_in = QLineEdit(DEFAULT_PAUSE_SEC)
        # allow reasonable range only (inches)
        self.figw_in.setValidator(QDoubleValidator(1.0, 50.0, 2))
        self.figh_in.setValidator(QDoubleValidator(1.0, 50.0, 2))

        self.cats_in = QPlainTextEdit(DEFAULT_CATEGORIES)
        self.cats_in.setFixedHeight(140)

        # values
        self.bars_in = QLineEdit(DEFAULT_BARS)   # bars values
        self.line2_in = QLineEdit(DEFAULT_LINE2) # line 2 values
        self.line1_in = QLineEdit(DEFAULT_LINE1) # line 1 values

        # style inputs
        self.bar_color_in = QLineEdit(DEFAULT_BAR_COLOR)
        self.ymax_in = QLineEdit(DEFAULT_YMAX)      # 0 = auto
        self.ytick_in = QLineEdit(DEFAULT_YTICK)    # step

        # toggles
        self.fade_labels_cb = QCheckBox("Fade-in bar labels")
        self.fade_labels_cb.setChecked(True)

        self.show_line1_cb = QCheckBox("Show Line 1")
        self.show_line1_cb.setChecked(True)

        self.show_line2_cb = QCheckBox("Show Line 2")
        self.show_line2_cb.setChecked(True)

        self.show_percent_cb = QCheckBox("Show %")
        self.show_percent_cb.setChecked(True)

        # legend label inputs
        self.label_bars_in = QLineEdit("Bars")
        self.label_line1_in = QLineEdit("Line 1")
        self.label_line2_in = QLineEdit("Line 2")

        # initialize state defaults
        self.label_bars = self.label_bars_in.text().strip() or "Bars"
        self.label_line1 = self.label_line1_in.text().strip() or "Line 1"
        self.label_line2 = self.label_line2_in.text().strip() or "Line 2"
        self.show_line1 = True
        self.show_line2 = True
        self.show_percent = True
        self.bar_color = DEFAULT_BAR_COLOR
        self.ymax = 0.0
        self.ytick = 10.0

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

        cl.addWidget(QLabel("Bars values (CSV):"))
        cl.addWidget(self.bars_in)

        cl.addWidget(QLabel("Line 1 values (CSV):"))
        cl.addWidget(self.line1_in)

        cl.addWidget(QLabel("Line 2 values (CSV):"))
        cl.addWidget(self.line2_in)

        cl.addWidget(QLabel("Bars color (hex like #7F3C57):"))
        cl.addWidget(self.bar_color_in)

        r_y = QWidget(); ly = QHBoxLayout(r_y)
        ly.addWidget(QLabel("Y max (0=auto)"))
        ly.addWidget(self.ymax_in)
        ly.addWidget(QLabel("Y tick step"))
        ly.addWidget(self.ytick_in)
        cl.addWidget(r_y)

        cl.addWidget(self.fade_labels_cb)
        cl.addWidget(self.show_line1_cb)
        cl.addWidget(self.show_line2_cb)
        cl.addWidget(self.show_percent_cb)

        cl.addWidget(QLabel("Legend labels:"))
        r_labels = QWidget()
        l_labels = QHBoxLayout(r_labels)
        l_labels.addWidget(QLabel("Bars"))
        l_labels.addWidget(self.label_bars_in)
        l_labels.addWidget(QLabel("Line 1"))
        l_labels.addWidget(self.label_line1_in)
        l_labels.addWidget(QLabel("Line 2"))
        l_labels.addWidget(self.label_line2_in)
        cl.addWidget(r_labels)

        btnrow = QWidget(); bl = QHBoxLayout(btnrow)
        bl.addWidget(self.start_btn)
        bl.addWidget(self.stop_btn)
        bl.addWidget(self.export_btn)
        bl.addWidget(self.plot_full_btn)
        cl.addWidget(btnrow)

        cl.addWidget(self.status)

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

        def schedule_heavy_rebuild():
            self.status.setText("⏳ Pending changes...")
            self.status.setStyleSheet("color: #444;")
            self._apply_debounce.start(AUTO_APPLY_DEBOUNCE_MS)

        # rebuild triggers
        text_inputs = [
            self.figw_in, self.figh_in, self.anim_sec_in, self.pause_sec_in,
            self.bars_in, self.line1_in, self.line2_in,
            self.bar_color_in, self.ymax_in, self.ytick_in,
            self.label_bars_in, self.label_line1_in, self.label_line2_in
        ]
        for w in text_inputs:
            w.textChanged.connect(schedule_heavy_rebuild)

        self.cats_in.textChanged.connect(schedule_heavy_rebuild)

        self.fade_labels_cb.toggled.connect(lambda _: self.canvas.draw_idle())
        self.show_line1_cb.toggled.connect(schedule_heavy_rebuild)
        self.show_line2_cb.toggled.connect(schedule_heavy_rebuild)
        self.show_percent_cb.toggled.connect(schedule_heavy_rebuild)

        # fullscreen
        self._plot_full_win = None
        self._placeholder = None

        # artists
        self._bars = None
        self._labels = []
        self._line1 = None
        self._line2 = None

        # data
        self.categories = []
        self.x = None
        self.bars_vals = None
        self.line1_vals = None
        self.line2_vals = None

        self.total_frames = 2
        self.pause_frames = 0
        self.total_frames_with_pause = 2

        self.apply_all()
        self.start_animation()
        self.resize(1200, 800)

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

    def _init_axes(self):
        ax = self.ax
        ax.clear()
        ax.set_facecolor("#FFFFFF")

        ax.grid(axis="y", linestyle="-", alpha=0.4)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=0)

        self.canvas.figure.patch.set_facecolor("#FFFFFF")

        self._bars = None
        self._labels = []
        self._line1 = None
        self._line2 = None

    def _apply_y_format(self, ax):
        # ticks spacing
        ax.yaxis.set_major_locator(MultipleLocator(self.ytick))

        # % on/off
        if self.show_percent:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}%"))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}"))

    def _compute_ymax_auto(self) -> float:
        mx = float(np.max([self.bars_vals.max(),
                           self.line1_vals.max(),
                           self.line2_vals.max()]))
        return max(5.0, mx * 1.15)

    def _build_artists(self):
        self._init_axes()
        ax = self.ax

        ymax = self.ymax if self.ymax > 0 else self._compute_ymax_auto()
        ax.set_ylim(0, ymax)

        ax.set_xticks(self.x)
        ax.set_xticklabels(self.categories, rotation=50, ha="right", fontsize=9)

        self._apply_y_format(ax)

        # optional lines
        self._line1 = None
        self._line2 = None

        if self.show_line1:
            (self._line1,) = ax.plot(
                self.x, self.line1_vals,
                linestyle="-", linewidth=1, color=DEFAULT_LINE_COLOR,
                marker="o", markersize=0,
                label=self.label_line1, alpha=0.9
            )

        if self.show_line2:
            (self._line2,) = ax.plot(
                self.x, self.line2_vals,
                linestyle="--", linewidth=1, color=DEFAULT_LINE_COLOR,
                marker="o", markersize=0,
                label=self.label_line2, alpha=0.9
            )

        # bars
        self._bars = ax.bar(
            self.x, np.zeros_like(self.x),
            color=self.bar_color, width=0.65,
            label=self.label_bars
        )

        # bar value labels
        self._labels = []
        suffix = "%" if self.show_percent else ""
        for i in range(len(self.categories)):
            t = ax.text(self.x[i], 0.2, f"0.0{suffix}", ha="center", va="bottom", fontsize=8, color="white")
            self._labels.append(t)

        # legend: only visible things
        handles, labs = ax.get_legend_handles_labels()
        by_label = {lab: h for h, lab in zip(handles, labs)}

        order = [self.label_bars]
        if self.show_line1:
            order.append(self.label_line1)
        if self.show_line2:
            order.append(self.label_line2)

        ax.legend([by_label[l] for l in order], order,
                  loc="upper center", bbox_to_anchor=(0.5, -0.30),
                  ncol=len(order), frameon=False, fontsize=10)

        self.canvas.figure.subplots_adjust(top=0.95, bottom=0.30, left=0.08, right=0.97)

    def apply_all(self):
        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        try:
            # figure sizing
            self.figw = parse_positive_float(self.figw_in.text(), "Figure width")
            self.figh = parse_positive_float(self.figh_in.text(), "Figure height")
            self.aspect_wrap.set_aspect(self.figw / self.figh)
            self.update_plot_widget_size()

            # animation timing
            anim_sec = parse_positive_float(self.anim_sec_in.text(), "Animation seconds")
            pause_sec = parse_positive_float(self.pause_sec_in.text(), "Pause seconds")
            self.total_frames = max(2, int(FPS * anim_sec))
            self.pause_frames = int(FPS * pause_sec)
            self.total_frames_with_pause = self.total_frames + self.pause_frames

            # labels
            self.label_bars = self.label_bars_in.text().strip() or "Bars"
            self.label_line1 = self.label_line1_in.text().strip() or "Line 1"
            self.label_line2 = self.label_line2_in.text().strip() or "Line 2"

            # toggles
            self.show_line1 = self.show_line1_cb.isChecked()
            self.show_line2 = self.show_line2_cb.isChecked()
            self.show_percent = self.show_percent_cb.isChecked()

            # style inputs
            self.bar_color = validate_hex_color(self.bar_color_in.text())
            self.ymax = parse_positive_float_allow_zero(self.ymax_in.text(), "Y max")
            self.ytick = parse_positive_float(self.ytick_in.text(), "Y tick step")

            # data
            self.categories = parse_categories(self.cats_in.toPlainText())
            self.bars_vals = parse_csv_floats(self.bars_in.text())
            self.line1_vals = parse_csv_floats(self.line1_in.text())
            self.line2_vals = parse_csv_floats(self.line2_in.text())

            n = len(self.categories)
            if len(self.bars_vals) != n or len(self.line1_vals) != n or len(self.line2_vals) != n:
                raise ValueError(
                    f"Lengths must match categories ({n}). "
                    f"Bars={len(self.bars_vals)}, Line1={len(self.line1_vals)}, Line2={len(self.line2_vals)}"
                )

            self.x = np.arange(n, dtype=float)

            # rebuild
            self.frame = 0
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

    def start_animation(self):
        self.timer.start(int(1000 / FPS))

    def stop_animation(self):
        self.timer.stop()

    def step(self):
        self.frame = (self.frame + 1) % self.total_frames_with_pause
        self.update_frame(self.frame)
        self.canvas.draw_idle()

    def update_frame(self, frame: int):
        if frame >= self.total_frames:
            progress = 1.0
        else:
            progress = frame / max(1, (self.total_frames - 1))

        fade = self.fade_labels_cb.isChecked()
        alpha = min(1.0, progress * 6.0) if fade else 1.0

        suffix = "%" if self.show_percent else ""

        for j, rect in enumerate(self._bars):
            h = float(self.bars_vals[j] * progress)
            rect.set_height(h)

            t = self._labels[j]
            t.set_text(f"{h:.1f}{suffix}")
            t.set_position((self.x[j], max(0.2, h * 0.10)))
            t.set_alpha(alpha)

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

    def export_mp4(self):
        was_running = self.timer.isActive()
        self.timer.stop()

        final_path, _ = QFileDialog.getSaveFileName(
            self, "Save animation as MP4",
            "bars_lines_anim.mp4",
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

            # style
            ax.set_facecolor("#FFFFFF")
            fig.patch.set_facecolor("#FFFFFF")
            ax.grid(axis="y", linestyle="-", alpha=0.4)
            ax.spines["left"].set_visible(False)
            ax.spines["bottom"].set_visible(True)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", length=0)

            ymax = self.ymax if self.ymax > 0 else self._compute_ymax_auto()
            ax.set_ylim(0, ymax)

            ax.set_xticks(self.x)
            ax.set_xticklabels(self.categories, rotation=50, ha="right", fontsize=9)

            ax.yaxis.set_major_locator(MultipleLocator(self.ytick))
            if self.show_percent:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}%"))
            else:
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}"))

            # optional lines
            if self.show_line1:
                ax.plot(self.x, self.line1_vals, linestyle="-", linewidth=1, color=DEFAULT_LINE_COLOR,
                        marker="o", markersize=0, label=self.label_line1, alpha=0.9)

            if self.show_line2:
                ax.plot(self.x, self.line2_vals, linestyle="--", linewidth=1, color=DEFAULT_LINE_COLOR,
                        marker="o", markersize=0, label=self.label_line2, alpha=0.9)

            bars = ax.bar(self.x, np.zeros_like(self.x), color=self.bar_color, width=0.65, label=self.label_bars)

            suffix = "%" if self.show_percent else ""
            labels = [
                ax.text(self.x[i], 0.2, f"0.0{suffix}", ha="center", va="bottom", fontsize=8, color="white")
                for i in range(len(self.categories))
            ]

            handles, labs = ax.get_legend_handles_labels()
            by_label = {lab: h for h, lab in zip(handles, labs)}
            order = [self.label_bars]
            if self.show_line1:
                order.append(self.label_line1)
            if self.show_line2:
                order.append(self.label_line2)

            ax.legend([by_label[l] for l in order], order,
                      loc="upper center", bbox_to_anchor=(0.5, -0.30),
                      ncol=len(order), frameon=False, fontsize=10)

            fig.subplots_adjust(top=0.95, bottom=0.30, left=0.08, right=0.97)

            writer = FFMpegWriter(
                fps=FPS,
                codec="libx264",
                bitrate=2500,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"]
            )

            fade = self.fade_labels_cb.isChecked()

            with writer.saving(fig, tmp_path, dpi=300):
                for frame in range(self.total_frames_with_pause):
                    if frame >= self.total_frames:
                        progress = 1.0
                    else:
                        progress = frame / max(1, (self.total_frames - 1))

                    alpha = min(1.0, progress * 6.0) if fade else 1.0

                    for j, rect in enumerate(bars):
                        h = float(self.bars_vals[j] * progress)
                        rect.set_height(h)
                        labels[j].set_text(f"{h:.1f}{suffix}")
                        labels[j].set_position((self.x[j], max(0.2, h * 0.10)))
                        labels[j].set_alpha(alpha)

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
