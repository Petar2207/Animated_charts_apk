import sys
import os
import tempfile
import uuid
import traceback
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QPlainTextEdit, QFileDialog,
    QSizePolicy, QScrollArea, QCheckBox
)
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QDoubleValidator

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
from matplotlib.ticker import MultipleLocator, FuncFormatter


# ===================== SETTINGS =====================
FPS = 30
PREVIEW_DPI = 95.0
AUTO_APPLY_DEBOUNCE_MS = 350

DEFAULT_FIGW = "12"
DEFAULT_FIGH = "5"
DEFAULT_ANIM_SEC = "15"
DEFAULT_PAUSE_SEC = "5"

DEFAULT_XMAX = "108"
DEFAULT_XTICK = "10"

DEFAULT_MIN_TEXT_PCT = "3.0"  # show text only if current visible segment >= this percent

DEFAULT_CATEGORIES = """Temperatur
Geräuschkulisse
Lichtverhältnisse
Sitzplatz
Technische Arbeitsmittel
Hygiene
"""

# Data rows, one CSV line per category, same number of columns each row
DEFAULT_DATA_ROWS = """11.8,47.1,11.8,23.5,5.9
5.9,11.8,29.4,29.4,23.5
35.3,41.2,17.6,5.9,0.0
23.5,52.9,17.6,0.0,5.9
64.7,23.5,11.8,0.0,0.0
17.6,64.7,11.8,5.9,0.0
"""

DEFAULT_SEG_LABELS = """sehr gut
gut
mittelmäßig
schlecht
sehr schlecht
"""

DEFAULT_SEG_COLORS = """#00b050
#92d050
#ffc000
#ff0000
#c00000
"""
# ====================================================


def parse_positive_float(text: str, name: str) -> float:
    try:
        v = float(text.strip())
    except Exception:
        raise ValueError(f"{name} must be a number.")
    if v <= 0:
        raise ValueError(f"{name} must be > 0.")
    return v


def parse_nonneg_float(text: str, name: str) -> float:
    try:
        v = float(text.strip())
    except Exception:
        raise ValueError(f"{name} must be a number.")
    if v < 0:
        raise ValueError(f"{name} must be >= 0.")
    return v


def parse_categories(text: str) -> list[str]:
    cats = [c.strip() for c in text.splitlines() if c.strip()]
    if not cats:
        raise ValueError("Provide at least 1 category.")
    return cats


def parse_lines(text: str) -> list[str]:
    return [c.strip() for c in text.splitlines() if c.strip()]


def validate_hex_color(s: str, name: str) -> str:
    s = s.strip()
    if not s.startswith("#") or len(s) != 7:
        raise ValueError(f"{name} must be hex like #00b050")
    int(s[1:], 16)
    return s


def parse_data_rows(text: str) -> np.ndarray:
    """
    Expects one row per line, each line is CSV floats.
    Returns shape (n_rows, n_cols)
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Provide at least 1 data row.")
    rows = []
    for i, ln in enumerate(lines, start=1):
        parts = [p.strip() for p in ln.split(",") if p.strip() != ""]
        if not parts:
            raise ValueError(f"Row {i} is empty.")
        try:
            rows.append([float(p) for p in parts])
        except Exception:
            raise ValueError(f"Row {i} contains non-numeric values.")
    ncols = len(rows[0])
    for i, r in enumerate(rows, start=1):
        if len(r) != ncols:
            raise ValueError(f"Row {i} has {len(r)} columns but expected {ncols}.")
    return np.array(rows, dtype=float)


class AspectRatioWidget(QWidget):
    """Keeps child centered and maintains a fixed aspect ratio (width/height)."""

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
        self.setWindowTitle("Animated Stacked Barh (Stable Export)")

        # -------- inputs --------
        self.figw_in = QLineEdit(DEFAULT_FIGW)
        self.figh_in = QLineEdit(DEFAULT_FIGH)
        self.anim_sec_in = QLineEdit(DEFAULT_ANIM_SEC)
        self.pause_sec_in = QLineEdit(DEFAULT_PAUSE_SEC)

        self.xmax_in = QLineEdit(DEFAULT_XMAX)
        self.xtick_in = QLineEdit(DEFAULT_XTICK)

        self.cats_in = QPlainTextEdit(DEFAULT_CATEGORIES)
        self.cats_in.setFixedHeight(130)

        self.data_in = QPlainTextEdit(DEFAULT_DATA_ROWS)
        self.data_in.setFixedHeight(140)

        self.seg_labels_in = QPlainTextEdit(DEFAULT_SEG_LABELS)
        self.seg_labels_in.setFixedHeight(110)

        self.seg_colors_in = QPlainTextEdit(DEFAULT_SEG_COLORS)
        self.seg_colors_in.setFixedHeight(110)

        self.show_percent_cb = QCheckBox("Show % sign")
        self.show_percent_cb.setChecked(True)

        self.show_values_cb = QCheckBox("Show segment values")
        self.show_values_cb.setChecked(True)

        self.min_text_in = QLineEdit(DEFAULT_MIN_TEXT_PCT)

        # allow reasonable range only (inches)
        self.figw_in.setValidator(QDoubleValidator(1.0, 50.0, 2))
        self.figh_in.setValidator(QDoubleValidator(1.0, 50.0, 2))

        # -------- buttons --------
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.export_btn = QPushButton("Export MP4")
        self.plot_full_btn = QPushButton("Plot Fullscreen")

        self.start_btn.clicked.connect(self.start_animation)
        self.stop_btn.clicked.connect(self.stop_animation)
        self.export_btn.clicked.connect(self.export_mp4)
        self.plot_full_btn.clicked.connect(self.open_plot_fullscreen)

        self.status = QLabel("")

        # -------- plot --------
        self.canvas = MplCanvas()
        self.ax = self.canvas.ax
        self.aspect_wrap = AspectRatioWidget(
            self.canvas, aspect=float(DEFAULT_FIGW) / float(DEFAULT_FIGH)
        )

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

        # -------- controls layout --------
        controls = QWidget()
        cl = QVBoxLayout(controls)

        r1 = QWidget()
        l1 = QHBoxLayout(r1)
        l1.addWidget(QLabel("Fig W"))
        l1.addWidget(self.figw_in)
        l1.addWidget(QLabel("Fig H"))
        l1.addWidget(self.figh_in)
        cl.addWidget(r1)

        r2 = QWidget()
        l2 = QHBoxLayout(r2)
        l2.addWidget(QLabel("Anim sec"))
        l2.addWidget(self.anim_sec_in)
        l2.addWidget(QLabel("Pause sec"))
        l2.addWidget(self.pause_sec_in)
        cl.addWidget(r2)

        r3 = QWidget()
        l3 = QHBoxLayout(r3)
        l3.addWidget(QLabel("X max"))
        l3.addWidget(self.xmax_in)
        l3.addWidget(QLabel("X tick"))
        l3.addWidget(self.xtick_in)
        cl.addWidget(r3)

        cl.addWidget(QLabel("Categories (one per line):"))
        cl.addWidget(self.cats_in)

        cl.addWidget(QLabel("Data rows (one row per category, CSV per line):"))
        cl.addWidget(self.data_in)

        cl.addWidget(QLabel("Segment labels (one per line):"))
        cl.addWidget(self.seg_labels_in)

        cl.addWidget(QLabel("Segment colors (one per line, hex):"))
        cl.addWidget(self.seg_colors_in)

        cl.addWidget(self.show_percent_cb)
        cl.addWidget(self.show_values_cb)

        r4 = QWidget()
        l4 = QHBoxLayout(r4)
        l4.addWidget(QLabel("Min segment width to show text (%)"))
        l4.addWidget(self.min_text_in)
        cl.addWidget(r4)

        btnrow = QWidget()
        bl = QHBoxLayout(btnrow)
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

        # -------- animation --------
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

        # triggers
        for w in [
            self.figw_in,
            self.figh_in,
            self.anim_sec_in,
            self.pause_sec_in,
            self.xmax_in,
            self.xtick_in,
            self.min_text_in,
        ]:
            w.textChanged.connect(schedule_heavy_rebuild)
        self.cats_in.textChanged.connect(schedule_heavy_rebuild)
        self.data_in.textChanged.connect(schedule_heavy_rebuild)
        self.seg_labels_in.textChanged.connect(schedule_heavy_rebuild)
        self.seg_colors_in.textChanged.connect(schedule_heavy_rebuild)
        self.show_percent_cb.toggled.connect(schedule_heavy_rebuild)
        self.show_values_cb.toggled.connect(lambda _: self.canvas.draw_idle())

        # fullscreen
        self._plot_full_win = None
        self._plot_index = -1

        # state
        self.categories = []
        self.data = None
        self.colors = []
        self.seg_labels = []

        self.proportions = None  # data/100
        self.n_rows = 0
        self.n_cols = 0
        self.y = None

        self.figw = float(DEFAULT_FIGW)
        self.figh = float(DEFAULT_FIGH)
        self.xmax = float(DEFAULT_XMAX)
        self.xtick = float(DEFAULT_XTICK)

        self.anim_frames = 2
        self.pause_frames = 0
        self.total_frames = 2

        self.min_text_pct = float(DEFAULT_MIN_TEXT_PCT)

        # artists
        self.bars = []   # list of BarContainer (one per segment)
        self.texts = []  # texts[col][row]

        self.apply_all()
        self.start_animation()
        self.resize(1300, 900)

    def update_plot_widget_size(self):
        w_px = int(self.figw * PREVIEW_DPI)
        h_px = int(self.figh * PREVIEW_DPI)
        w_px = max(w_px, 700)
        h_px = max(h_px, 350)

        # actual plot widget size (can be wider than viewport)
        self.aspect_wrap.setFixedSize(w_px, h_px)
        self.aspect_wrap.updateGeometry()

        # keep centering working + allow horizontal scroll when needed
        self.plot_inner.setMinimumWidth(w_px)
        self.plot_inner.setMinimumHeight(h_px)

        # prevent "ultra short": make the plot area show full height
        sb_h = self.plot_hscroll.horizontalScrollBar().sizeHint().height()
        self.plot_hscroll.setFixedHeight(h_px + sb_h + 4)

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
        ax.set_facecolor("white")
        ax.figure.patch.set_facecolor("white")

        for spine in ["right", "bottom", "left", "top"]:
            ax.spines[spine].set_visible(False)

        ax.grid(True, axis="x", linestyle="--", alpha=0.3)
        ax.tick_params(axis="x", which="major", length=0, labelsize=8)
        ax.tick_params(axis="y", which="major", length=0, labelsize=8)

    def _apply_x_format(self, ax):
        ax.set_xlim(0, self.xmax)
        ax.xaxis.tick_bottom()
        ax.xaxis.set_major_locator(MultipleLocator(self.xtick))
        if self.show_percent_cb.isChecked():
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))
        else:
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

    def _build_artists(self):
        ax = self.ax
        self._init_axes(ax)

        self.y = np.arange(self.n_rows, dtype=float)

        ax.set_ylim(-0.5, self.n_rows - 0.5)
        ax.set_yticks(range(self.n_rows))
        ax.set_yticklabels(self.categories, fontsize=15)
        ax.invert_yaxis()

        self._apply_x_format(ax)

        self.bars = []
        for j in range(self.n_cols):
            bc = ax.barh(
                range(self.n_rows),
                np.zeros(self.n_rows),
                color=self.colors[j],
                height=0.6,
                left=np.zeros(self.n_rows),
            )
            self.bars.append(bc)

        self.texts = [
            [
                ax.text(
                    0, i, "",
                    color="white", va="center", ha="center",
                    fontsize=8, weight="bold", alpha=1
                )
                for i in range(self.n_rows)
            ]
            for _ in range(self.n_cols)
        ]

        ax.legend(
            self.seg_labels,
            loc="lower center",
            ncol=min(self.n_cols, 8),
            bbox_to_anchor=(0.5, -0.15),
            fontsize=8,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
        )

        self.canvas.figure.subplots_adjust(left=0.27, right=0.97, top=0.93, bottom=0.18)

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

            # axis
            self.xmax = parse_positive_float(self.xmax_in.text(), "X max")
            self.xtick = parse_positive_float(self.xtick_in.text(), "X tick")
            self.min_text_pct = parse_nonneg_float(self.min_text_in.text(), "Min text width (%)")

            # inputs
            self.categories = parse_categories(self.cats_in.toPlainText())
            self.data = parse_data_rows(self.data_in.toPlainText())
            self.seg_labels = parse_lines(self.seg_labels_in.toPlainText())
            self.colors = parse_lines(self.seg_colors_in.toPlainText())

            # validate sizes
            self.n_rows = len(self.categories)
            if self.data.shape[0] != self.n_rows:
                raise ValueError(f"Data rows ({self.data.shape[0]}) must match categories ({self.n_rows}).")

            self.n_cols = self.data.shape[1]
            if len(self.seg_labels) != self.n_cols:
                raise ValueError(f"Segment labels ({len(self.seg_labels)}) must match data columns ({self.n_cols}).")
            if len(self.colors) != self.n_cols:
                raise ValueError(f"Segment colors ({len(self.colors)}) must match data columns ({self.n_cols}).")

            self.colors = [validate_hex_color(c, f"Color {i+1}") for i, c in enumerate(self.colors)]

            # proportions for timing
            self.proportions = self.data / 100.0

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

    def update_frame(self, frame: int):
        if frame >= self.anim_frames:
            progress = 1.0
        else:
            progress = frame / max(1, self.anim_frames)

        show_values = self.show_values_cb.isChecked()
        add_percent = self.show_percent_cb.isChecked()
        suffix = "%" if add_percent else ""

        cum_left = np.zeros(self.n_rows, dtype=float)

        for i in range(self.n_rows):
            bar_progress = progress
            cum = 0.0
            for j in range(self.n_cols):
                seg_len = float(self.proportions[i, j])
                start = cum
                end = cum + seg_len
                cum = end

                if seg_len == 0:
                    width = 0.0
                elif bar_progress <= start:
                    width = 0.0
                elif bar_progress >= end:
                    width = seg_len
                else:
                    width = (bar_progress - start)

                rect = self.bars[j][i]
                rect.set_x(cum_left[i])
                rect.set_width(width * 100.0)

                txt = self.texts[j][i]
                visible_pct = width * 100.0
                if show_values and visible_pct >= self.min_text_pct:
                    frac = (width / seg_len) if seg_len > 0 else 0.0
                    val = float(self.data[i, j] * frac)
                    x_mid = rect.get_x() + (visible_pct / 2.0)
                    txt.set_position((x_mid, i))
                    txt.set_text(f"{val:.1f}{suffix}")
                    txt.set_alpha(1.0)
                else:
                    txt.set_alpha(0.0)

                cum_left[i] += visible_pct

    def export_mp4(self):
        was_running = self.timer.isActive()
        self.timer.stop()

        final_path, _ = QFileDialog.getSaveFileName(
            self, "Save animation as MP4",
            "stacked_barh.mp4",
            "MP4 Video (*.mp4)",
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

            ax.set_facecolor("white")
            fig.patch.set_facecolor("white")
            for spine in ["right", "bottom", "left", "top"]:
                ax.spines[spine].set_visible(False)
            ax.grid(True, axis="x", linestyle="--", alpha=0.3)
            ax.tick_params(axis="x", which="major", length=0, labelsize=8)
            ax.tick_params(axis="y", which="major", length=0, labelsize=8)

            ax.set_ylim(-0.5, self.n_rows - 0.5)
            ax.set_yticks(range(self.n_rows))
            ax.set_yticklabels(self.categories, fontsize=15)
            ax.invert_yaxis()

            ax.set_xlim(0, self.xmax)
            ax.xaxis.set_major_locator(MultipleLocator(self.xtick))
            add_percent = self.show_percent_cb.isChecked()
            if add_percent:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}%"))
            else:
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))

            bars = []
            for j in range(self.n_cols):
                bc = ax.barh(
                    range(self.n_rows),
                    np.zeros(self.n_rows),
                    color=self.colors[j],
                    height=0.6,
                    left=np.zeros(self.n_rows),
                )
                bars.append(bc)

            texts = [
                [
                    ax.text(0, i, "", color="white", va="center", ha="center", fontsize=8, weight="bold", alpha=1)
                    for i in range(self.n_rows)
                ]
                for _ in range(self.n_cols)
            ]

            ax.legend(
                self.seg_labels,
                loc="lower center",
                ncol=min(self.n_cols, 8),
                bbox_to_anchor=(0.5, -0.15),
                fontsize=8,
                frameon=False,
                handletextpad=0.4,
                columnspacing=1.0,
            )

            fig.subplots_adjust(left=0.27, right=0.97, top=0.93, bottom=0.18)

            writer = FFMpegWriter(
                fps=FPS,
                codec="libx264",
                bitrate=2500,
                extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart",
                            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
            )

            show_values = self.show_values_cb.isChecked()
            suffix = "%" if add_percent else ""

            with writer.saving(fig, tmp_path, dpi=300):
                for frame in range(self.total_frames):
                    if frame >= self.anim_frames:
                        progress = 1.0
                    else:
                        progress = frame / max(1, self.anim_frames)

                    cum_left = np.zeros(self.n_rows, dtype=float)

                    for i in range(self.n_rows):
                        bar_progress = progress
                        cum = 0.0
                        for j in range(self.n_cols):
                            seg_len = float(self.proportions[i, j])
                            start = cum
                            end = cum + seg_len
                            cum = end

                            if seg_len == 0:
                                width = 0.0
                            elif bar_progress <= start:
                                width = 0.0
                            elif bar_progress >= end:
                                width = seg_len
                            else:
                                width = (bar_progress - start)

                            rect = bars[j][i]
                            rect.set_x(cum_left[i])
                            rect.set_width(width * 100.0)

                            txt = texts[j][i]
                            visible_pct = width * 100.0
                            if show_values and visible_pct >= self.min_text_pct:
                                frac = (width / seg_len) if seg_len > 0 else 0.0
                                val = float(self.data[i, j] * frac)
                                x_mid = rect.get_x() + (visible_pct / 2.0)
                                txt.set_position((x_mid, i))
                                txt.set_text(f"{val:.1f}{suffix}")
                                txt.set_alpha(1.0)
                            else:
                                txt.set_alpha(0.0)

                            cum_left[i] += visible_pct

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
            if "ffmpeg" in msg.lower() or "filenotfounderror" in msg.lower():
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

