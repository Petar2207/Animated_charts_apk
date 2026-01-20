import sys
import os
import traceback
import tempfile
import uuid
import numpy as np
from matplotlib.ticker import FuncFormatter

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QLabel, QPlainTextEdit, QFileDialog,
    QSizePolicy, QScrollArea, QSplitter, QComboBox, QCheckBox
)
from PySide6.QtCore import QTimer, Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FFMpegWriter
from PySide6.QtGui import QDoubleValidator


FPS = 30
DEFAULT_ANIMATION_DURATION = 10
DEFAULT_PAUSE_DURATION = 5

# Preview sizing: pixels = inches * PREVIEW_DPI
PREVIEW_DPI = 100.0

DEFAULT_COLOR_CYCLE = [
    "#548B54", "#4397A2", "#FF780C", "#00B0F0",
    "#8E44AD", "#C0392B", "#2C3E50", "#16A085"
]

DEFAULT_LINE_WIDTH = 3.0
DEFAULT_MARKER_SIZE = 17.0
DEFAULT_MARKER_EDGE_WIDTH = 6.0
DEFAULT_MARKER = "o"
DEFAULT_LINESTYLE = "-"

AUTO_APPLY_DEBOUNCE_MS = 350


def parse_labels(text: str) -> list[str]:
    labels = [t.strip() for t in text.split(",") if t.strip()]
    if len(labels) < 2:
        raise ValueError("Need at least 2 labels (comma-separated).")
    return labels


def parse_values_csv(text: str) -> list[float]:
    parts = [p.strip() for p in text.split(",")]
    out = []
    for p in parts:
        if p == "" or p.lower() == "nan":
            out.append(np.nan)
        else:
            out.append(float(p))
    return out


def parse_series_block(text: str, n_labels: int) -> list[tuple[str, np.ndarray]]:
    series = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError(f"Bad series line (missing ':'): {line}")

        name, values_str = line.split(":", 1)
        name = name.strip()
        values = parse_values_csv(values_str)

        if len(values) < n_labels:
            values = values + [np.nan] * (n_labels - len(values))
        elif len(values) > n_labels:
            values = values[:n_labels]

        series.append((name, np.array(values, dtype=float)))

    if not series:
        raise ValueError("Add at least one series line, e.g. 'A: 1,2,3'.")
    return series


def parse_colors_block(text: str) -> list[str]:
    colors = [c.strip() for c in text.splitlines() if c.strip()]
    if not colors:
        raise ValueError("Provide at least one color (one per line).")
    return colors


def parse_positive_float(text: str, field_name: str) -> float:
    v = float(text.strip())
    if v <= 0:
        raise ValueError(f"{field_name} must be > 0.")
    return v


class AspectRatioWidget(QWidget):
    """Keeps its child at a fixed aspect ratio (width/height)."""
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
        fig = Figure(figsize=(13, 5))
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
        self.setWindowTitle("Adaptive Labels + Series + Export (PySide + Matplotlib)")

        # Inputs
        self.labels_in = QLineEdit("2017, Mai 19, Sep 19, 2023, 2024, 2025")

        self.series_in = QPlainTextEdit(
            "Line 1: 42,30,18,40,46,38\n"
            "Line 2: 76,89,40,72,77,77\n"
            "Line 3: nan,nan,nan,52,72,49\n"
            "Line 4: nan,nan,nan,90,89,67\n"
        )
        self.series_in.setFixedHeight(110)

        self.ymin_in = QLineEdit("-50")
        self.ymax_in = QLineEdit("96")
        self.ystep_in = QLineEdit("20")

        self.figw_in = QLineEdit("13")
        self.figh_in = QLineEdit("5")

                # allow reasonable range only (inches)
        self.figw_in.setValidator(QDoubleValidator(1.0, 50.0, 2))
        self.figh_in.setValidator(QDoubleValidator(1.0, 50.0, 2))

        self.anim_dur_in = QLineEdit(str(DEFAULT_ANIMATION_DURATION))
        self.pause_dur_in = QLineEdit(str(DEFAULT_PAUSE_DURATION))

        self.show_percent_cb = QCheckBox("Show % on Y axis")
        self.show_percent_cb.setChecked(True)

        # Show only as many default colors as default series lines
        default_series_count = len([ln for ln in self.series_in.toPlainText().splitlines() if ln.strip()])
        self.colors_in = QPlainTextEdit("\n".join(DEFAULT_COLOR_CYCLE[:max(1, default_series_count)]))
        self.colors_in.setFixedHeight(90)

        # Buttons
        self.start_btn = QPushButton("Start")
        self.stop_btn = QPushButton("Stop")
        self.start_btn.clicked.connect(self.start_animation)
        self.stop_btn.clicked.connect(self.stop_animation)

        self.export_btn = QPushButton("Export MP4")
        self.export_btn.clicked.connect(self.export_mp4)

        self.plot_full_btn = QPushButton("Plot Fullscreen")
        self.plot_full_btn.clicked.connect(self.open_plot_fullscreen)

        self.status = QLabel("")

        # Plot
        self.canvas = MplCanvas()
        self.ax = self.canvas.ax
        self.aspect_wrap = AspectRatioWidget(self.canvas, aspect=13 / 5)

        # Inner wrapper that will center the plot
        self.plot_inner = QWidget()
        inner_lay = QHBoxLayout(self.plot_inner)
        inner_lay.setContentsMargins(0, 0, 0, 0)
        inner_lay.addStretch(1)
        inner_lay.addWidget(self.aspect_wrap)
        inner_lay.addStretch(1)

        # Scroll area: expands full width, but scrolls horizontally if plot is wider
        self.plot_hscroll = QScrollArea()
        self.plot_hscroll.setFrameShape(QScrollArea.NoFrame)
        self.plot_hscroll.setWidgetResizable(True)  # IMPORTANT for centering
        self.plot_hscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.plot_hscroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.plot_hscroll.setWidget(self.plot_inner)

        # Plot container
        self.plot_container = QWidget()
        pl = QVBoxLayout(self.plot_container)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.addWidget(self.plot_hscroll)  # ✅ no alignment here

        # Size policies so it doesn't become narrow
        self.plot_hscroll.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.plot_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


        # Controls panel
        controls = QWidget()
        cl = QVBoxLayout(controls)

        row_labels = QWidget()
        rl = QHBoxLayout(row_labels)
        rl.addWidget(QLabel("Labels:"))
        rl.addWidget(self.labels_in)
        cl.addWidget(row_labels)

        cl.addWidget(QLabel("Series (Name: v1,v2,... ; use nan):"))
        cl.addWidget(self.series_in)

        # Per-series style editor
        cl.addWidget(QLabel("Per-series style (live):"))
        self.style_panel = QWidget()
        self.style_layout = QVBoxLayout(self.style_panel)
        self.style_layout.setContentsMargins(0, 0, 0, 0)
        self.style_layout.setSpacing(6)

        self.style_scroll = QScrollArea()
        self.style_scroll.setWidgetResizable(True)
        self.style_scroll.setFrameShape(QScrollArea.NoFrame)
        self.style_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.style_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.style_scroll.setWidget(self.style_panel)
        self.style_scroll.setFixedHeight(170)
        cl.addWidget(self.style_scroll)

        row_y = QWidget()
        yl = QHBoxLayout(row_y)
        yl.addWidget(QLabel("Y min"))
        yl.addWidget(self.ymin_in)
        yl.addWidget(QLabel("Y max"))
        yl.addWidget(self.ymax_in)
        yl.addWidget(QLabel("Y step"))
        yl.addWidget(self.ystep_in)
        cl.addWidget(row_y)

        row_fig = QWidget()
        fl = QHBoxLayout(row_fig)
        fl.addWidget(QLabel("Fig W"))
        fl.addWidget(self.figw_in)
        fl.addWidget(QLabel("Fig H"))
        fl.addWidget(self.figh_in)
        cl.addWidget(row_fig)

        row_anim = QWidget()
        al = QHBoxLayout(row_anim)
        al.addWidget(QLabel("Anim (s)"))
        al.addWidget(self.anim_dur_in)
        al.addWidget(QLabel("Pause (s)"))
        al.addWidget(self.pause_dur_in)
        cl.addWidget(row_anim)

        cl.addWidget(QLabel("Colors (one per line): (auto-add when series increases)"))
        cl.addWidget(self.colors_in)

        cl.addWidget(self.show_percent_cb)

        btnrow = QWidget()
        bl = QHBoxLayout(btnrow)
        bl.addWidget(self.start_btn)
        bl.addWidget(self.stop_btn)
        bl.addWidget(self.export_btn)
        bl.addWidget(self.plot_full_btn)
        cl.addWidget(btnrow)

        cl.addWidget(self.status)
        cl.addStretch(1)

        # One-page scroll: controls + plot together
        page = QWidget()
        page_lay = QVBoxLayout(page)
        page_lay.addWidget(controls)
        page_lay.addWidget(self.plot_container)
        page_lay.addStretch(1)

        self.page_scroll = QScrollArea()
        self.page_scroll.setWidgetResizable(True)
        self.page_scroll.setFrameShape(QScrollArea.NoFrame)

        # ✅ only ONE vertical scrollbar
        self.page_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.page_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.page_scroll.setWidget(page)
        self.setCentralWidget(self.page_scroll)


        # Animation state
        self.frame = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.step)

        # Auto-apply debounce timer
        self._apply_debounce = QTimer(self)
        self._apply_debounce.setSingleShot(True)
        self._apply_debounce.timeout.connect(self.apply_all)

        # Data/artists
        self.labels = []
        self.series = []
        self.colors = DEFAULT_COLOR_CYCLE[:]
        self.lines = []
        self.points = []
        self.series_styles = []

        # Per-series style widgets + state
        self._style_widgets_by_name = {}
        self._style_state_by_name = {}

        # Fullscreen
        self._plot_full_win = None
        self._placeholder = None

        self._wire_auto_apply_inputs()

        self.apply_all()
        self.start_animation()
        self.resize(1200, 800)

    def _wire_auto_apply_inputs(self):
        def schedule():
            self.status.setText("⏳ Pending changes...")
            self.status.setStyleSheet("color: #444;")
            self._apply_debounce.start(AUTO_APPLY_DEBOUNCE_MS)

        self.labels_in.textChanged.connect(schedule)
        self.series_in.textChanged.connect(schedule)
        self.ymin_in.textChanged.connect(schedule)
        self.ymax_in.textChanged.connect(schedule)
        self.ystep_in.textChanged.connect(schedule)
        self.figw_in.textChanged.connect(schedule)
        self.figh_in.textChanged.connect(schedule)
        self.anim_dur_in.textChanged.connect(schedule)
        self.pause_dur_in.textChanged.connect(schedule)
        self.colors_in.textChanged.connect(schedule)
        self.show_percent_cb.toggled.connect(schedule)

    def update_plot_widget_size(self):
        w_px = int(self.figw * PREVIEW_DPI)
        h_px = int(self.figh * PREVIEW_DPI)
        w_px = max(w_px, 500)
        h_px = max(h_px, 300)

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

        # Detach from current parent
        self.plot_container.setParent(None)

        def restore():
            # Put it back under the scroll page
            page = self.page_scroll.widget()
            page.layout().insertWidget(1, self.plot_container)  # controls is index 0
            self._plot_full_win = None

        self._plot_full_win = PlotFullscreenWindow(self, self.plot_container, restore)
        self._plot_full_win.showFullScreen()


    def start_animation(self):
        self.timer.start(int(1000 / FPS))

    def stop_animation(self):
        self.timer.stop()

    def _clear_layout(self, layout: QVBoxLayout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _series_index_by_name(self, name: str) -> int:
        for i, (n, _) in enumerate(self.series):
            if n == name:
                return i
        return -1

    def _persist_current_style_ui(self):
        for name, widgets in list(self._style_widgets_by_name.items()):
            try:
                st = self._read_style_from_widgets(name, widgets)
                self._style_state_by_name[name] = dict(st)
            except Exception:
                pass

    def _read_style_from_widgets(self, name: str, widgets: dict) -> dict:
        linew = parse_positive_float(widgets["linew"].text(), f"Line width ({name})")
        markersize = parse_positive_float(widgets["markersize"].text(), f"Marker size ({name})")
        markeredgew = parse_positive_float(widgets["markeredgew"].text(), f"Marker edge width ({name})")
        marker = widgets["marker"].currentData()
        linestyle = widgets["linestyle"].currentData()
        return {
            "linew": linew,
            "markersize": markersize,
            "markeredgew": markeredgew,
            "marker": marker,
            "linestyle": linestyle,
        }

    def _on_style_changed_live(self, name: str):
        widgets = self._style_widgets_by_name.get(name)
        if not widgets:
            return
        try:
            st = self._read_style_from_widgets(name, widgets)
        except ValueError as e:
            self.status.setText(f"❌ {e}")
            self.status.setStyleSheet("color: red;")
            return

        self._style_state_by_name[name] = dict(st)

        idx = self._series_index_by_name(name)
        if idx < 0 or idx >= len(self.lines):
            return

        marker = "" if st["marker"] == "None" else st["marker"]
        linestyle = "None" if st["linestyle"] == "None" else st["linestyle"]

        self.lines[idx].set_linewidth(st["linew"])
        self.lines[idx].set_linestyle(linestyle)
        self.lines[idx].set_marker(marker)
        self.lines[idx].set_markersize(st["markersize"])
        self.lines[idx].set_markeredgewidth(st["markeredgew"])

        self.points[idx].set_marker(marker)
        self.points[idx].set_markersize(st["markersize"])
        self.points[idx].set_markeredgewidth(st["markeredgew"])

        self.canvas.draw_idle()
        self.status.setText("✅ Style updated.")
        self.status.setStyleSheet("color: green;")

    def _make_style_row(self, name: str):
        marker_options = [
            ("o", "Circle (o)"), ("s", "Square (s)"), ("^", "Triangle Up (^)"),
            ("v", "Triangle Down (v)"), (">", "Triangle Right (>)"), ("<", "Triangle Left (<)"),
            ("D", "Diamond (D)"), ("P", "Plus Filled (P)"), ("X", "X Filled (X)"),
            ("*", "Star (*)"), ("+", "Plus (+)"), ("None", "None"),
        ]
        linestyle_options = [
            ("-", "Solid (-)"), ("--", "Dashed (--)"), ("-.", "Dash-dot (-.)"),
            (":", "Dotted (:)"), ("None", "None"),
        ]

        st = self._style_state_by_name.get(name, {})
        lw = str(st.get("linew", DEFAULT_LINE_WIDTH))
        ms = str(st.get("markersize", DEFAULT_MARKER_SIZE))
        mew = str(st.get("markeredgew", DEFAULT_MARKER_EDGE_WIDTH))
        mk = st.get("marker", DEFAULT_MARKER)
        ls = st.get("linestyle", DEFAULT_LINESTYLE)

        row = QWidget()
        r = QHBoxLayout(row)
        r.setContentsMargins(0, 0, 0, 0)

        r.addWidget(QLabel(name))

        lw_in = QLineEdit(lw); lw_in.setFixedWidth(55)
        ms_in = QLineEdit(ms); ms_in.setFixedWidth(55)
        mew_in = QLineEdit(mew); mew_in.setFixedWidth(55)

        mk_cb = QComboBox()
        for val, label in marker_options:
            mk_cb.addItem(label, val)
        mk_cb.setCurrentIndex(max(0, mk_cb.findData(mk)))

        ls_cb = QComboBox()
        for val, label in linestyle_options:
            ls_cb.addItem(label, val)
        ls_cb.setCurrentIndex(max(0, ls_cb.findData(ls)))

        r.addWidget(QLabel("Line W")); r.addWidget(lw_in)
        r.addWidget(QLabel("Style"));  r.addWidget(ls_cb)
        r.addWidget(QLabel("Marker")); r.addWidget(mk_cb)
        r.addWidget(QLabel("Size"));   r.addWidget(ms_in)
        r.addWidget(QLabel("Edge"));   r.addWidget(mew_in)

        self._style_widgets_by_name[name] = {
            "linew": lw_in,
            "markersize": ms_in,
            "markeredgew": mew_in,
            "marker": mk_cb,
            "linestyle": ls_cb,
        }

        lw_in.textChanged.connect(lambda _=None, n=name: self._on_style_changed_live(n))
        ms_in.textChanged.connect(lambda _=None, n=name: self._on_style_changed_live(n))
        mew_in.textChanged.connect(lambda _=None, n=name: self._on_style_changed_live(n))
        mk_cb.currentIndexChanged.connect(lambda _=None, n=name: self._on_style_changed_live(n))
        ls_cb.currentIndexChanged.connect(lambda _=None, n=name: self._on_style_changed_live(n))

        return row

    def rebuild_style_editor(self, series_names: list[str]):
        self._clear_layout(self.style_layout)
        self._style_widgets_by_name = {}
        for name in series_names:
            self.style_layout.addWidget(self._make_style_row(name))
        self.style_layout.addStretch(1)

    def _styles_for_current_series_names(self, series_names: list[str]) -> list[dict]:
        out = []
        for name in series_names:
            st = self._style_state_by_name.get(name, {
                "linew": DEFAULT_LINE_WIDTH,
                "markersize": DEFAULT_MARKER_SIZE,
                "markeredgew": DEFAULT_MARKER_EDGE_WIDTH,
                "marker": DEFAULT_MARKER,
                "linestyle": DEFAULT_LINESTYLE,
            })
            out.append(dict(st))
        return out

    def _auto_add_missing_colors(self, series_count: int, colors: list[str]) -> list[str]:
        if series_count <= 0:
            return colors
        if len(colors) >= series_count:
            return colors

        extended = list(colors)
        start_idx = len(extended) % len(DEFAULT_COLOR_CYCLE)
        while len(extended) < series_count:
            extended.append(DEFAULT_COLOR_CYCLE[start_idx % len(DEFAULT_COLOR_CYCLE)])
            start_idx += 1

        self.colors_in.blockSignals(True)
        try:
            self.colors_in.setPlainText("\n".join(extended))
        finally:
            self.colors_in.blockSignals(False)

        return extended

    def apply_all(self):
        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        try:
            self._persist_current_style_ui()

            labels = parse_labels(self.labels_in.text())
            series = parse_series_block(self.series_in.toPlainText(), len(labels))

            ymin = float(self.ymin_in.text().strip())
            ymax = float(self.ymax_in.text().strip())
            if ymax <= ymin:
                raise ValueError("Y max must be greater than Y min.")
            ystep = parse_positive_float(self.ystep_in.text(), "Y step")

            figw = parse_positive_float(self.figw_in.text(), "Figure width")
            figh = parse_positive_float(self.figh_in.text(), "Figure height")

            anim_dur = parse_positive_float(self.anim_dur_in.text(), "Animation seconds")
            pause_dur = parse_positive_float(self.pause_dur_in.text(), "Pause seconds")

            colors = parse_colors_block(self.colors_in.toPlainText())
            colors = self._auto_add_missing_colors(len(series), colors)

            self.labels = labels
            self.series = series
            self.colors = colors

            self.ymin, self.ymax, self.ystep = ymin, ymax, ystep
            self.figw, self.figh = figw, figh
            self.anim_dur, self.pause_dur = anim_dur, pause_dur

            series_names = [name for name, _ in self.series]
            self.rebuild_style_editor(series_names)
            self.series_styles = self._styles_for_current_series_names(series_names)

            self.aspect_wrap.set_aspect(self.figw / self.figh)
            self.update_plot_widget_size()

            self.rebuild_axes_and_artists()

            self.status.setText("✅ Applied.")
            self.status.setStyleSheet("color: green;")

            self.frame = 0
            self.animate_frame(self.frame)
            self.canvas.draw_idle()
            figw = parse_positive_float(self.figw_in.text(), "Figure width")
            figh = parse_positive_float(self.figh_in.text(), "Figure height")

            if not (1.0 <= figw <= 50.0):
                raise ValueError("Figure width must be between 1 and 50 inches.")
            if not (1.0 <= figh <= 50.0):
                raise ValueError("Figure height must be between 1 and 50 inches.")

        except ValueError as e:
            self.status.setText(f"❌ {e}")
            self.status.setStyleSheet("color: red;")

        finally:
            if was_running:
                self.timer.start(int(1000 / FPS))

    def rebuild_axes_and_artists(self):
        self.ax.clear()
        self.canvas.figure.set_size_inches(self.figw, self.figh, forward=True)

        n = len(self.labels)
        self.x_pos = np.arange(n)
        self.x_shifted = self.x_pos - 0.5

        self.total_frames = int(self.anim_dur * FPS)
        self.frames_per_step = max(1, self.total_frames // (n - 1))
        self.pause_frames = int(self.pause_dur * FPS)
        self.total_frames_with_pause = self.total_frames + self.pause_frames

        ax = self.ax
        ax.set_xlim(-1, n - 1)
        ax.set_ylim(self.ymin, self.ymax)
        ax.set_yticks(np.arange(self.ymin, self.ymax + 1e-9, self.ystep))

        # ✅ read checkbox directly (no self.show_percent attribute needed)
        if self.show_percent_cb.isChecked():
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}%"))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{int(y)}"))

        ax.grid(True, alpha=0.3, color="#404040")
        ax.axhline(0, color="black", linewidth=1, alpha=0.7)

        self.canvas.figure.patch.set_facecolor("#eaeaef")
        ax.set_facecolor("#eaeaef")

        ax.set_xticks(self.x_pos)
        ax.set_xticks(self.x_shifted, minor=True)
        ax.set_xticklabels(self.labels, minor=True)

        ax.tick_params(axis="x", which="major", bottom=False, labelbottom=False)
        ax.tick_params(axis="x", which="minor", pad=10, colors="#404040")
        ax.tick_params(axis="y", pad=10)

        ax.grid(True, axis="x", which="major", color="#555557", linewidth=1)

        ax.spines["top"].set_visible(False)
        for spine in ax.spines.values():
            spine.set_color((0.25, 0.25, 0.25, 0.3))
            spine.set_linewidth(1)

        for tick in ax.xaxis.get_ticklabels() + ax.yaxis.get_ticklabels():
            tick.set_color("#555557")

        self.lines = []
        self.points = []

        if len(self.series_styles) != len(self.series):
            series_names = [name for name, _ in self.series]
            self.series_styles = self._styles_for_current_series_names(series_names)

        for i, (name, _) in enumerate(self.series):
            color = self.colors[i % len(self.colors)]
            st = self.series_styles[i]

            marker = "" if st["marker"] == "None" else st["marker"]
            linestyle = "None" if st["linestyle"] == "None" else st["linestyle"]

            line = ax.plot(
                [], [],
                marker=marker,
                linestyle=linestyle,
                linewidth=st["linew"],
                markersize=st["markersize"],
                markerfacecolor="white",
                markeredgewidth=st["markeredgew"],
                color=color,
                label=name
            )[0]

            pt = ax.plot(
                [], [],
                marker=marker,
                linestyle="None",
                linewidth=0,
                markersize=st["markersize"],
                markerfacecolor="white",
                markeredgewidth=st["markeredgew"],
                color=color
            )[0]

            self.lines.append(line)
            self.points.append(pt)

    def step(self):
        self.frame = (self.frame + 1) % self.total_frames_with_pause
        self.animate_frame(self.frame)
        self.canvas.draw_idle()

    def animate_frame(self, frame: int):
        n = len(self.labels)
        last_motion_frame = (n - 1) * self.frames_per_step - 1
        if last_motion_frame < 1:
            return
        if frame >= last_motion_frame:
            frame = last_motion_frame

        idx = frame / self.frames_per_step
        lower = int(np.floor(idx))
        upper = min(lower + 1, n - 1)
        frac = idx - lower

        x_now = self.x_shifted[lower] + frac * (self.x_shifted[upper] - self.x_shifted[lower])
        x_draw = np.append(self.x_shifted[:upper], x_now)

        def interp(data: np.ndarray):
            a, b = data[lower], data[upper]
            if np.isnan(a) or np.isnan(b):
                return np.nan
            return a + frac * (b - a)

        for i, (_, data) in enumerate(self.series):
            y_now = interp(data)
            y_draw = np.append(data[:upper], y_now)
            self.lines[i].set_data(x_draw, y_draw)
            if np.isnan(y_now):
                self.points[i].set_data([], [])
            else:
                self.points[i].set_data([x_now], [y_now])

    def export_mp4(self):
        was_running = self.timer.isActive()
        self.timer.stop()

        final_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save animation as MP4",
            "animation.mp4",
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

            writer = FFMpegWriter(
                fps=FPS,
                codec="libx264",
                bitrate=4000,
                extra_args=[
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"
                ]
            )

            with writer.saving(self.canvas.figure, tmp_path, dpi=150):
                for f in range(self.total_frames_with_pause):
                    self.animate_frame(f)
                    self.canvas.draw()
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

            self.status.setText(f"❌ Export failed: {e} (see console)")
            self.status.setStyleSheet("color: red;")

        finally:
            if was_running:
                self.timer.start(int(1000 / FPS))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


