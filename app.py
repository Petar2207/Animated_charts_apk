import sys
import os
import subprocess

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QMessageBox, QCheckBox
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon, QPixmap


# Put your preview images in:  <this file's folder>/previews/
# Example:
#   previews/bar.png
#   previews/line.png
#   previews/gauge.png
#   previews/pie.png
#   previews/horizontal.png
#   previews/fivecolor.png
SCRIPTS = [
    ("Vertical Bar (bars + lines)", "bar.py", "bar.png"),
    ("Line Chart (adaptive labels)", "final_code.py", "line.png"),
    ("Gauge", "gauge.py", "gauge.png"),
    ("Pie / Donut", "pie chart.py", "pie.png"),
    ("Horizontal Bars (two series)", "vertical_bar.py", "horizontal.png"),
    ("5 color Bars (two series)", "5_color_bar.py", "fivecolor.png"),
]


class Launcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Chart Launcher")

        self.base_dir = os.path.dirname(__file__)
        self.charts_dir = os.path.join(self.base_dir, "charts")
        self.previews_dir = os.path.join(self.base_dir, "previews")

        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        layout.addWidget(QLabel("Choose a chart:"))

        self.combo = QComboBox()
        self.combo.setIconSize(QSize(72, 40))  # size of icons shown in dropdown

        # Add items with icons + store (script_path, preview_path) as item data
        for title, script_name, preview_name in SCRIPTS:
            preview_path = os.path.join(self.previews_dir, preview_name)

            icon = QIcon(preview_path) if os.path.exists(preview_path) else QIcon()
            self.combo.addItem(icon, title, (script_name, preview_path))

        layout.addWidget(self.combo)

        # Large preview area
        self.preview = QLabel("Select a chart to see preview", alignment=Qt.AlignCenter)
        self.preview.setMinimumHeight(240)
        self.preview.setStyleSheet("border: 1px solid #999;")
        layout.addWidget(self.preview)

        # Update preview on selection change
        self.combo.currentIndexChanged.connect(self.update_preview)
        self.update_preview()  # show initial preview

        # Buttons row
        row = QHBoxLayout()
        self.launch_btn = QPushButton("Open")
        self.launch_btn.clicked.connect(self.launch_selected)

        self.close_launcher_cb = QCheckBox("Close launcher after opening")
        self.close_launcher_cb.setChecked(False)

        row.addWidget(self.launch_btn)
        row.addWidget(self.close_launcher_cb)
        row.addStretch(1)

        layout.addLayout(row)
        layout.addStretch(1)

        self.setMinimumWidth(560)

    def update_preview(self):
        data = self.combo.currentData()
        if not data:
            self.preview.setText("No preview")
            self.preview.setPixmap(QPixmap())
            return

        _, preview_path = data

        if not preview_path or not os.path.exists(preview_path):
            self.preview.setText("No preview image found")
            self.preview.setPixmap(QPixmap())
            return

        pix = QPixmap(preview_path)
        if pix.isNull():
            self.preview.setText("Preview failed to load")
            self.preview.setPixmap(QPixmap())
            return

        # Scale to fit the label
        scaled = pix.scaled(
            self.preview.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview.setPixmap(scaled)
        self.preview.setText("")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Re-scale preview when window changes size
        self.update_preview()

    def launch_selected(self):
        data = self.combo.currentData()
        if not data:
            return

        script_name, _ = data
        script_path = os.path.join(self.charts_dir, script_name)

        if not os.path.exists(script_path):
            QMessageBox.critical(self, "Not found", f"File not found:\n{script_path}")
            return

        try:
            subprocess.Popen([sys.executable, script_path], cwd=self.base_dir)
        except Exception as e:
            QMessageBox.critical(self, "Failed", f"Could not start:\n{script_path}\n\n{e}")
            return

        if self.close_launcher_cb.isChecked():
            self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Launcher()
    w.show()
    sys.exit(app.exec())
