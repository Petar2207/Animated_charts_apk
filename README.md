# 📊 Animated Chart Studio (PySide6 + Matplotlib)

A desktop application for **creating, previewing, animating, and exporting professional charts** using **Python, PySide6, and Matplotlib**.

The project consists of:
- a **launcher app** that lets you choose a chart visually
- multiple **specialized chart editors**, each with live preview, animation controls, fullscreen mode, and MP4 export

---

## ✨ Features

### General
- 🖥️ Native desktop GUI (PySide6)
- 🎞️ Smooth animated charts (30 FPS)
- 🔍 Live preview with fixed aspect ratio
- 🖱️ Fullscreen plot mode (ESC to exit)
- 🎥 Export animations to **MP4** (FFmpeg)
- ⏳ Auto-apply changes with debounce (no UI freezing)

---

## 🧭 Chart Launcher

The launcher (`app.py`) lets you:
- Select a chart type from a dropdown
- See a large preview image
- Launch the selected chart editor
- Optionally close the launcher after opening a chart

Charts are started as **independent Python processes**, so each tool runs cleanly and independently.

---

## 📈 Included Chart Tools

### 1️⃣ Animated Bars + Lines
**File:** `bar.py`

- Animated vertical bars
- Up to **two optional line series**
- Custom colors, labels, ticks, legend names
- Percent formatting toggle
- MP4 export

---

### 2️⃣ Diverging Horizontal Bars (2 Series)
**File:** `vertical_bar.py`

- Horizontal diverging bars (A vs B)
- Independent colors per side
- Optional value labels with `+` sign
- Fully configurable axis range and ticks
- Ideal for comparisons & surveys

---

### 3️⃣ Stacked Horizontal Bars (5 Colors)
**File:** `5_color_bar.py`

- Animated **stacked barh** chart
- Per-segment labels & colors
- Category editor + CSV data input
- Minimum segment size threshold for labels
- Clean, presentation-ready output

---

### 4️⃣ Donut Chart with Gradient Outer Ring
**File:** `pie chart.py`

- Donut chart with **high-resolution gradient ring**
- Harsh / sharp color blending near segment edges
- Adjustable animation duration
- Optional percent sign toggle
- Optimized preview vs export resolution

---

### 5️⃣ Semi-Circular Gauge
**File:** `gauge.py`

- Speedometer-style gauge (0–100)
- Animated fill arc
- Custom tick count, thickness, radius
- Optional decimal comma formatting
- Ideal for KPI / score visualization

---

### 6️⃣ Multi-Series Line Chart (Adaptive Labels)
**File:** `final_code.py`

- Multiple animated line series
- Supports `nan` values (gaps)
- Per-series live style editor
- Auto color expansion
- Clean adaptive labels
- MP4 export

---

## 📂 Project Structure

```text
project-root/
├─ app.py                # Chart launcher
├─ requirements.txt
├─ run.bat               # Windows helper
├─ bin/
│  └─ ffmpeg.exe         # FFmpeg binary (Windows)
├─ charts/
│  ├─ bar.py
│  ├─ vertical_bar.py
│  ├─ 5_color_bar.py
│  ├─ pie chart.py
│  ├─ gauge.py
│  └─ final_code.py
└─ previews/
   ├─ bar.png
   ├─ horizontal.png
   ├─ fivecolor.png
   ├─ pie.png
   ├─ gauge.png
   └─ line.png


🧪 Requirements
Python 3.9+ recommended
FFmpeg (required for MP4 export)
Python dependencies
numpy
matplotlib
PySide6

🎥 MP4 Export Notes
Export uses matplotlib.animation.FFMpegWriter
FFmpeg must be available in PATH

🚀 Quick Start (Windows – Recommended)

After setting up the project structure you only need to run:
run.bat
This will:
create / activate the virtual environment
install all dependencies
start the chart launcher
No manual Python commands required.
