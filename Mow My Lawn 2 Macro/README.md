# 🌿 Mow My Lawn 2 — AI Vision & Spatial Yield Autopilot

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV Vision](https://img.shields.io/badge/Vision-OpenCV%20%2B%20RapidOCR-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![GUI](https://img.shields.io/badge/UI-CustomTkinter%20Dark%20Mode-10B981)](https://github.com/TomSchimansky/CustomTkinter)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An external Computer-Vision & Spatial Potential-Field AI Autopilot for **[Mow My Lawn 2](https://www.roblox.com/games/115917693/MOW-MY-LAWN-2)** on Roblox.

Unlike simple key-loop macros that walk into walls or stay trapped in a small patch of grass, **MML2 AI Autopilot** combines real-time **OpenCV edge-texture segmentation**, a **16-Ray Polar Potential-Field AI Planner**, a **Shifting Ego-Centric Lawn Regrowth Grid**, and **asynchronous RapidOCR HUD tracking** inside a sleek dark-mode dashboard.

---

## ✨ Key Features

### 🧠 1. 16-Ray Potential-Field & Spatial Occupancy Grid AI ([`ai_brain.py`](file:///c:/Users/iford/OneDrive/Documents/Mow%20My%20Lawn%202%20Macro/ai_brain.py))
- **Ego-Centric Lawn Regrowth Map (`20×14` Grid)**: Remembers which sectors of the lawn have been recently cut (`0.0`), shifts the map opposite to your movement vector in real time, and models natural grass regrowth over time (`1.0+`) while projecting OpenCV-detected 3D grass blade clusters onto the grid.
- **16-Direction Polar Raycast Planner**: On every frame, evaluates 16 candidate movement rays (`22.5°` apart) by integrating:
  - **(+) Unmowed Grass Density & Regrowth Yield** along the ray
  - **(+) Full-Lawn Macro Sweep Momentum** (ensures your character traverses the entire Main Lawn from end to end without lingering in one spot)
  - **(+) Grill Explosion Food & Boss Token Drops**
  - **(-) House Door & Porch Repulsion Field** (keeps your character away from the house door)
  - **(-) Boss Red Telegraph Hazard Repulsion**
- **Online Bandit Auto-Tuner**: Continuously measures your grass intake rate over 10-second epochs and adapts its exploration vs. cluster-steering weight automatically.

### 📊 2. Bottom-Right Multi-Item Percentage & Tool Intake OCR ([`stats_tracker.py`](file:///c:/Users/iford/OneDrive/Documents/Mow%20My%20Lawn%202%20Macro/stats_tracker.py))
- Runs **RapidOCR** in a non-blocking background thread to read the **2 or more item progress percentages (`%`)** in the bottom-right HUD.
- Computes individual live countdown **ETAs to `100%`** (`Item #1`, `Item #2`, `Item #3`) and displays progress bars directly in the GUI.
- Gauges your currently equipped item's **grass intake per hit** (both `+% progress / hit` and `visual grass pixels cleared / hit`).

### 🛡️ 3. Boss "Tag & Survive" Protocol + Instant Hardware Emergency Stop
- **Lawn Boss Handling**: Automatically detects the top Lawn Boss health/timer bar, moves in briefly to land **1 tag hit** (securing your **4 Boss Tokens + EXP + Grill Explosion counter progress**), and then orbits the outer perimeter while dodging red ground telegraphs until the boss is defeated or the timer runs out.
- **200Hz Hardware Emergency Stop**: Polls Windows `GetAsyncKeyState` every `5ms`. Pressing **`F7`** or **`END`** (or moving your mouse to the top-left corner `(0,0)`) immediately halts and releases all keys—and you can press **`F6`** anytime to resume!

---

## 🚀 Quick Start

1. Clone or download this repository.
2. Double-click **[`run_macro.bat`](file:///c:/Users/iford/OneDrive/Documents/Mow%20My%20Lawn%202%20Macro/run_macro.bat)** (it will automatically create a local `.venv`, install all requirements from [`requirements.txt`](file:///c:/Users/iford/OneDrive/Documents/Mow%20My%20Lawn%202%20Macro/requirements.txt), and open the **MML2 AI Autopilot Dashboard**).
3. **In-Game Setup**:
   - Equip your mower/weapon (`1`).
   - Turn **OFF Shift-Lock** so `W`, `A`, `S`, `D` move freely in all 4 directions.
   - Zoom your camera out moderately at a ~45° downward angle with the House at the top of the screen.
4. Press **`F6`** (or click **▶ START / RESUME** in the dashboard) to engage the AI Autopilot!

---

## ⌨️ Hotkeys

| Hotkey | Action |
| :--- | :--- |
| **`F6`** | **Start / Resume / Pause** AI Autopilot |
| **`F7`** or **`END`** | **Emergency Stop** (Instant key release & halt; press `F6` anytime to restart) |
| **`Mouse -> Top-Left (0,0)`** | **Panic Failsafe Pause** |
| **`F9`** | Toggle manual **Boss Survival Evasion** mode |

---

## 📁 Project Architecture

- [`gui_app.py`](file:///c:/Users/iford/OneDrive/Documents/Mow%20My%20Lawn%202%20Macro/gui_app.py) — CustomTkinter Dark-Mode AI Control Center with embedded live radar feed, progress bars, and live tuning sliders.
- [`ai_brain.py`](file:///c:/Users/iford/OneDrive/Documents/Mow%20My%20Lawn%202%20Macro/ai_brain.py) — 16-Ray Polar Potential-Field planner, `20×14` ego-centric lawn occupancy/regrowth grid, and online bandit auto-tuner.
- [`vision.py`](file:///c:/Users/iford/OneDrive/Documents/Mow%20My%20Lawn%202%20Macro/vision.py) — High-speed `mss` + OpenCV pipeline for 3D grass blade texture detection, Grill Explosion food/drop tracking, house door repulsion, and boss hazard segmentation.
- [`stats_tracker.py`](file:///c:/Users/iford/OneDrive/Documents/Mow%20My%20Lawn%202%20Macro/stats_tracker.py) — Asynchronous RapidOCR engine for bottom-right item percentages (`%`), countdown ETAs, and per-hit tool intake analysis.
- [`controller.py`](file:///c:/Users/iford/OneDrive/Documents/Mow%20My%20Lawn%202%20Macro/controller.py) — DirectInput hardware controller & priority state machine.
- [`calibrate_colors.py`](file:///c:/Users/iford/OneDrive/Documents/Mow%20My%20Lawn%202%20Macro/calibrate_colors.py) — Interactive click-to-sample HSV calibration tool.
