"""
gui_app.py - Modern Dark-Mode CustomTkinter AI Control Center for Mow My Lawn 2 Macro

Features:
- Live embedded AI Vision Radar & Spatial Occupancy Heatmap feed
- Instant GUI buttons + 200Hz Win32 Hardware Hotkey Poller (F6 Start/Resume/Pause, F7 Emergency Stop)
- Live Tool Grass Intake Gauge (+%/hit, grass-px/hit, attacks/sec)
- Multi-Item Bottom-Right Percentage Progress Bars & Countdown ETAs
- Live AI Tuning Sliders (Sweep Width, Vision Steering Weight, Door Avoidance)
"""

import ctypes
import json
import os
import threading
import time
from typing import Dict, List

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image

from controller import MacroState, MML2Controller
from stats_tracker import ProgressionAnalyzer, format_eta, format_number
from vision import MML2Vision

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

user32 = ctypes.windll.user32

VK_MAP: Dict[str, int] = {
    "f1": 0x70,
    "f2": 0x71,
    "f3": 0x72,
    "f4": 0x73,
    "f5": 0x74,
    "f6": 0x75,
    "f7": 0x76,
    "f8": 0x77,
    "f9": 0x78,
    "f10": 0x79,
    "end": 0x23,
    "pause": 0x13,
}


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


def is_vk_down(vk_code: int) -> bool:
    return (user32.GetAsyncKeyState(vk_code) & 0x8000) != 0


def get_cursor_pos() -> tuple[int, int]:
    pt = POINT()
    user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


class MML2DashboardApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.title("Mow My Lawn 2 — AI Vision & Spatial Yield Suite")
        self.geometry("1160x765")
        self.minsize(1020, 680)

        self.config_data = load_config()
        self.vision = MML2Vision(self.config_data)
        self.tracker = ProgressionAnalyzer(self.config_data)
        self.controller = MML2Controller(self.config_data, on_swing_callback=self.tracker.record_weapon_swing)

        self.running = False
        self.manual_boss_override = False
        self.should_exit = False
        self._needs_camera_init = False
        self._latest_ctk_img = None

        self._build_ui()

        # Start 200Hz hardware hotkey poller thread
        self._hotkey_thread = threading.Thread(target=self._poll_hotkeys_loop, daemon=True)
        self._hotkey_thread.start()

        # Start AI & Vision background engine thread
        self._engine_thread = threading.Thread(target=self._engine_loop, daemon=True)
        self._engine_thread.start()

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(50, self._refresh_gui_loop)

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # =====================================================================
        # LEFT SIDEBAR: Controls, AI Status, Tool Intake, Sliders
        # =====================================================================
        sidebar = ctk.CTkFrame(self, width=350, corner_radius=12)
        sidebar.grid(row=0, column=0, padx=12, pady=12, sticky="nsew")
        sidebar.grid_propagate(False)

        title_lbl = ctk.CTkLabel(
            sidebar,
            text="MML2 AI AUTOPILOT",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color="#2EE59D",
        )
        title_lbl.pack(padx=16, pady=(16, 2), anchor="w")

        sub_lbl = ctk.CTkLabel(
            sidebar,
            text="16-Ray Potential-Field & Spatial Grid AI",
            font=ctk.CTkFont(size=12),
            text_color="#9BA3AE",
        )
        sub_lbl.pack(padx=16, pady=(0, 12), anchor="w")

        # Status Badge Card
        self.status_card = ctk.CTkFrame(sidebar, fg_color="#1A212B", corner_radius=10)
        self.status_card.pack(fill="x", padx=14, pady=6)

        self.state_lbl = ctk.CTkLabel(
            self.status_card,
            text="STATE: IDLE (Press F6 to Start)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color="#FFA726",
        )
        self.state_lbl.pack(padx=12, pady=(10, 2), anchor="w")

        self.ai_reason_lbl = ctk.CTkLabel(
            self.status_card,
            text="AI Brain: Standing by...",
            font=ctk.CTkFont(size=12),
            text_color="#B0BEC5",
        )
        self.ai_reason_lbl.pack(padx=12, pady=(0, 4), anchor="w")

        self.keys_lbl = ctk.CTkLabel(
            self.status_card,
            text="Active Keys: NONE   |   AI Yield Score: 100%",
            font=ctk.CTkFont(size=12, family="Consolas"),
            text_color="#80CBC4",
        )
        self.keys_lbl.pack(padx=12, pady=(0, 10), anchor="w")

        # Action Buttons (Start/Pause F6 & Emergency Stop F7)
        btn_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        btn_frame.pack(fill="x", padx=14, pady=8)

        self.btn_start = ctk.CTkButton(
            btn_frame,
            text="▶ START / RESUME [F6]",
            height=42,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#10B981",
            hover_color="#059669",
            command=self.toggle_running,
        )
        self.btn_start.pack(fill="x", pady=(0, 8))

        self.btn_estop = ctk.CTkButton(
            btn_frame,
            text="■ EMERGENCY STOP [F7 / END]",
            height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#EF4444",
            hover_color="#DC2626",
            command=lambda: self.trigger_emergency_halt("GUI Emergency Stop Clicked"),
        )
        self.btn_estop.pack(fill="x")

        # Tool Intake Telemetry Card
        intake_card = ctk.CTkFrame(sidebar, fg_color="#1A212B", corner_radius=10)
        intake_card.pack(fill="x", padx=14, pady=8)

        ctk.CTkLabel(
            intake_card,
            text="EQUIPPED TOOL INTAKE ANALYZER",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#64B5F6",
        ).pack(padx=12, pady=(10, 4), anchor="w")

        self.intake_hit_lbl = ctk.CTkLabel(
            intake_card,
            text="Progress / Hit: Measuring...",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.intake_hit_lbl.pack(padx=12, pady=2, anchor="w")

        self.intake_px_lbl = ctk.CTkLabel(
            intake_card,
            text="Visual Grass Cleared: 0 px/hit  (0.0 hits/s)",
            font=ctk.CTkFont(size=12),
            text_color="#90A4AE",
        )
        self.intake_px_lbl.pack(padx=12, pady=(0, 6), anchor="w")

        # Per-Item / Tool Attack Mode Selector (Spam Click vs Hold to Gather)
        default_item_modes = {
            "Custom / Equipped Tool": "spam_click",
            "Hammer / Sword / Scythe": "spam_click",
            "Bass Cleaver": "spam_click",
            "Thunderclap": "spam_click",
            "Lawnmower / Push Mower": "hold",
            "Weedwacker / Trimmer": "hold",
            "Chainsaw / Buzzsaw": "hold",
            "Flamethrower / Laser / Vacuum": "hold",
        }
        gen_cfg = self.config_data.setdefault("general", {})
        self.item_attack_modes: Dict[str, str] = gen_cfg.setdefault("item_attack_modes", default_item_modes)
        for k_item, v_mode in default_item_modes.items():
            self.item_attack_modes.setdefault(k_item, v_mode)

        selected_item = gen_cfg.get("selected_tool_profile", "Custom / Equipped Tool")
        if selected_item not in self.item_attack_modes:
            selected_item = "Custom / Equipped Tool"

        item_row_frame = ctk.CTkFrame(intake_card, fg_color="transparent")
        item_row_frame.pack(fill="x", padx=12, pady=(2, 4))

        ctk.CTkLabel(
            item_row_frame,
            text="Item:",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="#B0BEC5",
        ).pack(side="left", padx=(0, 6))

        self.tool_profile_menu = ctk.CTkOptionMenu(
            item_row_frame,
            values=list(self.item_attack_modes.keys()),
            height=26,
            font=ctk.CTkFont(size=11),
            command=self._on_tool_profile_selected,
        )
        self.tool_profile_menu.set(selected_item)
        self.tool_profile_menu.pack(side="left", fill="x", expand=True)

        self.attack_mode_seg = ctk.CTkSegmentedButton(
            intake_card,
            values=["⚡ Spam Click [F8]", "🔒 Hold to Gather [F8]"],
            font=ctk.CTkFont(size=12, weight="bold"),
            selected_color="#10B981",
            selected_hover_color="#059669",
            command=self._on_attack_mode_changed,
        )
        cur_mode = str(self.item_attack_modes.get(selected_item, gen_cfg.get("tool_attack_mode", "spam_click"))).lower()
        gen_cfg["tool_attack_mode"] = cur_mode
        self.attack_mode_seg.set("🔒 Hold to Gather [F8]" if cur_mode == "hold" else "⚡ Spam Click [F8]")
        self.attack_mode_seg.pack(fill="x", padx=12, pady=(2, 10))

        # Bottom-Right Multi-Item Percentage ETAs Card
        eta_card = ctk.CTkFrame(sidebar, fg_color="#1A212B", corner_radius=10)
        eta_card.pack(fill="x", padx=14, pady=8)

        ctk.CTkLabel(
            eta_card,
            text="BOTTOM-RIGHT ITEM % & ETAs",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#FFD54F",
        ).pack(padx=12, pady=(10, 6), anchor="w")

        self.item_rows: List[ tuple[ctk.CTkLabel, ctk.CTkProgressBar] ] = []
        for i in range(3):
            lbl = ctk.CTkLabel(
                eta_card,
                text=f"Item #{i+1}: Scanning bottom-right HUD...",
                font=ctk.CTkFont(size=12),
                anchor="w",
            )
            lbl.pack(fill="x", padx=12, pady=(2, 2))
            bar = ctk.CTkProgressBar(eta_card, height=8, progress_color="#FFD54F")
            bar.set(0.0)
            bar.pack(fill="x", padx=12, pady=(0, 8))
            self.item_rows.append((lbl, bar))

        # AI Hyperparameter Tuning Sliders
        tune_card = ctk.CTkFrame(sidebar, fg_color="#1A212B", corner_radius=10)
        tune_card.pack(fill="x", padx=14, pady=6)

        ctk.CTkLabel(
            tune_card,
            text="AI PATHING TUNING",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="#B39DDB",
        ).pack(padx=12, pady=(8, 4), anchor="w")

        self.sweep_lbl = ctk.CTkLabel(
            tune_card,
            text=f"Lawn Sweep Width: {self.config_data.get('mowing', {}).get('sweep_leg_duration_sec', 4.2):.1f}s",
            font=ctk.CTkFont(size=11),
        )
        self.sweep_lbl.pack(padx=12, anchor="w")

        self.sweep_slider = ctk.CTkSlider(
            tune_card,
            from_=1.8,
            to=7.5,
            number_of_steps=57,
            command=self._on_sweep_slider,
        )
        self.sweep_slider.set(float(self.config_data.get("mowing", {}).get("sweep_leg_duration_sec", 4.2)))
        self.sweep_slider.pack(fill="x", padx=12, pady=(2, 8))

        self.door_switch = ctk.CTkSwitch(
            tune_card,
            text="Repel From House Door",
            command=self._on_door_toggle,
        )
        if self.config_data.get("house_door_avoidance", {}).get("enabled", True):
            self.door_switch.select()
        self.door_switch.pack(padx=12, pady=(2, 10), anchor="w")

        # =====================================================================
        # RIGHT PANEL: Live AI Vision Radar + 16-Ray Rose + Spatial Heatmap
        # =====================================================================
        right_panel = ctk.CTkFrame(self, corner_radius=12)
        right_panel.grid(row=0, column=1, padx=(0, 12), pady=12, sticky="nsew")
        right_panel.grid_rowconfigure(1, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        header_bar = ctk.CTkFrame(right_panel, fg_color="transparent")
        header_bar.grid(row=0, column=0, padx=14, pady=(12, 4), sticky="ew")

        ctk.CTkLabel(
            header_bar,
            text="LIVE AI VISION RADAR  •  16-RAY POTENTIAL FIELD  •  SPATIAL OCCUPANCY HEATMAP",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#90CAF9",
        ).pack(side="left")

        self.boss_switch = ctk.CTkSwitch(
            header_bar,
            text="Force Boss Evasion [F9]",
            command=self._on_boss_switch,
        )
        self.boss_switch.pack(side="right")

        self.video_label = ctk.CTkLabel(right_panel, text="Initializing OpenCV AI Vision Stream...")
        self.video_label.grid(row=1, column=0, padx=12, pady=12, sticky="nsew")

    def _on_sweep_slider(self, val: float):
        v = round(float(val), 1)
        self.sweep_lbl.configure(text=f"Lawn Sweep Width: {v:.1f}s")
        self.config_data.setdefault("mowing", {})["sweep_leg_duration_sec"] = v
        self.controller.reload_config(self.config_data)

    def _on_door_toggle(self):
        enabled = bool(self.door_switch.get())
        self.config_data.setdefault("house_door_avoidance", {})["enabled"] = enabled
        self.vision.reload_config(self.config_data)
        self.controller.reload_config(self.config_data)

    def _on_boss_switch(self):
        self.manual_boss_override = bool(self.boss_switch.get())

    def _on_tool_profile_selected(self, item_name: str):
        gen_cfg = self.config_data.setdefault("general", {})
        gen_cfg["selected_tool_profile"] = item_name
        saved_mode = str(self.item_attack_modes.get(item_name, "spam_click")).lower()
        gen_cfg["tool_attack_mode"] = saved_mode
        self.attack_mode_seg.set("🔒 Hold to Gather [F8]" if saved_mode == "hold" else "⚡ Spam Click [F8]")
        self.controller.reload_config(self.config_data)
        try:
            save_config(self.config_data)
        except Exception:
            pass

    def _on_attack_mode_changed(self, seg_val: str):
        is_hold = "Hold" in seg_val
        new_mode = "hold" if is_hold else "spam_click"
        gen_cfg = self.config_data.setdefault("general", {})
        gen_cfg["tool_attack_mode"] = new_mode
        cur_item = self.tool_profile_menu.get()
        self.item_attack_modes[cur_item] = new_mode
        gen_cfg["item_attack_modes"] = self.item_attack_modes
        self.controller._lmb_currently_held = False
        self.controller.reload_config(self.config_data)
        try:
            save_config(self.config_data)
        except Exception:
            pass

    def toggle_attack_mode_hotkey(self):
        gen_cfg = self.config_data.setdefault("general", {})
        cur_mode = str(gen_cfg.get("tool_attack_mode", "spam_click")).lower()
        next_seg = "⚡ Spam Click [F8]" if cur_mode == "hold" else "🔒 Hold to Gather [F8]"
        self.attack_mode_seg.set(next_seg)
        self._on_attack_mode_changed(next_seg)

    def toggle_running(self):
        self.running = not self.running
        if self.running:
            self.controller.ai_brain.sweep_phase_start = time.time()
            if self.config_data.get("general", {}).get("auto_adjust_camera_on_start", True):
                self._needs_camera_init = True
        else:
            self.controller.release_all_keys()

    def trigger_emergency_halt(self, reason: str):
        self.running = False
        self.manual_boss_override = False
        self.controller.release_all_keys()
        self.controller.in_boss_encounter = False
        self.controller.ai_brain.reset()
        print(f"\n[AI MACRO] !!! EMERGENCY STOP ({reason}) — All keys released. Press F6 or Click Start to resume.")

    def _poll_hotkeys_loop(self):
        hk = self.config_data.get("hotkeys", {})
        vk_toggle = VK_MAP.get(hk.get("toggle_start_pause", "f6").lower(), 0x75)
        vk_stop = VK_MAP.get(hk.get("emergency_stop", "f7").lower(), 0x76)
        vk_atk_mode = VK_MAP.get(hk.get("toggle_overlay", "f8").lower(), 0x77)
        vk_boss = VK_MAP.get(hk.get("force_mode_cycle", "f9").lower(), 0x78)

        prev_f6 = False
        prev_f7 = False
        prev_f8 = False
        prev_f9 = False
        last_toggle = 0.0

        while not self.should_exit:
            now = time.time()

            # F7 or END: Instant Emergency Halt (Restartable via F6!)
            cur_f7 = is_vk_down(vk_stop) or is_vk_down(VK_MAP["end"])
            if cur_f7 and not prev_f7:
                self.trigger_emergency_halt("Hardware F7 / END Pressed")
            prev_f7 = cur_f7

            # F6: Start / Resume / Pause
            cur_f6 = is_vk_down(vk_toggle)
            if cur_f6 and not prev_f6 and (now - last_toggle > 0.25):
                last_toggle = now
                self.toggle_running()
            prev_f6 = cur_f6

            # F8: Toggle Current Item between Spam Click and Hold to Gather!
            cur_f8 = is_vk_down(vk_atk_mode)
            if cur_f8 and not prev_f8:
                self.after(0, self.toggle_attack_mode_hotkey)
            prev_f8 = cur_f8

            # F9: Boss Override
            cur_f9 = is_vk_down(vk_boss)
            if cur_f9 and not prev_f9:
                self.manual_boss_override = not self.manual_boss_override
            prev_f9 = cur_f9

            # Top-Left Mouse Corner Panic Halt
            if self.running:
                mx, my = get_cursor_pos()
                if mx <= 2 and my <= 2:
                    self.trigger_emergency_halt("Mouse Corner Failsafe (0,0)")

            time.sleep(0.005)

    def _engine_loop(self):
        target_fps = self.config_data.get("general", {}).get("loop_fps", 20)
        frame_time = 1.0 / float(max(target_fps, 5))

        while not self.should_exit:
            t0 = time.time()
            try:
                analysis = self.vision.analyze_frame(build_debug=True)
                if analysis.raw_frame is not None:
                    self.tracker.submit_frame(analysis.raw_frame)

                if self.manual_boss_override:
                    analysis.boss_active = True

                if self._needs_camera_init and self.running:
                    self._needs_camera_init = False
                    self.controller.adjust_camera_birds_eye(analysis.screen_width, analysis.screen_height)

                self.controller.update(analysis, running=self.running)

                if analysis.debug_frame is not None:
                    rgb = cv2.cvtColor(analysis.debug_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)
                    self._latest_ctk_img = ctk.CTkImage(
                        light_image=pil_img,
                        dark_image=pil_img,
                        size=(740, 440),
                    )
            except Exception:
                pass

            elapsed = time.time() - t0
            sleep_for = frame_time - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _refresh_gui_loop(self):
        if self.should_exit:
            return

        # Update Start/Pause Button & Status Colors
        if self.running:
            self.btn_start.configure(text="⏸ PAUSE AUTOPILOT [F6]", fg_color="#F59E0B", hover_color="#D97706")
            self.state_lbl.configure(text=f"STATE: {self.controller.state.value}", text_color="#2EE59D")
        else:
            self.btn_start.configure(text="▶ START / RESUME [F6]", fg_color="#10B981", hover_color="#059669")
            self.state_lbl.configure(text="STATE: IDLE (Press F6 to Start)", text_color="#FFA726")

        ai_tel = self.controller.latest_ai_telemetry
        self.ai_reason_lbl.configure(text=ai_tel.mode_reason)

        held_str = "".join(sorted(k.upper() for k in self.controller.held_keys)) or "NONE"
        self.keys_lbl.configure(
            text=f"Active Keys: {held_str:<5} | AI Yield Efficiency: {ai_tel.ai_efficiency_score:.0f}%"
        )

        # Update Tool Intake Stats
        snap = self.tracker.get_snapshot()
        if snap.primary_pct_per_hit > 0:
            self.intake_hit_lbl.configure(text=f"Progress / Hit: +{snap.primary_pct_per_hit:.3f}% per swing")
        elif snap.trimmings_per_hit > 0:
            self.intake_hit_lbl.configure(text=f"Trimmings / Hit: +{format_number(snap.trimmings_per_hit)} per swing")
        else:
            self.intake_hit_lbl.configure(text="Progress / Hit: Mowing to calibrate % delta...")

        self.intake_px_lbl.configure(
            text=f"Visual Grass Intake: {snap.visual_grass_px_per_hit:.0f} px/hit  ({snap.hits_per_sec:.1f} hits/s)"
        )

        # Update Bottom-Right Multi-Item Percentage Bars & ETAs + Auto-Register Discovered Items
        menu_updated = False
        for idx, (lbl, bar) in enumerate(self.item_rows):
            if idx < len(snap.upcoming_items):
                it = snap.upcoming_items[idx]
                if it.label and it.label not in self.item_attack_modes and not it.label.startswith("Item #"):
                    low_lbl = it.label.lower()
                    is_hold_item = any(
                        kw in low_lbl for kw in ("mower", "wacker", "trimmer", "saw", "laser", "flame", "vacuum", "beam")
                    )
                    self.item_attack_modes[it.label] = "hold" if is_hold_item else "spam_click"
                    menu_updated = True
                eta_txt = format_eta(it.eta_seconds, it.progress_pct)
                lbl.configure(
                    text=f"{it.label}:  {it.progress_pct:.1f}%   •   ETA: {eta_txt}  ({it.pct_per_sec:.2f}%/s)"
                )
                bar.set(min(1.0, max(0.0, it.progress_pct / 100.0)))
            else:
                lbl.configure(text=f"Item #{idx+1}: Waiting for bottom-right % HUD...")
                bar.set(0.0)

        if menu_updated:
            self.tool_profile_menu.configure(values=list(self.item_attack_modes.keys()))

        # Update Live Vision Radar Frame
        if self._latest_ctk_img is not None:
            self.video_label.configure(image=self._latest_ctk_img, text="")

        self.after(60, self._refresh_gui_loop)

    def _on_close(self):
        self.should_exit = True
        self.running = False
        self.controller.release_all_keys()
        self.tracker.stop()
        try:
            save_config(self.config_data)
        except Exception:
            pass
        self.destroy()


if __name__ == "__main__":
    app = MML2DashboardApp()
    app.mainloop()
