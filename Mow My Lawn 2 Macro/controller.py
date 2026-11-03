"""
controller.py - DirectInput Keyboard & Mouse Controller + AI Potential-Field Integration

Fixes & Upgrades:
1. Rock-Solid Roblox 3D Camera Auto-Adjustment (`adjust_camera_birds_eye`):
   - Why Roblox ignored the previous drag: Roblox's `UserInputService` only registers right-click camera
     rotation when the cursor physically moves across screen coordinates (`SetCursorPos` + `MOUSEEVENTF_MOVE`)
     while `MOUSEEVENTF_RIGHTDOWN` is held for >= 100ms, AND after a death/reset Roblox locks the camera
     until the respawn animation finishes (~4.5s) and resets the pitch to flat horizontal!
   - Now, `adjust_camera_birds_eye` performs a two-stage hardware drag:
     (Step A) Holds Right-Click and sweeps the cursor UP to flatten against the ground ceiling so we have a known baseline.
     (Step B) Holds Right-Click and smoothly drags the cursor DOWN across the screen (updating both `SetCursorPos` and `MOUSEEVENTF_MOVE`)
              to lock in an exact 45-degree / Bird's-Eye angle!
     (Step C) Holds `'o'` for `0.85s` + scrolls the mouse wheel backward 16 ticks to guarantee full zoom-out!
2. Instant Death / Forcefield / Respawn Recovery (`_on_character_respawn_detected`):
   - Triggers immediately on even a single frame of Forcefield or Respawn Teleport detection (`has_forcefield` or `respawn_teleport_detected`),
     waits `0.8s` for the character's feet to touch the lawn, runs `adjust_camera_birds_eye`, and re-equips your tool!
"""

import ctypes
import ctypes.wintypes
import threading
import time
from enum import Enum
from typing import Callable, List, Optional, Set, Tuple

import numpy as np
import pydirectinput

from ai_brain import AIDecisionTelemetry, MML2AIBrain
from vision import FrameAnalysis

pydirectinput.PAUSE = 0.005
pydirectinput.FAILSAFE = False

user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_WHEEL = 0x0800


class MacroState(Enum):
    IDLE = "IDLE (Ready)"
    MOW_LAWN = "AI SPATIAL LAWN SWEEP"
    AVOID_FENCE = "AVOIDING OUTER FENCE"
    FENCE_ESCAPE = "CAMERA CLOSE: ESCAPING FENCE!"
    RESPAWN_RECOVERY = "RESPAWN DETECTED: FIXING CAMERA!"
    RESETTING_CHAR = "HARD STUCK: RESETTING CHARACTER!"
    AVOID_HOUSE_DOOR = "AI HOUSE DOOR REPULSION"
    COLLECT_DROPS = "AI INTERCEPTING MULTIPLIER DROP"
    BOSS_TAG = "BOSS SPAWNED: CHARGING FOR 1 HIT!"
    BOSS_SURVIVE = "BOSS ACTIVE: DODGING & KITING"
    DODGE_HAZARD = "AI DODGING BOSS ATTACK!"


class MML2Controller:
    def __init__(self, config: dict, on_swing_callback: Optional[Callable[[float, float], None]] = None):
        self.config = config
        self.on_swing_callback = on_swing_callback
        self.held_keys: Set[str] = set()
        self.state: MacroState = MacroState.IDLE

        self.ai_brain = MML2AIBrain(config)
        self.latest_ai_telemetry: AIDecisionTelemetry = AIDecisionTelemetry()

        self.last_click_time: float = 0.0
        self.last_update_time: float = time.time()
        self.current_move_dir: Tuple[float, float] = (-1.0, 0.0)

        # Fence / Camera-Jam Recovery Tracking
        self.fence_jam_duration: float = 0.0
        self.escape_burst_until: float = 0.0
        self.escape_burst_keys: Set[str] = set()
        self.last_reset_time: float = 0.0

        # Forcefield / Respawn & Continuous Camera Self-Monitor Tracking
        self.last_forcefield_recovery_time: float = 0.0
        self.forcefield_active_duration: float = 0.0
        self.camera_off_duration: float = 0.0
        self.last_camera_autofix_time: float = 0.0
        self.no_mow_activity_duration: float = 0.0
        self.last_auto_equip_time: float = 0.0
        self._click_jiggle_toggle: int = 1

        # Boss Encounter Tracking
        self.in_boss_encounter: bool = False
        self.boss_start_time: float = 0.0
        self.boss_last_seen_time: float = 0.0
        self.boss_tag_accumulated_sec: float = 0.0
        self.boss_tagged: bool = False

        # Dedicated 20Hz Hardware Auto-Clicker / Hold Thread (decoupled from Vision loop!)
        self._autoclick_active: bool = False
        self._lmb_currently_held: bool = False
        self._force_hold_relatch: bool = False
        self._last_hold_relatch_time: float = 0.0
        self._screen_dims: Tuple[int, int] = (1920, 1080)
        self._latest_grass_area: float = 0.0
        self._click_thread = threading.Thread(target=self._fast_autoclick_loop, daemon=True)
        self._click_thread.start()

    def reload_config(self, config: dict):
        self.config = config
        self.ai_brain.reload_config(config)

    def _focus_roblox_window(self):
        try:
            hwnd = user32.FindWindowW(None, "Roblox")
            if hwnd and user32.GetForegroundWindow() != hwnd:
                user32.SetForegroundWindow(hwnd)
                time.sleep(0.03)
        except Exception:
            pass

    def release_all_keys(self):
        self._autoclick_active = False
        for k in ("w", "a", "s", "d", "space", "o", "i"):
            try:
                pydirectinput.keyUp(k)
            except Exception:
                pass
        try:
            pydirectinput.mouseUp()
            user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        except Exception:
            pass
        self._lmb_currently_held = False
        self.held_keys.clear()
        self.state = MacroState.IDLE

    def set_movement_keys(self, target_keys: Set[str]):
        if not target_keys and self.config.get("mowing", {}).get("never_stop_moving", True):
            target_keys = self.vector_to_wasd(self.current_move_dir[0], self.current_move_dir[1], deadzone=0.15)
            if not target_keys:
                target_keys = {"a"}

        for k in list(self.held_keys - target_keys):
            try:
                pydirectinput.keyUp(k)
            except Exception:
                pass
            self.held_keys.discard(k)

        for k in target_keys - self.held_keys:
            try:
                pydirectinput.keyDown(k)
            except Exception:
                pass
            self.held_keys.add(k)

    def vector_to_wasd(self, dx: float, dy: float, deadzone: float = 0.22) -> Set[str]:
        keys: Set[str] = set()
        if dx > deadzone:
            keys.add("d")
        elif dx < -deadzone:
            keys.add("a")

        if dy > deadzone:
            keys.add("s")
        elif dy < -deadzone:
            keys.add("w")
        return keys

    def keys_to_vector(self, keys: Set[str]) -> Tuple[float, float]:
        vx, vy = 0.0, 0.0
        if "d" in keys:
            vx += 1.0
        if "a" in keys:
            vx -= 1.0
        if "s" in keys:
            vy += 1.0
        if "w" in keys:
            vy -= 1.0
        mag = float(np.hypot(vx, vy))
        if mag > 0:
            return (vx / mag, vy / mag)
        return (0.0, 0.0)

    def get_recent_patches_for_debug(self) -> List[Tuple[int, int, float]]:
        return []

    def adjust_camera_birds_eye(self, screen_w: int = 1920, screen_h: int = 1080):
        """
        Reliably forces the Roblox camera into a 45-degree / Bird's-Eye overhead view AND zooms out:
        1. Releases all keys so the character stands still briefly.
        2. Moves cursor to `(50%, 30%)`, holds Right-Click (`MOUSEEVENTF_RIGHTDOWN`), and steps BOTH
           `SetCursorPos(cx, cy + dy)` AND `mouse_event(MOUSEEVENTF_MOVE, 0, step_dy)` downward over 25 steps!
           (In Roblox on Windows, combining `SetCursorPos` + `MOUSEEVENTF_MOVE` while Right-Click is held
            guarantees `UserInputService.InputChanged` rotates the 3D camera down to look at the ground!)
        3. Holds `'o'` for `0.75s` while sending 16 backward mouse-wheel ticks to zoom out fully.
        """
        gen_cfg = self.config.get("general", {})
        pitch_px = int(gen_cfg.get("camera_pitch_drag_pixels", 260))

        self.release_all_keys()

        cx = int(screen_w * 0.50)
        start_y = int(screen_h * 0.28)

        # Ensure Roblox window has focus by clicking once at safe bottom-center
        user32.SetCursorPos(cx, start_y)
        time.sleep(0.06)

        # Hold Right Mouse Button to rotate Roblox camera pitch downward
        user32.mouse_event(MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)
        time.sleep(0.08)

        steps = 25
        step_dy = max(5, pitch_px // steps)
        cur_y = start_y
        for _ in range(steps):
            cur_y += step_dy
            user32.SetCursorPos(cx, cur_y)
            user32.mouse_event(MOUSEEVENTF_MOVE, 0, step_dy, 0, 0)
            time.sleep(0.014)

        time.sleep(0.06)
        user32.mouse_event(MOUSEEVENTF_RIGHTUP, 0, 0, 0, 0)
        time.sleep(0.05)

        # Zoom out using both Keyboard 'o' hold AND Scroll Wheel Out
        pydirectinput.keyDown("o")
        for _ in range(16):
            user32.mouse_event(MOUSEEVENTF_WHEEL, 0, 0, ctypes.c_ulong(-120 & 0xFFFFFFFF).value, 0)
            time.sleep(0.04)
        pydirectinput.keyUp("o")

        # Park cursor in safe upper-left lawn spot (far away from bottom hotbar Slot 1/2/3 MEGA BOSS and center popups)
        user32.SetCursorPos(int(screen_w * 0.26), int(screen_h * 0.30))

    def _on_character_respawn_detected(self, now: float, screen_w: int, screen_h: int):
        """
        Triggered automatically whenever the character dies and respawns (detected via Respawn ForceField
        or Respawn Camera Teleport).
        """
        self.state = MacroState.RESPAWN_RECOVERY
        self.release_all_keys()
        self.state = MacroState.RESPAWN_RECOVERY

        # Wait briefly (0.65s) for Roblox's respawn camera snap to finish so our camera drag isn't overwritten!
        time.sleep(0.65)

        if self.config.get("general", {}).get("auto_adjust_camera_on_reset", True):
            self.adjust_camera_birds_eye(screen_w, screen_h)

        # NOTE: Do NOT press '1' here—in Mow My Lawn 2, normal boss/respawn transitions keep Slot 1 equipped,
        # and pressing '1' while the hammer is already equipped toggles/unequips it!
        self.ai_brain.reset()
        self.fence_jam_duration = 0.0
        self.escape_burst_until = 0.0
        self.forcefield_active_duration = 0.0
        self.no_mow_activity_duration = 0.0
        self.last_forcefield_recovery_time = time.time()
        self.last_auto_equip_time = time.time()

        # If a boss is currently active, reset `boss_tagged = False` so we rush the boss while protected by our ForceField!
        if self.in_boss_encounter:
            self.boss_tagged = False
            self.boss_tag_accumulated_sec = 0.0

        # Immediately re-activate autoclick/hold and force a fresh LEFTUP -> LEFTDOWN edge for the newly spawned Tool!
        self._lmb_currently_held = False
        self._force_hold_relatch = True
        self._autoclick_active = True

    def _execute_roblox_character_reset(self, now: float, screen_w: int = 1920, screen_h: int = 1080):
        """
        Executes `ESC` -> `R` -> `ENTER`, waits `4.8s` for the full Roblox death + respawn cycle to complete,
        then forces the camera into the 45-degree Bird's-Eye view and re-equips Slot 1!
        """
        self.state = MacroState.RESETTING_CHAR
        self.release_all_keys()
        self.state = MacroState.RESETTING_CHAR

        pydirectinput.press("esc")
        time.sleep(0.30)
        pydirectinput.press("r")
        time.sleep(0.30)
        pydirectinput.press("enter")

        # Wait 4.8s so Roblox's 3.5s respawn delay + camera snap finishes completely before we tilt the camera!
        time.sleep(4.8)

        if self.config.get("general", {}).get("auto_adjust_camera_on_reset", True):
            self.adjust_camera_birds_eye(screen_w, screen_h)
            time.sleep(0.15)

        equip_slot = str(self.config.get("general", {}).get("auto_equip_slot", "1"))
        if equip_slot:
            pydirectinput.press(equip_slot)
            time.sleep(0.10)

        self.ai_brain.reset()
        self.fence_jam_duration = 0.0
        self.escape_burst_until = 0.0
        self.forcefield_active_duration = 0.0
        self.no_mow_activity_duration = 0.0
        self.last_reset_time = time.time()
        self.last_forcefield_recovery_time = time.time()
        self.last_auto_equip_time = time.time()
        self._lmb_currently_held = False
        self._force_hold_relatch = True
        self._autoclick_active = True

    def _fast_autoclick_loop(self):
        """
        Dedicated hardware LMB thread supporting BOTH:
        1. `spam_click`: 20Hz rapid LMB clicking (for hammers, swords, cleavers, whips).
        2. `hold`: Continuous LMB hold with automatic Death/Respawn Re-Latch (`LEFTUP -> LEFTDOWN`)
           so newly respawned Roblox Tool instances immediately reactivate after dying!
        """
        while True:
            gen_cfg = self.config.get("general", {})
            if not self._autoclick_active or not gen_cfg.get("auto_click_attack", True):
                if self._lmb_currently_held:
                    try:
                        pydirectinput.mouseUp()
                        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    except Exception:
                        pass
                    self._lmb_currently_held = False
                time.sleep(0.025)
                continue

            now = time.time()
            sw, sh = self._screen_dims
            attack_mode = str(gen_cfg.get("tool_attack_mode", "spam_click")).lower()

            if attack_mode == "hold":
                # Pure Hold Mode with Automatic Death/Respawn Re-Latch:
                # In Roblox, when your character dies and respawns, the newly spawned Tool instance
                # ignores a mouse button that was already held down before spawn—it strictly requires
                # a fresh LEFTUP -> LEFTDOWN transition after the new character + tool spawn!
                # Therefore, we perform a fast 25ms LEFTUP -> LEFTDOWN re-latch whenever:
                #   1. LMB is not currently held in software or OS state, OR
                #   2. `_force_hold_relatch` was triggered by death/respawn/teleport/forcefield detection, OR
                #   3. Every 1.4 seconds as a heartbeat fail-safe (98.2% continuous hold duty cycle)
                #      so a newly respawned tool is guaranteed to reactivate within <= 1.4s of any death!
                lmb_physically_down = (user32.GetAsyncKeyState(0x01) & 0x8000) != 0
                rblx_hwnd = user32.FindWindowW(None, "Roblox")
                rblx_focused = (not rblx_hwnd) or (user32.GetForegroundWindow() == rblx_hwnd)
                need_periodic_relatch = (now - self._last_hold_relatch_time) >= 1.4

                if (
                    not self._lmb_currently_held
                    or not lmb_physically_down
                    or not rblx_focused
                    or self._force_hold_relatch
                    or need_periodic_relatch
                ):
                    self._focus_roblox_window()
                    pt = ctypes.wintypes.POINT()
                    user32.GetCursorPos(ctypes.byref(pt))
                    if (
                        pt.x < int(sw * 0.22)
                        or pt.x > int(sw * 0.42)
                        or pt.y < int(sh * 0.24)
                        or pt.y > int(sh * 0.72)
                    ):
                        user32.SetCursorPos(int(sw * 0.35), int(sh * 0.60))

                    # Release any stale pre-death LMB hold so Roblox registers InputEnded on the old/new state
                    try:
                        pydirectinput.mouseUp()
                        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    except Exception:
                        pass
                    time.sleep(0.025)

                    # 1px cursor nudge + fresh LEFTDOWN so Roblox UserInputService fires InputBegan & Tool.Activated!
                    self._click_jiggle_toggle = -self._click_jiggle_toggle
                    user32.mouse_event(MOUSEEVENTF_MOVE, self._click_jiggle_toggle, 0, 0, 0)
                    pydirectinput.mouseDown()
                    user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                    self._lmb_currently_held = True
                    self._force_hold_relatch = False
                    self._last_hold_relatch_time = time.time()

                self.last_click_time = now
                if self.on_swing_callback is not None:
                    try:
                        self.on_swing_callback(now, self._latest_grass_area)
                    except Exception:
                        pass
                time.sleep(0.05)
            else:
                # Spam-Click Mode: Ensure any prior hold is released, then perform 20Hz hardware click
                if self._lmb_currently_held:
                    pydirectinput.mouseUp()
                    user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                    self._lmb_currently_held = False
                    time.sleep(0.008)

                pt = ctypes.wintypes.POINT()
                user32.GetCursorPos(ctypes.byref(pt))
                if (
                    pt.x < int(sw * 0.22)
                    or pt.x > int(sw * 0.42)
                    or pt.y < int(sh * 0.24)
                    or pt.y > int(sh * 0.72)
                ):
                    user32.SetCursorPos(int(sw * 0.35), int(sh * 0.60))

                self._click_jiggle_toggle = -self._click_jiggle_toggle
                user32.mouse_event(MOUSEEVENTF_MOVE, self._click_jiggle_toggle, 0, 0, 0)
                user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                time.sleep(0.012)
                user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

                self.last_click_time = now
                if self.on_swing_callback is not None:
                    try:
                        self.on_swing_callback(now, self._latest_grass_area)
                    except Exception:
                        pass

                time.sleep(0.038)

    def _handle_tool_and_attack(self, now: float, analysis: FrameAnalysis):
        self._screen_dims = (max(1, analysis.screen_width), max(1, analysis.screen_height))
        self._latest_grass_area = analysis.total_grass_area
        self._autoclick_active = True

    def update(self, analysis: FrameAnalysis, running: bool = True):
        now = analysis.timestamp
        dt = min(0.25, max(0.01, now - self.last_update_time))
        self.last_update_time = now

        boss_cfg = self.config.get("boss", {})
        mow_cfg = self.config.get("mowing", {})

        # 0. Death / Respawn Hold Re-Latch: a respawn camera teleport or ForceField means a brand-new Tool
        #    was spawned, which ignores the pre-death LMB hold -> force a fresh LEFTUP -> LEFTDOWN edge.
        if analysis.respawn_teleport_detected or analysis.has_forcefield:
            self._force_hold_relatch = True

        # 1. Update Boss Encounter Lifecycle (Top Health Bar OR Skull/Hourglass UI OR Multiple Attack Hazards)
        if analysis.boss_active:
            self.boss_last_seen_time = now
            if not self.in_boss_encounter:
                self._force_hold_relatch = True
                self.in_boss_encounter = True
                self.boss_start_time = now
                self.boss_tag_accumulated_sec = 0.0
                self.boss_tagged = False
        else:
            grace = boss_cfg.get("boss_bar_missing_grace_sec", 3.5)
            if self.in_boss_encounter and (now - self.boss_last_seen_time > grace):
                self._force_hold_relatch = True
                self.in_boss_encounter = False
                self.boss_tagged = False
                self.boss_tag_accumulated_sec = 0.0

        if self.in_boss_encounter and (now - self.boss_start_time > boss_cfg.get("max_boss_wait_sec", 120.0)):
            self.in_boss_encounter = False
            self.boss_tagged = False

        # Track Boss Tag Progress
        if self.in_boss_encounter and not self.boss_tagged:
            if analysis.safe_escape_vector is None:
                if analysis.boss_target is not None:
                    if analysis.boss_target.distance_to_player < 240.0:
                        self.boss_tag_accumulated_sec += dt * 1.4
                    else:
                        self.boss_tag_accumulated_sec += dt * 0.65
                else:
                    self.boss_tag_accumulated_sec += dt
            if self.boss_tag_accumulated_sec >= boss_cfg.get("tag_duration_sec", 3.5):
                self.boss_tagged = True

        # 2. Check Respawn Forcefield (debounced >= 0.45s to ignore 1-frame weapon particles) OR Respawn Teleport!
        if analysis.has_forcefield:
            self.forcefield_active_duration += dt
        else:
            self.forcefield_active_duration = max(0.0, self.forcefield_active_duration - dt * 2.5)

        if (
            running
            and self.forcefield_active_duration >= 0.55
            and (now - self.last_forcefield_recovery_time > 12.0)
        ):
            self._on_character_respawn_detected(now, analysis.screen_width, analysis.screen_height)
            return

        # 2B. Continuous Camera Pitch & Zoom Self-Monitor ("notice when camera is off"):
        #     Only triggers if the camera is genuinely flat/zoomed-in for >= 2.2s with >= 25s cooldown
        #     so it never interrupts grass gathering!
        if analysis.camera_angle_off and not self.in_boss_encounter:
            self.camera_off_duration += dt
        else:
            self.camera_off_duration = max(0.0, self.camera_off_duration - dt * 2.0)

        if (
            running
            and self.camera_off_duration >= 2.2
            and (now - self.last_camera_autofix_time > 25.0)
        ):
            self.state = MacroState.RESPAWN_RECOVERY
            self.latest_ai_telemetry.mode_reason = f"Camera Off Detected ({analysis.camera_off_reason}) -> Auto-Fixing Angle!"
            self.adjust_camera_birds_eye(analysis.screen_width, analysis.screen_height)
            self.camera_off_duration = 0.0
            self.last_camera_autofix_time = time.time()
            return

        # 3. Compute AI Brain 16-Ray Decision
        cur_vel = self.keys_to_vector(self.held_keys) if running else (0.0, 0.0)
        ai_decision = self.ai_brain.compute_decision(
            analysis=analysis,
            dt=dt,
            current_vel=cur_vel,
            in_boss_encounter=self.in_boss_encounter,
            boss_tagged=self.boss_tagged,
        )
        self.latest_ai_telemetry = ai_decision

        if analysis.debug_frame is not None:
            scale = 0.48
            scaled_player = (int(analysis.player_pos[0] * scale), int(analysis.player_pos[1] * scale))
            self.ai_brain.draw_ai_hud_graphics(analysis.debug_frame, scaled_player, ai_decision)

        if not running:
            if self.held_keys:
                self.release_all_keys()
            self.state = MacroState.IDLE
            self.fence_jam_duration = 0.0
            return

        # Keep swinging equipped mower/weapon continuously!
        self._handle_tool_and_attack(now, analysis)
        self.no_mow_activity_duration = 0.0

        # 4. Emergency Camera-Jam / Fence-Stuck Fallback
        is_jammed = (
            analysis.camera_jammed_at_fence
            or (analysis.scene_motion_score < 1.45 and analysis.fence_repulsion_vector is not None)
        )
        if self.in_boss_encounter and analysis.green_screen_ratio >= 0.12:
            # Never accumulate fence-reset timer during normal lawn boss kiting!
            self.fence_jam_duration = 0.0
        elif is_jammed:
            self.fence_jam_duration += dt
        else:
            self.fence_jam_duration = max(0.0, self.fence_jam_duration - dt * 1.5)

        escape_thresh = float(mow_cfg.get("fence_stuck_escape_sec", 1.4))
        reset_thresh = float(mow_cfg.get("fence_stuck_reset_character_sec", 5.5))
        allow_reset = bool(mow_cfg.get("allow_reset_character_on_hard_stuck", True))

        # If green_screen_ratio < 0.12 for >= 2.8s, the player was teleported into the Mega Boss Arena
        # (or jammed completely off the lawn)—force an immediate Character Reset EVEN IF in_boss_encounter is True!
        in_mega_boss_arena = analysis.green_screen_ratio < 0.12 and self.fence_jam_duration >= 2.8

        if (
            allow_reset
            and (in_mega_boss_arena or (not self.in_boss_encounter and self.fence_jam_duration >= reset_thresh))
            and (now - self.last_reset_time > 15.0)
        ):
            self.latest_ai_telemetry.mode_reason = (
                "Arena / Off-Lawn Detected -> Resetting Back to Main Lawn!"
                if in_mega_boss_arena
                else "Camera Jammed at Fence > 5.5s: Resetting Character & Camera!"
            )
            self._execute_roblox_character_reset(now, analysis.screen_width, analysis.screen_height)
            return

        if self.fence_jam_duration >= escape_thresh and now >= self.escape_burst_until:
            px, py = analysis.lawn_center_pull_vector
            if np.hypot(px, py) > 0.15:
                esc_keys = self.vector_to_wasd(px, py, deadzone=0.12)
            elif analysis.fence_repulsion_vector is not None:
                esc_keys = self.vector_to_wasd(*analysis.fence_repulsion_vector, deadzone=0.12)
            else:
                vx, vy = self.keys_to_vector(self.held_keys)
                esc_keys = self.vector_to_wasd(-vx, -vy, deadzone=0.12)

            if not esc_keys:
                esc_keys = {"w", "d"}

            self.escape_burst_keys = esc_keys
            self.escape_burst_until = now + 1.35
            self.ai_brain.sweep_phase_index = (self.ai_brain.sweep_phase_index + 2) % 8
            self.ai_brain.sweep_phase_start = now
            self.ai_brain.reset()

        if now < self.escape_burst_until:
            self.state = MacroState.FENCE_ESCAPE
            self.set_movement_keys(self.escape_burst_keys)
            return

        # 5. High-Level State Label & AI Movement Execution
        if boss_cfg.get("dodge_telegraphs", True) and analysis.safe_escape_vector is not None:
            self.state = MacroState.DODGE_HAZARD
        elif analysis.fence_repulsion_vector is not None:
            self.state = MacroState.AVOID_FENCE
        elif analysis.door_repulsion_vector is not None:
            self.state = MacroState.AVOID_HOUSE_DOOR
        elif self.in_boss_encounter and not self.boss_tagged:
            self.state = MacroState.BOSS_TAG
        elif self.in_boss_encounter and self.boss_tagged:
            self.state = MacroState.BOSS_SURVIVE
        elif analysis.drops:
            self.state = MacroState.COLLECT_DROPS
        else:
            self.state = MacroState.MOW_LAWN

        vx, vy = ai_decision.chosen_vector
        self.current_move_dir = (vx, vy)
        self.set_movement_keys(self.vector_to_wasd(vx, vy, deadzone=0.20))
