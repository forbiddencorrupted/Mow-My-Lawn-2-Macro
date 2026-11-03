"""
vision.py - OpenCV Screen Capture & Detection Engine for Mow My Lawn 2 Macro

Fixes & Upgrades:
1. Unconditional Top Boss UI + Skull & Hourglass Detector (`_detect_boss_bar`):
   - Previously, `_detect_boss_bar` required `red_mask` to match specific red HSV values before it would
     accept the Skull & Hourglass icons—so if the boss health bar was Orange, Yellow, Purple, Dark, or Half-Empty,
     it completely failed to trigger Boss Mode!
   - Now, `_detect_boss_bar` detects ANY non-green Boss Health Bar (Red, Orange, Yellow, Purple, Pink, or Dark Frame)
     OR the presence of the Skull (death counter) & Hourglass (timer) icon/number row below the top meter!
2. Multi-Signal Death / Respawn & Forcefield Detector (`_detect_death_or_forcefield`):
   - Detects character death & respawn via THREE independent signals:
     (A) Glowing Blue/Cyan/Purple/White Respawn ForceField sphere around the character (`min_forcefield_pixel_ratio: 0.065`).
     (B) Sudden Scene Teleport / Respawn Flash (when your character dies at a fence or to a boss and teleports back to the spawn point).
     (C) Red Damage/Death Vignette or sudden weapon unequip on death.
3. Proactive 360-Degree Fence Detection & Boss Attack Dodging.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import mss
import numpy as np

from boss_classifier import BossProfile, WikiBossClassifier


@dataclass
class DetectedObject:
    label: str
    center: Tuple[int, int]       # Screen-space (x, y) within captured monitor
    area: float
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    distance_to_player: float
    priority_score: float = 0.0


@dataclass
class FrameAnalysis:
    timestamp: float
    screen_width: int
    screen_height: int
    player_pos: Tuple[int, int]
    raw_frame: Optional[np.ndarray] = None
    boss_active: bool = False
    boss_bar_fill_ratio: float = 0.0
    boss_skull_hourglass_detected: bool = False
    boss_target: Optional[DetectedObject] = None
    identified_boss: Optional[BossProfile] = None
    identified_boss_confidence: float = 0.0
    has_forcefield: bool = False
    respawn_teleport_detected: bool = False
    camera_angle_off: bool = False
    camera_off_reason: str = ""
    robux_modal_detected: bool = False
    danger_zones: List[DetectedObject] = field(default_factory=list)
    house_obstacles: List[DetectedObject] = field(default_factory=list)
    fence_obstacles: List[DetectedObject] = field(default_factory=list)
    drops: List[DetectedObject] = field(default_factory=list)
    grass_targets: List[DetectedObject] = field(default_factory=list)
    total_grass_area: float = 0.0
    green_screen_ratio: float = 1.0
    mowing_activity_under_player: bool = True
    camera_jammed_at_fence: bool = False
    scene_motion_score: float = 10.0
    safe_escape_vector: Optional[Tuple[float, float]] = None
    door_repulsion_vector: Optional[Tuple[float, float]] = None
    fence_repulsion_vector: Optional[Tuple[float, float]] = None
    lawn_center_pull_vector: Tuple[float, float] = (0.0, 0.0)
    debug_frame: Optional[np.ndarray] = None


class MML2Vision:
    def __init__(self, config: dict):
        self.config = config
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.width = self.monitor["width"]
        self.height = self.monitor["height"]

        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        self._prev_small_gray: Optional[np.ndarray] = None
        self._prev_green_ratio: float = 0.5
        self.boss_classifier = WikiBossClassifier()

        # Async boss-icon matcher state (keeps the vision loop from stalling during boss spawns)
        self._boss_ui_cached: Optional[Tuple[BossProfile, float]] = None
        self._boss_ui_busy: bool = False
        self._boss_ui_last_start: float = 0.0
        self._boss_last_active_time: float = 0.0

    def _maybe_start_boss_ui_match(self, boss_crop: np.ndarray):
        now = time.time()
        if self._boss_ui_cached is not None or self._boss_ui_busy or now - self._boss_ui_last_start < 3.0:
            return
        ch, cw = boss_crop.shape[:2]
        # The boss portrait sits in the horizontal center of the banner; matching only the center 40%
        # cuts the work by ~2.5x, and the worker thread means it never blocks movement either way.
        center = boss_crop[:, int(cw * 0.30) : int(cw * 0.70)]
        self._boss_ui_busy = True
        self._boss_ui_last_start = now

        def _worker(img: np.ndarray):
            try:
                prof, conf = self.boss_classifier.match_boss_in_top_ui(img)
                if prof is not None:
                    self._boss_ui_cached = (prof, conf)
            except Exception:
                pass
            finally:
                self._boss_ui_busy = False

        threading.Thread(target=_worker, args=(center.copy(),), daemon=True).start()

    def reload_config(self, config: dict):
        self.config = config

    def capture_screen(self) -> np.ndarray:
        raw = self.sct.grab(self.monitor)
        img = np.array(raw, dtype=np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _get_roi_pixel_bounds(self, roi_norm: List[float]) -> Tuple[int, int, int, int]:
        x1 = int(roi_norm[0] * self.width)
        y1 = int(roi_norm[1] * self.height)
        x2 = int(roi_norm[2] * self.width)
        y2 = int(roi_norm[3] * self.height)
        return x1, y1, x2, y2

    def analyze_frame(
        self,
        build_debug: bool = True,
        recent_patches: Optional[List[Tuple[int, int, float]]] = None,
        frame: Optional[np.ndarray] = None,
    ) -> FrameAnalysis:
        if frame is None:
            frame = self.capture_screen()
        h, w = frame.shape[:2]
        self.width, self.height = w, h

        reg_cfg = self.config.get("regions", {})
        pc_norm = reg_cfg.get("player_center", [0.50, 0.52])
        player_pos = (int(pc_norm[0] * w), int(pc_norm[1] * h))

        analysis = FrameAnalysis(
            timestamp=time.time(),
            screen_width=w,
            screen_height=h,
            player_pos=player_pos,
            raw_frame=frame,
        )

        # 1. Check Character Respawn Forcefield
        if self.config.get("forcefield_detection", {}).get("enabled", True):
            analysis.has_forcefield = self._detect_character_forcefield(frame, player_pos)

        # 2. Check Top Boss Bar + Skull (Deaths) & Hourglass (Timer) UI + Wiki Boss Icon Matcher
        boss_roi_norm = reg_cfg.get("top_boss_bar_roi", [0.22, 0.01, 0.78, 0.19])
        bx1, by1, bx2, by2 = self._get_roi_pixel_bounds(boss_roi_norm)
        boss_crop = frame[by1:by2, bx1:bx2]
        boss_active, boss_ratio, skull_hg_found = self._detect_boss_bar(boss_crop)
        if boss_active:
            # Masked template matching is very slow (measured ~19s on a full 1075x194 banner crop), which froze
            # the engine loop while movement keys stayed held. Run it on a background worker at most once every
            # 3s per encounter, on a small center crop, and reuse the cached result.
            self._boss_last_active_time = time.time()
            self._maybe_start_boss_ui_match(boss_crop)
            if self._boss_ui_cached is not None:
                analysis.identified_boss, analysis.identified_boss_confidence = self._boss_ui_cached
        elif time.time() - self._boss_last_active_time > 8.0:
            self._boss_ui_cached = None
        analysis.boss_active = boss_active
        analysis.boss_bar_fill_ratio = boss_ratio
        analysis.boss_skull_hourglass_detected = skull_hg_found

        # 3. Extract Play Area ROI
        play_roi_norm = reg_cfg.get("play_area_roi", [0.10, 0.18, 0.90, 0.88])
        px1, py1, px2, py2 = self._get_roi_pixel_bounds(play_roi_norm)
        play_crop = frame[py1:py2, px1:px2]
        play_hsv = cv2.cvtColor(play_crop, cv2.COLOR_BGR2HSV)

        char_mask = np.ones(play_crop.shape[:2], dtype=np.uint8) * 255
        rel_px = player_pos[0] - px1
        rel_py = player_pos[1] - py1
        cv2.circle(char_mask, (rel_px, rel_py), int(min(w, h) * 0.052), 0, -1)

        # Precompute 1/4-scale fast lawn trapezoid + distance transform ONCE per frame (< 3ms total!)
        self._cached_lawn = self._compute_fast_lawn_and_dist(play_hsv)

        # 4. Detect House Door Obstacles
        door_cfg = self.config.get("house_door_avoidance", {})
        if door_cfg.get("enabled", True):
            analysis.house_obstacles, analysis.door_repulsion_vector = self._detect_house_and_door(
                play_hsv, char_mask, (px1, py1), player_pos
            )

        # 5. Proactive 360-Degree Fence Detection & Repulsion
        fence_cfg = self.config.get("fence_avoidance", {})
        if fence_cfg.get("enabled", True):
            analysis.fence_obstacles, analysis.fence_repulsion_vector = self._detect_fences_and_boundaries(
                play_hsv, char_mask, (px1, py1), player_pos
            )

        # 6. Detect Boss Attack Telegraphs & Projectiles + Locate & Classify Boss Entity via Wiki Profiles
        danger_zones, boss_entity = self._detect_telegraphs_and_boss_entity(
            play_crop, play_hsv, char_mask, (px1, py1), player_pos, analysis.boss_active
        )
        analysis.danger_zones = danger_zones
        analysis.boss_target = boss_entity

        if boss_entity is not None and analysis.identified_boss is None:
            bx, by, bw, bh = boss_entity.bbox
            entity_crop = frame[max(0, by) : min(h, by + bh), max(0, bx) : min(w, bx + bw)]
            prof, conf = self.boss_classifier.classify_boss_crop(entity_crop, lawn_only=True)
            if prof is not None:
                analysis.identified_boss = prof
                analysis.identified_boss_confidence = conf
                boss_entity.label = f"{prof.name} (Lv.{prof.level})"
        elif boss_entity is not None and analysis.identified_boss is not None:
            boss_entity.label = f"{analysis.identified_boss.name} (Lv.{analysis.identified_boss.level})"

        # Also: if massive boss attack telegraphs are on the field, ensure boss_active is True!
        if len(danger_zones) >= 2 or sum(d.area for d in danger_zones) > 1400:
            analysis.boss_active = True

        if danger_zones:
            analysis.safe_escape_vector = self._compute_escape_vector(
                player_pos, danger_zones, (px1, py1, px2, py2)
            )

        # 7. Detect Multiplier Drops (Grill Food, Gold/Tokens, Multiplier Orbs)
        if self.config.get("drops_and_multipliers", {}).get("enabled", True):
            analysis.drops = self._detect_drops(
                play_hsv, char_mask, (px1, py1), player_pos, danger_zones, analysis.house_obstacles
            )

        # 8. Detect Unmowed Grass Clusters, Lawn Centroid, & Respawn Scene Teleport
        (
            grass_list,
            total_area,
            center_pull,
            green_ratio,
            cam_jammed,
            motion_score,
            respawn_jump,
            mow_active,
        ) = self._detect_grass_and_lawn_centroid(
            play_crop, play_hsv, char_mask, (px1, py1), player_pos, analysis.house_obstacles
        )
        analysis.grass_targets = grass_list
        analysis.total_grass_area = total_area
        analysis.lawn_center_pull_vector = center_pull
        analysis.green_screen_ratio = green_ratio
        analysis.camera_jammed_at_fence = cam_jammed
        analysis.scene_motion_score = motion_score
        analysis.respawn_teleport_detected = respawn_jump
        analysis.mowing_activity_under_player = mow_active

        # 9. Continuous Camera Pitch & Zoom Self-Monitor ("notice when camera is off")
        cam_off, cam_reason = self._check_if_camera_is_off(frame, play_hsv, green_ratio)
        analysis.camera_angle_off = cam_off
        analysis.camera_off_reason = cam_reason

        if build_debug:
            analysis.debug_frame = self._build_debug_overlay(
                frame, analysis, (bx1, by1, bx2, by2), (px1, py1, px2, py2)
            )

        return analysis

    def _detect_character_forcefield(self, frame: np.ndarray, player_pos: Tuple[int, int]) -> bool:
        """
        Detects if the player character at `player_pos` currently has a Roblox Respawn ForceField
        (glowing cyan/blue/purple/white translucent aura around the avatar).
        """
        ff_cfg = self.config.get("forcefield_detection", {})
        h, w = frame.shape[:2]
        rad = int(min(w, h) * ff_cfg.get("avatar_radius_ratio", 0.095))
        px, py = player_pos

        x1, y1 = max(0, px - rad), max(0, py - rad)
        x2, y2 = min(w, px + rad), min(h, py + rad)
        avatar_crop = frame[y1:y2, x1:x2]
        if avatar_crop.size == 0:
            return False

        hsv = cv2.cvtColor(avatar_crop, cv2.COLOR_BGR2HSV)

        ch, cw = avatar_crop.shape[:2]
        ring_mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.circle(ring_mask, (cw // 2, ch // 2), rad, 255, -1)
        cv2.circle(ring_mask, (cw // 2, ch // 2), int(rad * 0.25), 0, -1)

        b_low = np.array(ff_cfg.get("blue_cyan_hsv_lower", [80, 75, 145]), dtype=np.uint8)
        b_high = np.array(ff_cfg.get("blue_cyan_hsv_upper", [140, 255, 255]), dtype=np.uint8)
        w_low = np.array(ff_cfg.get("shimmer_white_hsv_lower", [0, 0, 215]), dtype=np.uint8)
        w_high = np.array(ff_cfg.get("shimmer_white_hsv_upper", [180, 55, 255]), dtype=np.uint8)

        cyan_mask = cv2.inRange(hsv, b_low, b_high)
        white_mask = cv2.inRange(hsv, w_low, w_high)

        combined = cv2.bitwise_or(cyan_mask, white_mask)
        combined = cv2.bitwise_and(combined, ring_mask)
        cyan_ring = cv2.bitwise_and(cyan_mask, ring_mask)

        ring_area = float(max(1, cv2.countNonZero(ring_mask)))
        ff_ratio = float(cv2.countNonZero(combined)) / ring_area
        cyan_ratio = float(cv2.countNonZero(cyan_ring)) / ring_area

        # A real Roblox Respawn ForceField is a 360-degree sphere surrounding all 4 quadrants of the avatar,
        # whereas a glowing blue hammer or floating white UFO pet only occupies 1-2 quadrants!
        mid_y, mid_x = ch // 2, cw // 2
        quad_ratios = []
        for q_cyan, q_ring in (
            (cyan_ring[:mid_y, :mid_x], ring_mask[:mid_y, :mid_x]),
            (cyan_ring[:mid_y, mid_x:], ring_mask[:mid_y, mid_x:]),
            (cyan_ring[mid_y:, :mid_x], ring_mask[mid_y:, :mid_x]),
            (cyan_ring[mid_y:, mid_x:], ring_mask[mid_y:, mid_x:]),
        ):
            q_area = float(max(1, cv2.countNonZero(q_ring)))
            quad_ratios.append(float(cv2.countNonZero(q_cyan)) / q_area)

        min_quad_cyan = min(quad_ratios) if quad_ratios else 0.0
        return (ff_ratio >= 0.18) and (cyan_ratio >= 0.15) and (min_quad_cyan >= 0.08)

    def _detect_boss_bar(self, boss_crop: np.ndarray) -> Tuple[bool, float, bool]:
        """
        Detects if a Lawn Boss is active at the top of the screen:
        In MML2:
        - Normal Mowing: Top bar is bright GREEN (grass progress meter) and there is NO Skull/Hourglass below it.
        - Boss Active:
          1. A Skull (shared deaths) and Hourglass (countdown timer) with white/gold text appear at the top!
          2. The top bar changes from the normal green grass meter to a Boss Health Bar (Red, Orange, Yellow, Purple, or Dark/Depleted).
        We trigger `boss_active = True` if EITHER the Skull/Hourglass status row is present OR a non-green Boss Health Bar is present!
        """
        if boss_crop.size == 0:
            return False, 0.0, False

        hsv = cv2.cvtColor(boss_crop, cv2.COLOR_BGR2HSV)
        crop_h, crop_w = boss_crop.shape[:2]

        # 1. Check if the normal bright-GREEN Grass Progress Meter is currently occupying the top bar
        green_bar_mask = cv2.inRange(
            hsv, np.array([35, 85, 95], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8)
        )
        green_bar_present = False
        g_cnts, _ = cv2.findContours(green_bar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in g_cnts:
            gx, gy, gw, gh = cv2.boundingRect(cnt)
            if gw > crop_w * 0.14 and (gw / float(max(gh, 1))) > 3.0 and gh >= 5:
                green_bar_present = True
                break

        # 2. Check for ANY Boss Health Bar colors (Red, Orange, Gold, Purple/Magenta)
        red1 = cv2.inRange(hsv, np.array([0, 110, 110], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))
        red2 = cv2.inRange(hsv, np.array([130, 110, 110], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
        boss_bar_mask = cv2.bitwise_or(red1, red2)

        bar_found = False
        fill_ratio = 0.0
        contours, _ = cv2.findContours(boss_bar_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect_ratio = cw / float(max(ch, 1))
            if cw > crop_w * 0.10 and aspect_ratio > 2.5 and ch >= 4:
                bar_found = True
                fill_ratio = max(fill_ratio, min(1.0, cw / float(crop_w)))

        # 3. Check for Skull (Death Count) & Hourglass (Timer) UI Icons + Digits in the top ROI
        #    Note: Only count Skull + Hourglass as an active boss if EITHER the red/orange/purple boss bar
        #    is visible OR the normal green grass progress bar is gone (replaced by a depleted/dark boss bar).
        lower_band = hsv[int(crop_h * 0.30) :, int(crop_w * 0.12) : int(crop_w * 0.88)]
        skull_hourglass_found = False
        if lower_band.size > 0:
            bone_white = cv2.inRange(
                lower_band, np.array([0, 0, 200], dtype=np.uint8), np.array([180, 50, 255], dtype=np.uint8)
            )
            hourglass_gold = cv2.inRange(
                lower_band, np.array([12, 105, 145], dtype=np.uint8), np.array([32, 255, 255], dtype=np.uint8)
            )
            bone_cnts, _ = cv2.findContours(bone_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            gold_cnts, _ = cv2.findContours(hourglass_gold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_bone = [c for c in bone_cnts if 16 <= cv2.contourArea(c) <= 1800]
            valid_gold = [c for c in gold_cnts if 14 <= cv2.contourArea(c) <= 1800]

            # Require both bone-white (Skull/digits) AND gold (Hourglass/timer) clusters, and NOT the normal green grass meter
            if len(valid_bone) >= 2 and len(valid_gold) >= 1 and (bar_found or not green_bar_present):
                skull_hourglass_found = True
                bar_found = True
                if fill_ratio == 0.0:
                    fill_ratio = 0.5

        return bar_found, fill_ratio, skull_hourglass_found

    def _compute_fast_lawn_and_dist(
        self, play_hsv: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, Tuple[int, int]]:
        """
        Computes `raw_green`, `lawn_poly`, `dist_map`, `max_dist`, and `max_loc` in < 3ms by running
        morphology, convex hull, House exclusion, and `cv2.distanceTransform` at 1/4 resolution!
        """
        h, w = play_hsv.shape[:2]
        mow_cfg = self.config.get("mowing", {})
        g_low = np.array(mow_cfg.get("grass_hsv_lower", [32, 20, 45]), dtype=np.uint8)
        g_high = np.array(mow_cfg.get("grass_hsv_upper", [96, 255, 235]), dtype=np.uint8)
        raw_green = cv2.inRange(play_hsv, g_low, g_high)

        sw, sh = max(64, w // 4), max(64, h // 4)
        small_green = cv2.resize(raw_green, (sw, sh), interpolation=cv2.INTER_NEAREST)
        small_hsv = cv2.resize(play_hsv, (sw, sh), interpolation=cv2.INTER_NEAREST)

        closed = cv2.morphologyEx(small_green, cv2.MORPH_CLOSE, self.kernel_small)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly_small = np.zeros_like(small_green)
        if cnts:
            largest = max(cnts, key=cv2.contourArea)
            if cv2.contourArea(largest) > 150:
                hull = cv2.convexHull(largest)
                cv2.drawContours(poly_small, [hull], -1, 255, -1)
            else:
                poly_small = closed

        # Carve out any House / Porch / Dark Roof structure from `poly_small` (with an ~85px safety margin)
        # and zero the outer frame so `cv2.distanceTransform` ALWAYS treats fences, borders, and the House as 0!
        dark_roof = cv2.inRange(
            small_hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 58], dtype=np.uint8)
        )
        white_siding = cv2.inRange(
            small_hsv, np.array([0, 0, 185], dtype=np.uint8), np.array([180, 35, 255], dtype=np.uint8)
        )
        dark_dil = cv2.dilate(dark_roof, self.kernel_small, iterations=2)
        house_edge_mask = cv2.bitwise_or(dark_roof, cv2.bitwise_and(white_siding, dark_dil))
        house_edge_mask = cv2.morphologyEx(house_edge_mask, cv2.MORPH_CLOSE, self.kernel_small)
        h_cnts, _ = cv2.findContours(house_edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        house_carve = np.zeros_like(poly_small)
        for h_cnt in h_cnts:
            if cv2.contourArea(h_cnt) >= 110:
                cv2.drawContours(house_carve, [h_cnt], -1, 255, -1)
        if cv2.countNonZero(house_carve) > 0:
            # Dilate House carve-out by ~85px in full-screen coordinates so grass targets never spawn near the House!
            house_carve = cv2.dilate(house_carve, self.kernel_large, iterations=2)
            poly_small[house_carve > 0] = 0

        poly_small[:1, :] = 0
        poly_small[-1:, :] = 0
        poly_small[:, :1] = 0
        poly_small[:, -1:] = 0

        dist_small = cv2.distanceTransform(poly_small, cv2.DIST_L2, 5) * 4.0
        _, max_dist, _, max_loc_small = cv2.minMaxLoc(dist_small)
        max_loc = (int(max_loc_small[0] * 4), int(max_loc_small[1] * 4))

        lawn_poly = cv2.resize(poly_small, (w, h), interpolation=cv2.INTER_NEAREST)
        dist_map = cv2.resize(dist_small, (w, h), interpolation=cv2.INTER_LINEAR)
        return raw_green, lawn_poly, dist_map, float(max_dist), max_loc

    def _fence_margin(self, max_dist: float) -> float:
        """Fence clearance in pixels, proportional to how big the lawn currently looks on screen."""
        fence_cfg = self.config.get("fence_avoidance", {})
        ratio = float(fence_cfg.get("margin_ratio_of_lawn_radius", 0.30))
        lo = float(fence_cfg.get("min_margin_px", 70))
        hi = float(fence_cfg.get("max_margin_px", 150))
        return float(np.clip(float(max_dist) * ratio, lo, hi))

    def _extract_main_lawn_trapezoid(self, play_hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        cached = getattr(self, "_cached_lawn", None)
        if cached is not None and cached[0].shape == play_hsv.shape[:2]:
            return cached[0], cached[1]
        raw_green, lawn_poly, _, _, _ = self._compute_fast_lawn_and_dist(play_hsv)
        return raw_green, lawn_poly

    def _detect_fences_and_boundaries(
        self,
        play_hsv: np.ndarray,
        char_mask: np.ndarray,
        offset: Tuple[int, int],
        player_pos: Tuple[int, int],
    ) -> Tuple[List[DetectedObject], Optional[Tuple[float, float]]]:
        """
        Uses a 360-degree Euclidean Distance Transform (`cv2.distanceTransform`) on `lawn_poly`
        so that ALL fences—including diagonal fences at any camera rotation—strongly repel the player
        toward the deepest interior point (pole of inaccessibility) of the lawn!
        """
        ox, oy = offset
        px, py = player_pos
        h, w = play_hsv.shape[:2]
        rel_px = int(np.clip(px - ox, 0, w - 1))
        rel_py = int(np.clip(py - oy, 0, h - 1))

        cached = getattr(self, "_cached_lawn", None)
        if cached is not None and cached[0].shape == (h, w):
            _, lawn_poly, dist_map, max_dist, max_loc = cached
        else:
            _, lawn_poly, dist_map, max_dist, max_loc = self._compute_fast_lawn_and_dist(play_hsv)

        if max_dist < 25.0:
            return [], None

        player_fence_dist = float(dist_map[rel_py, rel_px])

        # Margins scale with the on-screen lawn size. The old fixed 220-305px buffer was larger than the
        # whole lawn's half-height at the current zoom (max_dist ~280px), so Fence Guard fired on every
        # frame and the player only ever circled the middle strip.
        safe_fence_dist = self._fence_margin(max_dist)

        rx, ry = 0.0, 0.0
        fence_near = False
        obstacles: List[DetectedObject] = []

        # Probe 8 compass directions (including all 4 diagonals) slightly beyond the margin
        probe_radius = safe_fence_dist * 1.25
        probe_hit = safe_fence_dist * 0.35
        compass_8 = [
            (1.0, 0.0, "fence_E"),
            (0.7071, -0.7071, "fence_NE"),
            (0.0, -1.0, "fence_N"),
            (-0.7071, -0.7071, "fence_NW"),
            (-1.0, 0.0, "fence_W"),
            (-0.7071, 0.7071, "fence_SW"),
            (0.0, 1.0, "fence_S"),
            (0.7071, 0.7071, "fence_SE"),
        ]

        for dx, dy, lbl in compass_8:
            sx = int(np.clip(round(rel_px + dx * probe_radius), 0, w - 1))
            sy = int(np.clip(round(rel_py + dy * probe_radius), 0, h - 1))
            d_probe = float(dist_map[sy, sx])
            if d_probe < probe_hit:
                fence_near = True
                weight = (probe_hit - d_probe) / max(1.0, probe_hit)
                rx -= dx * weight * 5.0
                ry -= dy * weight * 5.0
                obstacles.append(
                    DetectedObject(
                        lbl,
                        (ox + sx, oy + sy),
                        3600.0,
                        (ox + sx - 30, oy + sy - 30, 60, 60),
                        max(1.0, player_fence_dist),
                    )
                )

        if player_fence_dist < safe_fence_dist or fence_near:
            # Pull strongly toward `max_loc` (the true deepest center of the fenced yard at any angle)
            to_cx = float(max_loc[0] - rel_px)
            to_cy = float(max_loc[1] - rel_py)
            c_mag = float(np.hypot(to_cx, to_cy))
            if c_mag > 1.0:
                urgency = max(0.35, (safe_fence_dist - player_fence_dist) / max(1.0, safe_fence_dist))
                rx += (to_cx / c_mag) * (4.0 + urgency * 5.0)
                ry += (to_cy / c_mag) * (4.0 + urgency * 5.0)

            mag = float(np.hypot(rx, ry))
            if mag > 0.05:
                return obstacles, (rx / mag, ry / mag)

        return obstacles, None

    def _detect_house_and_door(
        self,
        play_hsv: np.ndarray,
        char_mask: np.ndarray,
        offset: Tuple[int, int],
        player_pos: Tuple[int, int],
    ) -> Tuple[List[DetectedObject], Optional[Tuple[float, float]]]:
        door_cfg = self.config.get("house_door_avoidance", {})
        obstacles: List[DetectedObject] = []
        ox, oy = offset
        px, py = player_pos

        for z_norm in door_cfg.get("forbidden_screen_zones", []):
            zx1, zy1, zx2, zy2 = self._get_roi_pixel_bounds(z_norm)
            zw, zh = max(1, zx2 - zx1), max(1, zy2 - zy1)
            zcx, zcy = zx1 + zw // 2, zy1 + zh // 2
            dist = float(np.hypot(zcx - px, zcy - py))
            obstacles.append(
                DetectedObject(
                    label="house_door_zone",
                    center=(zcx, zcy),
                    area=float(zw * zh),
                    bbox=(zx1, zy1, zw, zh),
                    distance_to_player=dist,
                )
            )

        if door_cfg.get("avoid_visual_house_and_door", True):
            # In the POV reference screenshot, the House sits at the top edge of the lawn (x: 0.44..0.63)
            # and consists of bright white siding/pillars (S < 35, V > 180) paired directly with a black/dark roof
            # and dark open porch doorways (V < 62), plus the brown shed next to it.
            lower = np.array(door_cfg.get("house_wood_brick_hsv_lower", [5, 65, 45]), dtype=np.uint8)
            upper = np.array(door_cfg.get("house_wood_brick_hsv_upper", [20, 225, 195]), dtype=np.uint8)
            wood_mask = cv2.inRange(play_hsv, lower, upper)

            dark_door_mask = cv2.inRange(
                play_hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 62], dtype=np.uint8)
            )
            white_siding_mask = cv2.inRange(
                play_hsv, np.array([0, 0, 185], dtype=np.uint8), np.array([180, 35, 255], dtype=np.uint8)
            )
            # Combine dark porch doorways/roof with adjacent white porch pillars
            dark_dilated = cv2.dilate(dark_door_mask, self.kernel_large, iterations=2)
            house_porch_mask = cv2.bitwise_or(dark_door_mask, cv2.bitwise_and(white_siding_mask, dark_dilated))

            mask = cv2.bitwise_or(wood_mask, house_porch_mask)
            mask = cv2.bitwise_and(mask, char_mask)
            # Ignore only the center of the lawn so the House is detected whether it sits at the Top,
            # Right, Bottom-Left, or Left edge of the screen when the camera rotates!
            h_crop, w_crop = mask.shape[:2]
            mask[int(h_crop * 0.28) : int(h_crop * 0.72), int(w_crop * 0.24) : int(w_crop * 0.76)] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_large)

            min_area = door_cfg.get("min_house_contour_area", 1400)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cx = ox + x + w // 2
                cy = oy + y + h // 2
                # Measure distance to the NEAREST edge of the House bounding box (not its far interior center!)
                bx1, by1, bx2, by2 = ox + x, oy + y, ox + x + w, oy + y + h
                dx_box = max(bx1 - px, 0, px - bx2)
                dy_box = max(by1 - py, 0, py - by2)
                dist = float(np.hypot(dx_box, dy_box))
                obstacles.append(
                    DetectedObject(
                        label="house_structure",
                        center=(cx, cy),
                        area=area,
                        bbox=(ox + x, oy + y, w, h),
                        distance_to_player=dist,
                    )
                )

        rep_radius = int(door_cfg.get("repulsion_radius_px", 210))
        rx, ry = 0.0, 0.0
        near_door = False
        for ob in obstacles:
            if ob.label == "house_structure" and ob.distance_to_player < rep_radius:
                near_door = True
                bx, by, bw, bh = ob.bbox
                near_x = float(np.clip(px, bx, bx + bw))
                near_y = float(np.clip(py, by, by + bh))
                vx = float(px - near_x)
                vy = float(py - near_y)
                mag = max(1.0, float(np.hypot(vx, vy)))
                weight = (rep_radius - ob.distance_to_player) / rep_radius
                rx += (vx / mag) * weight
                ry += (vy / mag) * weight

        if near_door:
            mag = float(np.hypot(rx, ry))
            if mag > 1e-3:
                return obstacles, (rx / mag, ry / mag)

        return obstacles, None

    def _detect_telegraphs_and_boss_entity(
        self,
        play_crop: np.ndarray,
        play_hsv: np.ndarray,
        char_mask: np.ndarray,
        offset: Tuple[int, int],
        player_pos: Tuple[int, int],
        boss_active: bool,
    ) -> Tuple[List[DetectedObject], Optional[DetectedObject]]:
        boss_cfg = self.config.get("boss", {})
        l1 = np.array(boss_cfg.get("telegraph_red_hsv_lower1", [0, 130, 140]), dtype=np.uint8)
        u1 = np.array(boss_cfg.get("telegraph_red_hsv_upper1", [14, 255, 255]), dtype=np.uint8)
        l2 = np.array(boss_cfg.get("telegraph_red_hsv_lower2", [165, 130, 140]), dtype=np.uint8)
        u2 = np.array(boss_cfg.get("telegraph_red_hsv_upper2", [180, 255, 255]), dtype=np.uint8)

        mask = cv2.bitwise_or(cv2.inRange(play_hsv, l1, u1), cv2.inRange(play_hsv, l2, u2))

        if boss_active:
            o_l = np.array(boss_cfg.get("telegraph_orange_yellow_hsv_lower", [15, 175, 185]), dtype=np.uint8)
            o_u = np.array(boss_cfg.get("telegraph_orange_yellow_hsv_upper", [30, 255, 255]), dtype=np.uint8)
            p_l = np.array(boss_cfg.get("telegraph_purple_magic_hsv_lower", [130, 135, 155]), dtype=np.uint8)
            p_u = np.array(boss_cfg.get("telegraph_purple_magic_hsv_upper", [164, 255, 255]), dtype=np.uint8)
            w_l = np.array(boss_cfg.get("telegraph_white_shock_hsv_lower", [0, 0, 232]), dtype=np.uint8)
            w_u = np.array(boss_cfg.get("telegraph_white_shock_hsv_upper", [180, 35, 255]), dtype=np.uint8)

            mask = cv2.bitwise_or(mask, cv2.inRange(play_hsv, o_l, o_u))
            mask = cv2.bitwise_or(mask, cv2.inRange(play_hsv, p_l, p_u))
            mask = cv2.bitwise_or(mask, cv2.inRange(play_hsv, w_l, w_u))

        mask = cv2.bitwise_and(mask, char_mask)
        # Restrict hazard detection strictly to the interior of the green lawn using precomputed `dist_map`
        cached = getattr(self, "_cached_lawn", None)
        if cached is not None and cached[2].shape == mask.shape:
            mask[cached[2] < 45.0] = 0
        mask[:, : int(play_hsv.shape[1] * 0.14)] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_medium)

        min_area = boss_cfg.get("min_telegraph_area", 240)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ox, oy = offset
        px, py = player_pos
        hazards: List[DetectedObject] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            cx = ox + x + w // 2
            cy = oy + y + h // 2
            dist = float(np.hypot(cx - px, cy - py))
            hazards.append(
                DetectedObject(
                    label="boss_hazard",
                    center=(cx, cy),
                    area=area,
                    bbox=(ox + x, oy + y, w, h),
                    distance_to_player=dist,
                    priority_score=area / max(dist, 1.0),
                )
            )

        hazards.sort(key=lambda d: d.distance_to_player)

        boss_target: Optional[DetectedObject] = None
        if boss_active:
            mow_cfg = self.config.get("mowing", {})
            g_low = np.array(mow_cfg.get("grass_hsv_lower", [32, 45, 45]), dtype=np.uint8)
            g_high = np.array(mow_cfg.get("grass_hsv_upper", [85, 255, 235]), dtype=np.uint8)
            non_green = cv2.bitwise_not(cv2.inRange(play_hsv, g_low, g_high))
            non_green = cv2.bitwise_and(non_green, char_mask)
            bh, bw = non_green.shape[:2]
            margin_x, margin_y = int(bw * 0.14), int(bh * 0.16)
            inner_mask = np.zeros_like(non_green)
            inner_mask[margin_y : bh - margin_y, margin_x : bw - margin_x] = 255
            non_green = cv2.bitwise_and(non_green, inner_mask)
            non_green = cv2.morphologyEx(non_green, cv2.MORPH_OPEN, self.kernel_medium)
            non_green = cv2.morphologyEx(non_green, cv2.MORPH_CLOSE, self.kernel_large)

            b_cnts, _ = cv2.findContours(non_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            best_b_area = 450.0
            for cnt in b_cnts:
                area = cv2.contourArea(cnt)
                if area > best_b_area:
                    best_b_area = area
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    bcx = ox + x + cw // 2
                    bcy = oy + y + ch // 2
                    bdist = float(np.hypot(bcx - px, bcy - py))
                    boss_target = DetectedObject(
                        label="LAWN_BOSS",
                        center=(bcx, bcy),
                        area=area,
                        bbox=(ox + x, oy + y, cw, ch),
                        distance_to_player=bdist,
                        priority_score=area,
                    )

        return hazards, boss_target

    def _compute_escape_vector(
        self,
        player_pos: Tuple[int, int],
        danger_zones: List[DetectedObject],
        play_bounds: Tuple[int, int, int, int],
    ) -> Optional[Tuple[float, float]]:
        boss_cfg = self.config.get("boss", {})
        danger_radius = boss_cfg.get("danger_radius_px", 340)
        px, py = player_pos

        repulse_x = 0.0
        repulse_y = 0.0
        threat_found = False

        for hz in danger_zones:
            hz_radius = np.sqrt(hz.area / np.pi)
            effective_dist = max(1.0, hz.distance_to_player - hz_radius)

            if effective_dist < danger_radius:
                threat_found = True
                vx = px - hz.center[0]
                vy = py - hz.center[1]
                norm = np.hypot(vx, vy)
                if norm < 1e-3:
                    vx, vy, norm = 1.0, 0.0, 1.0
                perp_x, perp_y = -vy / norm, vx / norm
                weight = ((danger_radius - effective_dist) / danger_radius) ** 2 * (1.0 + hz.area / 1500.0)
                repulse_x += ((vx / norm) * 0.75 + perp_x * 0.25) * weight
                repulse_y += ((vy / norm) * 0.75 + perp_y * 0.25) * weight

        if not threat_found:
            return None

        mag = np.hypot(repulse_x, repulse_y)
        if mag < 1e-4:
            return (1.0, 0.0)

        return (repulse_x / mag, repulse_y / mag)

    def _is_inside_house_zone(self, cx: int, cy: int, house_obstacles: List[DetectedObject], margin: int = 35) -> bool:
        for ob in house_obstacles:
            ox, oy, ow, oh = ob.bbox
            if (ox - margin) <= cx <= (ox + ow + margin) and (oy - margin) <= cy <= (oy + oh + margin):
                return True
        return False

    def _detect_drops(
        self,
        play_hsv: np.ndarray,
        char_mask: np.ndarray,
        offset: Tuple[int, int],
        player_pos: Tuple[int, int],
        danger_zones: List[DetectedObject],
        house_obstacles: List[DetectedObject],
    ) -> List[DetectedObject]:
        drop_cfg = self.config.get("drops_and_multipliers", {})
        min_area = drop_cfg.get("min_drop_area", 80)
        max_area = drop_cfg.get("max_drop_area", 6500)
        profiles = drop_cfg.get("color_profiles", {})

        ox, oy = offset
        px, py = player_pos
        ph, pw = play_hsv.shape[:2]
        mow_cfg = self.config.get("mowing", {})
        g_low = np.array(mow_cfg.get("grass_hsv_lower", [32, 28, 45]), dtype=np.uint8)
        g_high = np.array(mow_cfg.get("grass_hsv_upper", [85, 255, 235]), dtype=np.uint8)
        lawn_green_mask = cv2.inRange(play_hsv, g_low, g_high)
        detected: List[DetectedObject] = []

        for label, bounds in profiles.items():
            lower = np.array(bounds["hsv_lower"], dtype=np.uint8)
            upper = np.array(bounds["hsv_upper"], dtype=np.uint8)
            mask = cv2.inRange(play_hsv, lower, upper)
            mask = cv2.bitwise_and(mask, char_mask)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel_small)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel_medium)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < min_area or area > max_area:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                aspect = max(w, h) / float(max(min(w, h), 1))
                if aspect > 3.2:
                    continue

                rel_cx = x + w // 2
                rel_cy = y + h // 2
                # Never chase anything in the upper 26% of the play area (near House / Arena Wall)
                if rel_cy < int(ph * 0.26):
                    continue

                cx = ox + rel_cx
                cy = oy + rel_cy
                dist = float(np.hypot(cx - px, cy - py))

                # Ignore anything within 195px of player_pos (eliminates chasing our own extended weapon tip!)
                if dist < 195.0 or dist > 460.0:
                    continue

                # Verify the object is sitting on pure green lawn (not part of another player's avatar or fence)
                bx1, by1 = max(0, rel_cx - 42), max(0, rel_cy - 42)
                bx2, by2 = min(pw, rel_cx + 42), min(ph, rel_cy + 42)
                surround = lawn_green_mask[by1:by2, bx1:bx2]
                if surround.size == 0 or (float(cv2.countNonZero(surround)) / float(surround.size)) < 0.72:
                    continue

                if self._is_inside_house_zone(cx, cy, house_obstacles, margin=55):
                    continue

                unsafe = False
                for hz in danger_zones:
                    hz_rad = np.sqrt(hz.area / np.pi) + 75
                    if np.hypot(cx - hz.center[0], cy - hz.center[1]) < hz_rad:
                        unsafe = True
                        break
                if unsafe:
                    continue

                dist = float(np.hypot(cx - px, cy - py))
                type_bonus = 1.5 if "food" in label or "multiplier" in label else 1.2
                score = (type_bonus * 1000.0) / max(dist, 25.0)

                detected.append(
                    DetectedObject(
                        label=label,
                        center=(cx, cy),
                        area=area,
                        bbox=(ox + x, oy + y, w, h),
                        distance_to_player=dist,
                        priority_score=score,
                    )
                )

        detected.sort(key=lambda d: d.priority_score, reverse=True)
        return detected

    def _detect_grass_and_lawn_centroid(
        self,
        play_crop: np.ndarray,
        play_hsv: np.ndarray,
        char_mask: np.ndarray,
        offset: Tuple[int, int],
        player_pos: Tuple[int, int],
        house_obstacles: List[DetectedObject],
    ) -> Tuple[List[DetectedObject], float, Tuple[float, float], float, bool, float, bool]:
        mow_cfg = self.config.get("mowing", {})
        cached = getattr(self, "_cached_lawn", None)
        if cached is not None and cached[0].shape == play_hsv.shape[:2]:
            raw_green, lawn_poly, dist_map, max_dist, max_loc = cached
        else:
            raw_green, lawn_poly, dist_map, max_dist, max_loc = self._compute_fast_lawn_and_dist(play_hsv)
        green_mask = cv2.bitwise_and(raw_green, char_mask)

        total_play_px = float(max(1, green_mask.shape[0] * green_mask.shape[1]))
        green_ratio = float(cv2.countNonZero(green_mask)) / total_play_px

        gh, gw = green_mask.shape[:2]
        top_green = float(cv2.countNonZero(green_mask[: gh // 2, :])) / (total_play_px * 0.5)
        bot_green = float(cv2.countNonZero(green_mask[gh // 2 :, :])) / (total_play_px * 0.5)
        left_green = float(cv2.countNonZero(green_mask[:, : gw // 2])) / (total_play_px * 0.5)
        right_green = float(cv2.countNonZero(green_mask[:, gw // 2 :])) / (total_play_px * 0.5)

        ox, oy = offset
        px, py = player_pos
        rel_px = int(np.clip(px - ox, 0, gw - 1))
        rel_py = int(np.clip(py - oy, 0, gh - 1))
        player_fence_dist = float(dist_map[rel_py, rel_px])
        safe_margin = self._fence_margin(max_dist)

        center_pull = (0.0, 0.0)
        if max_dist > 40.0:
            lawn_cx = ox + int(max_loc[0])
            lawn_cy = oy + int(max_loc[1])
            dx = float(lawn_cx - px)
            dy = float(lawn_cy - py)
            dist = float(np.hypot(dx, dy))
            # Only pull toward the center when approaching the fence margin (or very far off-center),
            # so the player is free to sweep across the entire safe interior of the lawn!
            if player_fence_dist < safe_margin * 1.35 and dist > 45.0:
                strength = min(1.0, (safe_margin * 1.35 - player_fence_dist) / max(1.0, safe_margin))
                center_pull = ((dx / dist) * strength, (dy / dist) * strength)
            elif dist > max(210.0, float(max_dist) * 0.85):
                strength = min(0.65, (dist - 210.0) / 180.0)
                center_pull = ((dx / dist) * strength, (dy / dist) * strength)
        else:
            qx = right_green - left_green
            qy = bot_green - top_green
            qmag = max(1e-3, float(np.hypot(qx, qy)))
            center_pull = (qx / qmag, qy / qmag)

        gray = cv2.cvtColor(play_crop, cv2.COLOR_BGR2GRAY)

        small_gray = cv2.resize(gray, (96, 64))
        motion_score = 10.0
        if self._prev_small_gray is not None and self._prev_small_gray.shape == small_gray.shape:
            diff = cv2.absdiff(small_gray, self._prev_small_gray)
            diff[22:42, 36:60] = 0
            motion_score = float(np.mean(diff))
        self._prev_small_gray = small_gray

        respawn_jump = (motion_score > 42.0) or (abs(green_ratio - self._prev_green_ratio) > 0.32 and motion_score > 25.0)
        self._prev_green_ratio = green_ratio

        zoom_thresh = float(mow_cfg.get("camera_zoomed_in_green_ratio_threshold", 0.14))
        edge_collapsed = min(top_green, bot_green, left_green, right_green) < 0.08 and max(
            top_green, bot_green, left_green, right_green
        ) > 0.30
        camera_jammed = (green_ratio < zoom_thresh) or (edge_collapsed and motion_score < 2.0)

        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        abs_grad = cv2.addWeighted(cv2.convertScaleAbs(grad_x), 0.5, cv2.convertScaleAbs(grad_y), 0.5, 0)
        _, step_edges = cv2.threshold(abs_grad, 15, 255, cv2.THRESH_BINARY)

        # Define `safe_lawn` using proportional `_fence_margin(max_dist)` (~85-130px from any fence or House)
        # so the entire safe interior of the yard is harvested while still keeping a strict buffer from fences!
        safe_lawn = np.where(dist_map >= safe_margin, 255, 0).astype(np.uint8)
        safe_lawn = cv2.bitwise_and(safe_lawn, raw_green)
        safe_lawn = cv2.bitwise_and(safe_lawn, char_mask)

        interior_lawn = np.where(dist_map >= 25.0, 255, 0).astype(np.uint8)
        mowed_steps = cv2.bitwise_and(step_edges, interior_lawn)
        mowed_steps = cv2.bitwise_and(mowed_steps, char_mask)

        local_cut_roi = mowed_steps[
            max(0, rel_py - 115) : min(gh, rel_py + 115),
            max(0, rel_px - 115) : min(gw, rel_px + 115),
        ]
        mowing_active = cv2.countNonZero(local_cut_roi) >= 18

        # Score 96x96 tiles (stride 72px) across `safe_lawn`:
        # - Every full green tile inside `safe_lawn` (outside the 105px player swing circle and House margin)
        #   is a valid mowing target, with a bonus for 3D grass layer steps (`step_px`)!
        tile_sz = 96
        stride = 72
        min_safe_px = int((tile_sz * tile_sz) * 0.42)

        patches: List[DetectedObject] = []
        for ty in range(20, gh - tile_sz, stride):
            for tx in range(20, gw - tile_sz, stride):
                safe_tile = safe_lawn[ty : ty + tile_sz, tx : tx + tile_sz]
                safe_px = cv2.countNonZero(safe_tile)
                if safe_px < min_safe_px:
                    continue

                step_tile = mowed_steps[ty : ty + tile_sz, tx : tx + tile_sz]
                step_px = cv2.countNonZero(step_tile)

                rel_cx = tx + tile_sz // 2
                rel_cy = ty + tile_sz // 2

                tile_fence_dist = float(dist_map[int(np.clip(rel_cy, 0, gh - 1)), int(np.clip(rel_cx, 0, gw - 1))])
                if tile_fence_dist < safe_margin * 0.92:
                    continue

                cx = ox + rel_cx
                cy = oy + rel_cy
                if self._is_inside_house_zone(cx, cy, house_obstacles, margin=45):
                    continue

                dist = float(np.hypot(cx - px, cy - py))
                if dist < 105.0:
                    continue

                # Sweet-spot forward lookahead (~200px away) + bonus for 3D grass layer steps (`step_px * 2.2`)
                lookahead_bell = float(np.exp(-(((dist - 200.0) / 150.0) ** 2)))
                center_safety = min(1.30, tile_fence_dist / max(1.0, safe_margin * 1.4))
                yield_val = float(safe_px) + float(step_px) * 2.2
                score = yield_val * (0.55 + 0.65 * lookahead_bell) * center_safety

                patches.append(
                    DetectedObject(
                        label="grass",
                        center=(cx, cy),
                        area=float(safe_px + step_px * 2),
                        bbox=(ox + tx, oy + ty, tile_sz, tile_sz),
                        distance_to_player=dist,
                        priority_score=score,
                    )
                )

        total_grass_px = float(cv2.countNonZero(safe_lawn))
        patches.sort(key=lambda g: g.priority_score, reverse=True)
        return patches, total_grass_px, center_pull, green_ratio, camera_jammed, motion_score, respawn_jump, mowing_active

    def _check_if_camera_is_off(
        self, frame: np.ndarray, play_hsv: np.ndarray, green_ratio: float
    ) -> Tuple[bool, str]:
        """
        Continuously monitors if the Roblox camera angle/zoom is "off" (i.e. NOT in the 45-degree
        overhead bird's-eye view):
        1. Flat / Horizontal Pitch Check:
           When the camera resets to flat horizontal after dying/respawning, the upper 35% of the screen
           is filled with Sky (bright blue/cyan/white/gray skybox) or distant background walls instead of
           green lawn! In a true 45-degree bird's-eye view, the upper half of `play_hsv` has substantial green lawn.
        2. Over-Zoomed Check:
           If the player's avatar fills the center or `green_ratio < 0.23`, the camera is zoomed in too close.
        """
        ph, pw = play_hsv.shape[:2]
        if ph == 0 or pw == 0:
            return False, ""

        mow_cfg = self.config.get("mowing", {})
        g_low = np.array(mow_cfg.get("grass_hsv_lower", [32, 45, 45]), dtype=np.uint8)
        g_high = np.array(mow_cfg.get("grass_hsv_upper", [85, 255, 235]), dtype=np.uint8)

        # Check green lawn presence in the upper 38% of the play area vs lower 50%
        upper_strip = play_hsv[: int(ph * 0.38), int(pw * 0.15) : int(pw * 0.85)]
        lower_strip = play_hsv[int(ph * 0.50) :, int(pw * 0.15) : int(pw * 0.85)]

        upper_green = float(cv2.countNonZero(cv2.inRange(upper_strip, g_low, g_high))) / float(
            max(1, upper_strip.shape[0] * upper_strip.shape[1])
        )
        lower_green = float(cv2.countNonZero(cv2.inRange(lower_strip, g_low, g_high))) / float(
            max(1, lower_strip.shape[0] * lower_strip.shape[1])
        )

        # Check skybox presence in the top 25% of the full screen (blue/cyan/white sky when looking horizontally)
        fh, fw = frame.shape[:2]
        sky_band = cv2.cvtColor(frame[int(fh * 0.05) : int(fh * 0.28), int(fw * 0.15) : int(fw * 0.85)], cv2.COLOR_BGR2HSV)
        sky_blue = cv2.inRange(sky_band, np.array([85, 35, 145], dtype=np.uint8), np.array([130, 255, 255], dtype=np.uint8))
        sky_bright = cv2.inRange(sky_band, np.array([0, 0, 205], dtype=np.uint8), np.array([180, 40, 255], dtype=np.uint8))
        sky_ratio = float(cv2.countNonZero(cv2.bitwise_or(sky_blue, sky_bright))) / float(
            max(1, sky_band.shape[0] * sky_band.shape[1])
        )

        # Case A: Upper view has almost no grass (< 6%) while sky is filling the top (> 55%)
        # -> Camera pitch is completely flat/horizontal instead of 45-degree bird's-eye!
        if (upper_green < 0.06 and lower_green > 0.25) or (sky_ratio > 0.55 and upper_green < 0.10):
            return True, f"Camera Pitch Flat (Upper Grass {upper_green*100:.0f}%, Sky {sky_ratio*100:.0f}%)"

        # Case B: Overall green lawn visibility collapsed (< 10%) -> Camera zoomed in too close!
        if green_ratio < 0.10:
            return True, f"Camera Zoomed In Too Close (Lawn {green_ratio*100:.0f}%)"

        return False, ""

    def _build_debug_overlay(
        self,
        frame: np.ndarray,
        analysis: FrameAnalysis,
        boss_rect: Tuple[int, int, int, int],
        play_rect: Tuple[int, int, int, int],
    ) -> np.ndarray:
        vis = frame.copy()

        bx1, by1, bx2, by2 = boss_rect
        px1, py1, px2, py2 = play_rect
        boss_box_col = (0, 0, 255) if analysis.boss_active else (180, 180, 0)
        cv2.rectangle(vis, (bx1, by1), (bx2, by2), boss_box_col, 2)
        if analysis.boss_active:
            lbl = "BOSS BAR + SKULL/HOURGLASS ACTIVE!" if analysis.boss_skull_hourglass_detected else "BOSS BAR ACTIVE!"
            cv2.putText(vis, lbl, (bx1, max(18, by1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.rectangle(vis, (px1, py1), (px2, py2), (120, 120, 120), 1)

        for fb in analysis.fence_obstacles:
            fx, fy, fw, fh = fb.bbox
            cv2.rectangle(vis, (fx, fy), (fx + fw, fy + fh), (0, 140, 255), 2)

        for ob in analysis.house_obstacles:
            ox, oy, ow, oh = ob.bbox
            cv2.rectangle(vis, (ox, oy), (ox + ow, oy + oh), (200, 80, 255), 2)

        if analysis.has_forcefield:
            cv2.circle(vis, analysis.player_pos, 48, (255, 255, 0), 3)
            cv2.putText(
                vis,
                "RESPAWN / FORCEFIELD DETECTED -> ADJUSTING CAMERA!",
                (analysis.player_pos[0] - 160, analysis.player_pos[1] - 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
        else:
            cv2.circle(vis, analysis.player_pos, 10, (255, 255, 255), 2)

        if analysis.boss_target is not None:
            bx, by, bw, bh = analysis.boss_target.bbox
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (255, 0, 255), 3)
            cv2.putText(
                vis,
                f"BOSS: {analysis.boss_target.label}",
                (bx, max(20, by - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 255),
                2,
            )
        elif analysis.identified_boss is not None:
            cv2.putText(
                vis,
                f"WIKI BOSS IDENTIFIED: {analysis.identified_boss.name} (Lv.{analysis.identified_boss.level})",
                (bx1, by2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 100, 255),
                2,
            )

        if analysis.camera_angle_off:
            cv2.putText(
                vis,
                f"CAMERA OFF: {analysis.camera_off_reason} -> AUTO-FIXING!",
                (px1 + 10, py1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
            )

        for idx, g in enumerate(analysis.grass_targets[:5]):
            x, y, w, h = g.bbox
            color = (0, 255, 0) if idx == 0 else (0, 160, 0)
            cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)

        for idx, d in enumerate(analysis.drops[:5]):
            x, y, w, h = d.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 215, 255), 2)
            cv2.putText(
                vis,
                f"DROP: {d.label}",
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 215, 255),
                2,
            )

        for hz in analysis.danger_zones:
            x, y, w, h = hz.bbox
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(
                vis,
                "ATTACK HAZARD!",
                (x, max(20, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
            )

        if analysis.safe_escape_vector is not None:
            vx, vy = analysis.safe_escape_vector
            end_pt = (
                int(analysis.player_pos[0] + vx * 160),
                int(analysis.player_pos[1] + vy * 160),
            )
            cv2.arrowedLine(vis, analysis.player_pos, end_pt, (255, 255, 0), 4)

        scale = 0.48
        small = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))
        return small
