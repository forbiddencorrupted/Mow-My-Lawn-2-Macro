"""
ai_brain.py - Adaptive Spatial Potential-Field & Multi-Ray Lawn Yield AI Engine

Upgrades:
1. 360-Degree Fence Repulsion Ring:
   - Outer border cells of the 20x14 spatial grid are penalized and `analysis.fence_repulsion_vector`
     + `analysis.lawn_center_pull_vector` apply strong inward forces so the AI stays strictly in the
     safe middle of the lawn and avoids fences altogether!
2. Boss Tagging & Attack Evasion Vectors:
   - During `BOSS_TAG` (when a boss spawns and hasn't been hit yet), the AI steers directly toward
     `analysis.boss_target` (or sweeps across the center arena) to guarantee landing at least 1 hit!
   - Whenever `analysis.safe_escape_vector` is active (or during `BOSS_SURVIVE`), the AI combines
     radial + tangential evasion with strong fence repulsion so dodging a boss attack NEVER pushes
     the player into a fence!
"""

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from vision import FrameAnalysis


@dataclass
class AIDecisionTelemetry:
    chosen_vector: Tuple[float, float] = (1.0, 0.0)
    chosen_angle_deg: float = 0.0
    confidence: float = 1.0
    mode_reason: str = "Initializing AI Engine..."
    ray_scores: List[Tuple[float, float, float]] = field(default_factory=list)
    heatmap_grid: Optional[np.ndarray] = None
    ai_efficiency_score: float = 100.0
    active_exploration_weight: float = 0.46


class MML2AIBrain:
    def __init__(self, config: dict):
        self.config = config
        self.grid_cols = 20
        self.grid_rows = 14

        self.occupancy_grid = np.ones((self.grid_rows, self.grid_cols), dtype=np.float32)
        self.subpixel_shift_x = 0.0
        self.subpixel_shift_y = 0.0

        self.num_rays = 16
        self.ray_dirs: List[Tuple[float, float]] = [
            (math.cos(2.0 * math.pi * i / self.num_rays), math.sin(2.0 * math.pi * i / self.num_rays))
            for i in range(self.num_rays)
        ]

        self.heading_x: float = -1.0
        self.heading_y: float = 0.0
        self.sweep_phase_index: int = 0
        self.sweep_phase_start: float = time.time()

        self.exploration_weight: float = 0.45
        self.last_tune_time: float = time.time()
        self.epoch_grass_cleared: float = 0.0
        self.best_epoch_rate: float = 1.0

    def reload_config(self, config: dict):
        self.config = config

    def reset(self):
        self.occupancy_grid.fill(1.0)
        self.sweep_phase_start = time.time()

    def _update_spatial_grid(self, analysis: FrameAnalysis, dt: float, current_vel: Tuple[float, float]):
        mow_cfg = self.config.get("mowing", {})
        regrowth_sec = max(3.0, float(mow_cfg.get("patch_cooldown_sec", 5.0)))

        cells_per_sec_x = 3.2
        cells_per_sec_y = 2.6
        self.subpixel_shift_x -= current_vel[0] * cells_per_sec_x * dt
        self.subpixel_shift_y -= current_vel[1] * cells_per_sec_y * dt

        shift_cols = int(round(self.subpixel_shift_x))
        shift_rows = int(round(self.subpixel_shift_y))

        if shift_cols != 0:
            self.subpixel_shift_x -= shift_cols
            self.occupancy_grid = np.roll(self.occupancy_grid, shift_cols, axis=1)
            if shift_cols > 0:
                self.occupancy_grid[:, :shift_cols] = 1.1
            else:
                self.occupancy_grid[:, shift_cols:] = 1.1

        if shift_rows != 0:
            self.subpixel_shift_y -= shift_rows
            self.occupancy_grid = np.roll(self.occupancy_grid, shift_rows, axis=0)
            if shift_rows > 0:
                self.occupancy_grid[:shift_rows, :] = 1.1
            else:
                self.occupancy_grid[shift_rows:, :] = 1.1

        self.occupancy_grid = np.clip(self.occupancy_grid + (dt / regrowth_sec), 0.0, 1.35)

        # Penalize outer perimeter ring of the grid so the AI prefers the interior lawn away from fences & house door
        self.occupancy_grid[0:2, :] *= 0.25
        self.occupancy_grid[-2:, :] *= 0.25
        self.occupancy_grid[:, 0:2] *= 0.25
        self.occupancy_grid[:, -2:] *= 0.25

        if self.config.get("house_door_avoidance", {}).get("enabled", True):
            self.occupancy_grid[0:3, 6:14] = 0.0

        cx = self.grid_cols // 2
        cy = self.grid_rows // 2
        y_indices, x_indices = np.ogrid[: self.grid_rows, : self.grid_cols]
        dist_sq = (x_indices - cx) ** 2 + (y_indices - cy) ** 2
        self.occupancy_grid[dist_sq <= 2.4] = 0.0

        w = max(1, analysis.screen_width)
        h = max(1, analysis.screen_height)
        for g in analysis.grass_targets[:12]:
            gx = int(np.clip((g.center[0] / w) * self.grid_cols, 0, self.grid_cols - 1))
            gy = int(np.clip((g.center[1] / h) * self.grid_rows, 0, self.grid_rows - 1))
            # Never overwrite cells that were just mowed in the last ~2.2 seconds (`< 0.45`)!
            if (gx - cx) ** 2 + (gy - cy) ** 2 > 3.2 and self.occupancy_grid[gy, gx] >= 0.45:
                bonus = min(0.85, g.area / 2000.0)
                self.occupancy_grid[gy, gx] = min(1.95, self.occupancy_grid[gy, gx] + bonus)

    def _get_macro_sweep_bias(self, now: float) -> Tuple[float, float]:
        """
        Compact, tight center-lawn figure-8 sweep (`1.55s` legs) that stays inside the safe middle
        of the lawn and turns back long before reaching any outer fence!
        """
        leg_dur = float(self.config.get("mowing", {}).get("sweep_leg_duration_sec", 1.55))
        shift_dur = float(self.config.get("mowing", {}).get("sweep_shift_duration_sec", 0.75))

        compass_cycle = [
            (-1.0, 0.0, leg_dur),          # Sweep Left across center
            (-0.35, 0.93, shift_dur),      # Step Down-Left
            (1.0, 0.0, leg_dur),           # Sweep Right across center
            (0.35, -0.93, shift_dur),      # Step Up-Right
            (-0.92, 0.38, leg_dur),        # Diagonal Left-Down
            (0.35, -0.93, shift_dur),      # Step Up-Right
            (0.92, -0.38, leg_dur),        # Diagonal Right-Up
            (-0.35, 0.93, shift_dur),      # Step Down-Left
        ]

        vx, vy, dur = compass_cycle[self.sweep_phase_index % len(compass_cycle)]
        if now - self.sweep_phase_start >= dur:
            self.sweep_phase_index = (self.sweep_phase_index + 1) % len(compass_cycle)
            self.sweep_phase_start = now
            vx, vy, _ = compass_cycle[self.sweep_phase_index]

        mag = math.hypot(vx, vy)
        return (vx / mag, vy / mag)

    def compute_decision(
        self,
        analysis: FrameAnalysis,
        dt: float,
        current_vel: Tuple[float, float],
        in_boss_encounter: bool,
        boss_tagged: bool,
    ) -> AIDecisionTelemetry:
        now = analysis.timestamp
        self._update_spatial_grid(analysis, dt, current_vel)

        sweep_bx, sweep_by = self._get_macro_sweep_bias(now)
        pull_x, pull_y = analysis.lawn_center_pull_vector
        cx = self.grid_cols / 2.0
        cy = self.grid_rows / 2.0
        px, py = analysis.player_pos

        raw_scores: List[float] = []

        # Re-rank visible uncut grass patches by forward alignment with current heading so the player
        # marches continuously forward into fresh uncut grass and NEVER flips 180 deg onto its own trench!
        forward_grass = []
        for g in analysis.grass_targets[:10]:
            gdx = g.center[0] - px
            gdy = g.center[1] - py
            gdist = max(1.0, math.hypot(gdx, gdy))
            if gdist > 95.0:
                align = (gdx / gdist) * self.heading_x + (gdy / gdist) * self.heading_y
                fwd_score = g.priority_score * (1.0 + 0.55 * align)
                forward_grass.append((fwd_score, gdx / gdist, gdy / gdist, g))
        forward_grass.sort(key=lambda item: item[0], reverse=True)

        boundary_rebound_active = (
            analysis.fence_repulsion_vector is not None
            or analysis.door_repulsion_vector is not None
            or analysis.safe_escape_vector is not None
        )

        for ux, uy in self.ray_dirs:
            score = 0.0

            # 1. Proactive 360-Degree Fence Avoidance + Angled Lane Banking (Never retrace the same mowed line!)
            if analysis.fence_repulsion_vector is not None:
                frx, fry = analysis.fence_repulsion_vector
                fence_dot = ux * frx + uy * fry
                score += fence_dot * 12.0
                if fence_dot < -0.10:
                    # Hard penalty against moving closer to a nearby fence!
                    score -= 10.0
                # Bank inward + tangentially into a fresh adjacent uncut lane rather than bouncing 180 deg back
                t1x, t1y = -fry, frx
                if t1x * self.heading_x + t1y * self.heading_y < 0:
                    t1x, t1y = fry, -frx
                tang_dot = ux * t1x + uy * t1y
                if fence_dot > 0.15 and tang_dot > 0.0:
                    score += tang_dot * 3.6

            # Strong Lawn Centering Pull toward the pole of inaccessibility (deepest lawn center)
            score += (ux * pull_x + uy * pull_y) * 5.2

            # 2. House Door Repulsion
            if analysis.door_repulsion_vector is not None:
                drx, dry = analysis.door_repulsion_vector
                door_dot = ux * drx + uy * dry
                score += door_dot * 9.5
                if door_dot < -0.10:
                    score -= 8.0

            # 3. Boss Attack Telegraph & Projectile Evasion (Immediate Survival!)
            if analysis.safe_escape_vector is not None:
                evx, evy = analysis.safe_escape_vector
                score += (ux * evx + uy * evy) * 12.0

            # 4. BOSS PHASE A (TAG BOSS ONCE): Steer directly toward the Boss Entity to land at least 1 attack!
            if in_boss_encounter and not boss_tagged:
                if analysis.boss_target is not None:
                    bdx = analysis.boss_target.center[0] - px
                    bdy = analysis.boss_target.center[1] - py
                    bdist = max(1.0, math.hypot(bdx, bdy))
                    score += (ux * (bdx / bdist) + uy * (bdy / bdist)) * 9.0
                else:
                    score += (ux * pull_x + uy * pull_y) * 3.5 + (ux * sweep_bx + uy * sweep_by) * 3.5
                raw_scores.append(score)
                continue

            # 5. BOSS PHASE B (SURVIVE & DODGE): Continuous mid-lawn kiting + grass mowing!
            if in_boss_encounter and boss_tagged:
                if analysis.boss_target is not None:
                    bdx = px - analysis.boss_target.center[0]
                    bdy = py - analysis.boss_target.center[1]
                    bdist = max(1.0, math.hypot(bdx, bdy))
                    rx_b, ry_b = bdx / bdist, bdy / bdist
                    tx_b, ty_b = -ry_b, rx_b
                    score += (ux * rx_b + uy * ry_b) * 4.0 + (ux * tx_b + uy * ty_b) * 3.5
                else:
                    score += (ux * sweep_bx + uy * sweep_by) * 3.5

            # 6. HIGH-PRIORITY GRASS GATHERING & SPATIAL LAWN SWEEP
            ray_yield = 0.0
            for step in (2.0, 3.2, 4.5, 6.0):
                gx = int(np.clip(round(cx + ux * step), 0, self.grid_cols - 1))
                gy = int(np.clip(round(cy + uy * step), 0, self.grid_rows - 1))
                cell_val = float(self.occupancy_grid[gy, gx])
                if cell_val < 0.40:
                    ray_yield -= 1.15
                else:
                    ray_yield += cell_val * (1.4 / math.sqrt(step))

            score += ray_yield * (1.0 - self.exploration_weight) * 3.8

            # Strong direct attraction toward the highest-priority forward uncut grass clusters!
            for idx, (_, gx_u, gy_u, _) in enumerate(forward_grass[:3]):
                weight = 7.5 if idx == 0 else (4.2 if idx == 1 else 2.4)
                score += (ux * gx_u + uy * gy_u) * weight

            score += (ux * sweep_bx + uy * sweep_by) * self.exploration_weight * 2.4

            mom_dot = ux * self.heading_x + uy * self.heading_y
            if not boundary_rebound_active:
                # Never reverse 180 degrees onto the freshly-mowed trench behind the player!
                if mom_dot < -0.25:
                    score -= 6.5
                else:
                    score += mom_dot * 2.8
            else:
                if mom_dot < -0.75:
                    score -= 1.5

            if analysis.drops:
                best_drop = analysis.drops[0]
                ddx = best_drop.center[0] - px
                ddy = best_drop.center[1] - py
                ddist = max(1.0, math.hypot(ddx, ddy))
                score += (ux * (ddx / ddist) + uy * (ddy / ddist)) * 5.8

            raw_scores.append(score)

        best_idx = int(np.argmax(raw_scores))
        best_ux, best_uy = self.ray_dirs[best_idx]

        blend = 0.90 if boundary_rebound_active else 0.65
        self.heading_x = blend * best_ux + (1.0 - blend) * self.heading_x
        self.heading_y = blend * best_uy + (1.0 - blend) * self.heading_y
        h_mag = max(1e-4, math.hypot(self.heading_x, self.heading_y))
        self.heading_x /= h_mag
        self.heading_y /= h_mag

        min_s = min(raw_scores)
        max_s = max(raw_scores)
        span = max(1e-4, max_s - min_s)
        norm_rays = [
            (self.ray_dirs[i][0], self.ray_dirs[i][1], (raw_scores[i] - min_s) / span)
            for i in range(self.num_rays)
        ]

        fresh_ratio = float(np.mean(self.occupancy_grid < 0.65)) * 100.0
        efficiency = min(100.0, 55.0 + fresh_ratio * 0.9)

        reason = "AI Potential-Field: Center-Lawn Grass Yield Optimization"
        boss_name_str = (
            f"{analysis.identified_boss.name} (Lv.{analysis.identified_boss.level})"
            if analysis.identified_boss is not None
            else "Lawn Boss"
        )
        if analysis.has_forcefield:
            reason = "Forcefield Detected: Respawn Recovery & Re-Centering!"
        elif analysis.camera_angle_off:
            reason = f"Camera Off ({analysis.camera_off_reason}) -> Auto-Adjusting!"
        elif analysis.safe_escape_vector is not None:
            reason = f"AI Hazard Evasion: Dodging {boss_name_str} Attack!"
        elif in_boss_encounter and not boss_tagged:
            reason = f"AI Boss Protocol: Charging {boss_name_str} for 1 Tag Hit!"
        elif in_boss_encounter and boss_tagged:
            reason = f"AI Boss Protocol: Kiting {boss_name_str} (Away from Fences)"
        elif analysis.fence_repulsion_vector is not None:
            reason = "AI Fence Guard: Repelling Away From Outer Fence!"
        elif analysis.door_repulsion_vector is not None:
            reason = "AI Boundary Guard: Repelling from House Door"
        elif analysis.drops:
            reason = f"AI Drop Intercept: Collecting {analysis.drops[0].label}"

        return AIDecisionTelemetry(
            chosen_vector=(best_ux, best_uy),
            chosen_angle_deg=math.degrees(math.atan2(best_uy, best_ux)),
            confidence=min(1.0, span / 4.5),
            mode_reason=reason,
            ray_scores=norm_rays,
            heatmap_grid=self.occupancy_grid.copy(),
            ai_efficiency_score=efficiency,
            active_exploration_weight=self.exploration_weight,
        )

    def draw_ai_hud_graphics(self, vis_frame: np.ndarray, player_pos: Tuple[int, int], telemetry: AIDecisionTelemetry):
        px, py = player_pos

        base_r = 38
        max_r = 105
        for ux, uy, norm_s in telemetry.ray_scores:
            length = int(base_r + norm_s * (max_r - base_r))
            ex = int(px + ux * length)
            ey = int(py + uy * length)
            g_col = int(norm_s * 245)
            r_col = int((1.0 - norm_s) * 220)
            b_col = int(norm_s * 140)
            thickness = 3 if norm_s > 0.92 else 1
            cv2.line(vis_frame, (px, py), (ex, ey), (b_col, g_col, r_col), thickness, cv2.LINE_AA)

        ch_x, ch_y = telemetry.chosen_vector
        arrow_end = (int(px + ch_x * 135), int(py + ch_y * 135))
        cv2.arrowedLine(vis_frame, (px, py), arrow_end, (0, 255, 180), 3, cv2.LINE_AA, tipLength=0.22)

        if telemetry.heatmap_grid is not None:
            fh, fw = vis_frame.shape[:2]
            map_w, map_h = 200, 140
            mx1 = fw - map_w - 16
            my1 = 104
            mx2 = mx1 + map_w
            my2 = my1 + map_h

            if my2 < fh and mx1 > 0:
                norm_grid = np.clip(telemetry.heatmap_grid / 1.5, 0.0, 1.0)
                grid_u8 = (norm_grid * 255).astype(np.uint8)
                colored_map = cv2.applyColorMap(grid_u8, cv2.COLORMAP_VIRIDIS)
                resized_map = cv2.resize(colored_map, (map_w, map_h), interpolation=cv2.INTER_NEAREST)

                roi = vis_frame[my1:my2, mx1:mx2]
                cv2.addWeighted(resized_map, 0.82, roi, 0.18, 0, roi)
                cv2.rectangle(vis_frame, (mx1, my1), (mx2, my2), (0, 230, 180), 2)
                cv2.putText(
                    vis_frame,
                    "AI SPATIAL YIELD MAP",
                    (mx1 + 6, my1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 200),
                    1,
                    cv2.LINE_AA,
                )
