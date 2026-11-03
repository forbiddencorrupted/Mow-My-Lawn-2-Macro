"""
stats_tracker.py - Dual-Corner (Bottom-Left & Bottom-Right) Percentage & Tool Intake OCR Analyzer

Fixes:
1. Scans BOTH Bottom-Left AND Bottom-Right HUD regions (or whichever corner has the tool/upgrade
   progress percentages `%`) and filters out non-item percentages (like 100% health, EXP, or static stats).
2. Prioritizes the active **Tool / Mower Upgrade Progress %** (percentages between `0.1%` and `99.9%`
   that are actively increasing as you cut grass, or labeled with tool/shop names).
3. Tracks every detected item's `%` growth rate (`% / sec` and `% / hit`) to give accurate live ETAs.
"""

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from rapidocr_onnxruntime import RapidOCR
    HAS_RAPIDOCR = True
except Exception:
    HAS_RAPIDOCR = False


SUFFIX_MAP: Dict[str, float] = {
    "": 1.0,
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "t": 1e12,
    "qa": 1e15,
    "qd": 1e15,
    "qi": 1e18,
    "qn": 1e18,
    "sx": 1e21,
    "sp": 1e24,
    "oc": 1e27,
    "no": 1e30,
    "dc": 1e33,
    "ud": 1e36,
    "dd": 1e39,
    "td": 1e42,
}

# Words that indicate a percentage is NOT an upcoming tool/item (e.g. HP, Health, EXP, Volume, Boost)
IGNORE_KEYWORDS = ("hp", "health", "life", "lives", "def", "crit", "dodge", "vol", "server")

PERCENT_PATTERN = re.compile(r"(\d{1,3}(?:\.\d{1,3})?)\s*%")
NUMBER_PATTERN = re.compile(
    r"(?:[\$\+\-]?\s*)(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*([a-zA-Z]{0,3})\b"
)


def parse_percentages(text: str) -> List[Tuple[float, str]]:
    cleaned = (
        text.replace("O%", "0%")
        .replace("o%", "0%")
        .replace("I%", "1%")
        .replace("l%", "1%")
    )
    results: List[Tuple[float, str]] = []
    for match in PERCENT_PATTERN.finditer(cleaned):
        try:
            val = float(match.group(1))
            if 0.0 <= val <= 100.0:
                results.append((val, match.group(0).strip()))
        except ValueError:
            continue
    return results


def parse_game_number(text: str) -> List[Tuple[float, str]]:
    if "%" in text:
        return []
    cleaned = text.replace("O", "0").replace("l", "1").replace("I", "1")
    results: List[Tuple[float, str]] = []
    for match in NUMBER_PATTERN.finditer(cleaned):
        num_str = match.group(1).replace(",", "")
        suffix_raw = match.group(2).lower()
        try:
            base_val = float(num_str)
        except ValueError:
            continue

        if suffix_raw in SUFFIX_MAP:
            val = base_val * SUFFIX_MAP[suffix_raw]
            results.append((val, match.group(0).strip()))
    return results


def format_number(val: float) -> str:
    if val <= 0:
        return "0"
    suffixes = [
        (1e33, "Dc"),
        (1e30, "No"),
        (1e27, "Oc"),
        (1e24, "Sp"),
        (1e21, "Sx"),
        (1e18, "Qi"),
        (1e15, "Qa"),
        (1e12, "T"),
        (1e9, "B"),
        (1e6, "M"),
        (1e3, "K"),
    ]
    for threshold, sfx in suffixes:
        if abs(val) >= threshold:
            return f"{val / threshold:.2f}{sfx}"
    if val >= 100:
        return f"{val:,.0f}"
    return f"{val:.2f}"


def format_eta(seconds: Optional[float], pct: float = 0.0) -> str:
    if pct >= 100.0:
        return "READY (100%)"
    if seconds is None:
        return "Measuring % rate..."
    if seconds <= 0.5:
        return "READY!"
    if seconds > 86400 * 7:
        return ">7d"
    sec = int(round(seconds))
    if sec < 60:
        return f"{sec}s"
    mins, s = divmod(sec, 60)
    if mins < 60:
        return f"{mins}m {s:02d}s"
    hrs, m = divmod(mins, 60)
    return f"{hrs}h {m:02d}m"


@dataclass
class ShopItemTarget:
    index: int
    label: str
    progress_pct: float
    pct_per_sec: float = 0.0
    pct_per_hit: float = 0.0
    eta_seconds: Optional[float] = None
    screen_pos: Tuple[int, int] = (0, 0)
    raw_text: str = ""


@dataclass
class ProgressionSnapshot:
    current_trimmings: Optional[float] = None
    trimmings_per_sec: float = 0.0
    trimmings_per_hit: float = 0.0
    primary_pct_per_hit: float = 0.0
    visual_grass_px_per_hit: float = 0.0
    hits_per_sec: float = 0.0
    upcoming_items: List[ShopItemTarget] = field(default_factory=list)
    raw_ocr_lines: List[str] = field(default_factory=list)
    last_ocr_update: float = 0.0


class ProgressionAnalyzer:
    def __init__(self, config: dict):
        self.config = config
        self.ocr_engine = RapidOCR() if HAS_RAPIDOCR else None

        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = True

        # Keyed by spatial bucket `(side_tag, row_bucket)` so each distinct UI bar is tracked cleanly
        self.item_pct_history: Dict[str, Deque[Tuple[float, float, str, Tuple[int, int]]]] = {}
        self.smoothed_pct_rates: Dict[str, float] = {}

        self.balance_history: Deque[Tuple[float, float]] = deque(maxlen=25)
        self.swing_history: Deque[Tuple[float, float]] = deque(maxlen=40)
        self.last_grass_total_area: float = 0.0

        self.snapshot = ProgressionSnapshot()

        self._worker = threading.Thread(target=self._ocr_loop, daemon=True)
        self._worker.start()

    def stop(self):
        self._running = False

    def reload_config(self, config: dict):
        self.config = config

    def record_weapon_swing(self, now: float, current_grass_area: float):
        cleared = max(0.0, self.last_grass_total_area - current_grass_area)
        self.last_grass_total_area = current_grass_area
        with self._lock:
            self.swing_history.append((now, cleared))

    def submit_frame(self, frame: np.ndarray):
        if self.ocr_engine is None:
            return
        with self._lock:
            self._latest_frame = frame

    def get_snapshot(self) -> ProgressionSnapshot:
        with self._lock:
            return self.snapshot

    def _ocr_loop(self):
        while self._running:
            prog_cfg = self.config.get("progression_ocr", {})
            interval = prog_cfg.get("scan_interval_sec", 1.1)

            frame = None
            with self._lock:
                if self._latest_frame is not None:
                    frame = self._latest_frame.copy()
                    self._latest_frame = None

            if frame is not None and self.ocr_engine is not None and prog_cfg.get("enabled", True):
                try:
                    self._process_hud_ocr(frame, prog_cfg)
                except Exception:
                    pass

            time.sleep(interval)

    def _estimate_pct_rate(
        self, history: Deque[Tuple[float, float, str, Tuple[int, int]]], slot_key: str
    ) -> float:
        if len(history) < 2:
            return self.smoothed_pct_rates.get(slot_key, 0.0)

        t0, p0, _, _ = history[0]
        t1, p1, _, _ = history[-1]
        dt = t1 - t0

        rate = 0.0
        if dt >= 0.8 and p1 > p0:
            rate = (p1 - p0) / dt
        else:
            step_rates = []
            for i in range(1, len(history)):
                dt_s = history[i][0] - history[i - 1][0]
                dp_s = history[i][1] - history[i - 1][1]
                if dt_s > 0.2 and 0.0 < dp_s < 50.0:
                    step_rates.append(dp_s / dt_s)
            if step_rates:
                rate = float(np.median(step_rates))

        prev = self.smoothed_pct_rates.get(slot_key, 0.0)
        if rate > 0:
            smoothed = (0.65 * rate + 0.35 * prev) if prev > 0 else rate
            self.smoothed_pct_rates[slot_key] = smoothed
            return smoothed
        else:
            decayed = prev * 0.93
            self.smoothed_pct_rates[slot_key] = decayed
            return decayed

    def _scan_roi_for_percentages(
        self, frame: np.ndarray, roi_norm: List[float], side_tag: str
    ) -> Tuple[List[Tuple[int, int, float, str, str, str]], List[Tuple[int, int, str]]]:
        """Scans a specific HUD ROI and returns (pct_detections, all_text_boxes)."""
        h, w = frame.shape[:2]
        rx1, ry1, rx2, ry2 = (
            int(roi_norm[0] * w),
            int(roi_norm[1] * h),
            int(roi_norm[2] * w),
            int(roi_norm[3] * h),
        )
        crop = frame[ry1:ry2, rx1:rx2]
        pct_detections: List[Tuple[int, int, float, str, str, str]] = []
        all_boxes: List[Tuple[int, int, str]] = []

        if crop.size == 0:
            return pct_detections, all_boxes

        ocr_res, _ = self.ocr_engine(crop)
        if not ocr_res:
            return pct_detections, all_boxes

        for item in ocr_res:
            box, text, score = item[0], item[1], float(item[2])
            if score < 0.40:
                continue
            bx = int(rx1 + (box[0][0] + box[2][0]) * 0.5)
            by = int(ry1 + (box[0][1] + box[2][1]) * 0.5)
            all_boxes.append((by, bx, text))

            pcts = parse_percentages(text)
            for p_val, p_str in pcts:
                name_part = text.replace(p_str, "").strip(" :-|[]()")
                pct_detections.append((by, bx, p_val, name_part, text, side_tag))

        return pct_detections, all_boxes

    def _process_hud_ocr(self, frame: np.ndarray, prog_cfg: dict):
        now = time.time()

        with self._lock:
            swings = list(self.swing_history)

        hits_per_sec = float(1.0 / max(self.config.get("general", {}).get("click_interval_sec", 0.12), 0.05))
        visual_px_per_hit = 0.0
        if len(swings) >= 2:
            dt_swings = max(0.1, swings[-1][0] - swings[0][0])
            hits_per_sec = (len(swings) - 1) / dt_swings
            positive_clears = [s[1] for s in swings if s[1] > 0]
            if positive_clears:
                visual_px_per_hit = float(np.mean(positive_clears))

        # Scan Bottom-Right first, and also Bottom-Left so whichever corner has the 2+ item % bars is captured!
        br_roi = prog_cfg.get("bottom_right_hud_roi", [0.58, 0.52, 1.0, 0.99])
        bl_roi = prog_cfg.get("bottom_left_hud_roi", [0.0, 0.52, 0.42, 0.99])

        br_pcts, br_boxes = self._scan_roi_for_percentages(frame, br_roi, "BR")
        bl_pcts, bl_boxes = self._scan_roi_for_percentages(frame, bl_roi, "BL")

        all_pcts = br_pcts + bl_pcts
        all_boxes = br_boxes + bl_boxes

        # Sort by side ("BR" first since user specified bottom-right, then "BL") and top-to-bottom row
        all_pcts.sort(key=lambda d: (0 if d[5] == "BR" else 1, d[0] // 26, d[1]))

        valid_items: List[ShopItemTarget] = []
        for py, px, p_val, name_hint, full_txt, side_tag in all_pcts:
            if not name_hint:
                for by, bx, btxt in all_boxes:
                    if "%" not in btxt and abs(by - py) < 36 and abs(bx - px) < 260 and len(btxt.strip()) >= 2:
                        name_hint = btxt.strip()
                        break

            combined_lower = f"{name_hint} {full_txt}".lower()
            # Filter out non-item percentages (e.g. HP 100%, Health, Server boost, etc.)
            if any(kw in combined_lower for kw in IGNORE_KEYWORDS):
                continue
            # Ignore static 100% bars that aren't item progression unless we already tracked them rising
            slot_key = f"{side_tag}_{py // 32}"
            if p_val >= 100.0 and slot_key not in self.item_pct_history:
                continue

            idx = len(valid_items) + 1
            clean_label = f"[{side_tag}] Item #{idx}" if not name_hint else f"[{side_tag}] {name_hint[:16]}"

            if slot_key not in self.item_pct_history:
                self.item_pct_history[slot_key] = deque(maxlen=25)

            hist = self.item_pct_history[slot_key]
            if hist and (hist[-1][1] - p_val) > 15.0:
                hist.clear()
                self.smoothed_pct_rates[slot_key] = 0.0

            hist.append((now, p_val, clean_label, (px, py)))
            pct_per_sec = self._estimate_pct_rate(hist, slot_key)
            pct_per_hit = (pct_per_sec / hits_per_sec) if hits_per_sec > 0 else 0.0

            remaining_pct = max(0.0, 100.0 - p_val)
            if remaining_pct <= 0.0:
                eta_sec = 0.0
            elif pct_per_sec > 1e-5:
                eta_sec = remaining_pct / pct_per_sec
            else:
                eta_sec = None

            valid_items.append(
                ShopItemTarget(
                    index=idx,
                    label=clean_label,
                    progress_pct=p_val,
                    pct_per_sec=pct_per_sec,
                    pct_per_hit=pct_per_hit,
                    eta_seconds=eta_sec,
                    screen_pos=(px, py),
                    raw_text=full_txt,
                )
            )

        # Sort items so actively progressing items (`pct_per_sec > 0`) or items closest to 100% appear at the top!
        valid_items.sort(key=lambda it: (1 if it.pct_per_sec > 0 else 0, it.progress_pct), reverse=True)
        for new_idx, it in enumerate(valid_items, start=1):
            it.index = new_idx

        primary_pct_per_hit = valid_items[0].pct_per_hit if valid_items else 0.0

        with self._lock:
            self.snapshot = ProgressionSnapshot(
                primary_pct_per_hit=primary_pct_per_hit,
                visual_grass_px_per_hit=visual_px_per_hit,
                hits_per_sec=hits_per_sec,
                upcoming_items=valid_items[:4],
                last_ocr_update=now,
            )
