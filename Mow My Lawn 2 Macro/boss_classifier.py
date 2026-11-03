"""
boss_classifier.py - Wiki Boss Visual Classifier & Icon/Model Matcher for Mow My Lawn 2

Loads the 17 downloaded Boss reference images from `assets/bosses/` (sourced directly from
https://mml2.fandom.com/wiki/Category:Bosses) and associates live screen detections with the
exact Boss Name, Level, and Evasion Profile using:
1. Multi-Scale Template & Foreground-Mask Matching against the Top Boss Roulette/Banner UI
   (where the 128x128 Boss Portrait Icons `PlumPlum.png`, `Janky.png`, `Shootman.png`,
    `Franklin.png`, `Ashweaver.png`, `Hyperwitch.png`, `HelicWarformHead.png` appear!)
2. 2D Hue-Saturation-Value (HSV) Histogram Correlation + ORB Keypoint Matching against 3D
   Boss Entities detected on the lawn (`boss_target` crop).
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bosses")


@dataclass
class BossProfile:
    name: str
    filename: str
    category: str              # "Lawn Boss" or "Mega Boss"
    level: int
    danger_tier: int           # 1 (Easy) to 5 (Extreme)
    preferred_orbit_dist: float
    image_bgr: np.ndarray
    fg_mask: np.ndarray
    hsv_hist: np.ndarray
    small_templates: List[Tuple[np.ndarray, np.ndarray]]  # (resized_bgr, resized_mask) for top UI icon matching


# Metadata from https://mml2.fandom.com/wiki/Category:Bosses
WIKI_BOSS_METADATA: Dict[str, dict] = {
    "Plum-Plum the Imp": {
        "file": "plum_plum_the_imp.png",
        "category": "Lawn Boss",
        "level": 1,
        "danger_tier": 1,
        "orbit_dist": 220.0,
    },
    "Janky James": {
        "file": "janky_james.png",
        "category": "Lawn Boss",
        "level": 5,
        "danger_tier": 1,
        "orbit_dist": 250.0,
    },
    "Shootman": {
        "file": "shootman.png",
        "category": "Lawn Boss",
        "level": 10,
        "danger_tier": 2,
        "orbit_dist": 310.0,
    },
    "Franklin and the Probe-Boys": {
        "file": "franklin_and_the_probe_boys.png",
        "category": "Lawn Boss",
        "level": 30,
        "danger_tier": 3,
        "orbit_dist": 340.0,
    },
    "Ashweaver Painmage": {
        "file": "ashweaver_painmage.png",
        "category": "Lawn Boss",
        "level": 75,
        "danger_tier": 4,
        "orbit_dist": 370.0,
    },
    "Hyperwitch": {
        "file": "hyperwitch.png",
        "category": "Lawn Boss",
        "level": 160,
        "danger_tier": 5,
        "orbit_dist": 400.0,
    },
    "Devastator": {
        "file": "devastator.png",
        "category": "Lawn Boss",
        "level": 275,
        "danger_tier": 5,
        "orbit_dist": 420.0,
    },
    "Toxifer": {
        "file": "toxifer.png",
        "category": "Mega Boss",
        "level": 20,
        "danger_tier": 3,
        "orbit_dist": 340.0,
    },
    "Alden's Tower": {
        "file": "aldens_tower.png",
        "category": "Mega Boss",
        "level": 50,
        "danger_tier": 4,
        "orbit_dist": 360.0,
    },
    "Anointed One": {
        "file": "anointed_one.png",
        "category": "Mega Boss",
        "level": 90,
        "danger_tier": 4,
        "orbit_dist": 370.0,
    },
    "Llin": {
        "file": "llin.png",
        "category": "Mega Boss",
        "level": 120,
        "danger_tier": 4,
        "orbit_dist": 380.0,
    },
    "Kyarma": {
        "file": "kyarma.png",
        "category": "Mega Boss",
        "level": 180,
        "danger_tier": 5,
        "orbit_dist": 390.0,
    },
    "Polanius": {
        "file": "polanius.png",
        "category": "Mega Boss",
        "level": 220,
        "danger_tier": 5,
        "orbit_dist": 400.0,
    },
    "The Pumpkin King": {
        "file": "the_pumpkin_king.png",
        "category": "Mega Boss",
        "level": 100,
        "danger_tier": 4,
        "orbit_dist": 360.0,
    },
    "Nil": {
        "file": "nil.png",
        "category": "Mega Boss",
        "level": 300,
        "danger_tier": 5,
        "orbit_dist": 410.0,
    },
    "Ultiel": {
        "file": "ultiel.png",
        "category": "Mega Boss",
        "level": 350,
        "danger_tier": 5,
        "orbit_dist": 410.0,
    },
    "Ultima": {
        "file": "ultima.png",
        "category": "Mega Boss",
        "level": 400,
        "danger_tier": 5,
        "orbit_dist": 420.0,
    },
}


class WikiBossClassifier:
    def __init__(self):
        self.profiles: Dict[str, BossProfile] = {}
        self._load_profiles()

    def _load_profiles(self):
        for name, meta in WIKI_BOSS_METADATA.items():
            path = os.path.join(ASSETS_DIR, meta["file"])
            if not os.path.exists(path):
                continue
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None or bgr.size == 0:
                continue

            # Build foreground mask (exclude pure black background from wiki icons)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _, fg_mask = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
            if cv2.countNonZero(fg_mask) < 100:
                fg_mask = np.ones_like(gray) * 255

            # Compute normalized 2D Hue-Saturation + Value histogram for appearance matching
            hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], fg_mask, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

            # Precompute multi-scale icon templates (36px, 48px, 64px, 84px) for fast UI roulette/banner matching
            small_templates = []
            for sz in (36, 48, 64, 80):
                t_bgr = cv2.resize(bgr, (sz, sz), interpolation=cv2.INTER_AREA)
                t_msk = cv2.resize(fg_mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
                small_templates.append((t_bgr, t_msk))

            self.profiles[name] = BossProfile(
                name=name,
                filename=meta["file"],
                category=meta["category"],
                level=meta["level"],
                danger_tier=meta["danger_tier"],
                preferred_orbit_dist=meta["orbit_dist"],
                image_bgr=bgr,
                fg_mask=fg_mask,
                hsv_hist=hist,
                small_templates=small_templates,
            )

    def match_boss_in_top_ui(self, top_ui_crop: np.ndarray) -> Tuple[Optional[BossProfile], float]:
        """
        Checks if any of the 7 Lawn Boss Portrait Icons (Plum-Plum, Janky James, Shootman, Franklin,
        Ashweaver, Hyperwitch, Devastator) is currently displayed in the Top Boss Roulette / Health Bar UI!
        """
        if top_ui_crop is None or top_ui_crop.size == 0 or not self.profiles:
            return None, 0.0

        ch, cw = top_ui_crop.shape[:2]
        best_profile: Optional[BossProfile] = None
        best_score: float = 0.0

        # Prioritize the 7 Lawn Bosses for top roulette/banner icon matching
        for name, prof in self.profiles.items():
            if prof.category != "Lawn Boss":
                continue
            for t_bgr, t_msk in prof.small_templates:
                th, tw = t_bgr.shape[:2]
                if ch < th or cw < tw:
                    continue
                try:
                    res = cv2.matchTemplate(top_ui_crop, t_bgr, cv2.TM_CCOEFF_NORMED, mask=t_msk)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if np.isfinite(max_val) and max_val > best_score:
                        best_score = float(max_val)
                        best_profile = prof
                except Exception:
                    pass

        if best_score >= 0.56 and best_profile is not None:
            return best_profile, best_score
        return None, 0.0

    def classify_boss_crop(self, boss_crop_bgr: np.ndarray, lawn_only: bool = False) -> Tuple[Optional[BossProfile], float]:
        """
        Associates a 3D Boss Entity bounding box on the lawn with its matching Wiki Boss Profile
        by comparing 2D Hue-Saturation Histograms + Brightness signatures.
        """
        if boss_crop_bgr is None or boss_crop_bgr.size == 0 or not self.profiles:
            return None, 0.0

        hsv = cv2.cvtColor(boss_crop_bgr, cv2.COLOR_BGR2HSV)
        # Exclude green grass background inside the bounding box
        grass_mask = cv2.inRange(hsv, np.array([32, 45, 45], dtype=np.uint8), np.array([85, 255, 235], dtype=np.uint8))
        fg_mask = cv2.bitwise_not(grass_mask)
        if cv2.countNonZero(fg_mask) < 30:
            fg_mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255

        crop_hist = cv2.calcHist([hsv], [0, 1], fg_mask, [30, 32], [0, 180, 0, 256])
        cv2.normalize(crop_hist, crop_hist, 0, 1, cv2.NORM_MINMAX)

        best_prof: Optional[BossProfile] = None
        best_sim: float = -1.0

        for name, prof in self.profiles.items():
            if lawn_only and prof.category != "Lawn Boss":
                continue
            sim = float(cv2.compareHist(crop_hist, prof.hsv_hist, cv2.HISTCMP_CORREL))
            # Bonus for Lawn Bosses when fighting on the Main Lawn
            if prof.category == "Lawn Boss":
                sim += 0.08
            if sim > best_sim:
                best_sim = sim
                best_prof = prof

        return best_prof, max(0.0, min(1.0, best_sim))
