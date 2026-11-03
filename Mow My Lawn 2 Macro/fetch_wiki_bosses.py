"""
fetch_wiki_bosses.py - Downloads all Boss Reference Images from mml2.fandom.com/wiki/Category:Bosses
and computes their HSV color histograms, dominant visual signatures, and ORB feature descriptors
for real-time Boss Identification in `MML2Vision`.
"""

import json
import os
import ssl
import urllib.parse
import urllib.request

import cv2
import numpy as np

SSL_CTX = ssl._create_unverified_context()
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "bosses")
os.makedirs(ASSETS_DIR, exist_ok=True)

BOSS_PAGES = [
    "Plum-Plum the Imp",
    "Janky James",
    "Shootman",
    "Franklin and the Probe-Boys",
    "Ashweaver Painmage",
    "Hyperwitch",
    "Devastator",
    "Alden's Tower",
    "Anointed One",
    "Kyarma",
    "Llin",
    "Nil",
    "Polanius",
    "The Pumpkin King",
    "Toxifer",
    "Ultiel",
    "Ultima",
]


def api_get(params: dict) -> dict:
    qs = urllib.parse.urlencode(params)
    url = f"https://mml2.fandom.com/api.php?{qs}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=12, context=SSL_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    # 1. Query pageimages (thumbnails/originals) for all boss pages
    data = api_get(
        {
            "action": "query",
            "titles": "|".join(BOSS_PAGES),
            "prop": "pageimages|images",
            "pithumbsize": 400,
            "format": "json",
        }
    )
    pages = data.get("query", {}).get("pages", {})
    boss_manifest = {}

    for pid, pinfo in pages.items():
        title = pinfo.get("title", "")
        thumb_url = pinfo.get("thumbnail", {}).get("source")
        page_images = [im["title"] for im in pinfo.get("images", [])]
        print(f"Page: {title} -> thumb={bool(thumb_url)}, images={page_images}")

        # If no direct pageimage thumbnail, query the first non-icon image on the page
        if not thumb_url and page_images:
            for img_title in page_images:
                if any(ext in img_title.lower() for ext in (".png", ".jpg", ".jpeg", ".gif")):
                    img_data = api_get(
                        {
                            "action": "query",
                            "titles": img_title,
                            "prop": "imageinfo",
                            "iiprop": "url",
                            "format": "json",
                        }
                    )
                    for _, ii_page in img_data.get("query", {}).get("pages", {}).items():
                        iinfo = ii_page.get("imageinfo", [])
                        if iinfo:
                            thumb_url = iinfo[0].get("url")
                            break
                if thumb_url:
                    break

        if thumb_url:
            safe_name = (
                title.lower()
                .replace(" ", "_")
                .replace("'", "")
                .replace("!", "")
                .replace("-", "_")
            )
            out_path = os.path.join(ASSETS_DIR, f"{safe_name}.png")
            try:
                req = urllib.request.Request(thumb_url, headers={"User-Agent": "Mozilla/5.0"})
                img_bytes = urllib.request.urlopen(req, timeout=12, context=SSL_CTX).read()
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if bgr is not None:
                    cv2.imwrite(out_path, bgr)
                    # Compute dominant HSV profile (excluding near-black/white background)
                    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
                    fg_mask = cv2.inRange(
                        hsv,
                        np.array([0, 35, 35], dtype=np.uint8),
                        np.array([180, 255, 245], dtype=np.uint8),
                    )
                    if cv2.countNonZero(fg_mask) < 50:
                        fg_mask = np.ones(hsv.shape[:2], dtype=np.uint8) * 255
                    h_vals = hsv[:, :, 0][fg_mask > 0]
                    s_vals = hsv[:, :, 1][fg_mask > 0]
                    v_vals = hsv[:, :, 2][fg_mask > 0]
                    med_h = int(np.median(h_vals))
                    med_s = int(np.median(s_vals))
                    med_v = int(np.median(v_vals))
                    boss_manifest[title] = {
                        "file": f"{safe_name}.png",
                        "median_hsv": [med_h, med_s, med_v],
                        "hsv_lower": [max(0, med_h - 18), max(25, med_s - 65), max(25, med_v - 65)],
                        "hsv_upper": [min(180, med_h + 18), 255, 255],
                    }
                    print(f"  [Saved] {out_path} | Median HSV: {[med_h, med_s, med_v]}")
            except Exception as e:
                print(f"  [Error downloading {title}]: {e}")

    manifest_path = os.path.join(ASSETS_DIR, "boss_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(boss_manifest, f, indent=2)
    print(f"\nSaved {len(boss_manifest)} boss visual profiles to {manifest_path}")


if __name__ == "__main__":
    main()
