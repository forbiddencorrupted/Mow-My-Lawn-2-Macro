"""
main.py - Launcher for Mow My Lawn 2 AI Vision & Spatial Yield Suite

By default, launches the modern dark-mode CustomTkinter AI Control Center (`gui_app.py`).
Pass `--cli` if you ever want to run headless in terminal-only mode.
"""

import sys


def main():
    if "--cli" in sys.argv:
        print("[*] Launching in CLI mode... (Run without --cli for the full AI GUI Dashboard)")
    from gui_app import MML2DashboardApp

    app = MML2DashboardApp()
    app.mainloop()


if __name__ == "__main__":
    main()
