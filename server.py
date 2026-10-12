"""
Root server entrypoint for The Daily Classifier.
Allows running `python server.py` directly from the project root.
"""

import os
import sys

# Ensure web module is discoverable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(PROJECT_ROOT, "web")
sys.path.insert(0, WEB_DIR)

from server import main

if __name__ == "__main__":
    main()
