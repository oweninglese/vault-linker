#!/usr/bin/env python3
import sys
import os

# Add the src directory to the path
src_path = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_path)

try:
    from app import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"Error: Could not find the core logic in 'src/'. {e}")
    sys.exit(1)
