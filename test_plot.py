import sys
import logging
import traceback
import subprocess

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

if __name__ == '__main__':
    try:
        subprocess.run(["python", "main.py"])
    except Exception as e:
        print(f"FAILED: {e}")
        traceback.print_exc()
