import subprocess
import time
import psutil
import os

def is_obsidian_running():
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == 'Obsidian':
            return True
    return False

def run_update_script():
    script_path = os.path.join(os.path.dirname(__file__), 'update_vector_store.py')
    subprocess.run(['python3', script_path])

def main():
    print("Monitoring for Obsidian app...")
    obsidian_was_running = False

    while True:
        obsidian_is_running = is_obsidian_running()

        if obsidian_is_running and not obsidian_was_running:
            print("Obsidian app opened. Running update script...")
            run_update_script()

        obsidian_was_running = obsidian_is_running
        time.sleep(5)  # Check every 5 seconds

if __name__ == "__main__":
    main()