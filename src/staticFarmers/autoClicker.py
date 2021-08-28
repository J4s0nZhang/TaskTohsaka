import pyautogui 
import time
import sys, signal

def signal_handler(signal, frame):
    print("\nprogram exiting gracefully")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

while(True):
    time.sleep(30)
    pyautogui.click(button="left") 
