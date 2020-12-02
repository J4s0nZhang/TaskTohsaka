import pyautogui
import time

print("Press Ctrl-C to quit")
try: 
    while(True): # click every 5 minutes
        time.sleep(5)
        pyautogui.click()
        time.sleep(1)
        pyautogui.click()
        time.sleep(1)
        pyautogui.click()
        time.sleep(1)
        pyautogui.click()

except KeyboardInterrupt:
    print("done")