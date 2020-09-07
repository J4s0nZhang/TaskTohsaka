"""
Controls the game by taking screen shots of the game (currently takes screen shot of the whole screen)
mouse and screen shots taken with pyautogui 

For my mac main monitor: attack button (1478, 822)
cards: (420, 705), x increases by 250 ish per card fuck we have to get the bounding boxes don't we oof
"""

import PIL 
import numpy as np 
import pyautogui
import matplotlib.pyplot as plt 
import sys 
import Quartz
import time
from Quartz import CGWindowListCopyWindowInfo, kCGWindowListExcludeDesktopElements, kCGNullWindowID
from Foundation import NSSet, NSMutableSet

# get the bounds of the bluestacks emulator window 
def getWindow():
    for window in Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements, Quartz.kCGNullWindowID):
        if window['kCGWindowOwnerName'] == "BlueStacks":
            print("found blue stacks at: ", window.valueForKey_('kCGWindowBounds'))
            return (window.valueForKey_('kCGWindowBounds')['X'], window.valueForKey_('kCGWindowBounds')['Y'],
                    window.valueForKey_('kCGWindowBounds')['Width'], window.valueForKey_('kCGWindowBounds')['Height'])
    print("Did not find bluestacks window")
    return None

# click at input location 
def click_at(loc):
    x = loc[0]
    y = loc[1]
    pyautogui.moveTo(x, y)
    pyautogui.click()

# prints the current mouse position 
def print_mouse_pos():
    print("Press Ctrl-C to quit")
    try: 
        while(True):
            x, y = pyautogui.position()
            print('X: %d, Y: %d' %(x,y))

    except KeyboardInterrupt:
        print("done")

class ScreenGrabber:
    def __init__(self, region=None):
        self.image = []
        self.region = (0,0,0,0)
        if region:
            self.region = region 

    def grab_screen(self):
        if self.region != (0,0,0,0):
            self.image = pyautogui.screenshot(region=region)
        else:
            self.image = pyautogui.screenshot()

    def show_screen(self):
        plt.imshow(self.image)
        plt.show()

if __name__ == "__main__":
    region = getWindow()
    print(region)
    screenGrabber = ScreenGrabber(region)
    screenGrabber.grab_screen()
    screenGrabber.show_screen()
    

