"""
Controls the game by taking screen shots of the game (currently takes screen shot of the whole screen)
mouse and screen shots taken with pyautogui. To make the system work on a new machine, the txt files need to
be provided and updated with the coordinates of the buttons. 

Run this program to continuously print out mouse coordinates to get the coords for each of the buttons
Jason Zhang 2020 
"""

import PIL 
import pyautogui
import sys 
import time 

# click at input location 
def click_at(loc):
    x = loc[0]
    y = loc[1]
    pyautogui.moveTo(x, y)
    pyautogui.click()

# click the image 
def click_img(img_path, mac):
    #try: 
    loc = pyautogui.locateOnScreen(img_path, confidence=0.9)
    if loc == None:
        return False
    else:
        loc = pyautogui.center(loc)
        if mac:
            loc = [i/2 for i in loc]
        click_at([loc[0], loc[1]])
    return True

# prints the current mouse position 
def print_mouse_pos():
    print("Press Ctrl-C to quit")
    try: 
        while(True):
            time.sleep(1)
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

if __name__ == "__main__":
    #region = getWindow()
    #print(region)
    #screenGrabber = ScreenGrabber(region)
    #screenGrabber.grab_screen()
    #screenGrabber.show_screen()
    print_mouse_pos()
    

