from gameControl import click_at
from gameControl import click_img
from gameControl import getWindow
from PIL import Image
import time
import pyautogui


supp_img_path = "./button_imgs/iri_full.png"
update_supp_path = "./button_imgs/update_supp.png"
yes_button_path = "./button_imgs/yes.png"

# wrapper work flow for continuous farming of a node
# features to add in the future: automatically replenish AP, dantes brave chain generator
class turnStruct():
    def __init__(self, raw_skills, skills, cards):
        self.raw_skills = raw_skills  # list of image paths or coords of skills to press 
        self.skills = skills          # list of tuples of image paths or coords of targetted skills
        self.cards = cards            # lists of coords for card locations to press (default left 2 for now)
        
class Farmer: 
    def __init__(self, instructions_folder):
        self.instructions = instructions_folder
        self.supp_img = supp_img_path
        self.update_supp_img = update_supp_path
        self.yes_button = yes_button_path

        # get the bottom of the bluestacks screen 
        window_bounds = getWindow()
        self.wind_bot = window_bounds[1] + window_bounds[-1] - 100
        self.wind_side = window_bounds[0] + window_bounds[-2] - 100
    def farmCycle(self):
        # the actual farming loop (after support selection and battle starts)
        for turn in instructions:
            self.farmTurn(turn)

    def farmTurn(self, turn):
        # perform the actions of the turn 

        # first click and wait for the raw skills to finish
        for img in turn.raw_skills:
            click_img(img)
            time.sleep(3) # sleep 3 seconds for the skill activation 
        
        # complete the targetted skills (if any)
        for imgs in turn.skills: 
            click_img(imgs[0]) # click the skill 
            time.sleep(0.5)
            click_img(imgs[1]) # select the servant to use it on 
            time.sleep(3)   # wait for the skill activation 

        # press space to attack 
        pyautogui.press('space')
        time.sleep(0.5)

        for pos in turn.cards: 
            click_at(pos)
            time.sleep(0.5)

        # sleep 25 seconds for the turn to finish 
        time.sleep(25)

    def findSupport(self):
        # tries to automatically find the support using the supp img path 
        counter = 0 
        status = False
        while(not status):
            time.sleep(1)
            status = click_img(self.supp_img)
            time.sleep(0.5)
            if status == False and (counter % 8 != 0): 
                # if we don't find the support, scroll down on the list 
                pyautogui.moveTo(self.wind_side, self.wind_bot)
                pyautogui.drag(0, -400, 1, button="left") # drag up
                pyautogui.move(0, 400)                   # return to original position
            elif status == False and (counter % 8 == 0) and counter != 0 :
                # if we've already tried 3 times, refresh the friendlist 
                click_img(self.update_supp_img)
                time.sleep(0.5)
                click_img(self.yes_button)
                time.sleep(2)

            counter += 1

if __name__ == "__main__":
    time.sleep(2)
    farmer = Farmer([])
    farmer.findSupport()