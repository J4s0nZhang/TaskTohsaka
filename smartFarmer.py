from gameControl import click_at
from gameControl import click_img
from gameControl import getWindow
from PIL import Image
import time
import pyautogui
import random

supp_img_path = "./button_imgs/skadi_with_ce.png"
update_supp_path = "./button_imgs/update_supp.png"
yes_button_path = "./button_imgs/yes.png"
golden_apple = "./button_imgs/golden_apple.png"

def load_np_locs(np_txt):
    # get the mouse coords of the 3 nps
    np_dict = {}
    np_list = open(np_txt).read().splitlines()
    for i in range(3):
        string = "np{}".format(i+1)
        np_dict[string] = [int(x) for x in np_list[i].split(',')]

    return np_dict

def load_master_skills(master_txt):
    master_sk_dict = {}
    ms_list = open(master_txt).read().splitlines()
    # get the location of the master skills button
    master_sk_dict['M_sk'] = [int(x) for x in ms_list[0].split(',')]
    # get the location of the 3 master skills
    for i in range(3):
        str1 = "ms_skill{}".format(i+1)
        master_sk_dict[str1] = [int(x) for x in ms_list[i+1].split(',')]
    # get the servant swap locations (just back rank4th and front 3rd)
    for i in range(2):
        str2 = "swap{}".format(i+3)
        master_sk_dict[str2] = [int(x) for x in ms_list[4+i].split(',')]
    
    master_sk_dict["replace"] = [int(x) for x in ms_list[-1].split(',')]
    return master_sk_dict

def load_skill_coords(skills_txt):
    # get the location of the skills of the servants
    skill_loc_dict = {}
    skill_txt = open(skills_txt, "r")
    skills = skill_txt.read().splitlines()

    serv_count = 1
    skill_count = 1
    for i in range(9):
        string = "s{}_s{}".format(serv_count, skill_count)
        skill_loc_dict[string] = [int(x) for x in skills[i].split(",")]
        if skill_count == 3:
            skill_count = 1
            serv_count += 1
        else:
            skill_count += 1
    return skill_loc_dict
# wrapper work flow for continuous farming of a node
# features to add in the future: automatically replenish AP, dantes brave chain generator
class turnStruct():
    def __init__(self, raw_skills=None, skills=None, cards=None, ms_skills=None):
        self.raw_skills = raw_skills  # list of image paths or coords of skills to press 
        self.skills = skills          # list of tuples of image paths or coords of targetted skills
        self.cards = cards            # lists of coords for card locations to press (default left 2 for now)
        self.ms_skills = ms_skills    # list of master skills 
        
class Farmer: 
    def __init__(self, instructions_list, coords=True):
        
        self.turn_list = []
        """
        if not coords:
            
            commands = open(instructions_list[0], "r")
            com_list = commands.read().splitlines()
            line_count = 1
            for line in com_list:
                split_line = line.split(' ')
                if line_count == 1: 
                    raw_skills = split_line
                    line_count += 1
                elif line_count == 2:
                    skills = [x.split(',') for x in split_line]
                    line_count += 1
                else: 
                    cards = split_line 
                    line_count = 1
                    self.turn_list.append(turnStruct(raw_skills, skills, cards))
                
        else:
            """
        # if the instructions are in coords
        self.np_dict = load_np_locs("np_locs.txt")
        self.ms_dict = load_master_skills("master_skill.txt")
        self.skill_dict = load_skill_coords("skill_locs.txt")

        # assuming 3 turn farming set up 
        for i in range(3):
            turn_code = open(instructions_list[i], "r").read().splitlines()
            self.turn_list.append(turn_code)
        
        self.supp_img = supp_img_path
        self.update_supp_img = update_supp_path
        self.yes_button = yes_button_path

        # get the bottom of the bluestacks screen 
        window_bounds = getWindow()
        self.region = window_bounds # to update: turn region into something pyautogui expects
        self.wind_bot = window_bounds[1] + window_bounds[-1] - 100
        self.wind_side = window_bounds[0] + window_bounds[-2] - 100

        # curr card number count, just for simplicity for now, will be upgraded to random int 
        self.curr_num = 1

    def find_refill(self):
        # look for golden apple, if you find one, click, if not do nothing
        refill =  click_img(golden_apple)
        if refill:
            time.sleep(1)
            # confirm refill 
            click_at([1088, 751])
            time.sleep(6)

    def farmCycle_coords(self):
        self.findSupport()
        time.sleep(12)
        # the actual farming loop (after support selection and battle starts)
        for turn in self.turn_list:
            self.farmTurn_coords(turn)
            time.sleep(32)

        # clear final screens
        for i in range(3):
            click_at([1290, 883])
            time.sleep(1)
        
        # press the next button
        click_at([1290, 883])
        time.sleep(1)
        # press the repeat button
        click_at([1034,764])
        time.sleep(1)
        # look for gold apple, if it shows up press it 
        self.find_refill()
        
    def farmTurn_coords(self, turn_codes):
        # to be implemented, with the coords txt
        # we just gotta loop through the commands element by element
        print(turn_codes)
        for action in turn_codes:
            label, instr = action.split(' ')
            if label == "raw":
                # instruction should contain key for skill dict 
                click_at(self.skill_dict[instr])
                time.sleep(3) # always wait 3 seconds after skill activations 
            elif label == "skill":
                # instruction should contain key for skill and location of servant to use it on 
                skill, target = instr.split(',')
                click_at(self.skill_dict[skill])
                time.sleep(0.5)
                click_at([550,631]) # assuming dps is at pos 1 always for now aka: 550,631
                time.sleep(3) # then wait 3 seconds for the skill activation

            elif label == "ms":
                 # for ms just click at the location specified, but time to wait changes
                click_at(self.ms_dict[instr])
                if instr == "M_sk":
                    time.sleep(0.5)
                elif instr == "ms_skill1" or instr == "ms_skill2": 
                    time.sleep(5)
                elif instr == "ms_skill3": # for now assume skill3 is the swap 
                    time.sleep(0.5)
                elif "swap" in instr:
                    time.sleep(0.5)
                elif "replace" in instr:
                    time.sleep(5)
                else:
                    time.sleep(3)
            elif label == "attack":
                # press space to attack 
                pyautogui.press('space')
                time.sleep(2)
            elif label == "cards":
                if instr == "random":
                    if self.curr_num >=5:
                        self.curr_num = 1
                    else:
                        self.curr_num += 1
                    pyautogui.press(str(self.curr_num))
                elif "np1" in instr:
                    click_at(self.np_dict['np1'])
                else:
                    print("why have you done this")
                time.sleep(0.5)
        return

    def farmTurn_imgs(self, turn):
        # perform the actions of the turn 

        # first click and wait for the raw skills to finish
        for img in turn.raw_skills:
            print("looking for: ", img)  
            click_img(img, self.region)
            time.sleep(3) # sleep 3 seconds for the skill activation 
        
        # complete the targetted skills (if any)
        for imgs in turn.skills: 
            click_img(imgs[0],  self.region) # click the skill 
            time.sleep(0.5)
            click_img(imgs[1],  self.region) # select the servant to use it on 
            time.sleep(3)   # wait for the skill activation 

        # press space to attack 
        pyautogui.press('space')
        time.sleep(0.5)

        # for turn cards, I think we may need actual coords...
        turn.cards[0] = pyautogui.locateOnScreen(turn.cards[0], confidence=0.9)
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
                # if we've already tried 7 times, refresh the friendlist 
                click_img(self.update_supp_img)
                time.sleep(0.5)
                click_img(self.yes_button)
                time.sleep(2)

            counter += 1

if __name__ == "__main__":
    time.sleep(2)
    command_list = ["turn1_cmds.txt", "turn2_cmds.txt", "turn3_cmds.txt"]
    farmer = Farmer(command_list)
    start_time = time.time()
    for i in range(10):
        farmer.farmCycle_coords()
    print("--- %s minutes ---" % (time.time() - start_time)/60)

    #sdict = load_np_locs("./np_locs.txt")
    #print(sdict)