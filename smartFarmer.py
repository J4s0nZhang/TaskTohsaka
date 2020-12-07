from gameControl import click_at
from gameControl import click_img
from PIL import Image
import time
import pyautogui
import random

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

def load_button_coords(button_coords_txt): 
    # get the mouse coords of the 3 nps
    # get the mouse coords of the 3 nps
    button_dict = {}
    button_list = open(button_coords_txt).read().splitlines()
    for i in range(9):
        string = button_list[i].split()[0]
        button_dict[string] = [int(x) for x in button_list[i].split()[1].split(',')]

    return button_dict
# wrapper work flow for continuous farming of a node
# features to add in the future: automatically replenish AP, dantes brave chain generator
class turnStruct():
    def __init__(self, raw_skills=None, skills=None, cards=None, ms_skills=None):
        self.raw_skills = raw_skills  # list of image paths or coords of skills to press 
        self.skills = skills          # list of tuples of image paths or coords of targetted skills
        self.cards = cards            # lists of coords for card locations to press (default left 2 for now)
        self.ms_skills = ms_skills    # list of master skills 
        
class Farmer: 
    def __init__(self, instructions_list, mac=True):
        
        self.turn_list = []
        self.mac = mac
        # if the instructions are in coords
        if(mac):
            c_prefix = "./mac_coords/"
            i_prefix = "./mac_imgs/"
        else:
            c_prefix = "./wind_coords/"
            i_prefix = "./wind_imgs/"

        self.np_dict = load_np_locs(c_prefix+"np_locs.txt")
        self.ms_dict = load_master_skills(c_prefix+"master_skill.txt")
        self.skill_dict = load_skill_coords(c_prefix+"skill_locs.txt")
        self.button_dict = load_button_coords(c_prefix+"button_coords.txt")
        
        self.supp_img = i_prefix + "skadi_full.png"
        self.update_supp_img = i_prefix + "update_supp.png"
        self.yes_button = i_prefix +  "yes.png"
        self.golden_apple = i_prefix + "golden_apple.png"

        # assuming 3 turn farming set up 
        for i in range(3):
            turn_code = open(instructions_list[i], "r").read().splitlines()
            self.turn_list.append(turn_code)

        # curr card number count, just for simplicity for now, will be upgraded to random int 
        self.curr_num = 1

    def find_refill(self):
        # look for golden apple, if you find one, click, if not do nothing
        refill =  click_img(self.golden_apple, self.mac)
        if refill:
            time.sleep(1)
            # confirm refill 
            click_at(self.button_dict['refill'])
            time.sleep(6)

    def farmCycle_coords(self, i, end):
        self.findSupport()
        if i == 0:
            time.sleep(1)
            click_at(self.button_dict['start_quest'])   # click start quest button (only needed one time)
            time.sleep(3)
        time.sleep(25)
        # the actual farming loop (after support selection and battle starts)
        for turn in self.turn_list:
            self.farmTurn_coords(turn)
            time.sleep(32)

        # clear final screens
        for x in range(3):
            click_at(self.button_dict['next'])
            time.sleep(1)
        
        # press the next button
        click_at(self.button_dict['next'])
        time.sleep(1)
        
        # look for gold apple, if it shows up press it (don't do it if its the last run)
        
        if i != end:
            # press the repeat button
            click_at(self.button_dict['repeat'])
            time.sleep(1)
            self.find_refill()
        else:
            # press the cancel button 
            click_at(self.button_dict['cancel'])
        
    def farmTurn_coords(self, turn_codes):
        # to be implemented, with the coords txt
        # we just gotta loop through the commands element by element
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
                click_at(self.button_dict['supp_port']) # assuming dps is at pos 1 always for now aka: 550,631
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

    def lotto_clicker(self, timeout):
        # function to continuously click the lottery and then refresh once the box is done
        # time is what decides how long to click for before resetting the box 
        time_start = time.time()

        # press continuously for timeout seconds 
        while(time.time() < time_start + timeout):
            pyautogui.press("2")
            time.sleep(0.2)        

        click_at(self.button_dict['reset']) # click the reset button 
        time.sleep(1)
        click_at(self.button_dict['confirmation'])  # click confirmation button
        time.sleep(2)
        click_at(self.button_dict['close'])  # click close button
        time.sleep(2)

    def findSupport(self):
        # tries to automatically find the support using the supp img path 
        total_tries = 0
        counter = 1
        status = False
        while(not status):
            time.sleep(1)
            status = click_img(self.supp_img, self.mac)
            time.sleep(0.5)
            print(status)
            if status == False and (counter % 8 != 0): 
                # if we don't find the support, scroll down on the list 
                print("scrolling")
                pyautogui.scroll(-100)

            elif status == False and (counter % 8 == 0) and counter != 0 :
                # if we've already tried 7 times, refresh the friendlist 
                click_img(self.update_supp_img, self.mac)
                time.sleep(0.5)
                click_img(self.yes_button, self.mac)
                time.sleep(2)

            counter += 1
            # if we check for the support too many times, break out of the loop and exit
            if counter >= 15:
                break
        if status == True:
            return True
        else:
            return False

if __name__ == "__main__":
    
    time.sleep(2)
    command_list = ["turn1_cmds.txt", "turn2_cmds.txt", "turn3_cmds.txt"]
    farmer = Farmer(command_list, mac=False)
    end = 1
    start_time = time.time()
    for i in range(end):
        print("starting loop: ", i+1)
        farmer.farmCycle_coords(i, end-1)
        #farmer.lotto_clicker(110)
    print("--- {:.2f} minutes ---".format((time.time() - start_time)/60))
   
    