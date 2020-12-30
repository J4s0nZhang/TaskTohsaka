# Overview

To use the quick and lotto automation systems, coordinates of specific buttons need to provided in wind_coords/ or mac_coords/ folders, depending on the system you are on. There are 4 txts that need to be filled. 

* button_coords.txt - the coordinates of various UI buttons 
* master_skill.txt - the coordinates of master skill locations (battle suit mystic code)
* np_locs.txt - the coordinates of the 3 servant noble phantasm locations 
* skill_locs.txt - the coordinates of the servant skills 

You can check out the current .txt files to see how they are formatted. 

### button coords.txt 
The keywords that correspond to the buttons are listed in the table. 

| keyword | button description |
| ------ | ------ |
| refill | refill button for using apples |
| start_quest | start quest button location |
| next | at the end of battle, the next button to move past drops and other screens location |
| repeat | the repeat quest button location to repeat the map |
| cancel | the cancel button to stop repeating the map |
| supp_port | the location to click the dps servant for targeted skills* |
| reset | reset lotto boxes button  |
| confirmation | confirmation button for lotto box reset |
| close | close button after lotto reset confirmation |
*the dps servant portrait location to click when using targeted skills like Skadi's quick buff 

### master_skill.txt

The master coords used in the quick automation script assume you are using the [Chaldea Combat Uniform](https://fategrandorder.fandom.com/wiki/Chaldea_Combat_Uniform). 
The coordinates listed in line order are: 
1. master skill button that reveals the 3 skills 
2. the first master skill location 
3. the second master skill location 
4. the third master skill location 
5. the frontline servant location when swapping with the 3rd master skill 
6. the back servant to swap with when using the 3rd master skill 
7. replace button for the 3rd master skill 

### np_locs.txt 

The np locations for the 3 servants, with the first servant's noble phantasm card location as the first line, the second servant's noble phantasm card as the second line, etc. 

Currently with the dantes team setup, only the first servant (which should be your aoe servant that does the actual damage) needs to be a real coordinate, the other two can be placeholders. 

### skill_locs.txt

The servant skill coordinates listed in line order are: 
1. servant 1, skill 1 
2. servant 1, skill 2
3. servant 1, skill 3
4. servant 2, skill 1
5. servant 2, skill 2
6. servant 2, skill 3
7. servant 3, skill 1
8. servant 3, skill 2
9. servant 3, skill 3 










