# Overview 

This doc outlines the different keywords used to construct turn commands in turn1_cmds.txt, turn2_cmds.txt and turn3_cmds.txt. The general structure is that each line in the txts is an action that the script will do based on the keyword and the object, separated by space. 

e.g skill s2_s1,dps --- where skill is the keyword, and s2_s1,dps is the object for the action

## Keywords

| keyword | action |
| ------ | ------ |
| raw | a servant skill that only needs to be clicked once to activate (no target selection needed, e.g dantes np gain buff) |
| skill | a servant skill that needs a target after clicking to activate (e.g Skadi's quick buff) |
| ms | a master skill button |
| attack | press the attack button to get to the card selection |
| cards | press a normal command card |

## object 

###Servant skill distinction system
For action keywords like raw and skill, the servant and the skill must be specified in the following format: 
s{servant_number}_s{skill_number} --- e.g for skill 1 of the first servant, it would be s1_s1

For skill specifically, the servant recieving the buff* (dps) needs to be specified afterward, separated by a comma like so: 

s1_s1,dps

### Master skill and card selection 
* the attack keyword needs a placeholder object: sp 
* card keyword has the random object, which selects any card. It also has the np1 keyword, which selects the noble phantasm card of the first servant 

Master skill objects and descriptions table
| object | description |
| ------ | ------ |
| M_sk | the master skill button that reveals the 3 skills  |
| ms_skill1 | first master skill |
| ms_skill2 | second master skill |
| ms_skill3 | third master skill |
| swap | swapping servant portrait |
| replace | press the replace button once both servants to swap are selected |






