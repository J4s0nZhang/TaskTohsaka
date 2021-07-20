from core import smartFarmer
import PySimpleGUI as sg 
import os
import time
import threading 

# globals 
command_list = ["turn1_cmds.txt", "turn2_cmds.txt", "turn3_cmds.txt"]

# main event loop 
repeat = True
sg.theme("Dark")
# try looking for previous history values
if os.path.exists("quickGUIhistory.txt"):
    with open("quickGUIhistory.txt", "r") as history:
        contents = history.readlines() 
    root_hist = contents[0].strip('\n')
    numTurns_hist = int(contents[1].strip('\n'))
else:
    root_hist = "./"
    numTurns_hist = 3

layout = [[sg.Text('Select root folder')],
        [sg.Input(default_text=root_hist), sg.FolderBrowse()],
        [sg.Text('Input number of turns to farm')],
        [sg.InputText(default_text=numTurns_hist)], 
        [sg.Radio("Windows", "RADIO1", default=True), sg.Radio("Mac", "RADIO1", default=False)],
        [sg.OK(), sg.Exit()]]

window = sg.Window("Get relevant info", layout)

while repeat: 
    event, values = window.Read()
    if event == "OK":
        root_dir = values[0]
        farm_turns = int(values[1])
        mac = values[3]
        # save input values if they changed
        if root_hist != root_dir or numTurns_hist != farm_turns:
            with open("quickGUIhistory.txt", "w") as file:
                file.writelines([root_dir + "\n", str(farm_turns)])
        # create pop up window while the farmer is running to let user know what to do
        sg.Popup("Farming turns, Switch to Emulator window with support screen", keep_on_top=True)
        # run the farmer 
        threading.Thread(target=smartFarmer.run_quick_turns, 
                        args=(command_list,root_dir, mac, farm_turns), daemon=True).start()
         

    if event == "Exit" or event == sg.WIN_CLOSED:
        repeat = False
        window.close()
        


