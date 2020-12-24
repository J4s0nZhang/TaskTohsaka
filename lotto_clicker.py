import smartFarmer 
import PySimpleGUI as sg 
import threading 

repeat = True
layout = [[sg.Text('Number of boxes to open')],
        [sg.InputText(default_text=9)], 
        [sg.Text('Seconds to wait')],
        [sg.InputText(default_text=110)],
        [sg.Text('Root Directory')],
        [sg.InputText(default_text="./"),],
        [sg.OK(), sg.Exit()]]
window = sg.Window("Lotto Opener")
while repeat:
    event, value = window.Read()
    if event == "OK":
        num_boxes = int(value[0])
        secs = int(value[1])
        root_dir = value[2]

        # create pop up window while the farmer is running to let user know what to do
        sg.Popup("Grabbing boxes", keep_on_top=True)
      
        threading.Thread(target=smartFarmer.get_lotto_boxes, 
                        args=(num_boxes, root_dir, secs), daemon=True).start()
    
    if event == "Exit" or event == sg.WIN_CLOSED:
        repeat = False
        window.close()