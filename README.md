# Task:Tohsaka

Task:Tohsaka is a collection of automation/ analysis tools for the mobile game Fate Grand/Order. Works on both Mac and Windows. Requires the [BlueStacks](https://www.bluestacks.com/) emulator since specific instructions are emulator specific. This is a personal project to automate farming during lottery events, and created in a way to prevent account banning due to 3rd party software being used. All scripts in this repo have no interactions with FGO itself, and thus cannot be traced. 

The features of this repo include:

  - quick team automation 
  - lotto box collection 
  - smart farming using deep learning 

# Newest Features
  - GUI for 3 turn quick team automation 
  - GUI for lotto box collection 

### Packages
To use the basic quick automation system without deep learning, you'll need the following packages: 

* pyautogui 0.9.48
* pysimplegui 4.32.1
* FGO emulated on BlueStacks

To check out the WIP deep learning card classification models, you need these additional packages: 

* tensorflow > 2.0.0
* numpy
* matplotlib 

### Installation and usage 

Recommended to install above package dependencies using an [Anaconda environment](https://www.anaconda.com/). 

Once dependencies are installed, run the dantes quick team loop GUI inside the env with:
```
python guis/dantes_quick.py 
```
run the lotto box clicker GUI with:
```
python guis/lotto_clicker.py
```

The current implementation requires a specific set of screen coordinates/ screenshots for in-game buttons. These will have to differ machine to machine due to varying screen size and resolutions. The coordinate information is stored in mac_coords and wind_coords respectively, and the images are stored in mac_imgs and wind_imgs. 

For more information on which coordinates need to be obtained and stored in what order, refer to [docs]().

The dantes quick loop instructions are stored on a per turn basis in turn<#>_cmds.txts. Txts for all 3 turns are required, and a 3 turn farming setup is assumed based on the nature of quick teams. 

For more information on setting up the turn command txts, refer to [docs]().
