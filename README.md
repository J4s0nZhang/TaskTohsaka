# Task: Tohsaka
A suite of helper bots and software to play the mobile game Fate Grand Order. This repo contains image classification models used to make decisions in the game, as well as classes which automate farming for certain events. 

Currently implemented is the baseline models for card type classifiction (arts, buster, quick), as well as a basic siemese network to differenate different servant face cards. 

The farmer class found in smartFarmer.py is supposed to be able to use these model for more automated farming in the future, but currently it executes 3turn dantes quick loop setup via coordinates and pyautogui. It reads the txts as instructions and coordinates, and can completely autonomously farm events that work with the 3turn farming setup. 