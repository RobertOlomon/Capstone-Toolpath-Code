import pyautogui
import keyboard
import time

'''
Press z to obtain the current coordinate of the cursor 
'''
stop=0 #set to specific time to stop program

while True:
    start=time.time()
    #Get the coordinates of mouse pointer or cursor everytime 'z' key is pressedz
    if keyboard.is_pressed('z') and (start-stop)>1:
        # Get the X and Y coordinates of the mouse pointer or curser
        position = pyautogui.position()
        print(position)
        stop=time.time()