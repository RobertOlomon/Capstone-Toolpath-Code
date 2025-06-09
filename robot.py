import abb_motion_program_exec as abb
from abb_robot_client.rws import RWS
import numpy as np
import pyautogui
import time
import os


"""
Enables python communication to the ABB robot.
"""

class Robot:
    def __init__(self):
        self.setup_client()
        self.tools_dict = {
            "laser": {
                "get_obs": "laser_tcp", 
                "move": self.laser_tcp,
                "home_pos": np.array([-430.25, -958.73, 431.79, 0, 0, -0.70710678, 0.70710678])
                },
            "lidar": {
                "get_obs": "lidar_tcp", 
                "move": self.lidar_tcp,
                "home_pos": np.array([-195.92, -945.16, 416.93, 0, 0, -0.70710678, 0.70710678]) 
                },
            "tool0": {
                "get_obs": "tool0", 
                "move": None,
                "home_pos": np.array([-367, -860, 559, 0, 0.70710678, 0.70710678, 0])
                }
            }
        self.action_space = 7
    
    def setup_client(self):
        self.client = abb.MotionProgramExecClient(base_url='http://192.168.100.50:80')
        self.rws_client = RWS(base_url='http://192.168.100.50:80')
        self.laser_tcp = abb.tooldata(True, 
                                      abb.pose([-98.839,-63.321,127],[0.5,0.5,-0.5,-0.5]), 
                                      abb.loaddata(9,[11.934,-5.704,132.771],[1,0,0,0],0,0,0))
        self.lidar_tcp = abb.tooldata(True, 
                                      abb.pose([-85.214,171.034,142.016],[0.5,0.5,-0.5,-0.5]), 
                                      abb.loaddata(9,[11.934,-5.704,132.771],[1,0,0,0],0,0,0))
    
    def get_obs(self, tool="tool0"):
        '''
        Get current robot coordinates and quaternions
        '''
        rob_pos = self.rws_client.get_robtarget(tool=self.tools_dict[tool]["get_obs"])
        trans = [0] * self.action_space
        trans[:3] = rob_pos.trans
        trans[3:] = rob_pos.rot
        return np.array(trans)
    
    def reset(self, tool="tool0"):
        return self.step(self.tools_dict[tool]["home_pos"], tool=tool)
    
    def step(self, action, tool="tool0"):
        '''
        Moves robot to a position given an array of points (x,y,z,q1,q2,q3,q4)
        '''

        #set action as a table_trans value if needed
        #action=table_trans(action)
        assert isinstance(action, np.ndarray) and  action.shape == (self.action_space,)
        target = abb.robtarget(action[:3], action[3:], abb.confdata(0.,0.,0.,0.), [9e9]*6)
        self.mp = abb.MotionProgram(tool=self.tools_dict[tool]["move"])
        self.mp.MoveL(target, abb.v30, abb.z100)
        self.client.execute_motion_program(self.mp)
        print('moved robot')
        return self.get_obs()

    def process_obs(self, obs):
        [w, x, y, z] = obs[3:]
        return obs[:3] / 1000, [x, y, z, w]

class Laser:
    def ablate():
        '''
        starts laser given laser IPGmark is windowed to the right side the of the screen 
        and start process window is untouched.
        Will automatically detect when laser is finished
        '''

        print('ablating')
        for point in program_coordinates:
            done_color=(229, 229, 229) #start button turns 

            pyautogui.moveTo(point[0],point[1])
            ablate_status='True'
            pyautogui.click()
            pixel_color = pyautogui.pixel(point[0],point[1])
            print('waiting for laser to finish')
            while  pixel_color != done_color:
                pixel_color = pyautogui.pixel(point[0],point[1])
            ablate_status='False'
            
def table_trans(cord):
    '''
    Convert coordinates relative to the table (left corner=orgin)
    to global coordinate system (robot base orgin)
    '''
    fixed_cord= robot.get_obs() # get current position as waypoint
    fixed_cord[0]=cord[0]+x_orgin
    fixed_cord[1]=cord[1]+y_orgin
    fixed_cord[2]=cord[2]+z_orgin  #relative to center of torpedo in experiemnt set up 

    return np.array(fixed_cord) 

def onepass(passes):
    '''
    Code for ablating though single lines given the 
    number of ablation patterns created(used for ablation testing)
    '''
    offset=127 #set offset from edge (set to 0 to ablate fully from the edge)
    waypoint = robot.get_obs() # get current position as waypoint
    delta = np.array([-30-offset, 0, 0, 0, 0, 0, 0]) # moves away from current position by deltas; positive x is left, positive y is backwards, positive z up
    print(waypoint+delta)
    robot.step(waypoint+delta) ##set to first ablation pattern 
    print('start first ablation')
    Laser.ablate()
    
    for i in range(passes-1):  # repeats ablation 
        waypoint = robot.get_obs() # get current position as waypoint
        delta = np.array([-60, 0, 0, 0, 0, 0, 0]) # moves away from current position by deltas; positive x is left, positive y is backwards, positive z up
        print(waypoint+delta)
        robot.step(waypoint+delta) # move by delta
        Laser.ablate()

def scan_folder_for_npy_files(folder_path):
    # Initialize an empty list to store file paths
    file_paths = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder_path):
        # Check if the file has a .npy extension
        if file_name.endswith('.npy'):
            file_path = os.path.join(folder_path, file_name)
            
            # Append the file path to the list
            file_paths.append(file_path)

    # Convert the list of file paths into a NumPy array (single-column matrix)
    file_paths_matrix = np.array(file_paths).reshape(-1, 1)

    return file_paths_matrix

if __name__ == "__main__":
    robot = Robot()

    ablate_status='False' #check for if laser is running

    #get current robot coordinates and test if step is working
    waypoint = robot.get_obs() # get current position as waypoint
    print(waypoint)
    robot.step(waypoint)

    x_orgin=454.15 #left corner of the table relative to base(mm)
    y_orgin=-681.11
    z_orgin=260.05 

    program_coordinates = [
        (1632, 717)  #start button for laser *Ensure screen is right windowed
    ]

    '''
    moves robot to coords (example code for changing coordinates)
    waypoint = robot.get_obs() # get current position as waypoint
    waypoint[0] = -122.97
    waypoint[1] = -1100.64
    waypoint[2] = 681.75
    robot.step(waypoint)
    '''

    robot.step(table_trans([0,0,0]))#orgin point of table or home position 
    #onepass(3)    #abalte a single line with 3 ablation scans 

    filepaths=scan_folder_for_npy_files(r"PUT FILE PATH HERE")
    num_of_files = len(filepaths) 

    for i in range(num_of_files):
        #convert loaded file to 1D array
        curr_file = np.load(filepaths[i][0])
        #print(curr_file) #check current coordinate

        num_of_points = len(curr_file)
        
        for i in range(num_of_points): #for each coordinate, run the robot and ablate location 
            
            robot.step(curr_file[i])
            Laser.ablate()

    print('end program')
