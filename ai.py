from ultralytics import YOLO
from shapely.geometry import Polygon, Point
import os
import sys
import math
import time
import pickle
from PIL import Image
import pandas as pd
import argparse
from pymavlink import mavutil
import cv2
import re
import shutil
import numpy as np

# Define your list of desired targets
desired_targets = [
    {'shape': 'Quatercircle', 'color': 'red', 'letter': 'd', 'id': 1},
    {'shape': 'Emergent', 'color': None, 'letter': None, 'id': 2},
    {'shape': 'Star', 'color': 'white', 'letter': 'a', 'id': 3},
    {'shape': 'Cross', 'color': 'purple', 'letter': 'z', 'id': 3},
    {'shape': 'Pentagon', 'color': 'green', 'letter': 'w', 'id': 5}
    # Add more desired targets here
]

# Default values for variables, if any defaults changed please change it in the below description also
defaults = {
    'grid': False,
    'suas': False,
    'alt': False,
    'fov': False,
    'lap': False,
    'test': False,
    'sitl': False
}

# Descriptions for each argument
descriptions = {
    'grid': 'Pass grid if you want to enable the inference inside the Search Grid',
    'suas': 'Pass if its actual SUAS mission',
    'alt': 'Pass if you want to inference the images only at certain height ',
    'fov': 'Pass if you want to get the cropped images from the specific circular range',
    'lap': 'Pass if you want to drop the bottles lap wise, and make sure the waypoint file is in the specific lap format',
    'test': 'Pass to skip inference, and directly from the CSV file',
    'sitl': 'Pass this to connect to the sitl'
}


# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Script for performing automation task.')

for key, value in defaults.items():
    if key in ['grid','suas','alt','fov','lap', 'test', 'sitl']:
        parser.add_argument(f'-{key}', action='store_true', help=f'{descriptions[key]}, Default value: \'{value}\'')

# Parse the command-line arguments
args = parser.parse_args()

# Get value from the argparse
grid = bool(args.grid)
SUAS = bool(args.suas)
alt_limiter = bool(args.alt)
FOV = bool(args.fov)
lap_en = bool(args.lap)
test_en = bool(args.test)
sitl_en = bool(args.sitl)


print('Value inside the test_en:',test_en)

if not test_en:
    try:
        shutil.rmtree('/home/edhitha/ultralytics/yo/output1/test4')
    except:
        pass

    try:
        shutil.rmtree('/home/edhitha/ultralytics/yo/output2/test5')
    except:
        pass

    try:
        shutil.rmtree('/home/edhitha/ultralytics/yo/output3/test6')
    except:
        pass

    try:
        os.makedirs( '/home/edhitha/ultralytics/yo/log_results', exist_ok=True)
        shutil.move('/home/edhitha/ultralytics/yo/output1/test4', '/home/edhitha/ultralytics/yo/log_results')
    except Exception as e:
        print(e)
        pass
# Initialize YOLO models
start_time  = time.time()
model1 = YOLO('/home/edhitha/ultralytics/runs/detect/weights/best.pt')
print('Time required to load the first model:', time.time() - start_time)
model2 = YOLO('/home/edhitha/ultralytics/runs/classify/colors_filt/weights/best.pt')
print('Time required to load the second model:', time.time() - start_time)
model3 = YOLO('/home/edhitha/ultralytics/runs/classify/letters/weights/best.pt')
print('Time required to load the third model:', time.time() - start_time)
model4 = YOLO('/home/edhitha/ultralytics/runs/segment/16mm_shapesegment/weights/best.pt')
print('Time required to load the fourth model:', time.time() - start_time)

# Define input and output folders
input_folder = '/home/edhitha/DCIM/test_cam_1000/images'
output_folder = '/home/edhitha/ultralytics/yo/output1'
output_folder2 = '/home/edhitha/ultralytics/yo/output2'
output_folder3 = '/home/edhitha/ultralytics/yo/output3'
output_folder4 = '/home/edhitha/ultralytics/yo/output4'
results_file = '/home/edhitha/ultralytics/yo/results.txt'
results_csv_file = '/home/edhitha/ultralytics/yo/output1/test4/crops/results/results.csv'
modified_csv_file = '/home/edhitha/ultralytics/yo/output1/test4/crops/results/results1.csv'

eliminate_csv_file = '/home/edhitha/ultralytics/yo/output1/test4/crops/Target/detected.csv'
Emergent_folder = os.path.join(output_folder, 'test4', 'crops', 'Emergent')
Target_folder = os.path.join(output_folder, 'test4', 'crops', 'Target')
# /home/edhitha/ultralytics/yo/output1/test4/crops/results')

# Necessary directories
# os.makedirs('/home/edhitha/ultralytics/yo/results', exist_ok=True)
os.makedirs('/home/edhitha/ultralytics/yo/output1/test4/crops/Target', exist_ok=True)
os.makedirs('/home/edhitha/ultralytics/yo/output1/test4/crops/results', exist_ok=True)
# Pickle path
pickle_path = '/home/edhitha/DCIM/test_cam_190/cam_wps.pickle'
# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Initialize a set to keep track of processed image file names
processed_images = set()

# Initialize a dictionary to keep track of class_name counts
class_name_counts = {}

# Global connection to drone
global the_connection

# Define the Threshould limit for inference
threshould_height = 29

# Global GSD
GSD = 0.3
rem = 0   

# ANSI escape code for red text
red_text = "\033[91m"
blue_text = "\033[94m"
green_text = "\033[92m"
purple_text = "\033[95m"
orange_text = "\033[93m"

# Reset the text color to default after printing
reset_color = "\033[0m"

removed_targets = []

# number of targets 
num_targets = len(desired_targets)

# Lat lon of the SUAS search grid
SUAS_search_grid = [
    (38.31442311312976, -76.54522971451763),
    (38.31421041772561, -76.54400246436776),
    (38.31440703962630, -76.54394394383165),
    (38.31461622313521, -76.54516993186949),
    (38.31442311312976, -76.54522971451763)
]

# Lat lon of the other location
test_search_grid = [
    (13.6155446, 77.5724234),
    (13.6159959, 77.5715540),
    (13.6158929, 77.5715031),
    (13.6154706, 77.5723788),
    (13.6155446, 77.5724234)
      
]

poly=None
if grid:
    if SUAS:
        poly =Polygon(SUAS_search_grid)
    else:
        poly=Polygon(test_search_grid)

def is_point_inside_polygon(point, polygon_points):
    poly = Polygon(polygon_points)
    point = Point(point)
    return poly.contains(point)

def remove_pattern_from_filename(filename):
    # Define the pattern to be removed
    pattern = r'\.[a-f0-9]{32}'

    # Use regular expression to substitute the pattern with an empty string
    new_filename = re.sub(pattern, '', filename)

    return new_filename

# Function to check if a target is in the desired targets list
def is_target_in_desired_list(target):
    for i, desired_target in enumerate(desired_targets):
        if (desired_target['shape'] == target['shape']
            and desired_target['color'] == target['color']
            and desired_target['letter'] == target['letter']):
            return True, None, desired_targets[i]
        elif (desired_target['shape'] == target['shape']
            and desired_target['color'] == target['color']): 
            return True, 'Replace Letter', desired_targets[i]
        # elif (desired_target['shape'] == target['shape']): 
        #     return True, 'Replace color', desired_targets[i]
        #elif (desired_target['color'] == target['color']): 
            #return True, 'Letter and shape Not found', desired_targets[i]
    return False, None, None

#Function to remove a target from the desired targets list
def remove_target_from_desired_list(target):
    for i, desired_target in enumerate(desired_targets):
        if (desired_target['shape'] == target['shape']
            and desired_target['color'] == target['color']
            and desired_target['letter'] == target['letter']):
            del desired_targets[i]
            break

def get_midpoint(image_path):
    # Open the image
    img = Image.open(image_path)
    # Get the width and height of the image
    width, height = img.size
    # Calculate the mid_x and mid_y coordinates
    mid_x = width // 2
    print("in midpoint fxn",file=sys.stderr)
    mid_y = height // 2
    return (mid_x, mid_y)

def get_image_dimensions(image_path):
    # Open the image
    img = Image.open(image_path)
    # Get the width and height of the image
    width, height = img.size
    return (width, height)

def drop(index):
    global rem
    # Set the system and component ID (replace with your system and component ID)
    system_id = the_connection.target_system
    component_id = the_connection.target_component

    # Set the auxiliary pin or relay channel (replace 0 with your channel number)
    channel = 7

    pwm_values = [1600,1300,1000,1000,1000]
    # Set the initial PWM value
    pwm_value = pwm_values[index-1]
    print(pwm_value)
    # Set the MAV_CMD_DO_SET_SERVO command parameters
    command = mavutil.mavlink.MAV_CMD_DO_SET_SERVO
    param1 = channel  
    param2 = pwm_value

    # Send the MAV_CMD_DO_SET_SERVO command
    the_connection.mav.command_long_send(
        system_id, component_id,
        command,
        0,  # Confirmation
        param1, param2, 0, 0, 0, 0, 0
    )

def parse_gps_entry(file_path):
    gps_str = os.environ.get(file_path)
    if gps_str:
        try:
            gps_data = gps_str.strip('()').split(',')
            if len(gps_data) == 6:
                latitude, longitude, altitude, heading, yaw, time = map(float, gps_data)
                return latitude, longitude, altitude, heading, yaw, time
            else: 
                return None, None, None, None, None, None
        except:
            return None, None, None, None, None, None
    else:
        return None, None, None, None, None, None

def newlatlon(lat , lon , hdg ,dist, movementHead):
    lati=math.radians(lat)
    longi=math.radians(lon)
    rade = 6367489
    AD = dist/rade
    print("5",file=sys.stderr)
    sumofangles = (hdg + movementHead)%360
    newheading = math.radians(sumofangles)
    newlati = math.asin(math.sin(lati)*math.cos(AD) + math.cos(lati)*math.sin(AD)*math.cos(newheading))
    newlongi = longi + math.atan2(math.sin(newheading)*math.sin(AD)*math.cos(lati), math.cos(AD)-math.sin(lati)*math.sin(newlati))
    ret_lat = int(round(math.degrees(newlati*1e7)))
    ret_lon = int(round(math.degrees(newlongi*1e7)))
    return ret_lat, ret_lon

print("Waiting for images")
while not os.path.exists(input_folder):
    continue

def read_dict_from_file(pickle_path=pickle_path):
    while True:
        try:
            # Synchronous code to read from the file
            with open(pickle_path, 'rb') as file:
                loaded_dict = pickle.load(file)
            # print(loaded_dict)
            return loaded_dict
        except Exception as e:
            print('Error occured while loading the cam_wps:',e)
            time.sleep(0.2133253665)

def get_blur_constant(img_path):
    # Read the image
    img = cv2.imread(img_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Calculate the gradient using the Sobel operator
    gradient_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the magnitude of the gradient
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Calculate the variance of the gradient magnitude
    laplacian_var = np.var(gradient_magnitude)

    return laplacian_var

def sort_dataframe_by_id(df):
    """
    Sorts a DataFrame based on the 'ID' column in ascending order.

    Parameters:
    - df: DataFrame to be sorted

    Returns:
    - Sorted DataFrame
    """
    return df.sort_values(by='ID', ascending=True)

def get_coordinates(image_name,image_path,pickle_path,point2, fov_circle):
    wps_list = read_dict_from_file(pickle_path)
    # print(wps_list)
    index = int(image_name[-9:-4])
    lat, lon, alt, head, yaw, time1 = wps_list[index] 
    
    if lat is not None and lon is not None:
        point1 = (center_x, center_y)
        pixel_distance = math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
        if FOV:
            if pixel_distance > fov_circle:
                return None,None,None
        angle = math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0])) - 90
        real_distance_gsd = (pixel_distance * GSD) / 100

        if lat == 0:
            print(f"NO LAT AND LON FOUND, {image_path}", file=sys.stderr)

        value = newlatlon(lat, lon, yaw, real_distance_gsd, angle)
        return value
    else:
        return None,None,None

def connect():
    print("Drone connecting ...",file=sys.stderr)
    global the_connection
    if sitl_en:
        the_connection = mavutil.mavlink_connection('udpin:192.168.1.21:14553')
    else:
        the_connection = mavutil.mavlink_connection('udpin:127.0.0.1:14553')

    # Wait for the heartbeat message to be received
    the_connection.wait_heartbeat()
    print('Got HeartBeat from the drone ...')

    return the_connection

def distance_lat_lon(lat1, lon1, lat2, lon2):
    '''distance between two points'''
    dLat = math.radians(lat2) - math.radians(lat1)
    dLon = math.radians(lon2) - math.radians(lon1)
    a = math.sin(0.5*dLat)**2 + math.sin(0.5*dLon)**2 * math.cos(lat1) * math.cos(lat2)
    c = 2.0 * math.atan2(math.sqrt(abs(a)), math.sqrt(abs(1.0-a)))
    ground_dist = 6371 * 1000 * c
    return ground_dist

def guided():
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,176, 0, 1, 4, 0, 0, 0, 0, 0)
    msg = the_connection.recv_match(type='COMMAND_ACK', blocking=True)
    print(msg,file=sys.stderr)

def auto():
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,176, 0, 1, 3, 0, 0, 0, 0, 0)
    msg = the_connection.recv_match(type='COMMAND_ACK', blocking=True)
    print(msg,file=sys.stderr)

def loiter():
    the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,
                                     176, 0, 1, 5, 0, 0, 0, 0, 0)
    msg = the_connection.recv_match(type='COMMAND_ACK', blocking=True)
    print(msg,file=sys.stderr)

def automation(value,target):
    print("target aquired",file=sys.stderr)
    # print('Value:', value)
    gps = the_connection.recv_match(type='GLOBAL_POSITION_INT', blocking = True)
    # print(gps)
    calculated_distance =  distance_lat_lon(value[0],value[1],gps.lat/1e7,gps.lon/1e7)

    kurrentalti = 30

    accuracy = 0.1

    # print(kurrentalti)
    print('Guided',file=sys.stderr)
    guided()
    time.sleep(1.5)
    the_connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_global_int_message(10, the_connection.target_system,the_connection.target_component, 6, 1024, int(value[0]*1e7), int(value[1]*1e7),kurrentalti , 0, 0, 0, 0, 0, 0, 0,0))
    # msg = the_connection.recv_match(type='COMMAND_ACK', blocking=True)
    print('Desired Latitude and Longitude sent ....', file=sys.stderr)
    print('Desired lat and lon:', value[0], value[1])
    print("Real_distance between lats and lons", calculated_distance,file=sys.stderr)
# Wait until the drone reaches the target location
    try:
        while True:
            msg = the_connection.recv_match(type=['GLOBAL_POSITION_INT'], blocking=True)
            current_lat = msg.lat / 1e7
            current_lon = msg.lon / 1e7
            distance = distance_lat_lon(current_lat,current_lon,value[0],value[1])
            print('distance between drone and target:',distance, 'for lat and lon:', current_lat, current_lon, file=sys.stderr)
            if distance <= accuracy:
                print('Started Dropping ...',file=sys.stderr)
                drop(target)
                
                the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,181, 0, 0, 1, 0, 0, 0, 0, 0)
                if sitl_en:
                    time.sleep(5)
                else:
                    time.sleep(20)
                    
                print('dropped',file=sys.stderr) 
                print('Current drop loc latitude:', msg.lat / 1e7,file=sys.stderr)
                print('Current drop loc longtitude:', msg.lon / 1e7,file=sys.stderr)
                auto() # Replace with your desired action
                break
    except Exception as e:
        print(e)
        pass

def get_wps_and_indicies(vehicle):
    # Request mission items
    vehicle.mav.mission_request_list_send(vehicle.target_system, vehicle.target_component)

    # Wait for the mission count message
    count_msg = vehicle.recv_match(type='MISSION_COUNT', blocking=True)
    num_items = count_msg.count

    # Request each mission item
    commands = []
    for seq in range(num_items):
        vehicle.mav.mission_request_send(vehicle.target_system, vehicle.target_component, seq)

        # Wait for the mission item message
        item_msg = vehicle.recv_match(type='MISSION_ITEM', blocking=True)
        commands.append(item_msg)

    # Create a DataFrame to store mission commands
    columns = ['Index', 'Latitude', 'Longitude', 'Altitude', 'Command', 'Param1', 'Param2', 'Param3', 'Param4', 'AutoContinue']
    commands_df = pd.DataFrame(columns=columns)

    # Populate DataFrame with mission command data
    for cmd in commands:
        commands_df.loc[len(commands_df)] = [
            cmd.seq,
            cmd.x,  # Coordinates are in degrees * 1e7
            cmd.y,
            cmd.z,
            cmd.command,
            cmd.param1,
            cmd.param2,
            cmd.param3,
            cmd.param4,
            cmd.autocontinue
        ]

    # Print the DataFrame
    print("Mission Commands DataFrame:")
    print(commands_df)

    # Find the WP index which is before the 177 command
    wp_last_occurrence = commands_df[commands_df['Command'] == 177].index[0] - 1

    # Find the last occurence of 177 command 
    search_wp_last_occurrence = commands_df[commands_df['Command'] == 177].index[-1] 
    while commands_df.loc[search_wp_last_occurrence]['Command'] != 16:
        search_wp_last_occurrence -= 1
    # Print the result
    print(f"wp_last_occurrence: {blue_text}{wp_last_occurrence}{reset_color}, search_wp_last_occurrence: {blue_text}{search_wp_last_occurrence}{reset_color}")
    # search_grid_wp_start = None
    # for index, row in commands_df.loc[second_occurrence:last_occurrence].iterrows():
    #     # Your processing logic for each row goes here
    #     print(row['Index'], row['Latitude'], row['Longitude'], row['Altitude'], row['Command'])
    #     if row['Command'] == 16 and row['Latitude'] and row['Longitude'] and search_grid_wp_start == None:
    #         search_grid_wp_start = index
    #     elif row['Command'] == 16 and row['Latitude'] and row['Longitude']:
    #         search_grid_wp_end = index

    # print('Search_grid_wp_start:', search_grid_wp_start)
    # print('Search_grid_wp_end:', search_grid_wp_end)
    return commands_df.loc[wp_last_occurrence], commands_df.loc[search_wp_last_occurrence], wp_last_occurrence, search_wp_last_occurrence, commands_df

def current_wp(the_connection):
    while True:
        # Wait for a new message from the autopilot
        msg = the_connection.recv_match()
        if not msg:
            continue

        # Check if the message is the current mission item
        if msg.get_type() == 'MISSION_CURRENT':
            print('Current mission item:', msg.seq)
            return msg.seq

def is_near_to_loc(lat1, lon1, lat2, lon2):
    '''distance between two points'''
    dLat = math.radians(lat2) - math.radians(lat1)
    dLon = math.radians(lon2) - math.radians(lon1)
    a = math.sin(0.5*dLat)**2 + math.sin(0.5*dLon)**2 * math.cos(lat1) * math.cos(lat2)
    c = 2.0 * math.atan2(math.sqrt(abs(a)), math.sqrt(abs(1.0-a)))
    ground_dist = 6371 * 1000 * c

    if 5 > ground_dist:
        return True
    else:
        return False

# Define column names
columns = ['ID','Image File', 'Image Path', 'Cropped Image', 'Object Type', 'Color', 'Letter', 'Coordinates_x', 'Coordinates_y', 'Lat', 'Lon', 'Cropped Image Path']
# Initialize an empty DataFrame with columns
results_df = pd.DataFrame(columns=columns)
eliminate_df = pd.DataFrame(columns=columns)
timeout_duration = 5
start_time = time.time()
dropped_list = []
flag = 1
wp_index = None
search_index = None
prev_files = None
mid_point = None
wp_row = None
search_wp_row = None
repeat_pattern = 0
if not lap_en:
    the_connection = None
if lap_en:
    the_connection = connect()
    try:
        wp_row, search_wp_row, wp_index, search_index, commands_df = get_wps_and_indicies(the_connection)
    except Exception as e:
        print('Error occured:',red_text, e, reset_color)
        print(orange_text,'Terminating the lap based ...', reset_color)
        lap_en = False 

while True and not test_en:
    # List all files in the input folder
    image_files = os.listdir(input_folder)
    initial_files = set(os.listdir(input_folder))
    # Filter out files that have already been processed
    new_image_files = [img for img in image_files if img not in processed_images]

    for image_file in new_image_files:
        image_path = os.path.join(input_folder, image_file)
        # Check if the image has not been processed
        if image_path not in processed_images:
            if grid or alt_limiter:
                try:
                    image_index = int(image_file[-9:-4])
                    loaded_dic = read_dict_from_file()
                    lat1, lon2, alt, head, yaw, time1 = loaded_dic[image_index]
                    if alt_limiter:
                        if threshould_height > alt:
                            print('Skipping image file:', image_file,orange_text,'ALT CONDITION',reset_color)
                            processed_images.add(image_path)
                            continue
                            
                    if grid:
                        if SUAS:
                            if not poly.contains(Point(lat1,lon2)):
                                print('Skipping image file:', image_file,orange_text,'SUAS GRID CONDITION',reset_color)
                                processed_images.add(image_path)
                                continue
                        else:
                            if not poly.contains(Point(lat1,lon2)):
                                print('Skipping image file:', image_file,orange_text,'TEST GRID CONDITION',reset_color)
                                processed_images.add(image_path)
                                continue
                except: 
                    continue

            # Perform object detection and save cropped images
            results = model1.predict(source=image_path, project=output_folder, name='test4', exist_ok=True ,device=0,
            save_crop=True)
            # Open the results file for appending
            
            if mid_point == None:
                mid_point = get_midpoint(image_path)
                fov_range = math.sqrt((mid_point[1]) ** 2 + (mid_point[1]) ** 2)

            for i, result in enumerate(results):
                for j, box in enumerate(result.boxes):
                    class_name = result.names[box.cls[0].item()]
                    print("this ia the detected target class")
                    print(class_name)
                    coords = box.xyxy[0].tolist()
                    coords = [round(x) for x in coords]
                    center_x = (coords[0] + coords[2]) / 2
                    center_y = (coords[1] + coords[3]) / 2

                    value = get_coordinates(image_file, image_path, pickle_path, mid_point, fov_range)
                    
                    if FOV:
                        if value[0] == None:
                            print('Skipping image file:', image_file,orange_text,'FOV CONDITION',reset_color)
                            processed_images.add(image_path)
                            continue

                    # Assuming class_name_counts is a dictionary that keeps track of counts for each class name
                    if class_name_counts.get(class_name, 0) > 0:
                        # If class_name exists in class_name_counts and its count is greater than 0
                        count = class_name_counts[class_name] + 1  # Increment the count
                        class_name_counts[class_name] = count  # Update the count in the dictionary
                        crop_name = f"{image_file[:-4]}{class_name}{count}.jpg"
                        real_name = f"{image_file[:-4]}{count}.jpg"
                    else:
                        # If class_name doesn't exist in class_name_counts or its count is 0
                        class_name_counts[class_name] = 1  # Initialize the count for this class name
                        crop_name = f"{image_file[:-4]}{class_name}.jpg"  # Append '.jpg' for the first occurrence
                        real_name = f"{image_file[:-4]}.jpg"
                    print(crop_name)
                    print("this is the crop name ccccccccccc")
                    print(crop_name)
                    
                    real_path = os.path.join(output_folder, 'test4', 'crops', class_name , real_name)
                    crop_path = os.path.join(output_folder, 'test4', 'crops', class_name, crop_name)
                    os.rename(real_path, crop_path)
                    
                    
                    lat = value[0] / 1e7
                    lon = value[1] / 1e7
                    print('Saving in the eliminate DF ...')
                    shape = None
                    letter = None
                    color = None
                    eliminate_df = pd.concat([eliminate_df, pd.DataFrame({
                        'ID': [None],
                        'Image File': [image_file],
                        'Image Path': [image_path],
                        'Cropped Image': [crop_name],
                        'Object Type': [shape],
                        'Color': [color],
                        'Letter': [letter],
                        'Coordinates_x': [center_x],
                        'Coordinates_y': [center_y],
                        'Lat': lat,
                        'Lon': lon,
                        'Cropped Image Path': [os.path.join(output_folder, 'test4', 'crops', class_name, crop_name)]
                    })], ignore_index=True)
                    print( crop_path)
                    if class_name == 'Emergent':
                        shape = 'Emergent'
                        color = None 
                        letter = None
                    
                    else:
                        results4 = model4.predict(source=crop_path, project=output_folder4, name='test7', exist_ok=True,device=0)
                        class_list = []
                        for i, results12 in enumerate(results4):
                            for j, box in enumerate(results12.boxes):
                                shape_name = results12.names[box.cls[0].item()]
                                # print(class_name)
                                class_list.append(shape_name)
                        if class_list:
                            shape = class_list[0]
                            results2 = model2.predict(source=crop_path, project=output_folder2, name='test5', exist_ok=True,device=0, save=True)
                            color = results2[0].names[results2[0].probs.top1]

                            results3 = model3.predict(source=crop_path, project=output_folder3, name='test6', exist_ok=True, save=True, device=0)
                            letter_indices = results3[0].probs.top5
                            letter_list= [results3[0].names[index] for index in letter_indices]
                            letter_list=letter_list[:-3]
                            for desired_target in desired_targets:
                                if (desired_target['shape']==class_name and desired_target['color']==color):
                                    matching_letter=next((letter for letter in letter_list if letter == desired_target['letter']), None)
                                    if matching_letter is not None:
                                        letter=matching_letter
                                    else:
                                        letter=letter_list[0]
                                        
                                    break
                            else:
                                letter=letter_list[0]
                        else:
                            shape = None
                            color = None 
                            letter = None

                    detected_target = {
                        'shape': shape,
                        'color': color,
                        'letter': letter   
                    }
                    if shape != None:
                        check, status, ret_target = is_target_in_desired_list(detected_target)
                    else: 
                        check = False
                    if check:
                        print('Status of detected target:', status)
                        
                        # Check if the detected target is in the desired targets list
                        remove_target_from_desired_list(ret_target)
                        removed_targets.append(ret_target)
                        rem += 1
                        # Only process and save coordinates for matching targets
                        lat = value[0] / 1e7
                        lon = value[1] / 1e7
                        if value:   
                            results_df = pd.concat([results_df, pd.DataFrame({
                                'ID': [ret_target['id']],
                                'Image File': [image_file],
                                'Image Path': [image_path],
                                'Cropped Image': [crop_name],
                                'Object Type': [shape],
                                'Color': [color],
                                'Letter': [letter],
                                'Coordinates_x': [center_x],
                                'Coordinates_y': [center_y],
                                'Lat': lat,
                                'Lon': lon,
                                'Cropped Image Path': [os.path.join(output_folder, 'test4', 'crops', class_name, crop_name)]
                            })], ignore_index=True)
                            results_df = sort_dataframe_by_id(results_df)
                            shutil.copy(os.path.join(output_folder, 'test4', 'crops', class_name, crop_name),'/home/edhitha/ultralytics/yo/output1/test4/crops/results')
                        else:
                            print(red_text,'ValueError',reset_color)

        # Add the image to the set of processed images
        processed_images.add(image_path)

        # Clear the class_name counts for the next image
        class_name_counts = {}
        # results_csv_file = '/home/edhitha/ultralytics/running/results.csv'
        results_df.to_csv(results_csv_file, index=False)
        eliminate_df.to_csv(eliminate_csv_file, index=False)

    if prev_files != image_files and not (image_files == processed_images) and not test_en:
        prev_files = image_files
        start_time = time.time()
        print('Number of detected targets:', rem)
        repeat_pattern += 1
        if lap_en and repeat_pattern == 2 and len(dropped_list) == 0 and rem != 0:
            msg = the_connection.recv_match(type=['GLOBAL_POSITION_INT'], blocking=True)
            current_lat = msg.lat / 1e7
            current_lon = msg.lon / 1e7
            dist = distance_lat_lon(current_lat, current_lon, search_wp_row['Latitude'], search_wp_row['Longitude'])
            print('Distance between the Last Search Waypoint and the Drone:', dist )
            if dist < 3:
                # Open a file in write mode ('w')
                print(green_text,'Reached the last Search grid Waypoint',reset_color)
                with open('/home/edhitha/DCIM/test_cam/done.txt', 'w') as file:
                    # Write some text to the file
                    file.write('gay!\n')
                len_dropped = len(dropped_list)
            repeat_pattern = 0
    else:
        flag = 0
        changes = abs(time.time() - start_time)
        print('Number of detected targets:', blue_text, rem, reset_color)
        print('Waiting for images for next:', changes, 's')
        time.sleep(0.5)
        if changes > 10:
            # Open a file in write mode ('w')
            with open('/home/edhitha/DCIM/test_cam/done.txt', 'w') as file:
                # Write some text to the file
                file.write('gay!\n')
            emergent_files = os.listdir(Emergent_folder)
            emergent_image_files = [f for f in emergent_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
            for x in emergent_image_files:
                source_path = os.path.join(Emergent_folder, x)
                print(source_path)
                target_path = os.path.join(Target_folder, x)
                print(target_path)
                shutil.copy(source_path, target_path)
                print('done done')                
            
            print('File has been successfully created and written.')
            print('Number of detected targets:', rem)

            if not lap_en and the_connection:
                the_connection = connect()
            
            in_temp = input("Press enter after the changes done in the CSV ...")
            print('Changed results_df\n', pd.read_csv(modified_csv_file))
            read_df = sort_dataframe_by_id(pd.read_csv(modified_csv_file))
            read_df.to_csv(modified_csv_file, index=False)
            print('Changed results_df\n', read_df)
            prev_item = -1
            break_wp = -1
            print('Got these indices:',read_df.index)
            for lap in range(1,num_targets+1):
                # print(read_df.loc[i])
                if lap_en and lap != 1:
                    while True:
                        if lap != 0:
                            msg = the_connection.recv_match(type=['GLOBAL_POSITION_INT', 'MISSION_CURRENT'], blocking=True)
                            if msg.get_type() == 'MISSION_CURRENT':
                                if prev_item != msg.seq:
                                    print('Current mission item:', msg.seq, 'Waiting for the WP index:', wp_index) 
                                    prev_item = msg.seq
                                time.sleep(0.2)
                                if break_wp == msg.seq:
                                    continue
                                else:
                                    break_wp = -1
                                if wp_index != msg.seq and msg.seq != prev_item:
                                    time.sleep(0.2)
                                    prev_item = msg.seq
                                    continue
                                elif msg.seq == wp_index:
                                    prev_item = msg.seq
                                    break_wp = msg.seq
                                    break
                            elif msg.get_type() == 'GLOBAL_POSITION_INT' and False:
                                current_lat = msg.lat / 1e7
                                current_lon = msg.lon / 1e7
                                print('Lat:', current_lat, 'Lon:', current_lon)
                                if is_near_to_loc(current_lat, current_lon, wp_row['Latitude'], wp_row['Longitude']):
                                    print('Reached the given Last Waypoint ...')
                                    time.sleep(0.2)
                                    break
                                else:
                                    print('Distance between the desired point and drone:',distance_lat_lon(current_lat, current_lon, wp_row['Latitude'], wp_row['Longitude']))
                                    time.sleep(0.2)
                        else:
                            break
                
                try:
                    if lap in read_df['ID'].values:
                        row = read_df[read_df['ID'] == lap].iloc[0]
                        print('Extracted Row:',row, lap)
                        row['ID'].astype(int)

                        Lat = row['Lat'].astype(float)
                        Lon = row['Lon'].astype(float)
                        print('REPOSITIONING TO COLOR:',row['Color'])
                        print('REPOSITIONING TO LETTER:',row['Letter'])
                        print('REPOSITIONING TO SHAPE:',row['Object Type'])
                        automation((Lat, Lon), lap)
                        time.sleep(4)

                    else:
                        print('Skipping the Bottle ID,',red_text, 'Not Found ...', reset_color)
                    
                except Exception as e:
                    print('Skipping the Bottle ID,',red_text, 'Not Found ...', reset_color, 'error', e)
            print('Mission Accomplished ...')
            
            exit()
        else:
            flag = 1

if test_en:
    the_connection = connect()

        
    in_temp = input("Press enter after the changes done in the CSV ...")
    print('Changed results_df\n', pd.read_csv(modified_csv_file))
    read_df = sort_dataframe_by_id(pd.read_csv(modified_csv_file))
    read_df.to_csv(modified_csv_file, index=False)
    print('Changed results_df\n', read_df)
    prev_item = -1
    break_wp = -1
    print('Got these indices:',read_df.index)
    for lap in range(1,num_targets+1):
        # print(read_df.loc[i])
        if lap_en and lap != 1:
            while True:
                if lap != 0:
                    msg = the_connection.recv_match(type=['GLOBAL_POSITION_INT', 'MISSION_CURRENT'], blocking=True)
                    if msg.get_type() == 'MISSION_CURRENT':
                        if prev_item != msg.seq:
                            print('Current mission item:', msg.seq, 'Waiting for the WP index:', wp_index) 
                            prev_item = msg.seq
                        time.sleep(0.2)
                        if break_wp == msg.seq:
                            continue
                        else:
                            break_wp = -1
                        if wp_index != msg.seq and msg.seq != prev_item:
                            time.sleep(0.2)
                            prev_item = msg.seq
                            continue
                        elif msg.seq == wp_index:
                            prev_item = msg.seq
                            break_wp = msg.seq
                            break
                    elif msg.get_type() == 'GLOBAL_POSITION_INT' and False:
                        current_lat = msg.lat / 1e7
                        current_lon = msg.lon / 1e7
                        print('Lat:', current_lat, 'Lon:', current_lon)
                        if is_near_to_loc(current_lat, current_lon, wp_row['Latitude'], wp_row['Longitude']):
                            print('Reached the given Last Waypoint ...')
                            time.sleep(0.2)
                            break
                        else:
                            print('Distance between the desired point and drone:',distance_lat_lon(current_lat, current_lon, wp_row['Latitude'], wp_row['Longitude']))
                            time.sleep(0.2)
                else:
                    break
        
        try:
            if lap in read_df['ID'].values:
                row = read_df[read_df['ID'] == lap].iloc[0]
                print('Extracted Row:',row, lap)
                row['ID'].astype(int)

                Lat = row['Lat'].astype(float)
                Lon = row['Lon'].astype(float)
                print('REPOSITIONING TO COLOR:',row['Color'])
                print('REPOSITIONING TO LETTER:',row['Letter'])
                print('REPOSITIONING TO SHAPE:',row['Object Type'])
                automation((Lat, Lon), lap)
                time.sleep(4)

            else:
                print('Skipping the Bottle ID,',red_text, 'Not Found ...', reset_color)
            
        except Exception as e:
            print('Skipping the Bottle ID,',red_text, 'Not Found ...', reset_color, 'error', e)
    print('Mission Accomplished ...')
    exit()
