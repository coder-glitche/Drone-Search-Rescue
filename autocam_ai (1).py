# Increasing string order
import os
import re
import csv
import cv2
import sys
import time
import math
import pickle
import shutil
import signal
import argparse
import subprocess
import pytesseract
import numpy  as np
import pandas as pd
from PIL              import Image
from pathlib          import Path
from ultralytics      import YOLO
from pymavlink        import mavutil
from sklearn.cluster  import KMeans
from collections      import defaultdict
from shapely.geometry import Point, Polygon
from multiprocessing  import Event, Process, Value

# Define your list of desired targets
desired_targets = [
    {'shape': 'Emergent', 'color': None, 'letter': None, 'id': 1},
    {'shape': 'Rectangle', 'color': 'red', 'letter': '4', 'id': 2},
    {'shape': 'Circle', 'color': 'orange', 'letter': 'E', 'id': 3},
    {'shape': 'Circle', 'color': 'blue', 'letter': 'I', 'id': 4},
    {'shape': 'Rectangle', 'color': 'red', 'letter': 'M', 'id': 5}
    # Add more desired targets here
]
desired_targets = [
    {'shape': 'Cross', 'color': 'white', 'letter': 'r', 'id': 1},
    {'shape': 'Semicircle', 'color': 'orange', 'letter': 'o', 'id': 2},
    {'shape': 'Star', 'color': 'red', 'letter': 'E', 'id': 3},
    {'shape': 'Circle', 'color': 'blue', 'letter': 'I', 'id': 4},
    {'shape': 'Quartercircle', 'color': 'white', 'letter': 'M', 'id': 5}
    # Add more desired targets here
]

# Dictionary to store letters corresponding to each unique combination of shape and color
letters_for_combo = defaultdict(list)

for target in desired_targets:
    shape = target['shape']
    color = target['color']
    letter = target['letter']
    # If both shape and color are not None
    if shape is not None and color is not None:
        # Create a key for the combination of shape and color
        combo_key = (shape, color)
        # Append the letter to the corresponding combo in the dictionary
        letters_for_combo[combo_key].append(letter)

# Calculate and store the number of elements in the letters_list as an attribute
for combo_key, letters_list in letters_for_combo.items():
    num_elements = len(letters_list)
    letters_for_combo[combo_key] = (letters_list, num_elements)

print(letters_for_combo)

# Default values for variables, if any defaults changed please change it in the below description also
defaults = {
    'grid': False,
    'suas': False,
    'alt': False,
    'fov': False,
    'lap': False,
    'resume': False,
    'test': False,
    'sitl': False,
    'ros': False,
    'drop': -1,
    'autocam': False,
    'csv': False,
    'top2': False
}


# Descriptions for each argument
descriptions = {
    'grid': 'Pass grid if you want to enable the inference inside the Search Grid',
    'suas': 'Pass if its actual SUAS mission',
    'alt': 'Pass if you want to inference the images only at certain height ',
    'fov': 'Pass if you want to get the cropped images from the specific circular range',
    'lap': 'Pass if you want to drop the bottles lap wise, and make sure the waypoint file is in the specific lap format',
    'resume': 'Pass to skip inference, and directly from the CSV file',
    'test': 'Pass to disable Jetson related features',
    'sitl': 'Pass this to connect to the sitl',
    'ros': 'Pass this to get the Geotag data from the C++ ROS',
    'drop': 'If you want to drop a specified bottle, pass this with the ID number ranging from 1 to 5',
    'autocam': 'Pass to start the camera capture, default it enables the lap, alt, grid',
    'csv': 'Read data from the csv file',
    'top2': 'Pass to disable top 2 color, incase of same shapes'
}

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Script for performing automation task.')

for key, value in defaults.items():
    if key in ['grid','suas','alt','fov','lap', 'test', 'sitl', 'ros', 'resume', 'autocam', 'csv', 'top2']:
        parser.add_argument(f'-{key}', action='store_true', help=f'{descriptions[key]}, Default value: \'{value}\'')
    else:
        parser.add_argument(f'-{key}', default=value, help=f'{descriptions[key]}, Default value: \'{value}\'')

# Parse the command-line arguments
args = parser.parse_args()

# Define column names
columns = ['ID','Image File', 'Image Path', 'Cropped Image', 'Object Type', 'Color', 'Letter', 'Coordinates_x', 'Coordinates_y', 'Lat', 'Lon', 'Cropped Image Path', 'Dropped']
cingo_df = pd.DataFrame(columns=columns)

# Get value from the argparse
grid = bool(args.grid)
SUAS = bool(args.suas)
alt_limiter = bool(args.alt)
FOV = bool(args.fov)
lap_en = bool(args.lap)
test_en = bool(args.test)
resume_en = bool(args.resume)
sitl_en = bool(args.sitl)
ros_en = bool(args.ros)
autocam_en = bool(args.autocam)
first_drop_ID = int(args.drop)
csv_en = bool(args.csv)
top2_en = bool(args.top2)

if autocam_en:
    print("Enabling lap, grid, and alt parameters by Default.")
    grid, lap_en, alt_limiter = True, True, True

print('Value inside the resume_en:',resume_en)

if not resume_en:
    try:
        shutil.rmtree('/home/AI_Integrate/work/output1/test4')
    except:
        pass

    try:
        shutil.rmtree('/homeAI_Integrate/work/output2/test5')
    except:
        pass

    try:
        shutil.rmtree('/home/AI_Integrate/work/output3/test6')
    except:
        pass
    try:
        shutil.rmtree('/home/AI_Integrate/work/output1')
    except:
        pass
    try:
        os.makedirs( '/home/AI_Integrate/work/log_results', exist_ok=True)
        shutil.move('/home/AI_Integrate/work/output1/test4', '/home/AI_Integrate/work/output1/log_results')
    except Exception as e:
        print(e)
        pass
# # Initialize YOLO models
# start_time  = time.time()
# model1 = YOLO('/home/edhitha/ultralytics/runs/detect/weights/best.pt')
# print('Time required to load the first model:', time.time() - start_time)
# model2 = YOLO('/home/edhitha/ultralytics/runs/classify/train3/weights/best.pt')
# print('Time required to load the second model:', time.time() - start_time)
# model3 = YOLO('/home/edhitha/ultralytics/runs/classify/train6/weights/best.pt')
# print('Time required to load the third model:', time.time() - start_time)
# model4 = YOLO('/home/edhitha/ultralytics/runs/segment/16mm_shapesegment/weights/best.pt')
# print('Time required to load the fourth model:', time.time() - start_time)

# Initialize YOLO models
start_time  = time.time()
model1 = YOLO('/home/runs/detect/emergent_target_model/weights/best.pt')
print('Time required to load the first model:', time.time() - start_time)
model2 = YOLO('/home/AI_Integrate/work/SUAS_synreal_colors_model/weights/best.pt')
print('Time required to load the second model:', time.time() - start_time)
model4 = YOLO('/home/runs/segment/train2/weights/best.pt')
print('Time required to load the fourth model:', time.time() - start_time)

if ros_en:
    input_folder = '/home/edhitha/Images_4_3_24'
elif csv_en:
    input_folder = '/home/AI_Integrate/work/ui_10_04_4/images'
else:
    input_folder = '/home/AI_Integrate/work/12_05_gauriBidanur/test_cam_128/images'

output_folder = '/home/AI_Integrate/work/output1'
output_folder2 = '/home/AI_Integrate/work/output2'
output_dir = "/home/AI_Integrate/work/output14"
Path(output_dir).mkdir(parents=True, exist_ok=True)
results_file = '/home/AI_Integrate/work/results.txt'
results_csv_file = '/home/AI_Integrate/work/output1/test4/crops/results/results.csv'
modified_csv_file = '/home/AI_Integrate/work/output1/test4/crops/results/results1.csv'
geoCSV_path = "/home/AI_Integrate/work/ui_10_04_4/results.csv"
ros_csv_file_path = "/home/AI_Integrate/work/ui_10_04_4/results.csv"
eliminate_csv_file = '/home/AI_Integrate/work/output1/test4/crops/Target/detected.csv'
Emergent_folder = os.path.join(output_folder, 'test4', 'crops', 'Emergent')
Target_folder = os.path.join(output_folder, 'test4', 'crops', 'Target')
# /home/edhitha/ultralytics/yo/output1/test4/crops/results')

# Necessary directories
# os.makedirs('/home/edhitha/ultralytics/yo/results', exist_ok=True)
os.makedirs('/home/AI_Integrate/work/output1/test4/crops/Target', exist_ok=True)
os.makedirs('/home/AI_Integrate/work/output1/test4/crops/results', exist_ok=True)
# Pickle path
pickle_path = '/home/AI_Integrate/work/12_05_gauriBidanur/test_cam_128/cam_wps.pickle'
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
test_search_grid = []

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timeout occurred")

def execute_with_timeout(func, timeout, *args, **kwargs):
    # Set the signal handler
    signal.signal(signal.SIGALRM, timeout_handler)
    # Set the timeout alarm
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    except TimeoutError:
        print("Function execution timed out")
        result = None
    finally:
        # Reset the alarm
        signal.alarm(0)
    return result

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

def detect_letters(seg_path, letters_list):
    original_image = cv2.imread(seg_path)
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(os.path.join(output_folder2, os.path.basename(seg_path)), gray_image)
    # Apply histogram equalization to increase contrast
    equalized_image = cv2.equalizeHist(gray_image)
    cv2.imwrite(os.path.join(output_folder2,"M.png"), gray_image)
    
    # Apply gamma correction to adjust highlights and exposure
    gamma = 0.5  # You can adjust the gamma value as needed
    gamma_corrected_image = np.uint8(cv2.pow(equalized_image / 255.0, gamma) * 255)
    
    # Decrease shadows by adjusting the intensity range
    alpha = 1.5  # Adjust the contrast level
    beta = -150   # Adjust the brightness level
    gray_image = cv2.convertScaleAbs(gamma_corrected_image, alpha=alpha, beta=beta)
    cv2.imwrite(os.path.join(output_folder2,os.path.basename(seg_path)), gray_image)
    
    rotation_step = 10
    # Convert alphabetic characters to a single string with both lowercase and uppercase versions included, excluding numbers
    to_detect = ''
    for i in letters_list:
        if str(i).isdigit():
            to_detect += str(i)
        else:
            to_detect += str(i).lower() + str(i).upper()

    # Remove repeated numerics
    to_detect = ''.join(sorted(set(to_detect), key=to_detect.index))
    print("letters:", to_detect)

    def rotate_and_detect(start_angle, end_angle, result, stop_event):
        for block_size in range(3, 22):
            try:
                thresholded_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, 2)

                for rotation_angle in range(start_angle, end_angle, rotation_step):
                    if stop_event.is_set():
                        return

                    rotated_image = np.copy(thresholded_image)
                    rows, cols = rotated_image.shape
                    rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_angle, 1)
                    rotated_image = cv2.warpAffine(rotated_image, rotation_matrix, (cols, rows))

                    custom_config = rf'--psm 10'
                    detected_text = pytesseract.image_to_string(rotated_image, config=custom_config)

                    for letter in letters_list:
                        if letter in detected_text:
                            result.value = ord(letter)  # Store ASCII code of the detected letter
                            stop_event.set()
                            return

            except cv2.error as e:
                print(f"Error: {e}. Trying next block size.")

    result = Value('i', 0)  # Shared memory for storing the ASCII code of the detected letter
    stop_event = Event()     # Event for signaling termination

    process_1 = Process(target=rotate_and_detect, args=(0, 180, result, stop_event))
    process_2 = Process(target=rotate_and_detect, args=(180, 360, result, stop_event))
    # Three processes parallely (0,120), (120, 240), and (240, 360)
    # process_3 = Process(target=rotate_and_detect, args=(240, 360, result, stop_event))

    process_1.start()
    process_2.start()
    # process_3.start()

    process_1.join()
    process_2.join()
    # process_3.join()

    detected_ascii = result.value
    if detected_ascii:
        detected_letter = chr(detected_ascii)  # Convert ASCII code back to the detected letter
        print("Detected Letter:", detected_letter)
        return detected_letter
    else:
        print(f"No letters in {letters_list} found for any block size.")
        return None
    
# Function to calculate aspect ratio of an image
def calculate_aspect_ratio(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
        aspect_ratio = width / height
    return aspect_ratio

# Function to check if a target is in the desired targets list
def is_target_in_desired_list1(target):
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

def check_all_true(df, column_name):
    """
    Function to check if all values in a DataFrame column are True.
    
    Parameters:
        df (DataFrame): The pandas DataFrame.
        column_name (str): The name of the column to check.
    
    Returns:
        bool: True if all values in the specified column are True, False otherwise.
    """
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
        return False
    
    # Check if all values in the column are True
    all_true = all(df[column_name])
    
    return all_true

# Function to check if a target is in the desired targets list
def is_target_in_desired_list2(target, image_file, image_path, crop_name, center_x, center_y, lat, lon, class_name):
    global cingo_df
    cingo_df = pd.concat([cingo_df, pd.DataFrame({
        'ID': [None],
        'Image File': [image_file],
        'Image Path': [image_path],
        'Cropped Image': [crop_name],
        'Object Type': target['shape'],
        'Color': target['color'],
        'Letter': None,
        'Coordinates_x': [center_x],
        'Coordinates_y': [center_y],
        'Lat': lat,
        'Lon': lon,
        'Cropped Image Path': [os.path.join(output_folder, 'test4', 'crops', class_name, crop_name)],
        'Dropped': False
    })], ignore_index=True)
    for i, desired_target in enumerate(desired_targets):
        if (desired_target['shape'] == target['shape']
            and desired_target['color'] == target['color']
            and desired_target['letter'] == target['letter']):
            return True, None, desired_targets[i]
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

    # Set the auxiliary pin or relay channel (replace 0 with your channel number) (Channel, PWM_VALUE)
    pwm_values = [(7,1000),(6,2000),(7,2000),(6,1000),(6,1500)]
    # Set the initial PWM value
    pwm_value = pwm_values[index-1]
    print(pwm_value)
    # Set the MAV_CMD_DO_SET_SERVO command parameters
    command = mavutil.mavlink.MAV_CMD_DO_SET_SERVO

    param1 = pwm_value[0]
    param2 = pwm_value[1]

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

def extract_data_from_file(file_path, target_file):
    # Initialize empty variables
    lat2 = None
    lon2 = None
    alt2 = None
    head2 = None
    yaw2 = None
    time2 = None

    # Open and read the CSV file
    with open(file_path, 'r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)

        # Skip the header row
        next(csv_reader)

        # Iterate through each row in the CSV file
        for row in csv_reader:
            # Check if the file name matches the target_file
            if row[0] == target_file:
                # Extract data from the matching row
                lat2 = float(row[1])
                lon2 = float(row[2])
                alt2 = float(row[3])
                head2 = float(row[4])
                yaw2 = float(row[5])
                time2 = float(row[6])

    # Return the extracted data as a dictionary or None if not found
    return lat2, lon2, alt2, head2, yaw2, time2

def extract_data_from_csv(file_path, target_file):
    # Initialize empty variables
    lat2 = None
    lon2 = None
    alt2 = None
    head2 = None
    yaw2 = None
    time2 = None

    # Open and read the CSV file
    with open(file_path, 'r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)

        # Skip the header row
        next(csv_reader)

        # Iterate through each row in the CSV file
        for row in csv_reader:
            # print(row)
            # Check if the file name matches the target_file
            if row[1] == target_file:
                # Extract data from the matching row
                lat2 = float(row[2])
                lon2 = float(row[3])
                alt2 = float(row[4])
                head2 = float(row[5])
                yaw2 = float(row[6])
                time2 = float(row[7])

    # Return the extracted data as a dictionary or None if not found
    return lat2, lon2, alt2, head2, yaw2, time2

def sort_dataframe_by_id(df):
    """
    Sorts a DataFrame based on the 'ID' column in ascending order.

    Parameters:
    - df: DataFrame to be sorted

    Returns:
    - Sorted DataFrame
    """
    return df.sort_values(by='ID', ascending=True)

def get_coordinates(image_name,image_path,pickle_path,point2, fov_circle, lat=None, lon=None, yaw=None):
    if lat is None or lon is None or yaw is None:
        if ros_en:
            # Open and read the CSV file
            with open(ros_csv_file_path, 'r') as csv_file:
                # Create a CSV reader object
                csv_reader = csv.reader(csv_file)

                # Skip the header row
                next(csv_reader)

                # Iterate through each row in the CSV file
                for row in csv_reader:
                    # Check if the file name matches the target_file
                    if row[0] == image_name:
                        # Extract data from the matching row
                        lat = float(row[1])
                        lon = float(row[2])
                        yaw = float(row[5])
        elif csv_en:
            # Open and read the CSV file
            with open(geoCSV_path, 'r') as csv_file:
                # Create a CSV reader object
                csv_reader = csv.reader(csv_file)

                # Skip the header row
                next(csv_reader)

                # Iterate through each row in the CSV file
                for row in csv_reader:
                    # Check if the file name matches the target_file
                    if row[1] == image_name:
                        # Extract data from the matching row
                        lat = float(row[2])
                        lon = float(row[3])
                        yaw = float(row[6])
        else:
            wps_list = read_dict_from_file(pickle_path)
            # print(wps_list)
            # for i, row in enumerate(wps_list):
            #     print(i, row)
            index = int(image_name[-9:-4])
            print(orange_text,wps_list[index],reset_color)
            if len(wps_list[index]) == 6:
                lat, lon, alt, head, yaw, time1 = wps_list[index]
            else:
                lat, lon, alt, time1 = wps_list[index]
                yaw = 165.31
    
    
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

# Function to calculate the angle between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    dLon = lon2 - lon1
    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    bearing = math.atan2(y, x)
    return math.degrees(bearing)

# Function to handle incoming messages
def target_bearing_int(target_lat, target_lon):
        msg = the_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        lat, lon = msg.lat / 1e7, msg.lon / 1e7
        yaw = msg.hdg / 100
        # Print heading and yaw
        if yaw is not None:
            # print(f"Heading: {heading}, Yaw: {yaw}, lat: {lat}, lon: {lon}")
            # Calculate and adjust yaw to point towards target
            if lat is not None and lon is not None:
                target_bearing = (calculate_bearing(lat, lon, target_lat, target_lon)+180) % 360
                yaw_error = target_bearing - yaw
                # if yaw_error > 180:
                #     yaw_error -= 360
                # elif yaw_error < -180:
                #     yaw_error += 360
                print(orange_text, "yaw:", yaw_error,reset_color, target_bearing)
                return int(0)

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

def guide_command(value, alt):
    the_connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_global_int_message(10, the_connection.target_system,the_connection.target_component, 6, 1024, int(value[0]*1e7), int(value[1]*1e7),alt , 0, 0, 0, 0, 0, 0, 0, target_bearing_int(value[0], value[1])))

def automation(value,target):
    print("target aquired",file=sys.stderr)
    # print('Value:', value)
    gps = the_connection.recv_match(type='GLOBAL_POSITION_INT', blocking = True)
    # print(gps)
    calculated_distance =  distance_lat_lon(value[0],value[1],gps.lat/1e7,gps.lon/1e7)

    kurrentalti = 30
    accuracy = 1

    # print(kurrentalti)
    print('changing Guided',file=sys.stderr)
    while True:
        try:
            execute_with_timeout(guided, 2)
            time.sleep(1)
            guide_command(value, alt=30)
            break
        except Exception as e:
            print(red_text,"Timeout Exception", reset_color)
            pass
    
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
            print('Distance between drone and target:',blue_text,distance,reset_color, 'for lat and lon:', current_lat, current_lon, end="\r")
            if distance <= accuracy:
                print('Started Dropping ...',file=sys.stderr)
                drop(target)
                
                the_connection.mav.command_long_send(the_connection.target_system, the_connection.target_component,181, 0, 0, 1, 0, 0, 0, 0, 0)
                if sitl_en or test_en:
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
    print("Sending Mission COUNT")
    count_msg = vehicle.recv_match(type='MISSION_COUNT', blocking=True)
    num_items = count_msg.count

    # Request each mission item
    commands = []
    for seq in range(num_items):
        vehicle.mav.mission_request_send(vehicle.target_system, vehicle.target_component, seq)

        # Wait for the mission item message
        item_msg = vehicle.recv_match(type='MISSION_ITEM', blocking=True)
        commands.append(item_msg)
    print("GOT ALL THE WAYPOINTS..")
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

    search_wp_first_occurrence = commands_df[commands_df['Command'] == 177].index[-2] 
    while commands_df.loc[search_wp_first_occurrence]['Command'] != 16:
        search_wp_first_occurrence += 1
    # Print the result
    print(f"wp_last_occurrence: {blue_text}{wp_last_occurrence}{reset_color}, search_wp_last_occurrence: {blue_text}{search_wp_last_occurrence}{reset_color}")
    return commands_df.loc[wp_last_occurrence], commands_df.loc[search_wp_last_occurrence], wp_last_occurrence, search_wp_first_occurrence, search_wp_last_occurrence, commands_df

def extract_polygon_vertices(commands_df):
    search_wp_last_occurrence = commands_df[commands_df['Command'] == 177].index[-1] 
    while commands_df.loc[search_wp_last_occurrence]['Command'] != 16:
        search_wp_last_occurrence -= 1

    search_wp_first_occurrence = commands_df[commands_df['Command'] == 177].index[-2] 
    while commands_df.loc[search_wp_first_occurrence]['Command'] != 16:
        search_wp_first_occurrence += 1

    print("First search grid occurence:",blue_text, search_wp_first_occurrence,reset_color, "Last search grid occurence:", blue_text,search_wp_last_occurrence, reset_color)
    latitudes, longitudes = [], []
    for idx in range(search_wp_first_occurrence, search_wp_last_occurrence, 1):
        latitudes.append(commands_df.loc[idx]['Latitude'])
        longitudes.append(commands_df.loc[idx]['Longitude'])
    min_lat, max_lat, min_lon, max_lon = min(latitudes), max(latitudes), min(longitudes), max(longitudes)
    sel_lat, sel_lon = [], []

    sel_lat.extend([min_lat, min_lat, max_lat, max_lat])
    sel_lon.extend([min_lon, max_lon, max_lon, min_lon])

    # Form a polygon from extracted points
    polygon = Polygon(zip(sel_lon, sel_lat))

    # Find the minimum number of vertices required to represent the polygon
    min_vertices = len(polygon.exterior.coords)

    return sel_lat, sel_lon, min_vertices

def increase_polygon_area(latitudes, longitudes, extension_factor):
    # Calculate the current diagonal length of the polygon
    current_diagonal_length = distance_lat_lon(latitudes[0], longitudes[0], latitudes[2], longitudes[2])

    # Calculate the new diagonal length after extension
    new_diagonal_length = current_diagonal_length * extension_factor

    # Calculate the ratio of the new diagonal length to the current diagonal length
    ratio = new_diagonal_length / current_diagonal_length

    # Calculate the new coordinates by extending each vertex from the centroid
    centroid_lat = sum(latitudes) / len(latitudes)
    centroid_lon = sum(longitudes) / len(longitudes)

    new_latitudes, new_longitudes = [], []
    for lat, lon in zip(latitudes, longitudes):
        # Calculate the distance and bearing from the centroid to the current vertex
        distance = distance_lat_lon(centroid_lat, centroid_lon, lat, lon)
        bearing = math.atan2(lon - centroid_lon, lat - centroid_lat)

        # Calculate the new coordinates based on the extended diagonal length
        new_distance = distance * ratio
        new_lat = centroid_lat + (new_distance * math.cos(bearing))
        new_lon = centroid_lon + (new_distance * math.sin(bearing))

        # Append the new coordinates to the lists
        new_latitudes.append(new_lat)
        new_longitudes.append(new_lon)

    return new_latitudes, new_longitudes

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

def command_exec(command):
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        # Continuously read the output of the process
        while True:
            output = process.stdout.readline()
            if not output:
                print(red_text,'Breaking the loop',reset_color)
                break

            print(output, end='')
        # Wait for the process to complete
        process.communicate()

    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
    except Exception as ex:
        print(f"An error occurred: {ex}")

if __name__ == '__main__':
    # Initialize an empty DataFrame with columns
    results_df = pd.DataFrame(columns=columns)
    eliminate_df = pd.DataFrame(columns=columns)
    timeout_duration = 5
    start_time = time.time()
    # None value initializers
    dropped_list, wp_index, search_index, search_first_idx, prev_files, mid_point, wp_row, search_wp_row, commands_df = [], None, None, None, None, None, None, None, None

    # location variables 
    lat1, lon2, yaw, perf_time = None, None, None, None

    # Integer value initializers
    flag, repeat_pattern = 1, 0

    # Boolean value initializers
    even_done, odd_done, first_drop = False, False, False

    if not lap_en:
        the_connection = None
    else:
        the_connection = connect()
        try:
            wp_row, search_wp_row, wp_index, search_first_idx, search_index, commands_df = get_wps_and_indicies(the_connection)
        except Exception as e:
            print('Error occured:',red_text, e, reset_color)
            print(orange_text,'Terminating the lap based ...', reset_color)
            lap_en = False 
    if autocam_en:
        prev_wp = -1
        if len(test_search_grid) == 0:
            # Call the function with your DataFrame 'commands_df'
            latitudes, longitudes, min_vertices = extract_polygon_vertices(commands_df)
            print("Extracted Latitudes:", latitudes, len(latitudes))
            print("Extracted Longitudes:", longitudes)
            print("Minimum Vertices Required:", min_vertices)

            # Extend the polygon by 3 meters from every corner
            extension = 0.0000115 # Adjust this value as needed
            extended_latitudes, extended_longitudes = increase_polygon_area(latitudes, longitudes, extension)
            for l, l2 in enumerate(extended_latitudes):
                test_search_grid.append((extended_latitudes[l], extended_longitudes[l]))
            test_search_grid.append((extended_latitudes[0], extended_longitudes[0]))
        # print(test_search_grid)
        while True:
            msg = the_connection.recv_match(type="MISSION_CURRENT", blocking=True)
            if msg.seq != prev_wp:
                print("Seq:", msg.seq+1, "WP_index:", wp_index)
                prev_wp = msg.seq
            if msg.seq + 1 == wp_index:
                break
        cam_command = "mavproxy.py"
        cam_process = Process(target=command_exec, args=(cam_command,))
        cam_process.start()
        while True:
            msg = the_connection.recv_match(type="MISSION_CURRENT", blocking=True)
            if msg.seq != prev_wp:
                print("Seq:", msg.seq, "Search_grid_lat_index:", search_index)
                prev_wp = msg.seq
            if msg.seq == search_index:
                while True:
                    msg = the_connection.recv_match(type="GLOBAL_POSITION_INT", blocking=True)
                    distance = distance_lat_lon(msg.lat/1e7, msg.lon/1e7, search_wp_row['Latitude'], search_wp_row['Longitude'])
                    print("Distance drone and last waypoint to stop the capturing of the images:", blue_text, distance, reset_color, 'm  ', end='\r')
                    if distance < 2:
                        break
                break
        try:
            cam_process.terminate()
            cam_process.kill()
        except Exception as e:
            print(e)
        time.sleep(2)
        # print(kurrentalti)
        print('Changing to Guided',file=sys.stderr)
        while True:
            try:
                execute_with_timeout(guided, 2)
                time.sleep(1)
                break
            except Exception as e:
                print(red_text,"Timeout Exception", reset_color)
                pass
        perf_time = time.time()
    poly=None
    if grid:
        if SUAS:
            poly =Polygon(SUAS_search_grid)
        else:
            poly=Polygon(test_search_grid)


    print("Waiting for images")
    while not os.path.exists(input_folder):
        continue

    

        
    while True and not resume_en:
        # List all files in the input folder
        image_files = os.listdir(input_folder)
        initial_files = set(os.listdir(input_folder))
        # Filter out files that have already been processed
        new_image_files = [img for img in image_files if img not in processed_images]

        new_image_files.sort()
        for image_file in new_image_files:
            image_index = int(image_file[-9:-4])

            if not even_done and not odd_done and image_index%2 == 1:
                continue 

            image_path = os.path.join(input_folder, image_file)
            # Check if the image has not been processed
            if image_path not in processed_images:
                if grid or alt_limiter:
                    try:
                        image_index = int(image_file[-9:-4])
                        if ros_en:
                            lat1, lon2, alt, head, yaw, time1 =  extract_data_from_file(ros_csv_file_path, image_file)
                        elif csv_en:
                            lat1, lon2, alt, head, yaw, time1 =  extract_data_from_csv(geoCSV_path, image_file)
                        else:
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
                print(purple_text)
                results = model1.predict(source=image_path, project=output_folder, name='test4', exist_ok=True ,device=0,
                save_crop=True)
                print(reset_color)
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

                        value = get_coordinates(image_file, image_path, pickle_path, mid_point, fov_range, lat1, lon2, yaw)
                        
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
                        # print("this is the crop name ccccccccccc")
                        # print(crop_name)
                        
                        real_path = os.path.join(output_folder, 'test4', 'crops', class_name , real_name)
                        crop_path = os.path.join(output_folder, 'test4', 'crops', class_name, crop_name)
                        os.rename(real_path, crop_path)
                        print(real_path, crop_path)
                        
                        lat = value[0] / 1e7
                        lon = value[1] / 1e7
                        print('Saving in the eliminate DF ...')
                        shape = None
                        letter = None
                        color = None
                        print( crop_path)
                        if class_name == 'Emergent':
                            shape = 'Emergent'
                            color = None 
                            letter = None
                        
                        else:
                            results4 = model4.predict(source=crop_path, device=0)
                            max_shape_name, output_path = None, None
                            for r in results4:
                                img = np.copy(r.orig_img)
                                img_name = Path(r.path).stem
                                print("i completed 1st image")
                                
                                # Lists to store shape names and corresponding confidences
                                shapes = []
                                confidences = []
                                
                                # iterate each object contour 
                                for ci, c in enumerate(r):
                                    label = c.names[c.boxes.cls.tolist().pop()]
                                    shape_name = c.names[c.boxes.cls[0].item()]
                                    confidence = c.boxes.conf[0].item()
                                    
                                    # Append shape name and confidence to lists
                                    shapes.append(shape_name)
                                    confidences.append(confidence)
                                if len(confidences) > 0:
                                    # Find the index of the shape with the highest confidence
                                    max_conf_idx = np.argmax(confidences)
                                    # Retrieve the shape name and corresponding confidence with the highest value
                                    max_shape_name = shapes[max_conf_idx]
                                    max_confidence = confidences[max_conf_idx]
                                    # Find the contour mask for the shape with the highest confidence
                                    contour = r[max_conf_idx].masks.xy.pop().astype(np.int32).reshape(-1, 1, 2)
                                    b_mask = np.zeros(img.shape[:2], np.uint8)
                                    _ = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)
                                    # Isolate object with transparent background (when saved as PNG)
                                    isolated = np.dstack([img, b_mask])
                                    # Get bounding box coordinates
                                    x1, y1, x2, y2 = r[max_conf_idx].boxes.xyxy.cpu().numpy().squeeze().astype(np.int32)
                                    # Save the isolated object as PNG
                                    isolated_crop = isolated[y1:y2, x1:x2]
                                    output_path = f"{output_dir}/{img_name}_object_{max_shape_name}.png"
                                    cv2.imwrite(output_path, isolated_crop)
                                    print(f"Saved: {output_path}")
                            
                                    
                            if shapes:
                                shape = max_shape_name
                                results2 = model2.predict(source=output_path, project=output_folder2, name='test5', exist_ok=True,device=0, save=True)
                                if top2_en:
                                    print("Taking Top 1 prediction")
                                    color = results2[0].names[results2[0].probs.top1]
                                else:
                                    color_indices = results2[0].probs.top5
                                    color_list= [results2[0].names[index] for index in color_indices]
                                    color_list=color_list[:-2]
                                    print(color_list)
                                    for desired_target in desired_targets:
                                        if (desired_target['shape']==shape):
                                            matching_color=next((color for color in color_list if color == desired_target['color']), None)
                                            if matching_color is not None:
                                                color=matching_color
                                            else:
                                                color=color_list[0]

                                            break
                                    else:
                                        color=color_list[0]
                                # Initialize a list to store the third elements
                                letters_list, len_letters_list = letters_for_combo.get((shape, color), ([], 0))
                                print("Letters list for", shape, color, ":", letters_list)

                                if letters_list:
                                    if len(letters_list) > 1:
                                        letter = detect_letters(output_path, letters_list)
                                        if letter != None:
                                            letters_list.remove(letter)

                                        # Update the dictionary with the modified list
                                        letters_for_combo[(shape, color)] = (letters_list, len_letters_list)
                                        print("Updated letters for each unique combination:")
                                        for combo, letters in letters_for_combo.items():
                                            print(f"For {combo[0]} {combo[1]}:", letters)
                                else:
                                    letter = None
                                                        
                                # for desired_target in desired_targets:
                                #     if (desired_target['shape']==class_name and desired_target['color']==color):
                                #         matching_letter=next((letter for letter in letter_list if letter == desired_target['letter']), None)
                                #         if matching_letter is not None:
                                #             letter=matching_letter
                                #         else:
                                #             letter=letter_list[0]
                                            
                                #         break
                                # else:
                                #     letter=letter_list[0]
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
                            if len_letters_list <= 1:
                                check, status, ret_target = is_target_in_desired_list1(detected_target)
                                letters_list = []
                            else: 
                                print("LIST2",detected_target)
                                check, status, ret_target = is_target_in_desired_list2(detected_target, crop_name, image_path, crop_name, center_x, center_y, lat, lon, class_name)
                                letters_list = []
                        else: 
                            check = False
                        
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
                            'Cropped Image Path': [os.path.join(output_folder, 'test4', 'crops', class_name, crop_name),],
                            'Dropped': False
                        })], ignore_index=True)
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
                                if first_drop_ID == -1:
                                    first_drop_ID = ret_target['id']
                                if not first_drop and first_drop_ID == ret_target['id']:
                                    drop_bottle = True
                                else:
                                    drop_bottle = False    
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
                                    'Cropped Image Path': [os.path.join(output_folder, 'test4', 'crops', class_name, crop_name)],
                                    'Dropped': drop_bottle
                                })], ignore_index=True)
                                results_df = sort_dataframe_by_id(results_df)
                                print(green_text, "Successfully added the target:", orange_text, "Shape:", blue_text, shape, orange_text, "Color:", blue_text, color, orange_text, "Letter:", blue_text, letter, reset_color)                            
                                shutil.copy(os.path.join(output_folder, 'test4', 'crops', class_name, crop_name),os.path.join(output_folder, 'test4', 'crops', 'results'))
                                if not first_drop and the_connection != None and first_drop_ID == ret_target['id']:
                                    print("Reposition after the last search waypoint ...")
                                    while True and not autocam_en:
                                            msg = the_connection.recv_match(type=['GLOBAL_POSITION_INT'], blocking=True)
                                            # print(msg)
                                            current_lat = msg.lat / 1e7
                                            current_lon = msg.lon / 1e7
                                            distance = distance_lat_lon(current_lat,current_lon,search_wp_row['Latitude'].astype(float),search_wp_row['Longitude'].astype(float))
                                            print('Distance between drone and wp point to change to guided:',blue_text,distance,reset_color,'m  ', end="\r")
                                            if distance < 5:
                                                time.sleep(3)
                                                break
                                    print('REPOSITIONING TO COLOR:',color)
                                    print('REPOSITIONING TO LETTER:',ret_target['letter'])
                                    print('REPOSITIONING TO SHAPE:',shape)
                                    if perf_time:
                                        print('Time took to inference for the first target to DROP:', blue_text, time.time() - perf_time, 's')
                                    automation((lat, lon), ret_target['id'])
                                    first_drop = True
                            else:
                                print(red_text,'ValueError',reset_color)
            # Add the image to the set of processed images
            processed_images.add(image_path)

            # Clear the class_name counts for the next image
            class_name_counts = {}
            # results_csv_file = '/home/edhitha/ultralytics/running/results.csv'
            results_df.to_csv(results_csv_file, index=False)
            eliminate_df.to_csv(eliminate_csv_file, index=False)

        if prev_files != image_files and not (image_files == processed_images) and not resume_en:
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
            print('Number of detected targets:', blue_text, rem, reset_color, '\tWaiting for images for next:', changes, 's')
            time.sleep(0.5)

            if not even_done:
                dynamic_time = 0.5 
            else:
                dynamic_time = 7
            if changes > dynamic_time:
                if rem != num_targets:
                    # print(rem, num_targets)
                    for combo, (letters_list, num_clusters) in letters_for_combo.items():
                        if num_clusters > 1:
                            # print(f"For {combo[0]} {combo[1]}:", letters_list)
                            for i, desired_target in enumerate(desired_targets):
                                if (desired_target['shape'] == combo[0] and desired_target['color'] == combo[1]):
                                    filtered_df = cingo_df[(cingo_df['Object Type'] == desired_target['shape']) & (cingo_df['Color'] == desired_target['color'])]
                                    lat_lon_data = filtered_df[['Lat', 'Lon']]
                                    print("lat_lon_data", lat_lon_data)
                                    if not lat_lon_data.empty and len(lat_lon_data) > 1:
                                        # Perform KMeans clustering
                                        kmeans = KMeans(n_clusters=num_clusters)
                                        kmeans.fit(lat_lon_data)

                                        # Get cluster centers and labels
                                        cluster_centers = kmeans.cluster_centers_
                                        cluster_labels = kmeans.labels_

                                        # Create a dictionary to store images belonging to each cluster
                                        cluster_images = {i: [] for i in range(num_clusters)}
                                        print("Labels:", cluster_labels)

                                        # Iterate through filtered_df and assign each image to its corresponding cluster
                                        for index, label in enumerate(cluster_labels):
                                            image_path = filtered_df.iloc[index]['Cropped Image']
                                            cluster_images[label].append(image_path)

                                        filtered_results_df = results_df[(results_df['Object Type'] == desired_target['shape']) & (results_df['Color'] == desired_target['color'])]
                                        print(filtered_results_df)

                                        if len(filtered_results_df) == 1:
                                            row2 = filtered_results_df.iloc[0]
                                            selected_crop_image = None
                                            selected_crop_cluster = None
                                            for j, cluster_image_list in cluster_images.items():
                                                if row2['Cropped Image'] not in cluster_image_list:
                                                    selected_crop_image = cluster_image_list[0]
                                                    selected_crop_cluster = j
                                                    break

                                            if selected_crop_image:
                                                check, status, ret_target = is_target_in_desired_list1({'shape': combo[0], 'color': combo[1], 'letter': None})
                                                remove_target_from_desired_list(ret_target)
                                                removed_targets.append(ret_target)
                                                rem += 1

                                                select_unique = eliminate_df[(eliminate_df['Cropped Image'] == selected_crop_image)]
                                                select_unique['ID'] = ret_target['id']
                                                # print("selected unique:", select_unique)
                                                # if select_unique['ID'] != None:
                                                results_df = pd.concat([select_unique, results_df])
                                                results_df = sort_dataframe_by_id(results_df)
                                                print(green_text, "Successfully added the target:", orange_text, "Shape:", blue_text, select_unique['Object Type'], orange_text, "Color:", blue_text, select_unique['Color'], orange_text, "Letter:", blue_text, select_unique['Letter'], reset_color)                          
                                                shutil.copy(os.path.join(output_folder, 'test4', 'crops', 'Target', selected_crop_image), os.path.join(output_folder, 'test4', 'crops', 'results'))

                                        elif len(filtered_results_df) == 0:
                                            # Russian rollete
                                            for k in range(num_clusters):
                                                selected_crop_image = cluster_images[k][0]
                                                selected_crop_cluster = k

                                                if selected_crop_image:
                                                    check, status, ret_target = is_target_in_desired_list1({'shape': combo[0], 'color': combo[1], 'letter': None})
                                                    remove_target_from_desired_list(ret_target)
                                                    removed_targets.append(ret_target)
                                                    rem += 1

                                                    select_unique = eliminate_df[(eliminate_df['Cropped Image'] == selected_crop_image)]
                                                    select_unique['ID'] = ret_target['id']
                                                    print("selected unique:", select_unique)
                                                    # if select_unique['ID'] != None:
                                                    results_df = pd.concat([select_unique, results_df])
                                                    results_df = sort_dataframe_by_id(results_df)
                                                    print(green_text, "Successfully added the target:", orange_text, "Shape:", blue_text, select_unique['Object Type'], orange_text, "Color:", blue_text, select_unique['Color'], orange_text, "Letter:", blue_text, select_unique['Letter'], reset_color)
                                                    shutil.copy(os.path.join(output_folder, 'test4', 'crops', 'Target', selected_crop_image), os.path.join(output_folder, 'test4', 'crops', 'results'))
                print("Updated targets, number of targets detected:", rem)
                if even_done:
                    odd_true = True
                elif rem != num_targets and not odd_done:
                    even_done = True
                    continue
                elif rem == num_targets:
                    print(green_text, "Successfully detected all the required targets ...", reset_color)
                    flag = 1
                results_df.to_csv(results_csv_file, index=False)
                # Open a file in write mode ('w')
                if not test_en:
                    with open('/home/edhitha/DCIM/test_cam/done.txt', 'w') as file:
                        # Write some text to the file
                        file.write('gay!\n')
                else: 
                    modified_csv_file = results_csv_file
                
                emergent_files = os.listdir(Emergent_folder) 
                emergent_image_files = [f for f in emergent_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                try:
                    for x in emergent_image_files:
                        source_path = os.path.join(Emergent_folder, x)
                        print(source_path)
                        target_path = os.path.join(Target_folder, x)
                        print(target_path)
                        shutil.copy(source_path, target_path)
                        print('done done')               
                except:
                    pass 
                
                print('File has been successfully created and written.')
                print('Number of detected targets:', rem)

                if not lap_en and the_connection == None:
                    the_connection = connect()
                
                print("$Waiting for the modified CSV FILE ...$")
                while(not os.path.exists(modified_csv_file)):
                    time.sleep(0.5)

                print('Changed results_df\n', pd.read_csv(modified_csv_file))
                read_df = sort_dataframe_by_id(pd.read_csv(modified_csv_file))
                read_df.to_csv(modified_csv_file, index=False)
                print('Changed results_df\n', read_df)
                prev_item = -1
                print('Got these indices:',read_df.index)
                lap = 1
                while lap <= num_targets:
                    # print("Inside the while lap and num_targets loop")
                    # print(read_df.loc[i])
                    if lap_en:
                        # print("Inside the lap_en")
                        msg = the_connection.recv_match(type=['MISSION_CURRENT'], blocking=True)
                        # print(msg.seq, wp_index)
                        if msg and msg.get_type() == 'MISSION_CURRENT':
                            if msg.seq == wp_index:
                                prev_item = msg.seq
                                while True:
                                    msg = the_connection.recv_match(type=['GLOBAL_POSITION_INT'], blocking=True)
                                    # print(msg)
                                    current_lat = msg.lat / 1e7
                                    current_lon = msg.lon / 1e7
                                    distance = distance_lat_lon(current_lat,current_lon,wp_row['Latitude'].astype(float),wp_row['Longitude'].astype(float))
                                    print('Distance between drone and wp point to change to guided:',blue_text,distance,reset_color,'m',end="\r")
                                    if distance <= 2:
                                        time.sleep(2)
                                        try:
                                            while lap in read_df['ID'].values:
                                                print("inside the read_df.")
                                                row = read_df[read_df['ID'] == lap].iloc[0]
                                                if row['Dropped'].astype(bool) == False:
                                                    print('Extracted Row:',row, lap)
                                                    row['ID'].astype(int)

                                                    Lat = row['Lat'].astype(float)
                                                    Lon = row['Lon'].astype(float)
                                                    print('REPOSITIONING TO COLOR:',row['Color'])
                                                    print('REPOSITIONING TO LETTER:',row['Letter'])
                                                    print('REPOSITIONING TO SHAPE:',row['Object Type'])
                                                    automation((Lat, Lon), lap)
                                                    row_index = read_df[read_df['ID'] == lap].index[0]
                                                    # read_df.at[row_index, 'Dropped'] = True
                                                    read_df.loc[row_index, 'Dropped'] = True
                                                    results_df.to_csv(results_csv_file, index=False)
                                                    read_df.to_csv(modified_csv_file, index=False)
                                                    time.sleep(4)
                                                    break
                                                else:
                                                    print("Bottle ID:", lap, "already dropped")
                                                    lap += 1
                                            
                                        except Exception as e:
                                            print('Skipping the Bottle ID,',red_text, 'Not Found ...', reset_color, 'error', e)
                                        
                                        lap += 1
                                        if check_all_true(read_df, 'Dropped'):
                                            print('Mission Accomplished ...')
                                            exit()
                                        break
                                    time.sleep(0.1)
                            elif msg.seq != prev_item:
                                print("Currently in the WP:", msg.seq, "Waiting for the WP:", wp_index)
                                prev_item = msg.seq

    if resume_en:
        the_connection = connect()

        if not test_en:
            with open('/home/edhitha/DCIM/test_cam/done.txt', 'w') as file:
                # Write some text to the file
                file.write('gay!\n')
        else: 
            modified_csv_file = results_csv_file

        emergent_files = os.listdir(Emergent_folder) 
        emergent_image_files = [f for f in emergent_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        try:
            for x in emergent_image_files:
                source_path = os.path.join(Emergent_folder, x)
                print(source_path)
                target_path = os.path.join(Target_folder, x)
                print(target_path)
                shutil.copy(source_path, target_path)
                print('done done')               
        except:
            pass 

        print('File has been successfully created and written.')
        print('Number of detected targets:', rem)

        if not lap_en and the_connection == None:
            the_connection = connect()

        print("$Waiting for the modified CSV FILE ...$")
        while(not os.path.exists(modified_csv_file)):
            time.sleep(0.5)

        print('Changed results_df\n', pd.read_csv(modified_csv_file))
        read_df = sort_dataframe_by_id(pd.read_csv(modified_csv_file))
        read_df.to_csv(modified_csv_file, index=False)
        print('Changed results_df\n', read_df)
        prev_item = -1
        print('Got these indices:',read_df.index)
        lap = 1
        while lap <= num_targets:
            # print("Inside the while lap and num_targets loop")
            # print(read_df.loc[i])
            if lap_en:
                # print("Inside the lap_en")
                msg = the_connection.recv_match(type=['MISSION_CURRENT'], blocking=True)
                # print(msg.seq, wp_index)
                if msg and msg.get_type() == 'MISSION_CURRENT':
                    if msg.seq == wp_index:
                        prev_item = msg.seq
                        while True:
                            msg = the_connection.recv_match(type=['GLOBAL_POSITION_INT'], blocking=True)
                            # print(msg)
                            current_lat = msg.lat / 1e7
                            current_lon = msg.lon / 1e7
                            distance = distance_lat_lon(current_lat,current_lon,wp_row['Latitude'].astype(float),wp_row['Longitude'].astype(float))
                            print('Distance between drone and wp point to change to guided:',blue_text,distance,reset_color,'m  ',end="\r")
                            if distance <= 2:
                                time.sleep(2)
                                try:
                                    while lap in read_df['ID'].values:
                                        print("inside the read_df.")
                                        row = read_df[read_df['ID'] == lap].iloc[0]
                                        if row['Dropped'].astype(bool) == False:
                                            print('Extracted Row:',row, lap)
                                            row['ID'].astype(int)

                                            Lat = row['Lat'].astype(float)
                                            Lon = row['Lon'].astype(float)
                                            print('REPOSITIONING TO COLOR:',row['Color'])
                                            print('REPOSITIONING TO LETTER:',row['Letter'])
                                            print('REPOSITIONING TO SHAPE:',row['Object Type'])
                                            automation((Lat, Lon), lap)
                                            row_index = read_df[read_df['ID'] == lap].index[0]
                                            # read_df.at[row_index, 'Dropped'] = True
                                            read_df.loc[row_index, 'Dropped'] = True
                                            # results_df.to_csv(results_csv_file, index=False)
                                            read_df.to_csv(modified_csv_file, index=False)
                                            time.sleep(4)
                                            break
                                        else:
                                            print("Bottle ID:", lap, "already dropped")
                                            lap += 1
                                    
                                except Exception as e:
                                    print('Skipping the Bottle ID,',red_text, 'Not Found ...', reset_color, 'error', e)
                                
                                lap += 1
                                if check_all_true(read_df, 'Dropped'):
                                    print('Mission Accomplished ...')
                                    exit()
                                break
                            time.sleep(0.1)
                    elif msg.seq != prev_item:
                        print("Currently in the WP:", msg.seq, "Waiting for the WP:", wp_index)
                        prev_item = msg.seq
