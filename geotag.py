import os
import re
import sys
import time
import math
import shutil
import bisect
import folium
import pickle
import socket
import pexpect
import hashlib
import argparse
import threading
import subprocess
import multiprocessing
import pandas as pd
from pymavlink import mavutil
from folium.features import DivIcon

# Default values for variables, if any defaults changed please change it in the below description also
defaults = {
    'ubuntu': True,
    'UDP': True,
    'wire_conn': False,
    'socket': False,
    'ssh': False,
    'arm': False,
    'avg': False,
    'heading': False,
    'interpolate': False,
    'delete': False,
    'pix_port': '/dev/ttyUSB0',
    'baud': 57600,
    'lap_ip': '192.168.1.45',
    'jet_ip': '192.168.1.21',
    'udp': '14550',
    'tcp': '5760',
    'soc_port': 6234,
    'rate': 5,
    'cam_z': 180,
    'cap_gap': 1500,
    'no_img': 1000,
    'l_path': '/home/edhitha/DCIM/test_cam',
    'exif_path': '/usr/bin/exiftool'
}

# Descriptions for each argument
descriptions = {
    'ubuntu': 'Operating System being used to run this Python Automation',
    'UDP': 'Want to use UDP Connection?',
    'wire_conn': 'Pass this to enable USB connection, (Debugging purpose)',
    'socket': 'Pass it to enable the socket connection for ground laptop',
    'ssh': 'Pass it to enable the SSH connection to get the Data',
    'arm': 'will start after the arming the drone',
    'avg': 'If it is passed, the average algorithm will kick in',
    'interpolate': 'If it is passed then interpolate algorithm will kick in',
    'delete': 'Pass it to Delete the previous test_cam instead of moving this to the new test_cam_x folder',
    'heading': 'If this is passed then the yaw will be using the heading value it self, in some cases increases the accuracy',
    'pix_port': 'For Windows it is COM ports, for Ubuntu it is in the tty* format, (Ubuntu cmd = \'ls /dev/tty*\')',
    'baud': 'Set Pixhawk Baud, (Known using Mission Planner)',
    'lap_ip': 'Laptop Static IP of Jetson with a Active Connection',
    'jet_ip': 'Jetson Static IP of Jetson with a Active Connection, (cmd = \'ifconfig\')',
    'udp': 'Desired UDP port',
    'tcp': 'Desired TCP port',
    'soc_port': 'Desired port, and must be used same as in the laptop receiving on ground',
    'rate': 'The rate at which the Global_Position_Int and Attitude is updating, should be above 2Hz, and not above 4Hz',
    'cam_z': 'Camera rotation angle (0, 90, 180, 270), all positive or clockwise rotation of camera direction from Pixhawk Direction',
    'cap_gap': 'Capture gap in milliseconds',
    'no_img': 'Number of Images to be taken in the automation run',
    'l_path': 'Local path, images will be saved in this path',
    'exif_path': 'Path to the ExifTool executable,(cmd = \'which exiftool\')'
}

# Create an ArgumentParser object
parser = argparse.ArgumentParser(description='Script for performing automation task.')

# Add arguments to override default values
for key, value in defaults.items():
    if key in ['wire_conn', 'socket', 'arm', 'avg','interpolate', 'heading', 'ssh', 'delete']:
        parser.add_argument(f'-{key}', action='store_true', help=f'{descriptions[key]}, Default value: \'{value}\'')
    else:
        parser.add_argument(f'-{key}', default=value, help=f'{descriptions[key]}, Default value: \'{value}\'')

# Parse the command-line arguments
args = parser.parse_args()

# Access the values of the arguments
Ubuntu = bool(args.ubuntu)
UDP = (args.UDP)
using_wire_connection = bool(args.wire_conn)
socket_en = bool(args.socket)
ssh_en = bool(args.ssh)
arm_en = bool(args.arm)
avg_en = bool(args.avg)
inter_en = bool(args.interpolate)
heading_en = bool(args.heading)
del_en = bool(args.delete)
pixhawk_port = args.pix_port
baud = args.baud
jetson_ip = args.jet_ip
udp_port = args.udp
tcp_port = args.tcp
socket_port = int(args.soc_port)
refresh_rate = int(args.rate)
cam_z_rotation = int(args.cam_z)
capture_gap = int(args.cap_gap)
no_images = int(args.no_img)
local_path = args.l_path

def make_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def move_and_create_unique_folder(folder_name):
    # Get a list of all existing folders in the current directory
    # print(os.listdir('/home/edhitha/DCIM')) 
    folder_path = '/home/edhitha/DCIM'
    existing_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    print(existing_folders)
    # Initialize a counter for postfixing the folder name
    counter = 1

    # Generate a new folder name with a postfixed number
    new_folder_name = f"{folder_name}_{counter}"
    new_name = os.path.basename(new_folder_name)
    print(new_name)
    # Keep incrementing the counter until a unique folder name is found
    while new_name in existing_folders:
        counter += 1
        new_folder_name = f"{folder_name}_{counter}"
        new_name = os.path.basename(new_folder_name)
    # Create the new folder
    os.makedirs(new_folder_name)

    print(f"Created a new folder: {new_folder_name}")

    # Check if the original folder exists
    if os.path.exists(folder_name):
        # Move the contents of the original folder to the new one
        for item in os.listdir(folder_name):
            item_path = os.path.join(folder_name, item)
            shutil.move(item_path, new_folder_name)

        print(f"Moved contents of {folder_name} to {new_folder_name}")
try:
    if not del_en:
        move_and_create_unique_folder(local_path)
    else:
        os.removedirs(local_path)
except Exception as e:
    print(e)
local_path_tagged = os.path.join(local_path,'Tagged_img')
local_path_images = os.path.join(local_path,'images')
make_directory(local_path_tagged)
make_directory(local_path_images)

# Global variables
SR = b'SR2'
socket_timeout = 1
gps_data = multiprocessing.Manager().list()
cam_waypoints = []
initial_time = multiprocessing.Manager().Queue()
mavlink_conn = multiprocessing.Manager().list()
Failure = 0

print('camera_image_path:', local_path_images, ',Tagged image directory:', local_path_tagged)

command = f'sudo mavlink-routerd -e 127.0.0.1:14569 -e 127.0.0.1:14550 {pixhawk_port}:{baud} -e <laptop_ip>:<udp_port>' 

def capture_image(l_path_images, l_path):
    global Failure
    global initial_time
    try:
        # Use subprocess to run the image capture command
        capture_command = f"cd && cd {l_path_images} && nvgstcapture-1.0 --automate --count {no_images} --capture-gap {capture_gap} --capture-auto {no_images} --image-res 12"
        process = subprocess.Popen(capture_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)

        # Continuously read the output of the process
        while True:
            output = process.stdout.readline()
            if not output:
                print(red_text,'Breaking the loop',reset_color)
                break
            
            # Check for the "Image Captured" message and capture the initial time
            if "Image Captured" in output and initial_time.empty():
                initial_time.put(time.time())
                print("Image Captured", time.time())

            print(purple_text,output,reset_color, end='')
        print(purple_text,'Trying to communicate',reset_color)
        # Wait for the process to complete
        process.communicate()
        # Check the return code of the process


        if initial_time is not None:
            print(f"Initial time captured: {initial_time}")
        else:
            print("No 'Image Captured' message received.")

        print("Image capture command has completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing the image capture command: {e}")
    except Exception as ex:
        print(f"An error occurred: {ex}")

def get_newest_image_path(l_path):
    image_extensions = ["png", "jpg", "jpeg", "gif"]
    images = []
    for ext in image_extensions:
        images.extend([os.path.join(l_path, image) for image in os.listdir(l_path) if image.lower().endswith(ext)])
    if not images:
        return None

    newest_image = max(images, key=os.path.getctime)
    return newest_image

def get_img_prefix_from_local(l_path):
    file_prefix = None  # Initialize file_prefix with a default value

    try:
        # List files in the local directory
        files = os.listdir(l_path)

        # Check if there are any files
        if files:
            max_suffix = -1
            for name in files:
                match = re.search(r'nvcamtest_(\d+)', name)
                if match:
                    suffix = int(match.group(1))
                    if suffix > max_suffix:
                        max_suffix = suffix
                        file = name
            if file:
                parts = file.split('_')
                file_prefix = parts[0]+'_'+parts[1]+'_'+parts[2]+'_'

        if file_prefix:
            print(f"File Prefix: {file_prefix}")
            return file_prefix
        else:
            print("No files found in the folder.")
            return None

    except FileNotFoundError:
        print("Local directory not found.")
        return None
    
def get_unique_img_prefix(local_path, previous_prefix):
    try:
        # List files in the local directory
        files = os.listdir(local_path)
        processed_prefixes = []

        # Check if there are any files
        if files:
            max_suffix = -1
            
            for name in files:
                match = re.search(r'nvcamtest_(\d+)_s\d+_(\d+)\.jpg', name)
                if match:
                    prefix, suffix = name.split('_')[0], int(match.group(1))
                    if suffix > max_suffix:
                        max_suffix = suffix
                        processed_prefixes.append(prefix+'_'+str(suffix)+'_')
                        if previous_prefix != prefix+'_'+str(suffix)+'_':
                            return prefix+'_'+str(suffix)+'_'
            print('No unique img prefix found inside the folder ...')
            return None
    except FileNotFoundError:
        print("Local directory not found.")
        return None
    
def is_drone_armed(master):
    try:
        # Wait for the HEARTBEAT message
        msg = None
        while msg is None or msg.get_type() != 'HEARTBEAT':
            msg = master.recv_msg()
            print('Waiting for the heartBeat ...')

        # Check if the drone is armed
        return bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)

    except Exception as e:
        print(f"Error: {e}")
        return False

def collect_gps_data():
    global gps_data
    global arm_en
    # Connect to the MAVLink system
    if using_wire_connection:
        mavlink_connection = mavutil.mavlink_connection(pixhawk_port, baud=baud)
        print(red_text,f'Waiting for Heartbeat ....',reset_color)
        mavlink_connection.wait_heartbeat()
        print(f"{green_text}HeartBeat in-sync ...{reset_color}")
    elif UDP:
        mavlink_connection = mavutil.mavlink_connection(f'udpin:127.0.0.1:14569')
        # mavlink_connection = mavutil.mavlink_connection(f'udpin:192.168.1.21:14569')
        print(red_text,f'Waiting for Heartbeat ....',reset_color)
        mavlink_connection.wait_heartbeat()
        print(f"{green_text}HeartBeat in-sync ...{reset_color}")
    else:
        mavlink_connection = mavutil.mavlink_connection(f'tcp:127.0.0.1:5760')
        print(red_text,f'Waiting for Heartbeat ....',reset_color)
        mavlink_connection.wait_heartbeat()
        print(f"{green_text}HeartBeat in-sync ...{reset_color}")

    # Send the parameter set command for sr2_extra1
    mavlink_connection.mav.param_set_send(
        mavlink_connection.target_system,
        mavlink_connection.target_component,
        SR+b'EXTRA1',  # Parameter name
        refresh_rate,  # Custom rate value
        mavutil.mavlink.MAV_PARAM_TYPE_UINT16  # Parameter type
    )
    # Send the parameter set command for sr2_position
    mavlink_connection.mav.param_set_send(
        mavlink_connection.target_system,
        mavlink_connection.target_component,
        SR+b'POSITION',  # Parameter name
        refresh_rate,  # Custom rate value
        mavutil.mavlink.MAV_PARAM_TYPE_UINT16  # Parameter type
    )
    print(f'{orange_text}{SR}_POSITION and {SR}_EXTRA1 set to {refresh_rate}Hz{reset_color}')
    if arm_en:
        print('Waitng for Arming the Drone ...')
        while not is_drone_armed(mavlink_connection):
            print('Waiting for the drone arm ...')
            time.sleep(0.5)
        print('Waiting for next 5 seconds before starting ...')
        time.sleep(5)
        arm_en = False
    mavlink_conn.append(1)
    print('Receiving GPS Stream ...')
    
    msg_list = ['GLOBAL_POSITION_INT']
    if heading_en:
        pass
    else:
        msg_list.append('ATTITUDE')

    while True:
        # For this type of collecting, make sure the refresh rate or the measure rate is set the same for all, recommended 2Hz
        msg = mavlink_connection.recv_match(type=msg_list, blocking=True)
        
        if msg.get_type() == 'GLOBAL_POSITION_INT':
            # Create a new GPS data entry
            if heading_en:
                gps_entry = (msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1e3, ((msg.hdg / 100)+cam_z_rotation) % 360,((msg.hdg / 100)+cam_z_rotation) % 360, time.time())
            else:
                gps_entry = (msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1e3, ((msg.hdg / 100)+cam_z_rotation) % 360)
            # Acquire the lock to update the shared gps_data list
            gps_data.append(gps_entry)

        elif msg.get_type() == 'ATTITUDE':
            # Check if gps_data is not empty before updating
            if len(gps_data) > 0:
                if len(gps_data[-1]) == 4:
                    yaw = math.degrees(msg.yaw)
                    # Ensure yaw_deg is within the range [0, 360)
                    if yaw < 0:
                        yaw += 360.0
                    yaw = (yaw+cam_z_rotation) % 360
                    timestamp = time.time()
                    # print(yaw)
                else:
                    continue
                # print(gps_data[-1][:4] + (yaw, ))
                gps_data[-1] = gps_data[-1][:4] + (yaw, timestamp)
        if gps_data:
                try:
                    parts = gps_data[-1]
                    parts[5]
                    print(orange_text,gps_data[-1],reset_color)
                except:
                    pass
        time.sleep((1/(refresh_rate*2)))

def recv_msg(client, server):
    while True:
        try:
            received_data = client.recv(1024).decode()
            message, received_checksum = received_data.split('\nChecksum: ')

            # Calculate checksum of received message
            calculated_checksum = hashlib.md5(message.encode()).hexdigest()

            if received_checksum == calculated_checksum:
                print(f"Received data from Jetson: {message}")
                print(f"{green_text}Integrity check: Message is valid.{reset_color}")
                # Acknowledgement that recv message
                client.send(b'ACK: 0')
                return message
            else:
                print(f"{red_text}Integrity check: Message is corrupted.{red_text}")
                print('Requesting to resend ...')
                # Acknowledgement to resend the data
                client.send(b'ACK: 1')
        except socket.timeout:
            print(f"Socket operation timed out ({socket_timeout}s).")
            print(f'{red_text}This message will be received in next iteration...{reset_color}')
            return -1
        except Exception as e:
            print(f"{red_text}Error while receiving data:{reset_color} {str(e)}")
            try:
                client.close()
            except:
                pass
            print('Closed the connection ...')
            return -1

def send_msg(message, client, server):
    while True:
        try:
            # Calculate checksum
            checksum = hashlib.md5(message.encode()).hexdigest()

            # Send the message and checksum to the client
            client.send(f"{message}\nChecksum: {checksum}".encode())

            received_data = client.recv(1024).decode()

            result = int(received_data.split("ACK: ")[1])
            # Check if the resend ACK is received: 
            if result == 1:
                continue
            else:
                return result
        except socket.timeout:
            print("Socket operation timed out.")
            return -1
        except Exception as e:
            print(f"Error while sending data: {str(e)}")
            print('Closing the connection ...')
            try:
                client.close()
            except:
                pass
            return -1

def socket_handler(init_time, socket_server, client, image_counter, cam_waypoints):
    if len(sent) == 0:
        index_to_be_sent = 0
    else:
        # print('length of tags:',str(len(sent)))
        index_to_be_sent = sent[-1] + 1
    print('image_to_be_sent:',index_to_be_sent,'image_counter', image_counter)
    # socket_server.settimeout(socket_timeout)
    while index_to_be_sent <= image_counter:
        local_image_name = f"{prefix}{index_to_be_sent:05d}.jpg"   
        # print('Sending the gps entry')
        while not isinstance(cam_waypoints[index_to_be_sent], tuple):
            time.sleep(0.05)
        check = -1
        while check != 0:
            check = send_msg(f'["{local_image_name}",{cam_waypoints[index_to_be_sent]}]',client, socket_server)
            if check == 0:
                print('Message sent successfully, still inside the loop ...')
                sent.append(index_to_be_sent)
                index_to_be_sent += 1
                time.sleep(capture_gap/(1000*20))
    return 1

# Function to process images and geo-tag them
def geotag_images(prefix, l_path, o_path):
    """Geotags a list of images using GPS data.
    Args:
        prefix, l_path, o_path
    Returns:
        None.
    """
    global sent
    global gps_data
    global client_from_process
    t_path = None
    client = None
    sent = []
    client_from_process = multiprocessing.Manager().Queue()
    initial_time_use = initial_time.get()
    image_counter = 0
    avg_time = []
    pickle_path = os.path.join(local_path,'cam_wps.pickle')
    if socket_en:
        print('Initializing the socket server ...')
        socket_server = server_init()
        print(f'{red_text}Initial connection to the UI laptop is needed to send the Exifdata to the AI{reset_color}')
        print(f'{green_text}Listening to the client connections ...{reset_color}')
        listen_for_clients(socket_server, None)
        client = client_from_process.get()
        client_active = 1
        print('Client initialized ...')
        print('Waiting for the newest image ..')
    while not t_path:
        t_path = get_newest_image_path(l_path)
        time.sleep(0.05)
    skips = int(time.time() - initial_time_use)/(capture_gap/1000) - 2
    print('Initial skips count:',skips)
    prev_size = 0
    while True:
        try:
            if initial_time_use is not None:
                local_image_name = f"{prefix}{image_counter:05d}.jpg"
                print(reset_color,'Started Geotagging for ',local_image_name)
                image_time = initial_time_use + (image_counter * capture_gap / 1000)
                while True:
                    try:
                        temp_data = list(gps_data)
                        try:
                            parts = temp_data[-1]
                            parts[5]
                        except:
                            temp_data = temp_data[:-1]

                        print('Getting the actual gps entry...')
                        gps_index = bisect.bisect_right([t[-1] for t in temp_data], image_time)
                        print('Image index for the image time:',blue_text,gps_index,reset_color,' Data length:',blue_text,len(temp_data),reset_color)
                        if inter_en:
                            print('Interpolating_GPS_Data for the index:',gps_index,' Data length:',len(temp_data))
                            gps_entry = interpolate_gps_data(gps_index, temp_data, image_time)
                        else:
                            temp_index = gps_index if abs(temp_data[gps_index][-1] - image_time) < abs(temp_data[gps_index -1 ][-1] - image_time) else gps_index - 1
                            gps_entry = temp_data[temp_index]
                        break
                    except IndexError as e:
                        print(e)
                        time.sleep(1/refresh_rate)
                cam_waypoints.append(gps_entry)
                # print('GPS Entry will be put in this pickle path:',pickle_path)
                while True:
                    try:
                        with open(pickle_path, 'wb') as file:
                            print('file_open ...')
                            pickle.dump(cam_waypoints, file)
                        file_size = os.path.getsize(pickle_path)
                        if prev_size != file_size:
                            prev_size = file_size
                            print('Current file size:', blue_text, file_size, reset_color)
                        else:
                            print(red_text, 'Error saving the pickle file terminating the program ...', reset_color)
                            exit()
                        break
                    except:
                        time.sleep(0.0176)
                try:
                    if socket_en:
                        # if client_active == -1:
                        #     check = listen_for_clients(socket_server, socket_timeout)
                        #     if check == 1:
                        #         client = client_from_process.get()
                        #         client_active = 1
                        if client_active == 1:
                            temp_wps = cam_waypoints
                            client_active = socket_handler(initial_time_use, socket_server, client, image_counter, temp_wps)
                            print('Sent the gps entry successfully ..')
                    if ssh_en:
                        print('$\'',local_image_name,gps_entry[0],gps_entry[1],gps_entry[2],gps_entry[3],gps_entry[4],gps_entry[5],'\'$')
                except Exception as e:
                    print(e)
                avg_time.append(abs(gps_entry[-1]-image_time))
                print('Avg time differences:',blue_text, sum(avg_time)*1000/len(avg_time), reset_color, 'ms')
                if no_images == image_counter:
                    print('No_images == Remote Image Counter')
                    break
                if skips > 0:
                    time.sleep(capture_gap/(10000))
                    skips = int((time.time() - initial_time_use)/(capture_gap/1000)) - image_counter - 1
                    print('Updated skips count:',skips)
                else:
                    skips -= 1
                    skips = int((time.time() - initial_time_use)/(capture_gap/1000)) - image_counter - 1
                    print('Updated skips count in the else condition:',skips)
                    time.sleep(capture_gap/(1000))
                img_count = len(os.listdir(local_path_images))
                if img_count + 1 < image_counter:
                    print(f'Number images in the folder ({img_count+1})< image_counter({image_counter})')
                    print(f'{red_text}Image Capturing process unexpectedly ended ...{reset_color}')
                    print(f'{blue_text}Rerun both scripts ....{reset_color} \nNow, exiting the program ...')
                    return None
                image_counter += 1

        except Exception as e:
            print(e)
            try:
                path_map(cam_waypoints)
            except:
                pass
            print(red_text,'Failed to geotag',reset_color)
            time.sleep(0.25)
            pass

def interpolate_gps_data(index, gps_data, target_time):
    """Interpolates GPS data between two timestamps using Newton interpolation.
    Args:
        index: The index of the GPS data entry to interpolate from.
        gps_data: A list of GPS data entries.
        target_time: The timestamp to interpolate to.
    Returns:
        A tuple of interpolated GPS data (latitude, longitude, relative altitude, heading, yaw, timestamp).
    """
    temp_data = list(gps_data)
    try:
        parts = temp_data[-1]
        parts[5]
    except IndexError:
        temp_data = temp_data[:-1]

    print('Interpolating GPS Data for the index (inside function):', index, ' Data length:', len(temp_data))
    # Check if the index is valid.
    if not (0 <= index < len(gps_data) - 1):
        return gps_data[index]  # Invalid index for interpolation.

    # Extract data from the entries.
    prev_entry = gps_data[index - 1]
    next_entry = gps_data[index]

    # Check if all timestamps are the same, return the value directly.
    if len(set(point[-1] for point in [prev_entry, next_entry])) == 1:
        return prev_entry

    # Calculate time differences.
    time_diff_prev = next_entry[-1] - prev_entry[-1]

    # Check for very small time differences to avoid float division by zero.
    if abs(time_diff_prev) < 1e-9:
        return prev_entry

    # Interpolate using Newton interpolation for the last 4 points.
    interpolated_data = tuple(
        newton_interpolation(gps_data[i - refresh_rate:i + 1], target_time) for i in range(3, len(gps_data) - 1)
    ) + (target_time,)

    return interpolated_data

def reboot_ubuntu(password):
    try:
        command = 'sudo -S reboot now'
        
        child = pexpect.spawn(command, timeout=10)
        # Expect 'Password:' and provide the password
        child.expect('password for')
        child.sendline(password)
        # Wait for the command to complete
        child.expect(pexpect.EOF)
        print("Reboot command sent successfully.")
            
    except pexpect.ExceptionPexpect as e:
        sys.exit(f"Error: {e}")

def newton_interpolation(points, x):
    """Newton interpolation for a set of points."""
    n = len(points)

    # Calculate divided differences table.
    divided_diff = [points[i][0] for i in range(n)]

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            divided_diff[i] = (divided_diff[i] - divided_diff[i - 1]) / (points[i][1] - points[i - j][1])

    # Initialize result as the first divided difference.
    result = divided_diff[-1]

    # Evaluate the interpolating polynomial.
    for i in range(n - 2, -1, -1):
        result = result * (x - points[i][1]) + divided_diff[i]

    return result


def path_map(waypoints):
    # Create a folium map centered at the first waypoint
    m = folium.Map(location=[waypoints[0][0], waypoints[0][1]], zoom_start=16)

    # Add waypoints and custom arrow markers
    for lat, lon, alt, heading, yaw, time1 in waypoints:
        # Create a custom arrow marker
        icon = DivIcon(
            icon_size=(20, 20),
            icon_anchor=(10, 10),
            html=f'<div style="transform: rotate({yaw}deg); color: red;">&#11015;</div>',
        )

        folium.Marker(
            [lat, lon],
            icon=icon,
            popup=f"Time: {time1}<br>Altitude: {alt}"
        ).add_to(m)
    # timestamp = time.localtime(time.time())

    web_view = os.path.join(local_path, 'waypoints_map.html')
    # Save the map to an HTML file
    m.save(web_view)
    print('Check out the updated HTML file for path review...')

def server_init():
        no_tries = 3
        try:
            server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # server_socket.settimeout(socket_timeout)
            server_socket.bind((jetson_ip, socket_port))
            server_socket.listen()

            print(f"Server waiting for a connection on {jetson_ip}:{socket_port} {red_text}for next {socket_timeout}s{reset_color}")
            return server_socket

        except Exception as e:
            no_tries -= 1
            print(f"{red_text}Error while initializing the server:{reset_color} {str(e)}")
            time.sleep(1)  # Wait for 1 seconds before retrying
            if no_tries == 0:
                print(f'{red_text}Check once your jetson IP (current ip: {blue_text}{jetson_ip}{red_text}), if the  with the argument like this:{reset_color}')
                print(f'{red_text}-jet_ip <JETSON IP>{reset_color}')
                print(f"{red_text}STOPPING THE PROGRAM, RERUN WITH THE CORRECT IP{reset_color}")
                print(f'{blue_text}If the Given IP is correct, then stop and Rerun both receiving Script on the UI laptop and this script{reset_color}')
                exit()
            pass

def listen_for_clients(server_socket, timeout):
    while True:
        try:
            print(f'Accepting socket connections ....')
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")
            print('Sending verification code and waiting for ACK.')
            ping_start = time.time()
            send_msg('#Edhitha', conn, server_socket)
            msg = recv_msg(conn, server_socket)
            if '#VERIFIED' == msg:
                ping = time.time() - ping_start
                send_msg(str(ping), conn, server_socket)
                print(f'Verified client with ping of {ping*1000}ms')
                client_from_process.put(conn)
                return 1  # Return the new connection after verification
            else:
                print('Client is not verified, closing the connection')
                conn.close()
                continue
        except socket.timeout:
            print("Socket operation timed out.")
            if timeout == None:
                print(f'{blue_text}Retrying with listening to the connection...')
                continue
            return -1
        except Exception as e:
            print(f"Error while accepting a new client: {str(e)}")
            time.sleep(1)  # Wait for 1 seconds before retrying
            print('Retrying to accept new connections...')
            pass

if __name__ == "__main__":
    # Important global variables

    # ANSI escape code for red text
    red_text = "\033[91m"
    blue_text = "\033[94m"
    green_text = "\033[92m"
    purple_text = "\033[95m"
    orange_text = "\033[93m"
    # Reset the text color to default after printing
    reset_color = "\033[0m"
    if not using_wire_connection:
            # Your text to be printed in red
            text_to_print = f'''To run the mavlink command:
Copy & Paste this in Jetson terminal or in the SSH terminal and authenticate with the password.
Change the laptop ip and UDP port for external mavlink connection
If you are using telem connection to Jetson you might also want change the Pixhawk port and buad ...
'''
            # Print the text in red
            print(red_text + text_to_print + blue_text + command + reset_color)

    if socket_en:
        print('Request received to start the Jetson Server')
    else:
        print(f'{red_text}Server is not enabled, if you want the server initiate to get GPS Data, close this script and run the following command:')
        print(f'{blue_text}python3 <this_script_name>.py --socket{reset_color}')
    
    collect_gps_data_thread = multiprocessing.Process(target=collect_gps_data)
    collect_gps_data_thread.start()

    while arm_en:
        time.sleep(1)

    while len(mavlink_conn) == 0:
        print(red_text,'Waiting for Mavlink Stream',reset_color)
        time.sleep(0.5)
        if using_wire_connection:
            print(blue_text,f'Check pixhawk port: {pixhawk_port}:{baud}',reset_color)
        elif UDP:
            print(blue_text,f'Check UDP port: 127.0.0.1:14569',reset_color)
        else:
            print(blue_text,f'Check TCP port: 127.0.0.1:5760', reset_color)


    capture_process = multiprocessing.Process(target=capture_image, args=(local_path_images, local_path))
    capture_process.start()

    while initial_time.empty():
        continue
    
    if len(mavlink_conn) == 1:
        # Get the prefix from existing local images
        prefix = get_img_prefix_from_local(local_path_images)
        print(prefix)

    geotag_images(prefix, local_path_images, local_path_tagged)
    print('Program done is all Job')
