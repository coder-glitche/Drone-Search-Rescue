import re
import os
import sys
import shutil
import pandas
import pandas
import paramiko
import subprocess
import multiprocessing

def move_and_create_unique_folder(folder_name, l_path):
    # Get a list of all existing folders in the current directory
    # print(os.listdir(folder_name))
    existing_folders = [folder for folder in os.listdir(folder_name) if os.path.isdir(os.path.join(folder_name, folder))]

    print(existing_folders)
    # Initialize a counter for postfixing the folder name
    counter = 1

    # Generate a new folder name with a postfixed number
    new_folder_name = f"{folder_name}_{counter}"
    new_name = os.path.basename(l_path)
    print(new_name)
    # Keep incrementing the counter until a unique folder name is found
    while new_name in existing_folders:
        counter += 1
        new_folder_name = f"{l_path}_{counter}"
        new_name = os.path.basename(new_folder_name)
    # Create the new folder
    print(new_folder_name)
    os.makedirs(os.path.join(new_folder_name))

    print(f"Created a new folder: {new_folder_name}")

    # Check if the original folder exists
    if os.path.exists(l_path):
        # Move the contents of the original folder to the new one
        for item in os.listdir(l_path):
            item_path = os.path.join(l_path, item)
            shutil.move(item_path, new_folder_name)

        print(f"Moved contents of {l_path} to {new_folder_name}")

def parse_string_to_variables(input_string):
    # Use regular expression to extract values from the string
    match = re.match(r"\$' (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) (\S+) '\$", input_string)
    
    if match:
        # Extract the matched groups
        file, lat, lon, alt, head, yaw, time = match.groups()
        
        # Convert numerical values to appropriate types
        lat, lon, alt, head, yaw, time = map(float, (lat, lon, alt, head, yaw, time))
        
        return file, lat, lon, alt, head, yaw, time
    else:
        return None

def run_python_file(file_path):
    try:
        subprocess.run(["python", file_path], check=True)
        print("Execution completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
    except Exception as e:
        print(f"Unexpected error occurred: {e}")
        
def ssh_and_run_commands(hostname, username, password, commands):
    # Create an SSH client
    ssh_client = paramiko.SSHClient()
    # Automatically add the server's host key (this is insecure and should be done with caution)
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        # Connect to the remote device
        ssh_client.connect(hostname, username=username, password=password)
        print('Connected to host ...')
        # Initialize a shell
        ssh_shell = ssh_client.invoke_shell()

        # Wait for the shell to be ready
        while not ssh_shell.recv_ready():
            pass

        # Send commands
        for command in commands:
            ssh_shell.send(command + '\n')
            # Wait for the command execution to complete
            while not ssh_shell.recv_ready():
                pass
            # Read and capture the output
            output = ssh_shell.recv(65535).decode('utf-8')
            print(output)  # Print the output of the command

        print('Ran all the commands ..')

        while True:
            # Read and capture the output
            output = ssh_shell.recv(65535).decode('utf-8')
            # print(output)
            # Extract lines starting with '$' and ending with '$'
            if "$Waiting for the modified CSV FILE ...$" in output:
                process = multiprocessing.Process(target=run_python_file, args=(assisted_UI_path,))
                process.start()
            sys.stdout.write(output)


    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    # Replace these values with your actual remote device details
    hostname = '192.168.1.21'
    username = 'edhitha'
    password = 'password'
    commands_to_run = [
    'source /home/edhitha/ultralytics/venv/bin/activate && python3.8 /home/edhitha/ultralytics/yo/emergent.py -alt -test -lap -sitl '
        # Add more commands as needed
    ]

    assisted_UI_path = '/home/yogesh/AI_UI/assisted_UI.py'

    # Run the SSH and command execution
    ssh_and_run_commands(hostname, username, password, commands_to_run)