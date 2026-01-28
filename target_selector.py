import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import paramiko
import os

def save_desired_targets():
    desired_targets = []
    for i in range(5):
        shape = shape_var[i].get()
        color = color_var[i].get()
        letter = letter_var[i].get()
        id_ = i + 1
        # Check if all fields are selected
        if shape is not None and color is not None and letter is not None:
            # Convert 'None' string to None
            if shape == "None":
                shape = None
            if color == "None":
                color = None
            if letter == "None":
                letter = None
            desired_targets.append({'shape': shape, 'color': color, 'letter': letter, 'id': id_})
        else:
            messagebox.showerror("Error", f"Target {i+1} is incomplete. Please select all fields.")
            return  # Exit the function if any target is incomplete

    with open('desired_targets.txt', 'w') as file:
        file.write("desired_targets = [\n")
        for target in desired_targets:
            file.write(f"    {target},\n")
        file.write("]")

    messagebox.showinfo("Success", "Desired targets saved successfully!")

    # SSH connection to the Jetson device
    jetson_ip = '192.168.1.21'
    username = 'edhitha'
    password = 'password'
    remote_code1_path = '/home/edhitha/ultralytics/yo/AI/autocam.py'
    local_code1_path = 'autocam.py'

    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(jetson_ip, username=username, password=password)

    # Download the existing code1.py from the Jetson
    sftp_client = ssh_client.open_sftp()
    sftp_client.get(remote_code1_path, local_code1_path)

    # Replace desired_targets in local copy of code1.py
    with open(local_code1_path, 'r') as file:
        code1_content = file.read()

    start_index = code1_content.find("desired_targets = ")
    end_index = code1_content.find("]", start_index) + 1

    new_desired_targets_code = "desired_targets = " + repr(desired_targets)

    # Clear existing targets
    modified_code1_content = code1_content[:start_index] + new_desired_targets_code + code1_content[end_index:]

    with open(local_code1_path, 'w') as file:
        file.write(modified_code1_content)

    print("Replacement completed successfully!")

    # Transfer the modified code1.py back to the Jetson, replacing the existing file
    sftp_client.put(local_code1_path, remote_code1_path)
    sftp_client.close()

    # Close SSH connection
    ssh_client.close()

    print("Modified code1.py has been transferred and replaced on the Jetson.")

root = tk.Tk()
root.title("Desired Targets Selector")

shapes = ['Sleeping', 'Circle', 'Cross', 'Pentagon', 'Quartercircle', 'Rectangle', 'Semicircle', 'Star', 'Triangle']
colors = [None, 'brown', 'blue', 'black', 'green', 'orange', 'purple', 'red', 'white']
letters = [None] + [str(i) for i in range(10)] + [chr(i) for i in range(65, 91)]

shape_var = [tk.StringVar() for _ in range(5)]
color_var = [tk.StringVar() for _ in range(5)]
letter_var = [tk.StringVar() for _ in range(5)]

for i in range(5):
    ttk.Label(root, text=f"Target {i+1}").grid(row=i, column=0)
    ttk.Label(root, text="Shape:").grid(row=i, column=1)
    ttk.Combobox(root, textvariable=shape_var[i], values=shapes).grid(row=i, column=2)
    ttk.Label(root, text="Color:").grid(row=i, column=3)
    ttk.Combobox(root, textvariable=color_var[i], values=colors).grid(row=i, column=4)
    ttk.Label(root, text="Letter:").grid(row=i, column=5)
    ttk.Combobox(root, textvariable=letter_var[i], values=letters).grid(row=i, column=6)

save_button = ttk.Button(root, text="Save", command=save_desired_targets)
save_button.grid(row=5, columnspan=7)

root.mainloop()
