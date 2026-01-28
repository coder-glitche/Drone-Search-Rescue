import os
import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import shutil
import csv
import paramiko

try:
    shutil.rmtree('/home/yogesh/AI_UI/results')
except:
    pass
try:
    shutil.rmtree('/home/yogesh/AI_UI/Target')
except:
    pass

class ImageSelector:
    def __init__(self, root, folder1_path, folder2_path, jetson_ip, jetson_username, jetson_password):
        self.root = root
        self.folder1_path = folder1_path
        self.folder2_path = folder2_path
        self.load_images()
        self.create_main_window()
        self.create_second_window()
        self.selected_images = []
        self.jetson_ip = jetson_ip
        self.jetson_username = jetson_username
        self.jetson_password = jetson_password
    def load_images(self):
        self.image_list_folder1 = []
        self.image_list_folder2 = []

        try:
            for filename in os.listdir(self.folder1_path):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    self.image_list_folder1.append(os.path.join(self.folder1_path, filename))
        except FileNotFoundError:
            print(f"Error: The folder '{self.folder1_path}' does not exist.")

        try:
            for filename in os.listdir(self.folder2_path):
                if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
                    self.image_list_folder2.append(os.path.join(self.folder2_path, filename))
        except FileNotFoundError:
            print(f"Error: The folder '{self.folder2_path}' does not exist.")

        # Read CSV file and create a mapping between 'Image name' and 'id'
        self.id_mapping = {}
        try:
            with open(os.path.join(self.folder2_path, 'results.csv'), 'r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    image_name = row['Cropped Image']
                    image_id = int(row['ID'])
                    self.id_mapping[image_name] = image_id
        except FileNotFoundError:
            print(f"Error: The CSV file 'results.csv' does not exist in folder '{self.folder2_path}'.")

    def create_main_window(self):
        self.root.title("----------------DETECTIONS----------------")
        rows, cols = 8, 8

        for i in range(rows):
            for j in range(cols):
                if i * cols + j < len(self.image_list_folder1):
                    image_path = self.image_list_folder1[i * cols + j]
                    img = Image.open(image_path)
                    img.thumbnail((100, 100))
                    img = ImageTk.PhotoImage(img)

                    label = tk.Label(self.root, image=img)
                    label.image = img
                    label.grid(row=i, column=j, padx=5, pady=5)

                    label.bind("<Button-1>", lambda event, path=image_path: self.on_image_click(path))
                else:
                    break

    def create_second_window(self):
        self.second_window = tk.Toplevel(self.root)
        self.second_window.title("----------------PREDECTIONS----------------")
        self.display_images_second()
        

    def display_images_second(self):
        # Display 5 positions in a single row in window 2
        cols = 5

        # Keep references to PhotoImage objects to prevent them from being garbage-collected
        self.photo_references = []

        for widget in self.second_window.winfo_children():
            widget.destroy()

        for j in range(cols):
            position = j + 1
            image_path = self.get_image_path_by_position(position)
            if image_path:
                img = Image.open(image_path)
                img.thumbnail((100, 100))
                img = ImageTk.PhotoImage(img)

                label = tk.Label(self.second_window, image=img)
                label.image = img
                label.grid(row=0, column=j, padx=5, pady=5)

                # Bind the click event and pass image_path, label, and id
                label.bind("<Button-1>", lambda event, path=image_path, label=label, id=position: self.on_image_select(event, path, label, id))

                # Keep references to PhotoImage objects
                self.photo_references.append(img)
            else:
                not_detected_label = tk.Label(self.second_window, text="Not Detected", padx=5, pady=5)
                not_detected_label.grid(row=0, column=j, padx=5, pady=5)

        delete_button = tk.Button(self.second_window, text="Select and Delete", command=self.select_and_delete_images)
        delete_button.grid(row=1, column=0, columnspan=cols, pady=10)
        submit_button = tk.Button(self.second_window, text="Submit", command=self.submit_targets)
        submit_button.grid(row=2, column=0, columnspan=5, pady=10)
    def get_image_path_by_position(self, position):
        for image_name, id in self.id_mapping.items():
            if id == position:
                image_path = os.path.join(self.folder2_path, image_name)
                if os.path.exists(image_path):
                    return image_path
        return None
    def submit_targets(self):
        try:
            # Open the local results.csv file for reading
            local_results_path = os.path.join(self.folder2_path, 'results.csv')
            with open(local_results_path, 'r') as local_results_file:
                local_results_reader = csv.reader(local_results_file)
                local_results_data = list(local_results_reader)

            # Connect to the Jetson and overwrite the remote results.csv file
            self.transfer_results_to_jetson(local_results_data)

            # Show a popup message
            tk.messagebox.showinfo("Submission Successful", "TARGETS LOCKED!! THANOS DISPATCHED!!!")
        except Exception as e:
            print(f"Error during submission: {e}")

    def transfer_results_to_jetson(self, local_results_data):
        remote_results_path = '/home/edhitha/ultralytics/yo/output1/test4/crops/results/results1.csv'

        transport = paramiko.Transport((self.jetson_ip, 22))
        transport.connect(username=self.jetson_username, password=self.jetson_password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        try:
            # Write the local results data to a temporary file
            temp_results_path = '/tmp/results_temp.csv'
            with open(temp_results_path, 'w', newline='') as temp_results_file:
                temp_results_writer = csv.writer(temp_results_file)
                temp_results_writer.writerows(local_results_data)

            # Transfer the temporary file to the Jetson and overwrite the remote results.csv file
            sftp.put(temp_results_path, remote_results_path)
            print(f"Successfully transferred 'results.csv' to Jetson.")
        finally:
            # Close connections and delete the temporary file
            sftp.close()
            transport.close()
            os.remove(temp_results_path)

    def on_image_select(self, event, image_path, label, id):
        if image_path in self.selected_images:
            # If image is already selected, deselect it
            self.selected_images.remove(image_path)
            label.config(bg="SystemButtonFace")
        else:
            # If image is not selected, select it
            self.selected_images.append(image_path)
            label.config(bg="red")

    def on_image_click(self, image_path):
        image_name = os.path.basename(image_path)

        # Check if image is already in CSV, if not prompt for position
        if image_name not in self.id_mapping:
            position = self.prompt_for_position(image_path)
            if position is None:
                return
        else:
            position = self.id_mapping[image_name]

        # Move image to folder 2
        self.move_image(image_path, self.folder2_path)

        # Append details to results.csv
        self.append_to_results_csv(image_name, position)

        # Reload images and display in Window 2
        self.load_images()
        self.display_images_second()

    def move_image(self, source_path, destination_folder):
        destination_path = os.path.join(destination_folder, os.path.basename(source_path))
        if os.path.exists(destination_path):
            print(f"Destination path '{destination_path}' already exists")
        else:
            shutil.copy(source_path, destination_path)

    def select_and_delete_images(self):
        # Create a copy of the selected_images list to avoid modification during iteration
        selected_images_copy = self.selected_images.copy()

        # Delete selected images from folder 2 and corresponding rows from results.csv
        for image_path in selected_images_copy:
            # Delete image
            os.remove(image_path)

            # Delete corresponding row from results.csv
            image_name = os.path.basename(image_path)
            self.delete_row_from_results_csv(image_name)

        # Clear the original selected_images list
        self.selected_images.clear()

        # Reload images from folder 2
        self.load_images()
        self.display_images_second()

    def delete_row_from_results_csv(self, image_name):
        results_csv_path = os.path.join(self.folder2_path, 'results.csv')
        rows_to_keep = []

        try:
            with open(results_csv_path, 'r') as results_file:
                results_reader = csv.DictReader(results_file)
                for row in results_reader:
                    if row['Cropped Image'] != image_name:
                        rows_to_keep.append(row)

            # Rewrite the results.csv file without the deleted row
            with open(results_csv_path, 'w', newline='') as results_file:
                results_writer = csv.DictWriter(results_file, fieldnames=results_reader.fieldnames)
                results_writer.writeheader()
                results_writer.writerows(rows_to_keep)
        except FileNotFoundError:
            print(f"Error: The CSV file 'results.csv' does not exist in folder '{self.folder2_path}'.")

    def append_to_results_csv(self, image_name, position):
        detected_csv_path = os.path.join(self.folder1_path, 'detected.csv')
        results_csv_path = os.path.join(self.folder2_path, 'results.csv')

        try:
            with open(detected_csv_path, 'r') as detected_file:
                detected_reader = csv.DictReader(detected_file)
                for row in detected_reader:
                    if row['Cropped Image'] == image_name:
                        # Replace 'id' with entered position value
                        row['ID'] = position

                        # Append to results.csv
                        with open(results_csv_path, 'a', newline='') as results_file:
                            results_writer = csv.DictWriter(results_file, fieldnames=detected_reader.fieldnames)
                            results_writer.writerow(row)
                        break
        except FileNotFoundError:
            print(f"Error: The CSV file 'detected.csv' does not exist in folder '{self.folder1_path}'.")

    def prompt_for_position(self, image_path):
        # Prompt the user for the position to display the image
        position = simpledialog.askinteger("Position", "Enter the position for the image in window 2:")
        if position is not None:
            # Update the id_mapping with the entered position
            self.id_mapping[os.path.basename(image_path)] = position
            return position
        return None

    @staticmethod
    def transfer_folders_from_jetson(local_path, jetson_ip, jetson_username, jetson_password, jetson_folders):
        transport = paramiko.Transport((jetson_ip, 22))
        transport.connect(username=jetson_username, password=jetson_password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        for folder in jetson_folders:
            remote_path = f'/home/edhitha/ultralytics/yo/output1/test4/crops/{folder}'
            local_folder_path = os.path.join(local_path, folder)

            try:
                # Create local directory if it doesn't exist
                os.makedirs(local_folder_path, exist_ok=True)

                # Transfer files from Jetson to local machine
                for file_name in sftp.listdir(remote_path):
                    remote_file_path = f'{remote_path}/{file_name}'
                    local_file_path = os.path.join(local_folder_path, file_name)
                    sftp.get(remote_file_path, local_file_path)
                    print(f"Successfully transferred '{file_name}' from Jetson to local machine.")
            except FileNotFoundError:
                print(f"Error: Folder '{folder}' not found on the Jetson.")
            except Exception as e:
                print(f"Error: {e}")

        sftp.close()
        transport.close()

if __name__ == "__main__":
    # Set the paths and Jetson connection details
    folder1_path = "Target"
    folder2_path = "results"
    local_path = "/home/yogesh/AI_UI"
    jetson_ip = "192.168.1.21"  # Replace with the actual IP address of your Jetson
    jetson_folders = [folder1_path, folder2_path]

    # Transfer folders from Jetson to local machine
    ImageSelector.transfer_folders_from_jetson(local_path, jetson_ip, "edhitha", "password", jetson_folders)

    # Create the GUI and start the main loop
    root = tk.Tk()
    app = ImageSelector(root, os.path.join(local_path, folder1_path), os.path.join(local_path, folder2_path), jetson_ip, "edhitha", "password")
    root.mainloop()

'''2 folders named Target and results should be present in  /home/edhitha/ultralytics/yo/'''
