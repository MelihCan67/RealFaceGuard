import cv2
import os
import time
import tkinter as tk
from tkinter import simpledialog, messagebox


class CameraCapture:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.directory = './dataset/new_users'  # Default save directory

    def capture_photos(self, profile_type, num_photos):
        print(f"Capturing {num_photos} photos for {profile_type} profile...")
        start_time = time.time()
        frame_count = 0
        capture_interval = 0.5  # Capture every 0.5 seconds

        while frame_count < num_photos:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame, exiting...")
                break

            if time.time() - start_time >= capture_interval * frame_count:
                frame_count += 1
                photo_name = os.path.join(self.directory, f"{profile_type}_{frame_count}.jpg")
                cv2.imwrite(photo_name, frame)
                print(f"Saved: {photo_name}")

                # Show camera on screen
                cv2.imshow('Camera', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        print(f"A total of {frame_count} {profile_type} profile photos were saved.")

    def start_capture(self):
        root = tk.Tk()
        root.withdraw()
        folder_name = simpledialog.askstring("Folder Name", "Please enter the name of the folder to save photos:")
        if folder_name is None:
            print("No folder name entered, exiting...")
            return

        self.directory = os.path.join('./dataset/new_users', folder_name)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        # Capture front profile photos
        self.capture_photos("front", 20)

        # Guide user
        messagebox.showinfo("Direction", "Please turn your face slightly to the right and click 'OK'.")
        time.sleep(2)

        # Capture right profile photos
        self.capture_photos("right_profile", 10)

        # Guide user
        messagebox.showinfo("Direction", "Please turn your face slightly to the left and click 'OK'.")
        time.sleep(2)

        # Capture left profile photos
        self.capture_photos("left_profile", 10)

        # Guide user
        messagebox.showinfo("Direction", "Please look slightly upwards and click 'OK'.")
        time.sleep(2)

        # Capture up profile photos
        self.capture_photos("up_profile", 10)

        # Guide user
        messagebox.showinfo("Direction", "Please look slightly downwards and click 'OK'.")
        time.sleep(2)

        # Capture down profile photos
        self.capture_photos("down_profile", 10)

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    camera = CameraCapture()
    camera.start_capture()
