import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import sys
sys.path.append('src/Prediction')
from facial_droop import Predictor
from crop_image import crop_face
import multiprocessing as mp

def strokeStatus(frame_queue, result_queue):
    pd = Predictor("model/svm_5_features.pkl")
    while True:
        if frame_queue.empty():
            continue
        frame = frame_queue.get()
        if frame is None:
            continue

        frame = frame_queue.get()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pd.predict(frame)
        result_queue.put(result)

class RejectPage:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.canvas = tk.Canvas(window, width=1280, height=720, bg="white")
        self.canvas.create_text(550, 40, text="Sorry\n\nYou don't appear to \nhave a webcam\n\nPlease use this application\non a device that has a camera",
                                font=('Helvetica', 16), fill="black")
        self.canvas.pack()

class WebcamApp:

    # Timer - if frame is "stroke" for at least 10 seconds, then it is a stroke
    stroke_timer = 0
    is_triggered = False

    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        self.video_source = 0
        self.vid = cv2.VideoCapture(self.video_source)

        self.isTriggered = False
        if self.vid is None or not self.vid.isOpened():
            self.app = RejectPage
            self.app(root, "No Webcam?")
        else:
            self.canvas = tk.Canvas(window, width=self.vid.get(3), height=self.vid.get(4), bg="black")
            self.canvas.pack()

            self.btn_stop = tk.Button(window, text="Stop", width=10, command=self.stop, bg="#758796", fg="white", relief=tk.FLAT, bd=0, font=('Helvetica', 12))
            self.btn_call = tk.Button(window, text="Call 9-1-1", width=10, bg="#758796", fg="white", relief=tk.FLAT, bd=0, font=('Helvetica', 12))

            self.btn_stop.pack(padx=20, pady=10, side=tk.LEFT)
            self.btn_call.pack(padx=20, pady=10, side=tk.RIGHT)

            self.show_rectangle = tk.BooleanVar()
            self.show_rectangle.set(True)  # Initial state is True

            self.checkbox = tk.Checkbutton(window, text="Show Face Tracking", variable=self.show_rectangle, onvalue = True, offvalue = False)
            self.checkbox.pack(pady=10)

            self.window.protocol("WM_DELETE_WINDOW", self.stop)

            self.photo = None
            self.displayedPhoto = None
            self.is_stroke = 42
            self.frame_queue = mp.Queue(maxsize=2)
            self.result_queue = mp.Queue(maxsize=2)

            self.prediction_process = mp.Process(target=strokeStatus, args=(self.frame_queue, self.result_queue))
            self.prediction_process.start()

            self.update()

    def stop(self):
        # First, close the queues
        self.frame_queue.close()
        self.result_queue.close()

        # Terminate the prediction process
        self.prediction_process.terminate()

        # Wait for the prediction process to finish
        self.prediction_process.join()

        # Release the video capture
        if self.vid.isOpened():
            self.vid.release()

        # Finally, destroy the Tkinter window
        self.window.destroy()

    def update(self):
        ret, frame = self.vid.read()

        if ret:

            frame_with_outline, face_crop = crop_face(frame) 

            # Display full image, however process only cropped image.

            if (self.show_rectangle.get()):
                self.displayedPhoto = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame_with_outline, cv2.COLOR_BGR2RGB)))
            else:
                self.displayedPhoto = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)))

            self.canvas.create_image(0, 0, image=self.displayedPhoto, anchor=tk.NW)
            self.canvas.create_text(300, 40, text="Please hold still during scanning", font=('Helvetica', 16), fill="white")

            if self.is_stroke == 1:
                self.stroke_timer += 1
                print("possible facial droop detected")
                # This stroke timer helps with false positives. Basically, 6 frames in a sequence need to be "stroke" for the system to say definite stroke. 
                if (self.stroke_timer > 10):
                    print("facial droop detected")
                    self.canvas.create_text(300, 450, text="Facial Droop Detected", font=('Helvetica', 16), fill="red")
                    if self.isTriggered is False:
                        self.warning_page()
                        self.isTriggered = True
                else: 
                    self.canvas.create_text(300, 450, text="Scanning In Progress", font=('Helvetica', 16), fill="white")

            elif self.is_stroke == 0:
                self.stroke_timer = 0
                print("no facial droop detected")
                self.canvas.create_text(300, 450, text="No Droop Detected", font=('Helvetica', 16), fill="green")
                # self.isTriggered = False

            # TODO - this is causing some issues. Usually it doesn't detect a face.
            else:
                if (self.stroke_timer > 0):
                    self.stroke_timer -= 1
                print("no face detected")
                self.canvas.create_text(300, 450, text="Scanning In Progress", font=('Helvetica', 16), fill="white")
                # self.isTriggered = False

            if self.result_queue.empty():
                if self.frame_queue.empty():
                    self.frame_queue.put(frame)

                self.window.after(5, self.update)
                return

            self.is_stroke = self.result_queue.get()

        self.window.after(5, self.update)

    def warning_page(self):
        # Calculate the center coordinates of the main window
        main_window_center_x = self.window.winfo_rootx() + self.window.winfo_width() // 2
        main_window_center_y = self.window.winfo_rooty() + self.window.winfo_height() // 2

        secondary_window = tk.Toplevel()
        secondary_window.title("Secondary Window")

        # Calculate the position to center the warning page on the main window
        x_position = main_window_center_x - 150  # Adjust as needed
        y_position = main_window_center_y - 100  # Adjust as needed

        secondary_window.geometry(f"300x200+{x_position}+{y_position}")
        secondary_window.config(bg="#FF6347")  # Red background
        secondary_window.label = tk.Label(secondary_window,
                                        text="WARNING!!!\nYOU MAY BE HAVING\nA STROKE\nPLEASE CONSIDER\nDIALLING 911",
                                        font=('Helvetica', 16), fg="white", bg="#FF6347")
        secondary_window.label.pack()

class StartPage:
    def __init__(self, master, app):
        self.master = master
        self.master.title("Start Page")

        self.label = tk.Label(master, text="Are You Having A Stroke?", font=('Helvetica', 24), fg="black")  # Darker text color
        self.label.pack(pady=20, padx=20)

        self.btn_start = tk.Button(master, text="Check for Stroke", width=15, height=2, command=self.start_webcam,
                                   bg="#4CAF50", fg="white", font=('Helvetica', 14), relief=tk.FLAT, bd=0)  # Green background
        self.btn_start.pack(pady=20)

        self.app = app

    def start_webcam(self):
        self.btn_start.pack_forget()
        self.app(self.master, "Are You Having A Stroke?")

if __name__ == "__main__":
    root = tk.Tk()
    start_page = StartPage(root, WebcamApp)
    root.mainloop()
