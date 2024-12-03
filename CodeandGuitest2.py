import tkinter as tk
from tkinter import messagebox
import cv2
import os
import numpy as np
from PIL import Image
import re

# Method for checking existence of path (i.e. the directory)
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# For detecting the faces in each frame, use Haarcascade Frontal Face default classifier of OpenCV
face_cascade_path = 'haarcascade_frontalface_default.xml'
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)

# Ensure that face_detector is loaded correctly before proceeding
if face_detector.empty():
    raise IOError("Failed to load the Haarcascade XML file. Please ensure the path is correct.")

# Initialize the face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()  # Define the face recognizer

# Method to get images and label data for training
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]  # Getting all file paths
    faceSamples = []  # Empty face sample initialized
    ids = []  # IDS for each individual

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # Convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8')  # Convert to numpy array

        # Get the image ID
        id = int(os.path.split(imagePath)[-1].split(".")[1])

        faces = face_detector.detectMultiScale(img_numpy)  # Detect faces

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y + h, x:x + w])  # Add face to sample
            ids.append(id)  # Add ID to list

    return faceSamples, ids

# Function for the login button
def login():
    banner = entry_username.get()  # Assuming 'entry_username' holds the BannerID

    # Check if the input is a 9-digit number
    if re.fullmatch(r"\d{9}", banner):  # Regex for exactly 9 digits
        messagebox.showinfo("Login Status", "Login Successful!")
        open_new_window()  # Call to open a new window on successful login
    else:
        messagebox.showerror("Login Status", "Invalid Credentials. Please enter a 9-digit number.")

# Function to open a new window after successful login
def open_new_window():
    new_window = tk.Toplevel(root)
    new_window.title("Welcome")
    new_window.geometry("300x200")  # Set size of the new window
    
    # Add a label in the new window
    welcome_label = tk.Label(new_window, text="Welcome to the application!", font=("Arial", 14))
    welcome_label.pack(pady=20)

    # Optionally, you can add a button to close the new window
    close_button = tk.Button(new_window, text="Close", command=new_window.destroy)
    close_button.pack(pady=10)

    # Start facial recognition after login
    start_face_recognition()

# Method to start facial recognition
def start_face_recognition():
    # Load the trained model
    recognizer.read('saved_model/s_model.yml')

    # Start the webcam
    vid_cam = cv2.VideoCapture(0)
    
    while True:
        _, image_frame = vid_cam.read()
        gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_region = gray[y:y + h, x:x + w]

            # Predict the face
            id, confidence = recognizer.predict(face_region)

            # Display the predicted ID and confidence
            cv2.putText(image_frame, f"ID: {id}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(image_frame, f"Confidence: {confidence:.2f}", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Draw rectangle around the face
            cv2.rectangle(image_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show the frame with face detection
        cv2.imshow("Face Recognition", image_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid_cam.release()
    cv2.destroyAllWindows()

# Create the main window
root = tk.Tk()
root.title("Login for Attendance")
root.geometry("800x600")  # Initial size, but will resize dynamically

# Define colors
navy_blue = "#000080"
gold_yellow = "#FFD700"
white_bg = "#FFFFFF"

# Frames for layout
navy_frame = tk.Frame(root, bg=navy_blue)
navy_frame.place(relwidth=0.33, relheight=1, anchor="nw", x=0, y=0)

# Good Morning label in the navy blue section
label_greeting = tk.Label(navy_frame, text="Good Morning", bg=navy_blue, fg="white", font=("Arial", 16))
label_greeting.pack(pady=(20, 5))

gold_frame = tk.Frame(root, bg=gold_yellow)
gold_frame.place(relwidth=0.08, relheight=1, anchor="nw", relx=0.33)

white_frame = tk.Frame(root, bg=white_bg)
white_frame.place(relwidth=0.59, relheight=1, anchor="nw", relx=0.41)

# Center the login form in the white frame using `place`
label_username = tk.Label(white_frame, text="Please enter your BannerID:", font=("Arial", 12))
label_username.place(relx=0.5, rely=0.4, anchor="center")

entry_username = tk.Entry(white_frame, font=("Arial", 12), bg="white")
entry_username.place(relx=0.5, rely=0.45, anchor="center")

login_button = tk.Button(white_frame, text="Login", font=("Arial", 12), bg=gold_yellow, fg="black", command=login)
login_button.place(relx=0.5, rely=0.55, anchor="center")

# Run the application
root.mainloop()
