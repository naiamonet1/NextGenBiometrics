import cv2
import numpy as np
import os
import getpass
import shutil
from cryptography.fernet import Fernet
import base64

# Method for checking existence of path i.e the directory
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# Function to create a key from the user's password
def generate_key(password):
    return base64.urlsafe_b64encode(password.encode('utf-8').ljust(32)[:32])

# Function to encrypt the folder
def encrypt_folder(password, folder_path):
    key = generate_key(password)
    fernet = Fernet(key)
    # Create a duplicate of the folder for encryption
    duplicate_folder = f"{folder_path}_encrypted"
    shutil.copytree(folder_path, duplicate_folder)

    # Compress the duplicated folder into a zip file for easier encryption
    shutil.make_archive(duplicate_folder, 'zip', duplicate_folder)

    with open(f'{duplicate_folder}.zip', 'rb') as file:
        original_file = file.read()

    encrypted_file = fernet.encrypt(original_file)

    with open(f'{duplicate_folder}.zip', 'wb') as encrypted_file_out:
        encrypted_file_out.write(encrypted_file)

    # Remove the original uncompressed duplicate folder after encryption
    shutil.rmtree(duplicate_folder)  # Delete the uncompressed duplicate folder
    print(f"Folder '{folder_path}' has been encrypted into '{duplicate_folder}.zip'.")

# Function to decrypt the folder
def decrypt_folder(password, folder_path):
    key = generate_key(password)
    fernet = Fernet(key)

    try:
        with open(f'{folder_path}.zip', 'rb') as encrypted_file:
            encrypted_content = encrypted_file.read()

        decrypted_content = fernet.decrypt(encrypted_content)

        # Create a folder for the decrypted content
        decrypted_folder = folder_path.replace("_encrypted", "_decrypted")
        os.makedirs(decrypted_folder, exist_ok=True)

        with open(f'{folder_path}.zip', 'wb') as decrypted_file_out:
            decrypted_file_out.write(decrypted_content)

        # Extract the decrypted zip file into the new folder
        shutil.unpack_archive(f'{folder_path}.zip', decrypted_folder)
        print(f"Folder has been decrypted into '{decrypted_folder}'.")
    except Exception as e:
        print("Incorrect password or decryption error:", str(e))

# Function to ask the user if they want to encrypt their folder
def manage_folder_encryption():
    folder_path = "Training_Data"
    encrypt_decide = input("Would you like to encrypt your image folder entitled 'Training_Data'? (yes/no): ").strip().lower()

    if encrypt_decide == 'yes':
        password = getpass.getpass("Enter your desired password to encrypt the folder: ")
        encrypt_folder(password, folder_path)
    elif encrypt_decide == 'no':
        password_protect_decide = input("Would you like to password-protect your image folder instead? (yes/no): ").strip().lower()
        
        if password_protect_decide == 'yes':
            password = getpass.getpass("Enter a password that will be used to unlock the folder later: ")
            print("Your folder is now password-protected.")
        else:
            print("No action taken.")
    else:
        print("Invalid option. Please enter 'yes' or 'no'.")

# Function to ask the user if they want to unlock the folder
def unlock_folder():
    folder_path = "Training_Data_encrypted"
    unlock_decide = input("Would you like to decrypt your image folder? (yes/no): ").strip().lower()

    if unlock_decide == 'yes':
        password = getpass.getpass("Enter the password to decrypt the folder: ")
        decrypt_folder(password, folder_path)
    else:
        print("Folder remains locked.")

# Function to get the face ID from the user
def get_face_id():
    while True:
        try:
            face_id = int(input("Please verify your selected ID: "))
            if face_id in range(1, 11):
                return face_id
            else:
                print("Invalid ID. Please enter a number between 1 and 10.")
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Function to get the user's name
def get_user_name():
    return input("Please verify your name: ")

# Main function for facial recognition and folder management
def main():
    # Ask the user to enter a face ID and name
    face_id = get_face_id()
    user_name = get_user_name()

    # Ask the user to manage folder encryption
    manage_folder_encryption()

    # Load the pre-trained face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    assure_path_exists("saved_model/")
    recognizer.read('saved_model/s_model.yml')

    # Load prebuilt classifier for Frontal Face detection
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)

    # Font style for displaying the name on the screen
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Start video capture
    cam = cv2.VideoCapture(0)

    while True:
        # Read the video frame
        ret, im = cam.read()
        if not ret:
            print("Failed to capture image")
            break

        # Convert the captured frame into grayscale
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = faceCascade.detectMultiScale(gray, 3, 5)

        # For each face, predict and display the name
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x-40, y-40), (x+w+40, y+h+40), (0, 255, 0), 4)
            Id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # Print the ID and confidence for debugging
            #print(f"Detected ID: {Id}, Confidence: {confidence:.2f}")

            # Use the user-defined face_id for display
            if Id == face_id:  # Check if recognized ID matches user-defined ID
                name = f"{user_name}"
                # (ID: {face_id}) {100 - confidence:.2f}%"
                cv2.putText(im, name, (x, y-40), font, 1, (255, 255, 255), 3)
                #cv2.putText(im, f"Name: {name}", (x, y+h+5), font, 0.8, (255, 255, 255), 2)
                cv2.putText(im, f"Confidence: {confidence:.2f}", (x, y+h+25), font, 0.8, (255, 255, 255), 2)

        # Display the video frame with the bounded rectangle
        cv2.imshow('Face Recognition', im)

        # Press 'q' to close the program
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cam.release()
    cv2.destroyAllWindows()

    # Ask the user if they want to unlock the folder after the facial recognition process
    unlock_folder()

if __name__ == "__main__":
    main()
