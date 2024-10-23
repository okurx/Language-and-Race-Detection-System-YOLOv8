#-------------------------------------------------
#
# Project: Language and Race Detection System
# Created by: Burak Okur and Ibrahim Halil Ozcakir
# Creation Date: 2024-06-18
#
#-------------------------------------------------

# Language and Race Detection System
# This project combines advanced deep learning techniques to predict the language spoken by a user and their race
# based on voice recordings and images. By processing both audio and image inputs, the system aims to output a tailored message
# welcoming users in their predicted language, such as "Welcome to the airport" in various languages (e.g., Korean, Spanish, Chinese, etc.).

# This project serves as a starting point for integrating language and facial recognition into larger applications.
# Feel free to modify and expand this code for your needs. For any questions or suggestions, contact me at [burak1837burak@gmail.com].

from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import QImage, QPixmap, QIcon
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5 import uic
import sys
import cv2
import numpy as np
from PIL import Image
import sounddevice as sd
from scipy.io import wavfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
from ultralytics import YOLO
import os

# Load the UI file
ui_path = "mymainw.ui"
Ui_MainWindow, MainWindowBase = uic.loadUiType(ui_path)

# Thread class to record audio in the background
class AudioRecordThread(QThread):
    finished = pyqtSignal()

    def __init__(self, duration, sr, file_path):
        super().__init__()
        self.duration = duration
        self.sr = sr
        self.file_path = file_path

    def run(self):
        # Start recording audio
        print("Recording started")
        audio_file_path = os.path.join(self.file_path, "recorded_audio.wav")
        self.record(audio_file_path, self.duration, self.sr)
        self.finished.emit()

    def record(self, file_path, duration, sr):
        # Record the audio using sounddevice library
        audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float64')
        sd.wait()  # Wait for the recording to finish
        wavfile.write(file_path, sr, audio_data)
        print(f"Audio saved at {file_path}.")
        pass

# Main window class for the application
class MainWindow(MainWindowBase, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)  # Set up the UI elements
        self.design()  # Customize button icons
        self.clear()  # Clear any initial values on the UI

        # Load the YOLO models for sound and face recognition
        self.model_sound = YOLO("ses.pt")
        self.model_face = YOLO("yuz.pt")

        # Start the camera for capturing video
        self.cap = cv2.VideoCapture(0)
        self.desired_size = (224, 224)  # Desired size of images

        # Create a QTimer to periodically update the camera frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)  # Connect timer to update_frame function
        self.timer.start(30)  # Update every 30ms

        # Connect buttons to their respective functions
        self.detectButton.clicked.connect(self.detect)  # Detect button to run detection
        self.recordButton.clicked.connect(self.record_audio)  # Record button to start recording

        # Create an audio recording thread
        self.audio_thread = AudioRecordThread(duration=5, sr=16000, file_path="C:/Users/burak/OneDrive/Masaüstü/pyt/test")
        self.audio_thread.finished.connect(self.on_audio_record_finished)  # Connect finished signal to its handler

        self.recall_timer = QTimer(self)
        self.recall_timer.timeout.connect(self.clear)  # Clear the screen after timeout


    def design(self):
        # Set custom icons for the buttons
        icon_path = 'C:/Users/burak/OneDrive/Masaüstü/pyt/test/tanima.png'
        icon = QIcon(icon_path)
        self.detectButton.setIcon(icon)  # Set icon for detect button
        icon_size = QPixmap(icon_path).size()
        self.detectButton.setIconSize(icon_size)

        icon_path2 = 'C:/Users/burak/OneDrive/Masaüstü/pyt/test/mikrofon.png'
        icon2 = QIcon(icon_path2)
        self.recordButton.setIcon(icon2)  # Set icon for record button
        icon_size2 = QPixmap(icon_path2).size()
        self.recordButton.setIconSize(icon_size2)


    def clear(self):
        # Clear the UI elements to initial state
        self.faceimg.clear()
        self.recordimg.clear()
        self.race_prediction.clear()
        self.language_prediction.clear()
        self.detectButton.setVisible(False)
        self.detecttext.setVisible(False)
        self.prediction.setText('Please click "Record" and speak for 5 seconds.')  # Set default text

    def detect(self):
        # Initialize prediction and welcome message
        prediction = 'Unknown'
        welcome_message = 'Welcome to the airport'  # Default message for "Others" category
        race_label, probs_race = self.detect_race()  # Detect race

        # Check if no face is detected
        if race_label is None:
            self.prediction.setText('No face detected. Please try again.')
            return

        # Get the highest probability for the detected race
        race_percent = max(probs_race) * 100
        self.race_prediction.setText(f'{race_label} ({race_percent:.2f}%)')

        # Detect the language
        language_label, probs_language = self.detect_language()
        language_percent = max(probs_language) * 100
        self.language_prediction.setText(f'{language_label} ({language_percent:.2f}%)')

        # Adjust welcome message based on race and language prediction
        if race_label == 'East_Asian':
            if max(probs_language) >= 0.8:
                prediction = language_label
            else:
                # Determine between Korean and Chinese
                if probs_language[3] < 0.25 and probs_language[2] < 0.25:
                    prediction = 'Others'
                elif probs_language[3] < probs_language[2] and probs_language[2] >= 0.25:
                    prediction = 'Korean'
                elif probs_language[3] > probs_language[2] and probs_language[3] >= 0.25:
                    prediction = 'Chinese'

            # Set welcome message based on language
            if prediction == 'Korean':
                welcome_message = '공항에 오신 것을 환영합니다'  # Korean
            elif prediction == 'Chinese':
                welcome_message = '欢迎来到机场'  # Chinese

        elif race_label == 'Latino':
            if max(probs_language) >= 0.8:
                prediction = language_label
            else:
                # Determine between Spanish and Italian
                if probs_language[1] < 0.25 and probs_language[0] < 0.25:
                    prediction = 'Others'
                elif probs_language[1] < probs_language[0] and probs_language[0] >= 0.25:
                    prediction = 'Spanish'
                elif probs_language[1] > probs_language[0] and probs_language[1] >= 0.25:
                    prediction = 'Italian'

            # Set welcome message based on language
            if prediction == 'Spanish':
                welcome_message = 'Bienvenido al aeropuerto'  # Spanish
            elif prediction == 'Italian':
                welcome_message = 'Benvenuto all\'aeroporto'  # Italian

        elif race_label == 'Others':
            if max(probs_language) >= 0.8:
                prediction = language_label
            else:
                prediction = 'Others'

            # Default message for "Others"
            if prediction == 'Others':
                welcome_message = 'Welcome to the airport'

        # Display the welcome message in the predicted language
        self.prediction.setText(welcome_message)

        # Restart detection after 20 seconds
        self.recall_timer.start(20000)
        self.recall_timer.setSingleShot(True)

    def detect_race(self):
        # Function to detect race from the camera frame
        ret, frame = self.cap.read()  # Read a frame from the camera
        if not ret:
            self.cap.release()
            cv2.destroyAllWindows()
            return

        # Detect face in the frame using OpenCV Haar Cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

        if len(faces) == 0:
            self.prediction.setText('No face detected. Please try again.')
            return None, None

        # Process detected faces
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            resized_img = cv2.resize(face_img, self.desired_size)  # Resize face image
            pil_image = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))  # Convert to PIL image
            results = self.model_face(pil_image)  # Use YOLO model for face classification
            names_dict = results[0].names
            probs = results[0].probs.data.tolist()
            highest_prob_index = np.argmax(probs)
            race_label = names_dict[highest_prob_index]

            # Show face image on the UI
            frame2 = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
            height, width, channel = frame2.shape
            bytesPerLine = 3 * width
            self.qImg = QImage(frame2.data, width, height, bytesPerLine, QImage.Format_RGB888)
            self.pixmap = QPixmap.fromImage(self.qImg)
            self.faceimg.setPixmap(self.pixmap)
            self.faceimg.setScaledContents(True)

        # Map detected race to pre-defined categories
        race_map = {
            'Black': 'Other',
            'East_Asian': 'East_Asian',
            'Indian': 'Other',
            'Latino': 'Latino',
            'White': 'Others'
        }
        mapped_race_label = race_map.get(race_label, 'Other')
        return mapped_race_label, probs

    def detect_language(self):
        # Function to detect language from the spectrogram of the recorded audio
        self.spectrogram_file_path = "C:/Users/burak/OneDrive/Masaüstü/pyt/test/recorded_audio.png"
        results = self.model_sound(self.spectrogram_file_path)  # Use YOLO model to predict language
        names_dict = results[0].names
        probs = results[0].probs.data.tolist()
        highest_prob_index = np.argmax(probs)
        language_label = names_dict[highest_prob_index]

        # Map detected language to readable labels
        language_map = {
            'es': 'Spanish',
            'it': 'Italian',
            'ko': 'Korean',
            'zh': 'Chinese'
        }
        mapped_language_label = language_map.get(language_label, 'Other')

        # Show spectrogram image on the UI
        self.pixmap = QPixmap(self.spectrogram_file_path)
        self.recordimg.setPixmap(self.pixmap)
        self.recordimg.setScaledContents(True)
        return mapped_language_label, probs

    def record_audio(self):
        # Start audio recording
        self.prediction.setText("Recording...")
        self.audio_thread.start()

    def on_audio_record_finished(self):
        # Handle completion of audio recording
        self.detectButton.setVisible(True)
        self.detecttext.setVisible(True)
        audio_file_path = "C:/Users/burak/OneDrive/Masaüstü/pyt/test/recorded_audio.wav"
        self.audio_to_spectrogram(audio_file_path, 224, 224)  # Convert recorded audio to spectrogram
        self.prediction.setText('Voice recorded. Please look at the camera and click "Detect".')

    def audio_to_spectrogram(self, file_path, target_height, target_width):
        # Convert audio file to a spectrogram image
        y, sr = librosa.load(file_path, sr=16000, duration=5)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
        spectrogram = librosa.power_to_db(spectrogram, ref=np.max)

        # Save the spectrogram as an image
        plt.figure(figsize=(2.90, 2.92), dpi=100)
        librosa.display.specshow(spectrogram, sr=16000)
        plt.axis('off')
        plt.savefig(file_path.replace('.wav', '.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

    def update_frame(self):
        # Periodically update the camera frame in the UI
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            cv2.destroyAllWindows()
            return

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        bytesPerLine = 3 * width
        self.qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(self.qImg)
        self.cam.setPixmap(self.pixmap)
        self.cam.setScaledContents(True)


if __name__ == "__main__":
    # Run the application
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

