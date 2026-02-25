import cv2
import librosa
import numpy as np
import os
import ffmpeg
from PIL import Image

class Preprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def extract_frames(self, video_path, sample_rate=1):
        """
        Extract frames from a video at a given sample rate (frames per second).
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / sample_rate) if fps > 0 else 1
        
        frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if count % frame_interval == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
            
            count += 1
        
        cap.release()
        return np.array(frames)

    def generate_spectrogram(self, audio_path):
        """
        Generate a mel-spectrogram from an audio file.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = librosa.load(audio_path)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Normalize and resize to target size for consistency
        S_dB_norm = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
        S_image = Image.fromarray((S_dB_norm * 255).astype(np.uint8))
        S_image = S_image.resize(self.target_size)
        
        return np.array(S_image)

    def preprocess_image(self, image_path):
        """
        Resize and normalize an image for model input.
        """
        img = Image.open(image_path).convert('RGB')
        img = img.resize(self.target_size)
        return np.array(img)
