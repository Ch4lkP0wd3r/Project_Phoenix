import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np

class DeepfakeDetector:
    def __init__(self, device='cpu'):
        self.device = device
        # Using EfficientNet-B0 as a base for visual analysis
        self.visual_model = models.efficientnet_b0(pretrained=False)
        self.visual_model.classifier[1] = nn.Linear(self.visual_model.classifier[1].in_features, 1)
        self.visual_model.to(self.device).eval()
        
        # Audio forensic model (CNN + LSTM) stub
        self.audio_model = self._build_audio_model().to(self.device).eval()

    def _build_audio_model(self):
        class AudioNet(nn.Module):
            def __init__(self):
                super(AudioNet, self).__init__()
                self.conv = nn.Sequential(
                    nn.Conv2d(1, 16, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2)
                )
                self.lstm = nn.LSTM(32 * 56 * 56, 128, batch_first=True) # Assumes 224x224 input
                self.fc = nn.Linear(128, 1)
            
            def forward(self, x):
                # Assuming x shape: (B, 1, 224, 224)
                x = self.conv(x)
                x = x.view(x.size(0), -1)
                x = x.unsqueeze(1) # B, 1, Features
                x, _ = self.lstm(x)
                x = self.fc(x[:, -1, :])
                return x
        
        return AudioNet()

    def predict_visual(self, frames):
        """
        Predict authenticity score for a list of frames.
        Returns a score between 0 and 1 (1 = authentic, 0 = manipulated).
        """
        if len(frames) == 0:
            return 1.0
        
        # Normalize and convert to tensor
        frames_tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        frames_tensor = frames_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = torch.sigmoid(self.visual_model(frames_tensor))
            score = outputs.mean().item()
        
        return score

    def predict_audio(self, spectrogram):
        """
        Predict authenticity score for a spectrogram.
        Returns a score between 0 and 1 (1 = authentic, 0 = manipulated).
        """
        spec_tensor = torch.from_numpy(spectrogram).unsqueeze(0).unsqueeze(0).float() / 255.0
        spec_tensor = spec_tensor.to(self.device)
        
        with torch.no_grad():
            output = torch.sigmoid(self.audio_model(spec_tensor))
            score = output.item()
        
        return score

    def fuse_scores(self, visual_score, audio_score, visual_weight=0.7, audio_weight=0.3):
        """
        Fuse visual and audio scores using a weighted average.
        In a real scenario, this could be a Bayesian fusion layer.
        """
        if visual_score is None:
            return audio_score
        if audio_score is None:
            return visual_score
            
        fused_score = (visual_score * visual_weight) + (audio_score * audio_weight)
        return fused_score

    def get_authenticity_percentage(self, score):
        """
        Convert 0-1 score to 0-100 percentage.
        """
        return round(score * 100, 2)
