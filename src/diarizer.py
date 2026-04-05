import os
import shutil
import torch
import torch.nn.functional as F
import numpy as np
from src import config

# Monkeypatch SpeechBrain symlinking issues on Windows
def _custom_link(src, dst, *args, **kwargs):
    shutil.copy(src, dst)

try:
    import speechbrain.utils.fetching as fetching
    fetching.link_with_strategy = _custom_link
except ImportError:
    pass

from speechbrain.inference.speaker import SpeakerRecognition

class SpeakerDiarizer:
    def __init__(self, threshold=0.35):
        self.threshold = threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.speakers = [] # List of dicts: {"id": "S1", "centroid": Tensor}
        self.enabled = False

    def _load_model(self):
        if self.model is None:
            print(f"\033[93m[INFO] Loading Speaker Diarization model on {self.device}...\033[0m")
            self.model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir="tmp_ecapa",
                run_opts={"device": self.device}
            )
            print("\033[92m[SUCCESS] Speaker Diarization model ready.\033[0m")
        
    def set_enabled(self, enabled: bool):
        self.enabled = enabled
        if self.enabled:
            self._load_model()
            
    def _extract_embedding(self, audio_np: np.ndarray) -> torch.Tensor:
        # audio_np is 1D float32 at 16kHz
        # SpeechBrain expects [batch, time]
        signal = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_batch(signal)
            # embedding shape is [1, 1, 192], flatten to [192]
            return embedding.squeeze()

    def identify_speaker(self, audio_np: np.ndarray) -> str:
        if not self.enabled:
            return ""
        # Ignore extremely short chunks (less than 0.5s) as embeddings are unreliable
        if len(audio_np) < 16000 * 0.5:
            return ""

        emb = self._extract_embedding(audio_np)
        
        if not self.speakers:
            sid = "Speaker 1"
            self.speakers.append({"id": sid, "centroid": emb})
            return sid
            
        # Compute cosine similarity with known centroids
        best_score = -1.0
        best_idx = -1
        
        for i, spk in enumerate(self.speakers):
            score = F.cosine_similarity(emb.unsqueeze(0), spk["centroid"].unsqueeze(0)).item()
            if score > best_score:
                best_score = score
                best_idx = i
                
        if best_score >= self.threshold:
            # Update centroid with running average
            alpha = 0.1
            self.speakers[best_idx]["centroid"] = (1 - alpha) * self.speakers[best_idx]["centroid"] + alpha * emb
            self.speakers[best_idx]["centroid"] = F.normalize(self.speakers[best_idx]["centroid"], dim=-1)
            return self.speakers[best_idx]["id"]
        else:
            # Create new speaker
            new_id = f"Speaker {len(self.speakers) + 1}"
            self.speakers.append({"id": new_id, "centroid": emb})
            return new_id
