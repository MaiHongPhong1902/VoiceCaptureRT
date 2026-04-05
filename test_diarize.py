import torch
import speechbrain.utils.fetching as fetching
import shutil

# Monkeypatch symlink to copy to fix Windows permission error
def custom_link(src, dst, strategy):
    shutil.copy(src, dst)
fetching.link_with_strategy = custom_link

from speechbrain.inference.speaker import SpeakerRecognition

def test():
    print("Loading model...")
    verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_ecapa")
    print("Model loaded.")
    
    # 1 second of random noise at 16kHz
    signal = torch.randn(1, 16000)
    emb1 = verification.encode_batch(signal)
    emb2 = verification.encode_batch(signal)
    
    sim = torch.nn.functional.cosine_similarity(emb1, emb2, dim=-1)
    print("Shape:", emb1.shape)
    print("Similarity of identical noise:", sim.item())
    
if __name__ == "__main__":
    test()
