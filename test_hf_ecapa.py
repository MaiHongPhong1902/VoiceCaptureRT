import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioXVector

def test():
    try:
        extractor = AutoFeatureExtractor.from_pretrained("speechbrain/spkrec-ecapa-voxceleb")
        model = AutoModelForAudioXVector.from_pretrained("speechbrain/spkrec-ecapa-voxceleb")
        print("Model loaded via Transformers!")
    except Exception as e:
        print("Failed:", e)

test()
