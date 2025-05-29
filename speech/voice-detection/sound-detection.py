import torch
import torchaudio
import numpy as np

# Load the model
model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad",
                              model="silero_vad",
                              force_reload=True,
                              trust_repo=True)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
model = model.to(device)

# Get VAD function
get_speech_timestamps = utils[0]

# Load audio
wav, sr = torchaudio.load("conversation.wav")
wav = wav.to(device)

# Get speech timestamps
speech_timestamps = get_speech_timestamps(
    wav[0],           # Audio samples
    model,            # VAD model
    sampling_rate=sr, # Sample rate (16000)
    threshold=0.3,    # Speech threshold (0.0-1.0, lower = more sensitive)
    min_speech_duration_ms=250,  # Minimum speech segment length
    min_silence_duration_ms=100, # Minimum silence between segments
    window_size_samples=512      # Size of each chunk to process
)

# Print results
print("\nSpeech segments detected:")
for i, segment in enumerate(speech_timestamps):
    start_time = segment["start"] / sr
    end_time = segment["end"] / sr
    print(f"Segment {i+1}: {start_time:.1f}s -> {end_time:.1f}s")

