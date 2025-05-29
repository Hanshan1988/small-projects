uv run python -c '
import sounddevice as sd
import soundfile as sf

# Record audio
duration = 10  # seconds
fs = 16000     # sample rate
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
print("Recording... speak for 10 seconds")
sd.wait()

# Save as WAV file
sf.write("conversation.wav", recording, fs)
print("Saved recording to conversation.wav")
'
