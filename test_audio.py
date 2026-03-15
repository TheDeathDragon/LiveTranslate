"""Quick test: capture loopback audio and check if data is non-zero."""
import pyaudiowpatch as pyaudio
import numpy as np

pa = pyaudio.PyAudio()

# Find loopback device
wasapi = None
for i in range(pa.get_host_api_count()):
    info = pa.get_host_api_info_by_index(i)
    if "WASAPI" in info["name"]:
        wasapi = info
        break

print(f"WASAPI: {wasapi['name']}")

loopback = None
for i in range(pa.get_device_count()):
    dev = pa.get_device_info_by_index(i)
    if dev.get("isLoopbackDevice", False):
        loopback = dev
        print(f"  Loopback: [{i}] {dev['name']} ch={dev['maxInputChannels']} rate={dev['defaultSampleRate']}")

if loopback is None:
    print("No loopback device found!")
    pa.terminate()
    exit(1)

channels = loopback["maxInputChannels"]
rate = int(loopback["defaultSampleRate"])
chunk = int(rate * 0.5)

print(f"\nCapturing from: {loopback['name']}")
print(f"Config: {rate}Hz, {channels}ch, chunk={chunk}")

stream = pa.open(
    format=pyaudio.paFloat32,
    channels=channels,
    rate=rate,
    input=True,
    input_device_index=loopback["index"],
    frames_per_buffer=chunk,
)

print("\nReading 6 chunks (3 seconds)...")
for i in range(6):
    data = stream.read(chunk, exception_on_overflow=False)
    audio = np.frombuffer(data, dtype=np.float32)
    mono = audio.reshape(-1, channels).mean(axis=1) if channels > 1 else audio
    rms = np.sqrt(np.mean(mono**2))
    print(f"  Chunk {i}: samples={len(mono)}, rms={rms:.6f}, max={np.abs(mono).max():.6f}")

stream.stop_stream()
stream.close()
pa.terminate()
print("\nDone.")
