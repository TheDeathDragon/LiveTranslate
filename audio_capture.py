import logging
import threading
import queue
import time
import numpy as np
import pyaudiowpatch as pyaudio

log = logging.getLogger("LiveTranslate.Audio")

DEVICE_CHECK_INTERVAL = 2.0  # seconds


def list_output_devices():
    """Return list of WASAPI output device names."""
    pa = pyaudio.PyAudio()
    devices = []
    wasapi_idx = None
    for i in range(pa.get_host_api_count()):
        info = pa.get_host_api_info_by_index(i)
        if "WASAPI" in info["name"]:
            wasapi_idx = info["index"]
            break
    if wasapi_idx is not None:
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == wasapi_idx
                and dev["maxOutputChannels"] > 0
                and not dev.get("isLoopbackDevice", False)
            ):
                devices.append(dev["name"])
    pa.terminate()
    return devices


def list_input_devices():
    """Return list of WASAPI input (microphone) device names."""
    pa = pyaudio.PyAudio()
    devices = []
    wasapi_idx = None
    for i in range(pa.get_host_api_count()):
        info = pa.get_host_api_info_by_index(i)
        if "WASAPI" in info["name"]:
            wasapi_idx = info["index"]
            break
    if wasapi_idx is not None:
        for i in range(pa.get_device_count()):
            dev = pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == wasapi_idx
                and dev["maxInputChannels"] > 0
                and not dev.get("isLoopbackDevice", False)
            ):
                devices.append(dev["name"])
    pa.terminate()
    return devices


class AudioCapture:
    """Capture system audio via WASAPI loopback using pyaudiowpatch."""

    def __init__(self, device=None, sample_rate=16000, chunk_duration=0.5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.audio_queue = queue.Queue(maxsize=100)
        self._stream = None
        self._running = False
        self._device_name = device
        self._pa = pyaudio.PyAudio()
        self._read_thread = None
        self._native_channels = 2
        self._native_rate = 44100
        self._current_device_name = None
        self._loopback_disabled = False
        self._lock = threading.Lock()
        self._restart_event = threading.Event()
        # Microphone input
        self._mic_device_name = None
        self._mic_stream = None
        self._mic_native_rate = 44100
        self._mic_native_channels = 1
        self._mic_restart_event = threading.Event()
        self._mic_buf = np.array([], dtype=np.float32)

    def _get_wasapi_info(self):
        for i in range(self._pa.get_host_api_count()):
            info = self._pa.get_host_api_info_by_index(i)
            if "WASAPI" in info["name"]:
                return info
        return None

    def _get_default_output_name(self):
        wasapi_info = self._get_wasapi_info()
        if wasapi_info is None:
            return None
        default_idx = wasapi_info["defaultOutputDevice"]
        default_dev = self._pa.get_device_info_by_index(default_idx)
        return default_dev["name"]

    @staticmethod
    def _query_current_default():
        """Create a fresh PA instance to get the actual current default device."""
        pa = pyaudio.PyAudio()
        try:
            for i in range(pa.get_host_api_count()):
                info = pa.get_host_api_info_by_index(i)
                if "WASAPI" in info["name"]:
                    default_idx = info["defaultOutputDevice"]
                    dev = pa.get_device_info_by_index(default_idx)
                    return dev["name"]
        finally:
            pa.terminate()
        return None

    def _find_loopback_device(self):
        """Find WASAPI loopback device for the default output."""
        wasapi_info = self._get_wasapi_info()
        if wasapi_info is None:
            raise RuntimeError("WASAPI host API not found")

        default_output_idx = wasapi_info["defaultOutputDevice"]
        default_output = self._pa.get_device_info_by_index(default_output_idx)
        log.info(f"Default output: {default_output['name']}")

        target_name = self._device_name or default_output["name"]

        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev["hostApi"] == wasapi_info["index"] and dev.get(
                "isLoopbackDevice", False
            ):
                if target_name in dev["name"]:
                    return dev

        # Fallback: any loopback device
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False):
                return dev

        raise RuntimeError("No WASAPI loopback device found")

    def _open_stream(self):
        """Open stream for current default loopback device."""
        loopback_dev = self._find_loopback_device()
        self._native_channels = loopback_dev["maxInputChannels"]
        self._native_rate = int(loopback_dev["defaultSampleRate"])
        self._current_device_name = loopback_dev["name"]

        log.info(f"Loopback device: {loopback_dev['name']}")
        log.info(
            f"Native: {self._native_rate}Hz, {self._native_channels}ch -> {self.sample_rate}Hz mono"
        )

        native_chunk = int(self._native_rate * self.chunk_duration)

        self._stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._native_channels,
            rate=self._native_rate,
            input=True,
            input_device_index=loopback_dev["index"],
            frames_per_buffer=native_chunk,
        )

    def _close_stream(self):
        if self._stream:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _find_mic_device(self):
        """Find WASAPI input device matching _mic_device_name, or system default."""
        wasapi_info = self._get_wasapi_info()
        if wasapi_info is None:
            raise RuntimeError("WASAPI host API not found")
        if self._mic_device_name in ("__default__", "default"):
            default_idx = wasapi_info["defaultInputDevice"]
            dev = self._pa.get_device_info_by_index(default_idx)
            if dev["maxInputChannels"] > 0:
                return dev
            raise RuntimeError("Default input device has no input channels")
        for i in range(self._pa.get_device_count()):
            dev = self._pa.get_device_info_by_index(i)
            if (
                dev["hostApi"] == wasapi_info["index"]
                and dev["maxInputChannels"] > 0
                and not dev.get("isLoopbackDevice", False)
                and dev["name"] == self._mic_device_name
            ):
                return dev
        raise RuntimeError(f"Mic device not found: {self._mic_device_name}")

    def _open_mic_stream(self):
        """Open microphone input stream."""
        dev = self._find_mic_device()
        self._mic_native_channels = dev["maxInputChannels"]
        self._mic_native_rate = int(dev["defaultSampleRate"])
        native_chunk = int(self._mic_native_rate * self.chunk_duration)
        log.info(
            f"Mic device: {dev['name']} ({self._mic_native_rate}Hz, {self._mic_native_channels}ch)"
        )
        self._mic_stream = self._pa.open(
            format=pyaudio.paFloat32,
            channels=self._mic_native_channels,
            rate=self._mic_native_rate,
            input=True,
            input_device_index=dev["index"],
            frames_per_buffer=native_chunk,
        )

    def _close_mic_stream(self):
        if self._mic_stream:
            try:
                self._mic_stream.stop_stream()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None

    def set_mic_device(self, device_name):
        """Change mic device at runtime. None = disabled."""
        if device_name == self._mic_device_name:
            return
        log.info(f"Mic device changed: {self._mic_device_name} -> {device_name}")
        self._mic_device_name = device_name
        if self._running:
            self._mic_restart_event.set()

    def set_device(self, device_name):
        """Change capture device at runtime. None = system default, '__disabled__' = off."""
        if device_name == self._device_name:
            return
        log.info(f"Audio device changed: {self._device_name} -> {device_name}")
        self._device_name = device_name
        self._loopback_disabled = device_name == "__disabled__"
        if self._running:
            self._restart_event.set()

    def _resample_to_mono(self, data, native_channels, native_rate):
        """Convert raw bytes to mono float32 at self.sample_rate."""
        audio = np.frombuffer(data, dtype=np.float32)
        if native_channels > 1:
            audio = audio.reshape(-1, native_channels).mean(axis=1)
        if native_rate != self.sample_rate:
            ratio = self.sample_rate / native_rate
            n_out = int(len(audio) * ratio)
            indices = np.arange(n_out) / ratio
            indices = np.clip(indices, 0, len(audio) - 1)
            idx_floor = indices.astype(np.int64)
            idx_ceil = np.minimum(idx_floor + 1, len(audio) - 1)
            frac = (indices - idx_floor).astype(np.float32)
            audio = audio[idx_floor] * (1 - frac) + audio[idx_ceil] * frac
        return audio

    def _restart_stream(self):
        """Restart stream with new default device."""
        with self._lock:
            self._close_stream()
            self._pa.terminate()
            self._pa = pyaudio.PyAudio()
            if not self._loopback_disabled:
                self._open_stream()
            else:
                log.info("Loopback disabled (mic-only mode)")
            # Re-open mic if active
            if self._mic_device_name:
                self._close_mic_stream()
                try:
                    self._open_mic_stream()
                except Exception as e:
                    log.warning(f"Failed to re-open mic after restart: {e}")

    def _read_loop(self):
        """Synchronous read loop in a background thread."""
        last_device_check = time.monotonic()

        while self._running:
            # Handle pending loopback restart request
            if self._restart_event.is_set():
                self._restart_event.clear()
                try:
                    self._restart_stream()
                    while not self.audio_queue.empty():
                        try:
                            self.audio_queue.get_nowait()
                        except queue.Empty:
                            break
                    log.info(f"Audio capture restarted on: {self._current_device_name}")
                except Exception as e:
                    log.error(f"Restart after device change failed: {e}")
                    time.sleep(0.5)
                continue

            # Handle mic device change
            if self._mic_restart_event.is_set():
                self._mic_restart_event.clear()
                self._close_mic_stream()
                self._mic_buf = np.array([], dtype=np.float32)
                if self._mic_device_name:
                    try:
                        self._open_mic_stream()
                        log.info(f"Mic stream opened: {self._mic_device_name}")
                    except Exception as e:
                        log.error(f"Failed to open mic: {e}")
                else:
                    log.info("Mic disabled")

            # Auto-switch only when using system default
            now = time.monotonic()
            if now - last_device_check > DEVICE_CHECK_INTERVAL:
                last_device_check = now
                if self._device_name is None:
                    try:
                        current_default = self._query_current_default()
                        if (
                            current_default
                            and self._current_device_name
                            and current_default not in self._current_device_name
                        ):
                            log.info(
                                f"System default output changed: "
                                f"{self._current_device_name} -> {current_default}"
                            )
                            log.info("Restarting audio capture for new device...")
                            self._restart_stream()
                            log.info(
                                f"Audio capture restarted on: {self._current_device_name}"
                            )
                    except Exception as e:
                        log.warning(f"Device check error: {e}")

            # Read loopback chunk or generate silence for mic-only mode
            loopback_audio = None
            if self._loopback_disabled:
                time.sleep(self.chunk_duration)
                n_samples = int(self.sample_rate * self.chunk_duration)
                loopback_audio = np.zeros(n_samples, dtype=np.float32)
            else:
                native_chunk = int(self._native_rate * self.chunk_duration)
                try:
                    data = None
                    with self._lock:
                        if not self._stream:
                            time.sleep(0.005)
                            continue
                        if self._stream.get_read_available() >= native_chunk:
                            data = self._stream.read(
                                native_chunk, exception_on_overflow=False
                            )
                    if data is not None:
                        loopback_audio = self._resample_to_mono(
                            data, self._native_channels, self._native_rate
                        )
                except Exception as e:
                    if self._restart_event.is_set():
                        continue
                    log.warning(f"Read error (device may have changed): {e}")
                    try:
                        time.sleep(0.5)
                        self._restart_stream()
                        log.info("Stream restarted after read error")
                    except Exception as re:
                        log.error(f"Restart failed: {re}")
                        time.sleep(1)
                    continue

            # Drain all available mic data into buffer
            if self._mic_stream:
                try:
                    avail = self._mic_stream.get_read_available()
                    if avail > 0:
                        mic_data = self._mic_stream.read(
                            avail, exception_on_overflow=False
                        )
                        mic_16k = self._resample_to_mono(
                            mic_data, self._mic_native_channels, self._mic_native_rate
                        )
                        self._mic_buf = np.concatenate([self._mic_buf, mic_16k])
                except Exception as e:
                    log.warning(f"Mic read error: {e}")

            if loopback_audio is None:
                time.sleep(0.005)
                continue

            # Mix: take matching length from mic buffer
            audio = loopback_audio
            mic_rms = None
            if len(self._mic_buf) > 0:
                n = len(loopback_audio)
                if len(self._mic_buf) >= n:
                    mic_chunk = self._mic_buf[:n]
                    self._mic_buf = self._mic_buf[n:]
                else:
                    mic_chunk = np.zeros(n, dtype=np.float32)
                    mic_chunk[: len(self._mic_buf)] = self._mic_buf
                    self._mic_buf = np.array([], dtype=np.float32)
                mic_rms = float(np.sqrt(np.mean(mic_chunk**2)))
                audio = loopback_audio + mic_chunk

            try:
                self.audio_queue.put_nowait((audio, mic_rms))
            except queue.Full:
                self.audio_queue.get_nowait()
                self.audio_queue.put_nowait((audio, mic_rms))

    def start(self):
        self._loopback_disabled = self._device_name == "__disabled__"
        if not self._loopback_disabled:
            self._open_stream()
        else:
            log.info("Loopback disabled (mic-only mode)")
        if self._mic_device_name:
            try:
                self._open_mic_stream()
            except Exception as e:
                log.warning(f"Failed to open mic on start: {e}")
        self._running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        log.info("Audio capture started")

    def stop(self):
        self._running = False
        if self._read_thread:
            self._read_thread.join(timeout=3)
        self._close_stream()
        self._close_mic_stream()
        log.info("Audio capture stopped")

    def get_audio(self, timeout=1.0):
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __del__(self):
        if self._pa:
            self._pa.terminate()
