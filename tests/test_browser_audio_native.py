"""Opt-in Windows check using two quiet synthetic audio processes, no recordings."""
import os
import subprocess
import sys
import time
import wave
from unittest.mock import patch

import numpy as np
import pytest

from process_audio import ProcessStream

pytestmark = pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("LIVETRANSLATE_TEST_NATIVE_AUDIO") != "1",
    reason="Set LIVETRANSLATE_TEST_NATIVE_AUDIO=1 on Windows to play quiet test tones",
)


def measure(stream):
    stream.read(stream.get_read_available())
    chunks = []
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        frames = stream.get_read_available()
        if frames:
            chunks.append(np.frombuffer(stream.read(frames), dtype="float32").reshape(-1, 2).mean(axis=1))
        time.sleep(.02)
    assert chunks and not stream.error
    samples = np.concatenate(chunks)
    spectrum = np.abs(np.fft.rfft(samples * np.hanning(len(samples))))
    frequencies = np.fft.rfftfreq(len(samples), 1 / 48000)
    peaks = {freq: float(np.max(spectrum[np.abs(frequencies - freq) < 3])) for freq in (440, 880)}
    return float(np.sqrt(np.mean(samples**2))), peaks


def test_native_process_isolation(tmp_path):
    player = tmp_path / "player.py"
    player.write_text(
        "import sys, time, winsound\nfrom pathlib import Path\n"
        "control = Path(sys.argv[2])\n"
        "playing = False\nwhile True:\n"
        "    enabled = not control.exists()\n"
        "    if enabled != playing:\n"
        "        winsound.PlaySound(sys.argv[1] if enabled else None, "
        "winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP if enabled else 0)\n"
        "        playing = enabled\n"
        "    time.sleep(.05)\n", encoding="utf-8")
    players = []
    stream = None
    try:
        for freq in (440, 880):
            path = tmp_path / f"tone-{freq}.wav"
            tone = (.03 * np.sin(2 * np.pi * freq * np.arange(48000) / 48000) * 32767).astype("int16")
            with wave.open(str(path), "wb") as fp:
                fp.setparams((1, 2, 48000, 0, "NONE", "not compressed"))
                fp.writeframes(tone.tobytes())
            players.append(subprocess.Popen(
                [sys.executable, "-X", "utf8", "-B", str(player), str(path), str(tmp_path / f"stop-{freq}")],
                creationflags=subprocess.CREATE_NO_WINDOW))
        with patch("process_audio.browser_pid", return_value=players[0].pid):
            stream = ProcessStream("synthetic-target.exe")
            time.sleep(.7)
            both_rms, peaks = measure(stream)
            (tmp_path / "stop-440").touch()
            time.sleep(.5)
            other_rms, _ = measure(stream)
            assert both_rms > .0001
            assert peaks[880] < peaks[440] * .01
            assert other_rms < both_rms * .01
            (tmp_path / "stop-440").unlink()
            time.sleep(.5)
            resumed_rms, resumed_peaks = measure(stream)
            assert resumed_rms > .0001
            assert resumed_peaks[880] < resumed_peaks[440] * .01
            print(f"Target RMS={both_rms:.6f}; excluded tone ratio={peaks[880]/peaks[440]:.6f}; "
                  f"excluded-only RMS={other_rms:.6f}; resumed RMS={resumed_rms:.6f}")
    finally:
        try:
            if stream:
                stream.close()
        finally:
            for proc in players:
                if proc.poll() is None:
                    proc.terminate()
                proc.wait(5)
