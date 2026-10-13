import ctypes as c
import os
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import torch  # noqa: F401
import numpy as np
import pytest
import yaml
from PyQt6.QtWidgets import QApplication

import audio_capture
import control_panel
import process_audio as native


@pytest.fixture
def processes(monkeypatch):
    def install(rows, active=(), idle=()):
        procs = [SimpleNamespace(pid=pid, info={"pid": pid, "name": name, "ppid": parent})
                 for pid, name, parent in rows]
        monkeypatch.setattr(native.psutil, "process_iter", lambda fields: iter(procs))
        monkeypatch.setattr(native, "active_audio_pids",
                            lambda active_only=True: set(active if active_only else idle))
    return install


@pytest.mark.parametrize("executable", ["chrome.exe", "SogouExplorer.exe"])
def test_actual_audio_child_excludes_voice_chat(processes, executable):
    processes([(10, executable, 1), (11, executable.upper(), 10), (20, "KOOK.exe", 1)],
              active=(11, 20))
    assert native.browser_pid(executable) == 11


def test_paused_audio_child_stays_selected(processes):
    processes([(10, "SogouExplorer.exe", 1), (11, "SogouExplorer.exe", 10)], idle=(11,))
    assert native.browser_pid("SogouExplorer.exe") == 11


def test_detached_sogou_helpers_do_not_hide_browser_tree(processes):
    processes([(10, "SogouExplorer.exe", 1), (11, "SogouExplorer.exe", 10),
               (20, "SogouExplorer.exe", 2), (21, "SogouExplorer.exe", 3)])
    assert native.browser_pid("SogouExplorer.exe") == 10


@pytest.mark.parametrize("active", [(11, 31), ()])
def test_ambiguous_instances_fail_closed(processes, active):
    processes([(10, "chrome.exe", 1), (11, "chrome.exe", 10),
               (30, "chrome.exe", 1), (31, "chrome.exe", 30)], active=active)
    with pytest.raises(RuntimeError, match="Multiple"):
        native.browser_pid("chrome.exe")


def test_absent_browser_does_not_select_another_app(processes):
    processes([(20, "KOOK.exe", 1)], active=(20,))
    assert native.browser_pid("chrome.exe") is None


def test_session_enumeration_failure_keeps_browser_scope(processes, monkeypatch):
    processes([(10, "chrome.exe", 1), (20, "KOOK.exe", 1)])
    monkeypatch.setattr(native, "active_audio_pids", Mock(side_effect=OSError("session failure")))
    assert native.browser_pid("chrome.exe") == 10


def capture_fixture():
    capture = audio_capture.AudioCapture.__new__(audio_capture.AudioCapture)
    capture._pa = None
    capture._device_name = "__process__:chrome.exe"
    capture._running = False
    capture._lock = threading.Lock()
    capture._restart_event = threading.Event()
    capture._mic_restart_event = threading.Event()
    capture._mic_device_name = None
    capture._mic_stream = None
    capture._mic_buf = np.array([], dtype=np.float32)
    capture._loopback_disabled = False
    capture._native_rate = 48000
    capture._native_channels = 2
    capture.sample_rate = 16000
    capture.chunk_duration = .032
    capture.audio_queue = queue.Queue(maxsize=100)
    return capture


def test_browser_open_disables_mic_and_discards_mic_buffer(monkeypatch):
    capture = capture_fixture()
    capture._mic_buf = np.ones(100, dtype=np.float32)
    capture.set_mic_device("__default__")
    assert capture._mic_device_name is None
    closed = Mock()
    capture._close_mic_stream = closed
    stream = SimpleNamespace(channels=2, rate=48000)
    factory = Mock(return_value=stream)
    monkeypatch.setattr(native, "ProcessStream", factory)
    capture._open_stream()
    factory.assert_called_once_with("chrome.exe")
    assert closed.called and capture._stream is stream and not capture._mic_buf.size


def test_browser_open_failure_cannot_select_system_loopback(monkeypatch):
    capture = capture_fixture()
    capture._find_loopback_device = Mock(side_effect=AssertionError("System audio must remain excluded"))
    monkeypatch.setattr(native, "ProcessStream", Mock(side_effect=OSError("unsupported API")))
    with pytest.raises(OSError, match="unsupported"):
        capture._open_stream()
    capture._find_loopback_device.assert_not_called()


def test_runtime_failure_does_not_retry_or_fall_back():
    capture = capture_fixture()
    capture._running = True
    capture._stream = SimpleNamespace(error=OSError("capture failure"))
    capture._restart_stream = Mock(side_effect=AssertionError("Must not retry"))
    def close():
        capture._stream = None
        capture._running = False
    capture._close_stream = Mock(side_effect=close)
    capture._read_loop()
    capture._close_stream.assert_called_once()
    capture._restart_stream.assert_not_called()


def test_source_change_discards_queued_and_in_flight_audio():
    capture = capture_fixture()
    capture._device_name = None
    capture._running = True
    capture.audio_queue.put((np.ones(512), None))
    capture._stream = SimpleNamespace(get_read_available=lambda: 1536,
                                     read=lambda *a, **kw: bytes(1536 * 8))
    def resample(*args):
        capture.set_device("__process__:SogouExplorer.exe")
        capture._running = False
        return np.ones(512)
    capture._resample_to_mono = resample
    capture._read_loop()
    assert capture.audio_queue.empty() and capture._restart_event.is_set()


def test_default_device_read_still_delivers_audio():
    capture = capture_fixture()
    capture._device_name = None
    capture._running = True
    capture._stream = SimpleNamespace(get_read_available=lambda: 1536,
                                     read=lambda *a, **kw: bytes(1536 * 8))
    def resample(*args):
        capture._running = False
        return np.ones(512)
    capture._resample_to_mono = resample
    capture._read_loop()
    samples, mic_rms = capture.get_audio(timeout=0)
    assert np.all(samples == 1) and mic_rms is None


@pytest.mark.parametrize("source", ["__process__:chrome.exe", "__process__:SogouExplorer.exe"])
def test_panel_restores_and_saves_browser_source_without_mic(source, tmp_path, monkeypatch):
    qt = QApplication.instance() or QApplication([])
    monkeypatch.setattr(control_panel, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr(control_panel.ControlPanel, "_refresh_cache", lambda self: None)
    monkeypatch.setattr(audio_capture, "list_output_devices", lambda: ["Headphones"])
    monkeypatch.setattr(audio_capture, "list_input_devices", lambda: ["Microphone"])
    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    panel = control_panel.ControlPanel(config, {"audio_device": source, "mic_device": "__default__"})
    try:
        assert panel._audio_device.currentData() == source
        assert panel._mic_device.currentIndex() == 0 and not panel._mic_device.isEnabled()
        panel._apply_settings()
        assert panel.get_settings()["audio_device"] == source
        assert panel.get_settings()["mic_device"] is None
        panel._audio_device.setCurrentIndex(1)
        assert panel._mic_device.isEnabled()
        panel._apply_settings()
        assert panel.get_settings()["audio_device"] is None
        panel._audio_device.setCurrentIndex(panel._audio_device.findText("Headphones"))
        panel._apply_settings()
        assert panel.get_settings()["audio_device"] == "Headphones"
    finally:
        panel._save_timer.stop()
        panel.close()
        qt.processEvents()


def test_capture_backlog_is_bounded():
    stream = native.ProcessStream.__new__(native.ProcessStream)
    stream._data = bytearray()
    stream._lock = threading.Lock()
    stream._append(bytes(48000 * 8 * 3))
    assert stream.get_read_available() == 96000
    assert len(stream.read(1536)) == 1536 * 8
    assert stream.get_read_available() == 96000 - 1536


def test_failed_com_initialization_is_reported_without_uninitialize(monkeypatch):
    ole = SimpleNamespace(CoInitializeEx=lambda *a: -2147467259, CoUninitialize=Mock())
    monkeypatch.setattr(native.c, "OleDLL", lambda *a: ole)
    stream = native.ProcessStream.__new__(native.ProcessStream)
    stream._ready = threading.Event()
    stream._run()
    assert isinstance(stream.error, OSError) and stream._ready.is_set()
    ole.CoUninitialize.assert_not_called()


def test_start_failure_releases_both_native_interfaces(monkeypatch):
    def fake_call(obj, index, result, types, *args):
        address = obj if isinstance(obj, int) else obj.value
        if address == 44 and index == 3:
            c.cast(args[0], c.POINTER(native.HR))[0] = 0
            c.cast(args[1], c.POINTER(native.P))[0] = native.P(123)
        if address == 123 and index == 14:
            c.cast(args[1], c.POINTER(native.P))[0] = native.P(456)
        if address == 123 and index == 10:
            return -2147467259
        return 0
    calls = Mock(side_effect=fake_call)
    monkeypatch.setattr(native, "call", calls)
    def activate(path, iid, params, handler, operation):
        c.cast(operation, c.POINTER(native.P))[0] = native.P(44)
        table = c.cast(handler, c.POINTER(c.POINTER(native.P))).contents
        c.WINFUNCTYPE(native.HR, native.P, native.P)(table[3])(handler, native.P(44))
        return 0
    monkeypatch.setattr(native.c, "WinDLL", lambda *a: SimpleNamespace(ActivateAudioInterfaceAsync=Mock(side_effect=activate)))
    stream = native.ProcessStream.__new__(native.ProcessStream)
    with pytest.raises(OSError):
        stream._activate(10)
    released = [args[0].value for args, _ in calls.call_args_list if args[1] == 2]
    assert released == [44, 456, 123]


def test_worker_waits_reconnects_and_stops_without_system_fallback(monkeypatch):
    ole = SimpleNamespace(CoInitializeEx=lambda *a: 0, CoUninitialize=Mock())
    monkeypatch.setattr(native.c, "OleDLL", lambda *a: ole)
    clock = [10.0]
    monkeypatch.setattr(native.time, "monotonic", lambda: clock[0])
    selected = iter([None, 10, None, 11])
    monkeypatch.setattr(native, "browser_pid", lambda *a: next(selected))
    stream = native.ProcessStream.__new__(native.ProcessStream)
    stream.executable = "chrome.exe"
    stream._data = bytearray(b"old")
    stream._lock = threading.Lock()
    stream._ready = threading.Event()
    stream.error = None
    stopped = [False]
    def tick(*a):
        clock[0] += 1.1
        stopped[0] = clock[0] > 14
    stream._stop = SimpleNamespace(is_set=lambda: stopped[0], wait=tick)
    stream._activate = Mock(side_effect=lambda pid: (native.P(pid * 10), native.P(pid * 10 + 1)))
    calls = Mock(return_value=0)
    monkeypatch.setattr(native, "call", calls)
    stream._run()
    assert stream.error is None and stream._ready.is_set()
    assert [args[0] for args, _ in stream._activate.call_args_list] == [10, 11]
    released = [args[0].value for args, _ in calls.call_args_list if args[1] == 2]
    assert released == [101, 100, 111, 110]
    ole.CoUninitialize.assert_called_once()
