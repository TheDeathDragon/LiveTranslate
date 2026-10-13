"""Windows application loopback, using only ctypes and existing psutil/numpy."""
import ctypes as c
from ctypes import wintypes as w
import logging
import threading
import time
import uuid

import numpy as np
import psutil

log = logging.getLogger("LiveTranslate.ProcessAudio")
P = c.c_void_p
HR = c.c_long
_failed_activations = []


class GUID(c.Structure):
    _fields_ = [("bytes", c.c_ubyte * 16)]

    def __init__(self, value):
        super().__init__()
        self.bytes[:] = uuid.UUID(value).bytes_le


class Activation(c.Structure):
    _fields_ = [("kind", w.DWORD), ("pid", w.DWORD), ("mode", w.DWORD)]


class Blob(c.Structure):
    _fields_ = [("size", w.ULONG), ("data", P)]


class Variant(c.Structure):
    _fields_ = [("vt", w.WORD), ("reserved", w.WORD * 3), ("blob", Blob)]


class WaveFormat(c.Structure):
    _pack_ = 1
    _fields_ = [("tag", w.WORD), ("channels", w.WORD), ("rate", w.DWORD),
                ("bytes_per_sec", w.DWORD), ("block", w.WORD),
                ("bits", w.WORD), ("extra", w.WORD)]


def check(hr):
    if hr < 0:
        raise OSError(f"Application audio HRESULT 0x{hr & 0xffffffff:08X}")


def call(obj, index, result, types, *args):
    table = c.cast(obj, c.POINTER(c.POINTER(P))).contents
    return c.WINFUNCTYPE(result, P, *types)(table[index])(obj, *args)


def active_audio_pids(active_only=True):
    ole = c.OleDLL("ole32")
    check(ole.CoInitializeEx(None, 0))
    enumerator = devices = None
    output = set()
    try:
        enumerator = P()
        clsid = GUID("bcde0395-e52f-467c-8e3d-c4579291692e")
        iid = GUID("a95664d2-9614-4f35-a746-de8db63617e6")
        check(ole.CoCreateInstance(c.byref(clsid), None, 23, c.byref(iid), c.byref(enumerator)))
        devices = P()
        check(call(enumerator, 3, HR, [c.c_int, w.DWORD, c.POINTER(P)], 0, 1, c.byref(devices)))
        count = w.UINT()
        check(call(devices, 3, HR, [c.POINTER(w.UINT)], c.byref(count)))
        for index in range(count.value):
            device, manager, sessions = P(), P(), P()
            try:
                check(call(devices, 4, HR, [w.UINT, c.POINTER(P)], index, c.byref(device)))
                manager_iid = GUID("77aa99a0-1bd6-484f-8bc7-2c654c9a9b6f")
                check(call(device, 3, HR, [c.POINTER(GUID), w.DWORD, P, c.POINTER(P)],
                           c.byref(manager_iid), 23, None, c.byref(manager)))
                check(call(manager, 5, HR, [c.POINTER(P)], c.byref(sessions)))
                n = c.c_int()
                check(call(sessions, 3, HR, [c.POINTER(c.c_int)], c.byref(n)))
                for i in range(n.value):
                    control, control2 = P(), P()
                    try:
                        check(call(sessions, 4, HR, [c.c_int, c.POINTER(P)], i, c.byref(control)))
                        iid2 = GUID("bfb7ff88-7239-4fc9-8fa2-07c950be9c6d")
                        check(call(control, 0, HR, [c.POINTER(GUID), c.POINTER(P)], c.byref(iid2), c.byref(control2)))
                        pid, state = w.DWORD(), c.c_int()
                        check(call(control2, 14, HR, [c.POINTER(w.DWORD)], c.byref(pid)))
                        check(call(control2, 3, HR, [c.POINTER(c.c_int)], c.byref(state)))
                        if state.value != 2 and (not active_only or state.value == 1) and pid.value:
                            output.add(pid.value)
                    finally:
                        for ptr in (control2, control):
                            if ptr:
                                call(ptr, 2, w.ULONG, [])
            finally:
                for ptr in (sessions, manager, device):
                    if ptr:
                        call(ptr, 2, w.ULONG, [])
        return output
    finally:
        for ptr in (devices, enumerator):
            if ptr:
                call(ptr, 2, w.ULONG, [])
        ole.CoUninitialize()


def browser_pid(executable):
    """Prefer the actual audio-session process, including Sogou audio children."""
    matches = {}
    for proc in psutil.process_iter(["pid", "name", "ppid"]):
        if (proc.info["name"] or "").lower() == executable.lower():
            matches[proc.pid] = proc.info
    try:
        sounding = active_audio_pids().intersection(matches)
    except OSError:
        log.warning("Could not enumerate audio sessions; waiting on browser root", exc_info=True)
        sounding = set()
    if len(sounding) == 1:
        return next(iter(sounding))
    if len(sounding) > 1:
        raise RuntimeError(f"Multiple {executable} audio processes; play audio in one browser instance")
    try:
        idle = active_audio_pids(active_only=False).intersection(matches)
    except OSError:
        idle = set()
    if len(idle) == 1:
        return next(iter(idle))
    roots = [p for p in matches.values() if p["ppid"] not in matches]
    if len(roots) > 1:
        # Sogou keeps detached helper roots; its browser tree has matching children.
        trees = [p for p in roots if any(child["ppid"] == p["pid"]
                                        for child in matches.values())]
        if len(trees) == 1:
            roots = trees
    # Multiple independent roots cannot safely be represented by one tree.
    if len(roots) > 1:
        raise RuntimeError(f"Multiple independent {executable} processes; use one browser instance")
    return roots[0]["pid"] if roots else None


class ProcessStream:
    """PyAudio-shaped float32 stereo stream. Missing browser produces silence."""
    rate = 48000
    channels = 2

    def __init__(self, executable):
        self.executable = executable
        self._data = bytearray()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        if not self._ready.wait(12):
            self.close()
            raise TimeoutError("Application audio initialization timed out")
        if self.error:
            self.close()
            raise self.error

    def get_read_available(self):
        with self._lock:
            return len(self._data) // 8

    def read(self, frames, exception_on_overflow=False):
        with self._lock:
            data = bytes(self._data[:frames * 8])
            del self._data[:frames * 8]
            return data

    def stop_stream(self):
        self._stop.set()

    def close(self):
        self._stop.set()
        self._thread.join(12)
        if self._thread.is_alive():
            raise TimeoutError("Application audio worker failed to stop")

    def _activate(self, pid):
        iid = GUID("1cb9ad4c-dbfa-4c32-b178-c2f568a703b2")
        supported = {uuid.UUID(x).bytes_le for x in (
            "00000000-0000-0000-c000-000000000046",
            "41d949ab-9862-444a-80f6-c261334da5eb",
            "94ea2b94-e9cc-49e0-c0ff-ee64ca8f5b90")}
        done = threading.Event()
        result, audio = HR(), P()
        refs = [1]

        def query(this, requested, output):
            if c.string_at(requested, 16) in supported:
                output[0] = this
                refs[0] += 1
                return 0
            output[0] = None
            return -2147467262

        def addref(this):
            refs[0] += 1
            return refs[0]

        def release(this):
            refs[0] -= 1
            return refs[0]

        def completed(this, operation):
            try:
                hr = call(operation, 3, HR, [c.POINTER(HR), c.POINTER(P)],
                          c.byref(result), c.byref(audio))
                if hr < 0:
                    result.value = hr
            except Exception:
                result.value = -2147467259
                log.exception("Could not retrieve application audio activation result")
            finally:
                done.set()
            return 0

        callbacks = [c.WINFUNCTYPE(HR, P, P, c.POINTER(P))(query),
                     c.WINFUNCTYPE(w.ULONG, P)(addref),
                     c.WINFUNCTYPE(w.ULONG, P)(release),
                     c.WINFUNCTYPE(HR, P, P)(completed)]
        table = (P * 4)(*[c.cast(cb, P).value for cb in callbacks])
        handler = c.pointer(c.cast(table, P))
        params = Activation(1, pid, 0)
        variant = Variant(65, (w.WORD * 3)(), Blob(c.sizeof(params), c.addressof(params)))
        operation = P()
        activate = c.WinDLL("Mmdevapi").ActivateAudioInterfaceAsync
        activate.argtypes = [w.LPCWSTR, c.POINTER(GUID), c.POINTER(Variant), P, c.POINTER(P)]
        activate.restype = HR
        check(activate("VAD\\Process_Loopback", c.byref(iid), c.byref(variant), handler,
                       c.byref(operation)))
        # Keep callback storage alive through the asynchronous COM reference.
        self._activation = (callbacks, table, handler, params, variant, operation)
        if not done.wait(10):
            # shortcut: retain timed-out callbacks until exit; revisit if activation becomes retryable.
            _failed_activations.append(self._activation)
            raise TimeoutError("Application loopback activation timed out")
        call(operation, 2, w.ULONG, [])
        self._activation = None
        capture = P()
        try:
            check(result.value)
            if not audio:
                raise OSError("Application audio activation returned no interface")
            fmt = WaveFormat(3, 2, 48000, 384000, 8, 32, 0)
            check(call(audio, 3, HR, [c.c_int, w.DWORD, c.c_longlong, c.c_longlong, P, P],
                       0, 0x80020000, 200000, 0, c.byref(fmt), None))
            capture_iid = GUID("c8adbd64-e71e-48a0-a4de-185c395cd317")
            check(call(audio, 14, HR, [c.POINTER(GUID), c.POINTER(P)],
                       c.byref(capture_iid), c.byref(capture)))
            check(call(audio, 10, HR, []))
            return audio, capture
        except Exception:
            for ptr in (capture, audio):
                if ptr:
                    call(ptr, 2, w.ULONG, [])
            raise

    def _append(self, data):
        with self._lock:
            self._data.extend(data)
            # Bound capture backlog to two seconds after a paused/stalled consumer.
            if len(self._data) > self.rate * 8 * 2:
                del self._data[:-self.rate * 8 * 2]

    def _run(self):
        ole = c.OleDLL("ole32")
        audio = capture = None
        pid = None
        initialized = False
        try:
            check(ole.CoInitializeEx(None, 0))
            initialized = True
            last_check = 0
            silence_time = time.monotonic()
            while not self._stop.is_set():
                now = time.monotonic()
                if now - last_check >= 1:
                    last_check = now
                    current = browser_pid(self.executable)
                    if current != pid:
                        if audio:
                            call(audio, 11, HR, [])
                            call(capture, 2, w.ULONG, [])
                            call(audio, 2, w.ULONG, [])
                            audio = capture = None
                        with self._lock:
                            self._data.clear()
                        if current:
                            audio, capture = self._activate(current)
                            log.info("Capturing application tree %s PID=%s", self.executable, current)
                        else:
                            log.info("Waiting for %s; system audio remains excluded", self.executable)
                        pid = current
                    self._ready.set()
                if capture:
                    available = w.UINT()
                    check(call(capture, 5, HR, [c.POINTER(w.UINT)], c.byref(available)))
                    while available.value and not self._stop.is_set():
                        data, frames, flags = P(), w.UINT(), w.DWORD()
                        check(call(capture, 3, HR, [c.POINTER(P), c.POINTER(w.UINT),
                                                  c.POINTER(w.DWORD), P, P],
                                   c.byref(data), c.byref(frames), c.byref(flags), None, None))
                        try:
                            packet = (bytes(frames.value * 8) if flags.value & 2 else
                                      c.string_at(data, frames.value * 8))
                            self._append(packet)
                        finally:
                            check(call(capture, 4, HR, [w.UINT], frames.value))
                        check(call(capture, 5, HR, [c.POINTER(w.UINT)], c.byref(available)))
                    silence_time = now
                elif now - silence_time >= .032:
                    self._append(np.zeros((1536, 2), dtype=np.float32).tobytes())
                    silence_time = now
                self._stop.wait(.005)
        except Exception as exc:
            self.error = exc
            log.exception("Application audio capture failed; no system-audio fallback")
            self._ready.set()
        finally:
            if audio:
                call(audio, 11, HR, [])
                call(capture, 2, w.ULONG, [])
                call(audio, 2, w.ULONG, [])
            if initialized:
                ole.CoUninitialize()
