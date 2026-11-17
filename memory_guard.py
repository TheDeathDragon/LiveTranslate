import threading
from contextlib import contextmanager

_lock = threading.Lock()
_suspensions = 0


@contextmanager
def suspend_memory_check():
    global _suspensions
    with _lock:
        _suspensions += 1
    try:
        yield
    finally:
        with _lock:
            _suspensions -= 1


def memory_check_suspended() -> bool:
    with _lock:
        return _suspensions > 0
