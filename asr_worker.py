import gc
import inspect
import logging
import sys
import traceback
from typing import Any

import numpy as np

log = logging.getLogger("LiveTranslate.ASRWorker")


def _setup_logging():
    if logging.getLogger().handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    logging.basicConfig(level=logging.INFO, handlers=[handler])
    logging.getLogger("LiveTranslate").setLevel(logging.DEBUG)


def _error_response(msg_id: str | None, exc: BaseException, recoverable: bool) -> dict:
    return {
        "id": msg_id,
        "ok": False,
        "type": "error",
        "error": {
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "recoverable": recoverable,
        },
    }


def _ok_response(msg_id: str | None, response_type: str, payload: Any = None) -> dict:
    return {
        "id": msg_id,
        "ok": True,
        "type": response_type,
        "payload": payload,
    }


def _parse_device(device: str) -> tuple[str, int]:
    device = str(device or "cpu").split(" (", 1)[0].strip()
    if device.startswith("cuda:"):
        index = int(device.split(":", 1)[1])
        return "cuda", index
    return device, 0


def _load_engine(config: dict):
    from model_manager import (
        ASR_ENGINE_PROFILES,
        MODELS_DIR,
        apply_cache_env,
        get_local_model_path,
    )

    apply_cache_env()

    engine_type = config["engine_type"]
    device = config.get("device", "cpu")
    hub = config.get("hub", "ms")
    language = config.get("language", "auto")
    pad_seconds = config.get("pad_seconds")

    profile = ASR_ENGINE_PROFILES.get(engine_type)
    if profile and profile["language_locked"]:
        language = profile["language_locked"]
        # Remember the lock so a runtime set_language command cannot undo it.
        config["language"] = language
        config["language_locked"] = True

    parsed_device, device_index = _parse_device(device)
    worker_device = parsed_device if parsed_device == "cpu" else f"cuda:{device_index}"
    compute_type = config.get("compute_type", "float16")
    if parsed_device == "cpu" and compute_type == "float16":
        compute_type = "int8"

    match engine_type:
        case "funasr":
            from asr_funasr import FunASREngine

            engine = FunASREngine(
                model_key=config.get("funasr_model"),
                device=device,
                hub=hub,
                pad_seconds=pad_seconds,
            )
        case "anime-whisper":
            from asr_anime_whisper import AnimeWhisperEngine

            engine = AnimeWhisperEngine(device=worker_device, hub=hub)
        case "qwen3-asr":
            from asr_qwen3 import Qwen3ASREngine

            engine = Qwen3ASREngine(device=worker_device, hub=hub)
        case "buzzasr-korean":
            from asr_engine import ASREngine

            ct2_path = get_local_model_path(engine_type)
            if not ct2_path:
                raise RuntimeError(
                    "BuzzASR Korean CTranslate2 model is not cached; "
                    "run the download step first"
                )
            engine = ASREngine(
                model_size=ct2_path,
                device=parsed_device,
                device_index=device_index,
                compute_type=compute_type,
                language=language,
                pad_seconds=pad_seconds,
            )
        case _:
            from asr_engine import ASREngine

            download_root = config.get("download_root")
            if not download_root:
                download_root = str((MODELS_DIR / "huggingface" / "hub").resolve())
            engine = ASREngine(
                model_size=config["model_size"],
                device=parsed_device,
                device_index=device_index,
                compute_type=compute_type,
                language=language,
                download_root=download_root,
                pad_seconds=pad_seconds,
            )

    if hasattr(engine, "set_language"):
        engine.set_language(language)
    return engine


def _transcribe(engine, payload: dict):
    audio = payload.get("audio")
    if not isinstance(audio, np.ndarray):
        raise TypeError("transcribe payload audio must be a numpy.ndarray")

    kwargs = {}
    signature = inspect.signature(engine.transcribe)
    if "word_timestamps" in signature.parameters:
        kwargs["word_timestamps"] = bool(payload.get("word_timestamps", False))
    return engine.transcribe(audio, **kwargs)


def _cleanup_engine(engine):
    if engine is not None and hasattr(engine, "unload"):
        try:
            engine.unload()
        except Exception:
            log.warning("ASR engine unload failed", exc_info=True)
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def worker_main(conn, config: dict):
    _setup_logging()
    engine = None
    try:
        log.info(
            "ASR worker loading: "
            f"{config.get('engine_type')} on {config.get('device')} "
            f"(pid config={config.get('display_name', '')})"
        )
        engine = _load_engine(config)
        conn.send(
            _ok_response(
                None,
                "ready",
                {
                    "engine_type": config.get("engine_type"),
                    "display_name": config.get("display_name"),
                    "device": config.get("device"),
                },
            )
        )
    except BaseException as exc:
        log.error(f"ASR worker load failed: {exc}", exc_info=True)
        try:
            conn.send(_error_response(None, exc, recoverable=False))
        finally:
            _cleanup_engine(engine)
            conn.close()
        return

    try:
        while True:
            try:
                msg = conn.recv()
            except EOFError:
                break

            msg_id = msg.get("id")
            msg_type = msg.get("type")
            payload = msg.get("payload") or {}

            try:
                if msg_type == "shutdown":
                    conn.send(_ok_response(msg_id, "shutdown"))
                    break
                if msg_type == "transcribe":
                    result = _transcribe(engine, payload)
                    conn.send(_ok_response(msg_id, "result", result))
                    continue
                if msg_type == "set_language":
                    if not config.get("language_locked") and hasattr(
                        engine, "set_language"
                    ):
                        engine.set_language(payload.get("language", "auto"))
                    conn.send(_ok_response(msg_id, "ack"))
                    continue
                if msg_type == "set_input_padding":
                    if hasattr(engine, "set_input_padding"):
                        engine.set_input_padding(payload.get("pad_seconds"))
                    conn.send(_ok_response(msg_id, "ack"))
                    continue
                raise ValueError(f"Unknown ASR worker command: {msg_type}")
            except Exception as exc:
                log.error(f"ASR worker command failed: {msg_type}: {exc}", exc_info=True)
                conn.send(_error_response(msg_id, exc, recoverable=True))
    finally:
        _cleanup_engine(engine)
        try:
            conn.close()
        except Exception:
            pass
        log.info("ASR worker stopped")
