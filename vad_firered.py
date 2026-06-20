import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("LiveTranslate.FireRedVAD")

_SAMPLE_RATE = 16000
_FRAME_LENGTH_SAMPLE = 400  # 25ms at 16kHz
_FRAME_SHIFT_SAMPLE = 160  # 10ms at 16kHz
_VALID_AGGREGATIONS = {"max", "latest", "mean"}


class FireRedVadAdapter:
    """Rolling-frame adapter for FireRedVAD streaming confidence."""

    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.5,
        smooth_window_size: int = 5,
        use_gpu: bool = False,
        frame_aggregation: str = "max",
    ):
        self.model_dir = str(model_dir)
        self.threshold = float(threshold)
        self.smooth_window_size = max(1, int(smooth_window_size))
        self.use_gpu = bool(use_gpu)
        self.frame_aggregation = str(frame_aggregation or "max").lower()
        if self.frame_aggregation not in _VALID_AGGREGATIONS:
            log.warning(
                "Unknown FireRedVAD frame aggregation '%s', using max",
                self.frame_aggregation,
            )
            self.frame_aggregation = "max"

        model_path = Path(self.model_dir)
        if not model_path.is_dir():
            raise RuntimeError(
                f"FireRedVAD model directory does not exist: {model_path}. "
                "Download FireRedTeam/FireRedVAD and select its Stream-VAD directory."
            )
        missing = [
            name
            for name in ("cmvn.ark", "model.pth.tar")
            if not (model_path / name).is_file()
        ]
        if missing:
            raise RuntimeError(
                f"Invalid FireRedVAD Stream-VAD model directory: {model_path}. "
                f"Missing: {', '.join(missing)}."
            )

        try:
            from fireredvad import FireRedStreamVad, FireRedStreamVadConfig
        except Exception as exc:
            raise RuntimeError(
                "FireRedVAD Python package is not available. "
                "Install project dependencies or run: uv pip install fireredvad>=0.0.2"
            ) from exc

        try:
            config = FireRedStreamVadConfig(
                use_gpu=self.use_gpu,
                smooth_window_size=self.smooth_window_size,
                speech_threshold=self.threshold,
            )
            self._stream_vad = FireRedStreamVad.from_pretrained(
                str(model_path), config
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FireRedVAD Stream-VAD model from {model_path}: {exc}"
            ) from exc

        self._raw_buffer = np.empty(0, dtype=np.float32)
        self._last_confidence = 0.0
        log.info(
            "FireRedVAD loaded: model=%s, device=%s, smooth_window=%s, aggregation=%s",
            model_path,
            "gpu" if self.use_gpu else "cpu",
            self.smooth_window_size,
            self.frame_aggregation,
        )

    def confidence_for_chunk(self, audio_chunk: np.ndarray) -> float:
        if self._stream_vad is None:
            return self._last_confidence

        chunk = np.asarray(audio_chunk, dtype=np.float32).reshape(-1)
        if chunk.size == 0:
            return self._last_confidence

        self._raw_buffer = np.concatenate((self._raw_buffer, chunk))
        confidences: list[float] = []

        while self._raw_buffer.size >= _FRAME_LENGTH_SAMPLE:
            frame = self._raw_buffer[:_FRAME_LENGTH_SAMPLE]
            self._raw_buffer = self._raw_buffer[_FRAME_SHIFT_SAMPLE:]
            firered_frame = (
                np.clip(frame, -1.0, 1.0).astype(np.float32, copy=False) * 32768.0
            )
            try:
                result = self._stream_vad.detect_frame(firered_frame)
            except Exception:
                log.exception("FireRedVAD detect_frame failed")
                raise
            confidence = getattr(result, "smoothed_prob", None)
            if confidence is None:
                confidence = getattr(result, "raw_prob", 0.0)
            confidences.append(float(confidence))

        if not confidences:
            return self._last_confidence

        if self.frame_aggregation == "latest":
            confidence = confidences[-1]
        elif self.frame_aggregation == "mean":
            confidence = float(sum(confidences) / len(confidences))
        else:
            confidence = max(confidences)

        self._last_confidence = max(0.0, min(1.0, confidence))
        return self._last_confidence

    def reset(self):
        self._raw_buffer = np.empty(0, dtype=np.float32)
        self._last_confidence = 0.0
        if self._stream_vad is not None:
            try:
                self._stream_vad.reset()
            except Exception:
                log.warning("FireRedVAD reset failed", exc_info=True)

    def unload(self):
        self._stream_vad = None
        self._raw_buffer = np.empty(0, dtype=np.float32)
        self._last_confidence = 0.0
        if self.use_gpu:
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass
