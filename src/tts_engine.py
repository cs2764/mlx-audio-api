"""TTS inference engine wrapping mlx_audio."""

import base64
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)


class TTSEngine:
    def __init__(self, model_path: str):
        """Load MLX quantized model from local path."""
        from mlx_audio.tts.utils import load_model

        logger.info(f"Loading TTS model from: {model_path}")
        self._model = load_model(Path(model_path), trust_remote_code=True)
        self._model_type: str = getattr(self._model, "model_type", "") or ""
        logger.info("TTS model loaded successfully")

    def generate(
        self,
        text: str,
        *,
        references: list[dict] | None = None,
        ref_audio_path: str | None = None,
        format: str = "wav",
        chunk_length: int = 200,
        temperature: float = 0.0,
        top_p: float = 0.7,
        top_k: int = 30,
        repetition_penalty: float = 1.1,
        max_new_tokens: int = 1024,
        seed: int | None = None,
        normalize: bool = True,
        streaming: bool = False,
        cfg_scale: float | None = None,
        flow_steps: int | None = None,
        sigma: float | None = None,
        instruct: str | None = None,
    ) -> "bytes | Generator[bytes, None, None]":
        """
        Execute TTS inference.

        Non-streaming: returns complete audio bytes.
        Streaming: returns a generator yielding WAV audio chunks.
        """
        # Qwen3 VoiceDesign requires instruct; apply default if not provided
        if self._model_type == "qwen3_tts" and not instruct:
            instruct = "A sexy female voice"

        # Resolve reference audio path
        resolved_ref_path = self._resolve_ref_audio(references, ref_audio_path)
        resolved_ref_text = self._resolve_ref_text(references)

        if streaming:
            return self._generate_streaming(
                text=text,
                ref_audio_path=resolved_ref_path,
                ref_text=resolved_ref_text,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                seed=seed,
                cfg_scale=cfg_scale,
                flow_steps=flow_steps,
                sigma=sigma,
                instruct=instruct,
            )
        else:
            return self._generate_full(
                text=text,
                ref_audio_path=resolved_ref_path,
                ref_text=resolved_ref_text,
                format=format,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                seed=seed,
                cfg_scale=cfg_scale,
                flow_steps=flow_steps,
                sigma=sigma,
                instruct=instruct,
            )

    # ── internal helpers ──────────────────────────────────────────────────────

    def _resolve_ref_audio(
        self,
        references: list[dict] | None,
        ref_audio_path: str | None,
    ) -> str | None:
        """
        Return a filesystem path to the reference audio, or None.

        If `references` is provided, decode the first entry's base64 audio
        into a temp file and return its path.  The caller is responsible for
        cleanup (we use NamedTemporaryFile with delete=False so the path
        survives the context manager).
        """
        if ref_audio_path:
            return ref_audio_path

        if references:
            first = references[0]
            audio_field = first.get("audio", "")
            if audio_field:
                audio_bytes = base64.b64decode(audio_field)
                tmp = tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                )
                try:
                    tmp.write(audio_bytes)
                    tmp.flush()
                    tmp.close()
                    return tmp.name
                except Exception:
                    tmp.close()
                    os.unlink(tmp.name)
                    raise

        return None

    def _resolve_ref_text(self, references: list[dict] | None) -> str | None:
        if references:
            return references[0].get("text") or None
        return None

    def _run_model(
        self,
        text: str,
        ref_audio_path: str | None,
        ref_text: str | None,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        seed: int | None,
        stream: bool = False,
        streaming_interval: float = 2.0,
        cfg_scale: float | None = None,
        flow_steps: int | None = None,
        sigma: float | None = None,
        instruct: str | None = None,
    ):
        """Call model.generate and return the results iterator."""
        from mlx_audio.tts.generate import load_audio

        import mlx.core as mx

        if seed is not None:
            mx.random.seed(seed)

        ref_audio_array = None
        if ref_audio_path:
            normalize_vol = (
                hasattr(self._model, "model_type")
                and self._model.model_type == "spark"
            )
            ref_audio_array = load_audio(
                ref_audio_path,
                sample_rate=self._model.sample_rate,
                volume_normalize=normalize_vol,
            )

        gen_kwargs: dict = dict(
            text=text,
            ref_audio=ref_audio_array,
            ref_text=ref_text,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            verbose=False,
            stream=stream,
            streaming_interval=streaming_interval,
        )
        # Ming Omni flow-matching params
        if cfg_scale is not None:
            gen_kwargs["cfg_scale"] = cfg_scale
        if flow_steps is not None:
            gen_kwargs["ddpm_steps"] = flow_steps
        if sigma is not None:
            gen_kwargs["sigma"] = sigma
        # Qwen3 VoiceDesign: voice style description
        if instruct is not None:
            gen_kwargs["instruct"] = instruct

        return self._model.generate(**gen_kwargs)

    def _generate_full(
        self,
        text: str,
        ref_audio_path: str | None,
        ref_text: str | None,
        format: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        seed: int | None,
        cfg_scale: float | None = None,
        flow_steps: int | None = None,
        sigma: float | None = None,
        instruct: str | None = None,
    ) -> bytes:
        """Run inference and return complete audio bytes in the requested format."""
        try:
            results = self._run_model(
                text=text,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                seed=seed,
                cfg_scale=cfg_scale,
                flow_steps=flow_steps,
                sigma=sigma,
                instruct=instruct,
            )

            # Collect all audio segments
            audio_segments = []
            sample_rate = None
            for result in results:
                audio_segments.append(np.array(result.audio))
                sample_rate = result.sample_rate

            if not audio_segments:
                raise RuntimeError("Model returned no audio")

            audio_data = (
                np.concatenate(audio_segments, axis=0)
                if len(audio_segments) > 1
                else audio_segments[0]
            )

            return self._encode_audio(audio_data, sample_rate, format)

        finally:
            self._cleanup_temp_ref(ref_audio_path)

    def _generate_streaming(
        self,
        text: str,
        ref_audio_path: str | None,
        ref_text: str | None,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        seed: int | None,
        cfg_scale: float | None = None,
        flow_steps: int | None = None,
        sigma: float | None = None,
        instruct: str | None = None,
    ) -> Generator[bytes, None, None]:
        """Run inference in streaming mode, yielding WAV chunks."""
        from mlx_audio.audio_io import write as audio_write

        try:
            results = self._run_model(
                text=text,
                ref_audio_path=ref_audio_path,
                ref_text=ref_text,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                seed=seed,
                stream=True,
                streaming_interval=2.0,
                cfg_scale=cfg_scale,
                flow_steps=flow_steps,
                sigma=sigma,
                instruct=instruct,
            )

            for result in results:
                audio_data = np.array(result.audio)
                buf = io.BytesIO()
                audio_write(buf, audio_data, result.sample_rate, format="wav")
                yield buf.getvalue()

        finally:
            self._cleanup_temp_ref(ref_audio_path)

    def _encode_audio(
        self, audio_data: np.ndarray, sample_rate: int, format: str
    ) -> bytes:
        """Encode numpy audio array to the requested format bytes."""
        from mlx_audio.audio_io import write as audio_write

        if format == "wav":
            buf = io.BytesIO()
            audio_write(buf, audio_data, sample_rate, format="wav")
            return buf.getvalue()

        elif format == "pcm":
            # Strip WAV header — return raw int16 PCM bytes
            if audio_data.dtype in (np.float32, np.float64):
                audio_data = np.clip(audio_data, -1.0, 1.0)
                audio_data = (audio_data * 32767).astype(np.int16)
            elif audio_data.dtype != np.int16:
                audio_data = audio_data.astype(np.int16)
            return audio_data.tobytes()

        elif format == "mp3":
            # Convert WAV → MP3 via pydub
            buf_wav = io.BytesIO()
            audio_write(buf_wav, audio_data, sample_rate, format="wav")
            buf_wav.seek(0)

            from pydub import AudioSegment

            segment = AudioSegment.from_wav(buf_wav)
            buf_mp3 = io.BytesIO()
            segment.export(buf_mp3, format="mp3")
            return buf_mp3.getvalue()

        else:
            raise ValueError(f"Unsupported format: {format}")

    @staticmethod
    def _cleanup_temp_ref(path: str | None) -> None:
        """Remove a temp reference audio file if it was created by us."""
        if path and os.path.exists(path):
            # Only delete files in the system temp directory
            try:
                tmp_dir = tempfile.gettempdir()
                if os.path.commonpath([path, tmp_dir]) == tmp_dir:
                    os.unlink(path)
            except Exception:
                pass
