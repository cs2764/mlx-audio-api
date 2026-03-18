"""TTS inference engine wrapping mlx_audio."""

import base64
import io
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)


def _release_mlx_cache() -> None:
    """Return cached MLX Metal GPU memory back to the system.

    MLX keeps a pool of allocated Metal buffers for reuse across ops.
    Under concurrent load each worker accumulates its own cache, causing
    apparent memory growth that is not a true leak but still exhausts RAM.
    Calling this after every inference request keeps the footprint bounded.
    """
    try:
        mx.metal.clear_cache()
    except Exception:
        pass


@dataclass
class _BatchParams:
    """Parameters extracted from a TTS request for use in batch inference."""
    text: str
    format: str
    instruct: str | None
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    max_new_tokens: int
    streaming: bool = False
    seed: int | None = None


class TTSEngine:
    def __init__(self, model_path: str):
        """Load MLX quantized model from local path."""
        from mlx_audio.tts.utils import load_model

        logger.info(f"Loading TTS model from: {model_path}")
        self._model = load_model(Path(model_path), trust_remote_code=True)
        self._model_type: str = getattr(self._model, "model_type", "") or ""
        logger.info("TTS model loaded successfully")

    @property
    def supports_batch(self) -> bool:
        """True if the underlying model supports batch_generate (Qwen3 only)."""
        return self._model_type == "qwen3_tts" and hasattr(self._model, "batch_generate")

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

    def generate_batch(self, requests: list) -> list:
        """
        Run a batch of inference requests in a single forward pass (Qwen3 only).

        Each element of `requests` is a callable handler(engine) -> bytes that
        was originally enqueued via RequestQueue.  We extract the params from
        each handler's closure, call model.batch_generate(), then encode and
        return results in the same order.

        Falls back to sequential generate() for non-Qwen3 models or if any
        request uses streaming (streaming is not supported in batch mode).

        Returns a list of (bytes | Exception) aligned with the input list.
        """
        if not self.supports_batch or len(requests) == 1:
            # Single item or non-batch model: run sequentially
            results = []
            for req in requests:
                try:
                    results.append(req(self))
                except Exception as e:
                    results.append(e)
            _release_mlx_cache()
            return results

        # Extract BatchParams from each handler closure
        params_list: list[_BatchParams] = []
        fallback_indices: list[int] = []

        for i, req in enumerate(requests):
            p = getattr(req, "_batch_params", None)
            if p is None or p.streaming:
                # Streaming or non-batch-aware request: run separately after batch
                fallback_indices.append(i)
            else:
                params_list.append(p)

        # If all requests need fallback, just run sequentially
        if not params_list:
            results = []
            for req in requests:
                try:
                    results.append(req(self))
                except Exception as e:
                    results.append(e)
            return results

        # Determine batch indices (those NOT in fallback)
        batch_indices = [i for i in range(len(requests)) if i not in fallback_indices]

        # Build batch inputs
        texts = [p.text for p in params_list]
        instructs = [p.instruct or "A calm female voice" for p in params_list]
        # Use params from first request for shared sampling settings
        p0 = params_list[0]

        batch_results: list[bytes | Exception] = [None] * len(requests)  # type: ignore

        try:
            import mlx.core as mx
            if p0.seed is not None:
                mx.random.seed(p0.seed)

            gen = self._model.batch_generate(
                texts=texts,
                instructs=instructs,
                temperature=p0.temperature,
                max_tokens=p0.max_new_tokens,
                top_k=p0.top_k,
                top_p=p0.top_p,
                repetition_penalty=p0.repetition_penalty,
                stream=False,
                verbose=False,
            )
            # batch_generate yields one BatchGenerationResult per sequence
            # Collect raw audio first, then encode in parallel
            raw_audio: dict[int, tuple[np.ndarray, int]] = {}
            for result in gen:
                seq_idx = result.sequence_idx
                raw_audio[seq_idx] = (np.array(result.audio), result.sample_rate)

            # Parallel audio encoding (mp3 especially benefits from this)
            from concurrent.futures import ThreadPoolExecutor as _EncPool

            def _encode_one(seq_idx: int) -> tuple[int, bytes]:
                audio_data, sr = raw_audio[seq_idx]
                fmt = params_list[seq_idx].format
                return seq_idx, self._encode_audio(audio_data, sr, fmt)

            with _EncPool(max_workers=min(len(raw_audio), 4)) as enc_pool:
                encoded = dict(enc_pool.map(_encode_one, raw_audio.keys()))

            for local_idx, global_idx in enumerate(batch_indices):
                batch_results[global_idx] = encoded.get(
                    local_idx, RuntimeError(f"No audio returned for sequence {local_idx}")
                )
        except Exception as e:
            for global_idx in batch_indices:
                batch_results[global_idx] = e

        # Run fallback requests sequentially
        for global_idx in fallback_indices:
            try:
                batch_results[global_idx] = requests[global_idx](self)
            except Exception as e:
                batch_results[global_idx] = e

        _release_mlx_cache()
        return batch_results

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
        # Ming Omni flow-matching params (not applicable to Qwen3)
        if self._model_type != "qwen3_tts":
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
            _release_mlx_cache()

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
            _release_mlx_cache()

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
