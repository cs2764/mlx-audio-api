import re
import shutil
from pathlib import Path


class ConflictError(Exception):
    pass


class NotFoundError(Exception):
    pass


_ID_PATTERN = re.compile(r'^[a-zA-Z0-9\-_ ]{1,255}$')


class ReferenceManager:
    def __init__(self, storage_dir: str = "./references"):
        self._storage = Path(storage_dir)
        self._storage.mkdir(parents=True, exist_ok=True)

    def validate_id(self, ref_id: str) -> bool:
        return bool(_ID_PATTERN.match(ref_id))

    async def add(self, ref_id: str, audio: bytes, text: str) -> None:
        ref_dir = self._storage / ref_id
        if ref_dir.exists():
            raise ConflictError(f"Reference '{ref_id}' already exists")
        ref_dir.mkdir(parents=True)
        (ref_dir / "audio.wav").write_bytes(audio)
        (ref_dir / "text.txt").write_text(text, encoding="utf-8")

    async def list_all(self) -> list[str]:
        return [d.name for d in self._storage.iterdir() if d.is_dir()]

    async def delete(self, ref_id: str) -> None:
        ref_dir = self._storage / ref_id
        if not ref_dir.exists():
            raise NotFoundError(f"Reference '{ref_id}' not found")
        shutil.rmtree(ref_dir)

    async def update(self, old_id: str, new_id: str) -> None:
        old_dir = self._storage / old_id
        if not old_dir.exists():
            raise NotFoundError(f"Reference '{old_id}' not found")
        new_dir = self._storage / new_id
        if new_dir.exists():
            raise ConflictError(f"Reference '{new_id}' already exists")
        old_dir.rename(new_dir)

    def get_path(self, ref_id: str) -> Path | None:
        ref_dir = self._storage / ref_id
        audio_path = ref_dir / "audio.wav"
        return audio_path if audio_path.exists() else None
