"""All persistent I/O. Paths are relative to a base URI (file:// now, gs:// from Stage 3), so
business logic never knows where data lives."""

import io
import uuid

import fsspec
import pandas as pd


class Storage:
    def __init__(self, base_uri: str) -> None:
        self.fs, root = fsspec.core.url_to_fs(base_uri)
        self.root = root.rstrip("/")

    def _abs(self, path: str) -> str:
        return f"{self.root}/{path.lstrip('/')}"

    def _write_bytes(self, data: bytes, path: str) -> None:
        """Write to a temp object then move, so readers never see a half-written file."""
        target = self._abs(path)
        self.fs.makedirs(target.rsplit("/", 1)[0], exist_ok=True)
        tmp = f"{target}.tmp-{uuid.uuid4().hex}"
        with self.fs.open(tmp, "wb") as f:
            f.write(data)
        self.fs.mv(tmp, target)

    def exists(self, path: str) -> bool:
        return bool(self.fs.exists(self._abs(path)))

    def list(self, prefix: str) -> list[str]:
        """All files under `prefix`, as sorted paths relative to the base URI."""
        base = self._abs(prefix)
        if not self.fs.exists(base):
            return []
        files = self.fs.find(base)
        root = self.fs._strip_protocol(self.root)
        return sorted(f[len(root) :].lstrip("/") for f in files)

    def write_parquet(self, df: pd.DataFrame, path: str) -> None:
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow", compression="snappy")
        self._write_bytes(buf.getvalue(), path)

    def read_parquet(self, path: str) -> pd.DataFrame:
        with self.fs.open(self._abs(path), "rb") as f:
            return pd.read_parquet(f, engine="pyarrow")

    def write_text(self, text: str, path: str) -> None:
        self._write_bytes(text.encode(), path)

    def read_text(self, path: str) -> str:
        with self.fs.open(self._abs(path), "rb") as f:
            data: bytes = f.read()
        return data.decode()
