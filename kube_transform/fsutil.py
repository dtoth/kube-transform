import os
import fsspec


def _resolve_path(path: str) -> str:
    """Resolves the path relative to the DATA_DIR environment variable."""
    data_dir = os.getenv("DATA_DIR", "")
    if not data_dir:
        raise ValueError("DATA_DIR environment variable not set.")
    return os.path.join(data_dir, path)


def listdir(path: str):
    """Lists filenames in the given directory, like `os.listdir()`."""
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    files = fs.glob(full_path + "/*")
    return [os.path.basename(f) for f in files]


def isfile(path: str) -> bool:
    """Checks if a path is a file."""
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    return fs.isfile(full_path)


def isdir(path: str) -> bool:
    """Checks if a path is a directory."""
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    return fs.isdir(full_path)


def exists(path: str) -> bool:
    """Checks if a path exists."""
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    return fs.exists(full_path)


def mkdir(path: str, exist_ok=True):
    """Creates a directory (no-op for S3, since folders are virtual)."""
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    if not fs.exists(full_path):
        fs.mkdir(full_path)
    elif not exist_ok:
        raise FileExistsError(f"Directory {path} already exists.")


def remove(path: str):
    """Deletes a file."""
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    fs.rm(full_path)


def rmdir(path: str):
    """Removes a directory and its contents."""
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    fs.rm(full_path, recursive=True)


def stat(path: str):
    """Returns file metadata."""
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    return fs.stat(full_path)


def glob(pattern: str):
    """Returns all matching files using wildcard patterns."""
    full_pattern = _resolve_path(pattern)
    fs = fsspec.open(full_pattern).fs
    return [os.path.basename(f) for f in fs.glob(full_pattern)]


def open(path: str, mode="r", **kwargs):
    """Opens a file, supporting both local and S3 paths."""
    full_path = _resolve_path(path)
    return fsspec.open(full_path, mode, **kwargs).open()


def read(path: str, encoding="utf-8") -> str:
    """Reads the entire file as a string."""
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write(path: str, data: str, encoding="utf-8"):
    """Writes a string to a file."""
    with open(path, "w", encoding=encoding) as f:
        f.write(data)


def append(path: str, data: str, encoding="utf-8"):
    """Appends a string to a file."""
    with open(path, "a", encoding=encoding) as f:
        f.write(data)


def join(*paths: str) -> str:
    """Joins multiple path components intelligently."""
    return os.path.join(*paths)
