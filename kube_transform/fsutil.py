import os
import fsspec
from typing import List, Any, Optional


def _resolve_path(path: str) -> str:
    """Resolves the path relative to the DATA_DIR environment variable.

    Args:
        path (str): The relative path.

    Returns:
        str: The absolute path resolved against DATA_DIR.

    Raises:
        ValueError: If DATA_DIR environment variable is not set.
    """
    data_dir = os.getenv("DATA_DIR", "")
    if not data_dir:
        raise ValueError("DATA_DIR environment variable not set.")
    return os.path.join(data_dir, path)


def listdir(path: str) -> List[str]:
    """Lists filenames in the given directory, like `os.listdir()`.

    Args:
        path (str): The directory path.

    Returns:
        List[str]: List of filenames in the directory.
    """
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    files = fs.glob(full_path + "/*")
    return [os.path.basename(f) for f in files]


def isfile(path: str) -> bool:
    """Checks if a path is a file.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path is a file, False otherwise.
    """
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    return fs.isfile(full_path)


def isdir(path: str) -> bool:
    """Checks if a path is a directory.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path is a directory, False otherwise.
    """
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    return fs.isdir(full_path)


def exists(path: str) -> bool:
    """Checks if a path exists.

    Args:
        path (str): The path to check.

    Returns:
        bool: True if the path exists, False otherwise.
    """
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    return fs.exists(full_path)


def mkdir(path: str, exist_ok: bool = True) -> None:
    """Creates a directory (no-op for S3, since folders are virtual).

    Args:
        path (str): The directory path to create.
        exist_ok (bool, optional): If False, raise an error if the directory exists. Defaults to True.

    Raises:
        FileExistsError: If the directory exists and exist_ok is False.
    """
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    if not fs.exists(full_path):
        fs.mkdir(full_path)
    elif not exist_ok:
        raise FileExistsError(f"Directory {path} already exists.")


def remove(path: str) -> None:
    """Deletes a file.

    Args:
        path (str): The file path to delete.
    """
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    fs.rm(full_path)


def rmdir(path: str) -> None:
    """Removes a directory and its contents.

    Args:
        path (str): The directory path to remove.
    """
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    fs.rm(full_path, recursive=True)


def stat(path: str) -> Any:
    """Returns file metadata.

    Args:
        path (str): The path to stat.

    Returns:
        Any: File metadata as returned by the filesystem.
    """
    full_path = _resolve_path(path)
    fs = fsspec.open(full_path).fs
    return fs.stat(full_path)


def glob(pattern: str) -> List[str]:
    """Returns all matching files using wildcard patterns.

    Args:
        pattern (str): The glob pattern.

    Returns:
        List[str]: List of matching filenames.
    """
    full_pattern = _resolve_path(pattern)
    fs = fsspec.open(full_pattern).fs
    return [os.path.basename(f) for f in fs.glob(full_pattern)]


def open(path: str, mode: str = "r", **kwargs) -> Any:
    """Opens a file, supporting both local and S3 paths.

    Args:
        path (str): The file path to open.
        mode (str, optional): The mode in which to open the file. Defaults to "r".
        **kwargs: Additional keyword arguments passed to fsspec.open.

    Returns:
        Any: A file-like object.
    """
    full_path = _resolve_path(path)
    return fsspec.open(full_path, mode, **kwargs).open()


def read(path: str, encoding: str = "utf-8") -> str:
    """Reads the entire file as a string.

    Args:
        path (str): The file path to read.
        encoding (str, optional): The encoding to use. Defaults to "utf-8".

    Returns:
        str: The file contents.
    """
    with open(path, "r", encoding=encoding) as f:
        return f.read()


def write(path: str, data: str, encoding: str = "utf-8") -> None:
    """Writes a string to a file.

    Args:
        path (str): The file path to write.
        data (str): The string data to write.
        encoding (str, optional): The encoding to use. Defaults to "utf-8".
    """
    with open(path, "w", encoding=encoding) as f:
        f.write(data)


def append(path: str, data: str, encoding: str = "utf-8") -> None:
    """Appends a string to a file.

    Args:
        path (str): The file path to append to.
        data (str): The string data to append.
        encoding (str, optional): The encoding to use. Defaults to "utf-8".
    """
    with open(path, "a", encoding=encoding) as f:
        f.write(data)


def join(*paths: str) -> str:
    """Joins multiple path components intelligently.

    Args:
        *paths (str): Path components to join.

    Returns:
        str: The joined path.
    """
    return os.path.join(*paths)
