"""
This module provides a fast and space-efficient way to serialize co-occurrence dictionaries.

A co-occurrence dictionary is a nested dictionary that maps IDs (can be int, 16-char str, or 2-3 char str)
to dictionaries of IDs to integer counts. These can be serialized to .ddnl files (dict-of-dicts numpy lz4).

A count dictionary is a dictionary of IDs to integer counts. These can be serialized to .dnl files (dict numpy lz4).

Serializing to .ddnl is significantly faster (~10x) and more space-efficient than using JSON or pickle for the
data that we serialize in the Spotify MPC project.
"""

import numpy as np
import lz4.frame


def save_ddnl(data, filename):
    """Saves a dict of dicts to a compressed file."""
    key_dtype, subkey_dtype, keys, subkeys, values, offsets = _dict_of_dicts_to_numpy(
        data
    )
    _save_numpy_lz4_ddnl(
        key_dtype, subkey_dtype, keys, subkeys, values, offsets, filename
    )


def load_ddnl(filename):
    """Loads a dict of dicts from a compressed file."""
    with lz4.frame.open(filename) as f:
        key_dtype = np.load(f)[0].decode("utf-8")  # Read key dtype
        subkey_dtype = np.load(f)[0].decode("utf-8")  # Read subkey dtype
        keys = np.load(f).astype(key_dtype)
        subkeys = np.load(f).astype(subkey_dtype)
        values = np.load(f)
        offsets = np.load(f)
    return _numpy_to_dict_of_dicts(keys, subkeys, values, offsets)


def save_dnl(data, filename):
    """Saves a dict to a compressed file."""
    key_dtype, keys, values = _dict_to_numpy(data)
    _save_numpy_lz4_dnl(key_dtype, keys, values, filename)


def load_dnl(filename):
    """Loads a dict from a compressed file."""
    with lz4.frame.open(filename) as f:
        key_dtype = np.load(f)[0].decode("utf-8")  # Read and decode dtype
        keys = np.load(f).astype(key_dtype)  # Convert back to detected dtype
        values = np.load(f)
    return _numpy_to_dict(keys, values)


### Helper Functions ###


def _detect_key_dtype(keys):
    """Detects the correct dtype for keys (int64 or fixed-length string)."""
    try:
        if all(int(k) == k for k in keys):
            return "int64"
    except ValueError:
        pass

    if all(
        (isinstance(k, str) or isinstance(k, np.bytes_)) and len(k) == 16 for k in keys
    ):
        return "S16"  # Store as bytes for efficiency

    if all(
        (isinstance(k, str) or isinstance(k, np.str_)) and len(k) in [2, 3]
        for k in keys
    ):
        return "<U32"  # We need unicode strings to capture emojis etc.
    else:
        raise ValueError(
            "Mixed key types detected (integers and variable-length strings)."
        )


def _dict_to_numpy(data):
    """Converts a dict into numpy arrays (keys, subkeys, values, offsets).
    The top-level dict must have integer or 16-char keys and integer values.
    """
    # Python 3.7+ guarantees insertion order, so we can rely on this order
    keys = list(data.keys())
    values = list(data.values())

    key_dtype = _detect_key_dtype(keys)
    keys = np.array(keys, dtype=key_dtype)
    values = np.array(values, dtype=np.int64)

    return key_dtype, keys, values


def _dict_of_dicts_to_numpy(data):
    """Converts a dict of dicts into numpy arrays (keys, subkeys, values, offsets).
    The top-level dict must have integer or 16-char keys and dict values.
    The sub-dicts must have integer or 16-char keys and values.
    """
    keys = []
    subkeys = []
    values = []
    offsets = [0]  # Track where each new key's subkeys start

    for key, subdict in data.items():
        keys.append(key)
        subkeys.extend(subdict.keys())
        values.extend(subdict.values())
        offsets.append(len(subkeys))  # Mark where this key's subkeys end

    key_dtype = _detect_key_dtype(keys)
    subkey_dtype = _detect_key_dtype(subkeys)

    keys = np.array(keys, dtype=key_dtype)
    subkeys = np.array(subkeys, dtype=subkey_dtype)
    values = np.array(values, dtype=np.int64)  # Adjust dtype based on your values
    offsets = np.array(
        offsets[:-1], dtype=np.int64
    )  # Remove last offset for correct size

    return key_dtype, subkey_dtype, keys, subkeys, values, offsets


def _save_numpy_lz4_ddnl(
    key_dtype, subkey_dtype, keys, subkeys, values, offsets, filename
):
    """Saves numpy arrays with key dtype metadata using LZ4 compression."""
    with lz4.frame.open(filename, "wb", compression_level=0) as f:
        np.save(f, np.array([key_dtype], dtype="S16"))  # Store key dtype
        np.save(f, np.array([subkey_dtype], dtype="S16"))  # Store subkey dtype
        np.save(f, keys)
        np.save(f, subkeys)
        np.save(f, values)
        np.save(f, offsets)


def _save_numpy_lz4_dnl(key_dtype, keys, values, filename):
    """Saves numpy arrays with key dtype metadata using LZ4 compression."""
    with lz4.frame.open(filename, "wb", compression_level=0) as f:
        np.save(f, np.array([key_dtype], dtype="S16"))  # Store dtype as a byte string
        np.save(f, keys)
        np.save(f, values)


def _numpy_to_dict_of_dicts(keys, subkeys, values, offsets):
    """Converts numpy arrays back into a dict of dicts."""
    data = {}
    for i, key in enumerate(keys):
        start = offsets[i]
        end = offsets[i + 1] if i + 1 < len(offsets) else len(subkeys)
        data[key] = {subkeys[j]: values[j] for j in range(start, end)}
    return data


def _numpy_to_dict(keys, values):
    """Converts numpy arrays back into a dict of dicts."""
    return dict(zip(keys, values))
