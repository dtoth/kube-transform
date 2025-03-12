"""This module is intended to handle all file saving/loading operations in Kube Transform.

It dynamically looks up the DATA_DIR from an environment variable.
It can handle that DATA_DIR being either a local directory or an S3 bucket.

All load and save methods should accept paths that are relative to the DATA_DIR.

You may need to extend this module to support additional file formats.
"""

import os
import json
import boto3
import pandas as pd
from generic import coo_dict_serialization as dnl
from urllib.parse import urlparse
import io
import time
import threading
import numpy as np


s3_client = boto3.client("s3")


def get_data_dir():
    """Returns the base data directory, which may be local or an S3 bucket."""
    return os.getenv("DATA_DIR", "/app/data")


def is_s3():
    """Check if the data directory is in S3."""
    return get_data_dir().startswith("s3://")


def parse_s3_path(relative_path):
    """Convert a relative path into an S3 bucket and key."""
    base_path = get_data_dir()
    bucket, base_prefix = urlparse(base_path).netloc, urlparse(base_path).path.lstrip(
        "/"
    )
    key = f"{base_prefix}/{relative_path}".lstrip("/")
    return bucket, key


def _list_all_s3_objects(bucket, prefix):
    all_objects = []
    continuation_token = None

    while True:
        list_kwargs = {"Bucket": bucket, "Prefix": prefix}
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token

        response = s3_client.list_objects_v2(**list_kwargs)

        # Append the retrieved objects to the list
        if "Contents" in response:
            all_objects.extend(response["Contents"])

        # Check if more objects are available
        if response.get("IsTruncated"):
            continuation_token = response["NextContinuationToken"]
        else:
            break

    return all_objects


def get_filepaths_in_directory(directory):
    """Return full paths of all files in the given directory."""
    filenames = get_filenames_in_directory(directory)
    return [join(directory, f) for f in filenames]


def get_filenames_in_directory(directory):
    """List files in a directory for both local and S3."""
    if is_s3():
        bucket, prefix = parse_s3_path(directory)
        if not prefix.endswith("/"):
            prefix += "/"

        response_contents = _list_all_s3_objects(bucket, prefix)
        fns = set([c["Key"].split(prefix)[1].split("/")[0] for c in response_contents])
        fns = [fn for fn in fns if len(fn) > 0]
        return fns
    else:
        full_dir = os.path.join(get_data_dir(), directory)
        if not os.path.exists(full_dir):
            return []
        return [o for o in os.listdir(full_dir) if not o.startswith(".")]


def join(*args):
    """Join paths, supporting both S3 and local."""
    if is_s3():
        return "/".join(arg.strip("/") for arg in args)
    else:
        return os.path.join(*args)


def data_generator(paths=None, directory=None, columns=None):
    """Yield data from a list of file paths or a directory."""
    if directory is not None:
        paths = get_filepaths_in_directory(directory)
    for path in paths:
        yield load_data(path, columns=columns)


def load_data(path, columns=None):
    """Load data from either local storage or S3."""
    if is_s3():
        return _load_s3_data(path, columns=columns)
    else:
        return _load_local_data(path, columns=columns)


def _load_local_data(path, columns=None):
    """Load local files based on format."""
    full_path = os.path.join(get_data_dir(), path)
    if path.endswith(".json"):
        with open(full_path, "r") as f:
            return json.load(f)
    elif path.endswith(".txt"):
        with open(full_path, "r") as f:
            return f.read()
    elif path.endswith(".parquet"):
        return pd.read_parquet(full_path, columns=columns)
    elif path.endswith(".dnl"):
        return dnl.load_dnl(full_path)
    elif path.endswith(".ddnl"):
        return dnl.load_ddnl(full_path)
    else:
        raise NotImplementedError(f"Loading {path} not yet supported.")


def _load_s3_data(path, columns=None):
    """Load data from S3."""
    bucket, key = parse_s3_path(path)
    obj = s3_client.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()

    if path.endswith(".json"):
        return json.loads(body.decode("utf-8"))
    elif path.endswith(".txt"):
        return body.decode("utf-8")
    elif path.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(body), columns=columns)
    elif path.endswith(".dnl"):
        return dnl.load_dnl(io.BytesIO(body))
    elif path.endswith(".ddnl"):
        return dnl.load_ddnl(io.BytesIO(body))
    else:
        raise NotImplementedError(f"Loading {path} not yet supported.")


def save_data(data, path):
    """Save data to either local storage or S3."""
    if is_s3():
        _save_s3_data(data, path)
    else:
        _save_local_data(data, path)


def _save_local_data(data, path):
    """Save data locally based on format."""
    full_path = os.path.join(get_data_dir(), path)
    os.makedirs(os.path.dirname(full_path), exist_ok=True)

    if path.endswith(".json"):
        with open(full_path, "w") as f:
            json.dump(data, f)
    elif path.endswith(".txt"):
        with open(full_path, "w") as f:
            f.write(data)
    elif path.endswith(".parquet"):
        data.to_parquet(full_path)
    elif path.endswith(".md"):
        with open(full_path, "w") as f:
            f.write(data)
    elif path.endswith(".dnl"):
        dnl.save_dnl(data, full_path)
    elif path.endswith(".ddnl"):
        dnl.save_ddnl(data, full_path)
    else:
        raise NotImplementedError(f"Saving {path} not yet supported.")


def _cpu_keep_alive(stop_event):
    """Runs lightweight CPU work while S3 write is happening.
    This prevents Karpenter from terminating the node
    as "underutilized" while waiting for the write to complete.
    """
    # TO OPTIMIZE: A better way to handle this is to remove
    # the need for keep-alive by using a custom node pool
    # with disuption.consolidateAfter=3m instead of the
    # default 30s. This will allow the node to stay alive
    # for 3 minutes after CPU usage drops below the threshold.
    print("CPU keep-alive running...")
    while not stop_event.is_set():  # Run until stop_event is triggered
        np.mean([x * x for x in range(2000000)])
        time.sleep(0.5)


def _save_s3_data(data, path):
    """Save data to S3 while running CPU keep-alive until write completes."""
    bucket, key = parse_s3_path(path)

    # Create an event to signal when to stop the CPU task
    stop_event = threading.Event()

    # Start background CPU keep-alive thread
    cpu_thread = threading.Thread(
        target=_cpu_keep_alive, args=(stop_event,), daemon=True
    )
    cpu_thread.start()

    try:
        # Perform the S3 write
        if path.endswith(".json"):
            s3_client.put_object(Bucket=bucket, Key=key, Body=json.dumps(data))
        elif path.endswith(".txt"):
            s3_client.put_object(Bucket=bucket, Key=key, Body=data.encode("utf-8"))
        elif path.endswith(".parquet"):
            buffer = io.BytesIO()
            data.to_parquet(buffer)
            s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        elif path.endswith(".md"):
            s3_client.put_object(Bucket=bucket, Key=key, Body=data)
        elif path.endswith(".dnl"):
            buffer = io.BytesIO()
            dnl.save_dnl(data, buffer)
            s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        elif path.endswith(".ddnl"):
            buffer = io.BytesIO()
            dnl.save_ddnl(data, buffer)
            s3_client.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
        else:
            raise NotImplementedError(f"Saving {path} not yet supported.")
    finally:
        # Signal the CPU thread to stop and wait for it to exit
        stop_event.set()
        cpu_thread.join()
