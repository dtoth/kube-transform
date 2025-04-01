[![PyPI version](https://img.shields.io/pypi/v/kube-transform.svg)](https://pypi.org/project/kube-transform/)

## kube-transform

**kube-transform** is a lightweight open-source framework for writing and deploying distributed, batch-oriented data pipelines on Kubernetes.

It is intentionally thin — only \~1,000 lines of code — and designed to be an easy-to-understand layer between your code and Kubernetes. Unlike heavyweight orchestration platforms like Apache Airflow (which has hundreds of thousands of lines), kube-transform aims to be simple to reason about, debug, and extend.

It focuses on simplicity and flexibility:

- Minimal required configuration
- No persistent control plane components
- Vendor-agnostic: compatible with any Kubernetes cluster, image, and file store that meet a few basic requirements

If you're looking for a quick way to get started, check out [`kube-transform-starter-kit`](https://github.com/dtoth/kube-transform-starter-kit) for reusable setup resources like Dockerfiles, Terraform, and RBAC templates. But using the starter kit is entirely optional.

---

## Requirements

Your setup must meet these basic requirements:

### 1. Deployment Inputs

To run a pipeline, you must provide:

- **`pipeline_spec`**: A Python dictionary that conforms to the [`KTPipeline`](kube_transform/spec.py) schema. You can optionally write your spec using the `KTPipeline` Pydantic model directly, and then call `.model_dump()` to convert it to a dict.
- **`image_path`**: A string path to your Docker image
- **`data_dir`**: A string path to your file store (must be valid from the perspective of pods running in the cluster)

### 2. Docker Image

- Must include Python 3.11+
- Must have `kube-transform` installed (e.g. via pip)
- Must include your code in `/app/kt_functions/`, which should be an importable module containing the functions referenced in your pipeline

### 3. File Store ("DATA\_DIR")

- This is the directory that all pipeline jobs and the controller will read from and write to. It will be passed as the `data_dir` argument to `run_pipeline()`.
- Can be a local folder (e.g. `/mnt/data`) or a cloud object store (e.g. `s3://some-bucket/`)
- Must be readable and writable by all pods in your cluster
- Compatible with anything `fsspec` supports

> The KT controller internally uses `fsspec.open()` to write pipeline metadata to `DATA_DIR`. You must ensure that `DATA_DIR` is transparently accessible to all pods (including the controller). This typically means one of the following:
>
> - You're using a mounted volume that is accessible at `/mnt/data`
> - You've configured access via IRSA (for S3) or Workload Identity (for GCS), so the `kt-pod` service account has permissions to access your object store

> In a single-node cluster, simply mounting a local folder to `/mnt/data` will work. All KT pods will have access to `/mnt/data`.

### 4. Kubernetes Cluster

- Must be able to pull your Docker image (e.g. via ECR, DockerHub, etc.)
- Must be able to access your file store
- Must include a service account named `kt-pod` in the default namespace, with permission to create Kubernetes Jobs
- Your deployment machine must be able to connect to the cluster (e.g. via `kubectl`)

> For a working example setup with autoscaling, IAM roles, and RBAC configuration, see [`kube-transform-starter-kit`](https://github.com/dtoth/kube-transform-starter-kit).

---

## How to Run

Once your inputs are ready:

```python
from kube_transform import run_pipeline

run_pipeline(pipeline_spec, image_path, data_dir)
```

This will:

1. Launch a temporary `kt-controller` Job in your Kubernetes cluster
2. Submit all pipeline jobs in the correct dependency order
3. Shut down automatically when the pipeline completes (or fails)

You can view progress using:

```bash
kubectl get pods
```

Make sure you have the same version of kube-transform running locally (for the `run_pipeline` function) as you have in your image.

---

## Autoscaling & Resource Management

`kube-transform` is compatible with both fixed-size and autoscaling clusters:

- If your cluster supports autoscaling, KT will take advantage of it automatically.
- If your cluster is fixed-size, jobs will remain in `Pending` state until resources are available.

> For help setting up either configuration, see [`kube-transform-starter-kit`](https://github.com/dtoth/kube-transform-starter-kit).

---

Questions? Feature requests? Open an issue on [GitHub](https://github.com/dtoth/kube-transform/issues).

