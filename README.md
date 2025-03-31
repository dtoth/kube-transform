> ## Notice
> This project is currently on version 0.1.0-rc. Prior to version 1.0.0, documentation may be limited.

## kube-transform

kube-transform is a lightweight open-source framework for writing and deploying distributed batch-oriented data transformations on Kubernetes.

kube-transform requires very little configuration. It's recommended that you check out kube-transform-starter-kit for examples (and reusable code) to help create and configure a Kubernetes cluster to use with kube-transform. But that is entirely optional, as this package will work with any image, file store, and Kuberenetes cluster that meet the following criteria:

1. Docker Image
    - Has Python 3.7+ installed
    - Has kube-transform installed (e.g. via pip)
    - Has your code (anything referenced in your DAGs) in /app/kt_functions/, which should be an importable module.

2. File Store
    - Can be a disk-based file path (e.g. /app/data/), large object store (e.g. s3://some-bucket/), etc. Basically, anything that `fsspec` can interact with. The path will be used by pods within your k8s cluster.
        - All KT pods will have access to /mnt/data on their Kubernetes node, so for a single-node setup, if you mount a local folder to /mnt/data when starting your k8s cluster, you can use /mnt/data as your file store.

3. Kubernetes Cluster
    - Can access your docker image (e.g. ECR access)
    - Can access your file store (e.g. S3 access)
    - Has a service account called "kt-controller" which is allowed to create Jobs in the default namespace. This will be used by the kt-controller pods. For instructions on how to configure this, see `kube-transform-starter-kit`. Or simply apply the RBAC file provided there.
    - Ensure your deployment machine can access your k8s cluster (i.e. you can run kubectl)

4. Deployment Setup
    - Your `pipeline_spec` should be a python dictionary that adheres to the `KTPipeline` schema (defined in `kube_transform.spec`).
    - `image_path` should point to your Docker image.
    - `data_dir` should point to your file store (from the perspective of pods in your cluster).


Then you can run:
```
from kube_transform import run_pipeline
...
run_pipeline(pipeline_spec, image_path, data_dir)
```

This will create a kt-controller Job in your cluster, which will manage the execution of your pipeline, and then shut down when complete. The kt-controller will submit jobs in an appropriate order, as defined in your pipeline spec.

Note that autoscaling is optional. The kt-controller will simply submit jobs to your cluster. If your cluster is configured to autoscale, it will do so.  If not, pods will be marked as `Pending` until they can be run. For help configuring remote-autoscaling and/or local-fixed Kubernetes clusters, see `kube-transform-starter-kit`.
