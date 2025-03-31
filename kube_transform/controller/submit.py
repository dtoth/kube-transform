import uuid
from kubernetes import config

from kube_transform.controller.k8s import create_controller_job


def run_pipeline(pipeline_spec, image_path, data_dir, namespace="default"):
    """Submits a pipeline for execution by creating a KTController Kubernetes Job."""
    pipeline_run_id = f"ktpr{str(uuid.uuid4())[:8]}"

    config.load_kube_config()

    create_controller_job(
        pipeline_run_id=pipeline_run_id,
        pipeline_spec=pipeline_spec,
        image_path=image_path,
        data_dir=data_dir,
        namespace=namespace,
    )
