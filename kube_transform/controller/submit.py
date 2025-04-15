import uuid
from kubernetes import config
from typing import Dict, Any

from kube_transform.controller.k8s import create_controller_job


def run_pipeline(
    pipeline_spec: Dict[str, Any],
    image_path: str,
    data_dir: str,
    namespace: str = "default",
) -> None:
    """
    Submits a pipeline for execution by creating a KTController Kubernetes Job.

    Args:
        pipeline_spec (Dict[str, Any]): The pipeline specification.
        image_path (str): The path to the container image.
        data_dir (str): The directory for data storage.
        namespace (str, optional): The Kubernetes namespace to create the job in. Defaults to "default".
    """
    pipeline_run_id = f"ktpr{str(uuid.uuid4())[:8]}"

    config.load_kube_config()

    create_controller_job(
        pipeline_run_id=pipeline_run_id,
        pipeline_spec=pipeline_spec,
        image_path=image_path,
        data_dir=data_dir,
        namespace=namespace,
    )
