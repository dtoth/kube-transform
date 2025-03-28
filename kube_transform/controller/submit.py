import uuid
from kubernetes import config

from kube_transform.controller.k8s import create_controller_job


def submit_dag(dag_function, dag_params, image_path, data_dir, namespace="default"):
    """Submits a DAG for execution by creating a KTController Kubernetes Job."""
    dag_uuid = f"dagrun{str(uuid.uuid4())[:8]}"
    dag_spec = dag_function(**dag_params)

    config.load_kube_config()

    create_controller_job(
        dag_uuid=dag_uuid,
        dag_spec=dag_spec,
        image_path=image_path,
        data_dir=data_dir,
        namespace=namespace,
    )
