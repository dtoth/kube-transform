from kubernetes import client
import json
import os
import logging
from typing import Any, Dict

import kube_transform.fsutil as fs


def create_controller_job(
    pipeline_run_id: str,
    pipeline_spec: Dict[str, Any],
    image_path: str,
    data_dir: str,
    namespace: str,
) -> None:
    """
    Creates a Kubernetes Job to run the KTController.

    Args:
        pipeline_run_id (str): The unique ID for the pipeline run.
        pipeline_spec (Dict[str, Any]): The pipeline specification.
        image_path (str): The path to the container image for all worker pods.
        data_dir (str): The directory for data storage (can be S3, local, etc.).
        namespace (str): The Kubernetes namespace to create the job in.
    """
    job_name = f"kt-controller"

    # Create a ConfigMap with the pipeline spec
    config_map_name = f"pipeline-config-{pipeline_run_id}"
    config_map = client.V1ConfigMap(
        metadata=client.V1ObjectMeta(name=config_map_name),
        data={"pipeline_spec.json": json.dumps(pipeline_spec)},
    )
    core_v1 = client.CoreV1Api()
    core_v1.create_namespaced_config_map(namespace=namespace, body=config_map)

    create_k8s_job(
        job_name,
        image_path,
        namespace,
        config_map_name,
        data_dir,
        pipeline_run_id,
        job_args={},
        service_account="kt-pod",
        command=["python", "-m", "kube_transform.controller.controller"],
        memory="1Gi",
        cpu="250m",
    )


def create_static_job(
    job_name: str,
    pipeline_run_id: str,
    job_spec: Dict[str, Any],
    image_path: str,
    namespace: str,
) -> None:
    """
    Creates a Kubernetes Job to run a static job.

    Args:
        job_name (str): The name of the job.
        pipeline_run_id (str): The unique ID for the pipeline run.
        job_spec (Dict[str, Any]): The job specification.
        image_path (str): The path to the container image.
        namespace (str): The Kubernetes namespace to create the job in.
    """

    # Write job config for task index lookup
    task_config_path = (
        f"kt-metadata/{pipeline_run_id}/static_job_config_maps/{job_name}.json"
    )
    logging.info(f"Writing task config to {task_config_path}")
    fs.write(task_config_path, json.dumps({"tasks": job_spec["tasks"]}, indent=2))

    memory = job_spec.get("memory", "1Gi")
    cpu = job_spec.get("cpu", "500m")

    create_k8s_job(
        job_name=job_name,
        image_path=image_path,
        namespace=namespace,
        config_map_name=None,  # only used for controller job
        data_dir=os.environ["DATA_DIR"],
        pipeline_run_id=pipeline_run_id,
        job_args={"task_config_path": task_config_path},
        service_account="kt-pod",
        command=[
            "python",
            "-c",
            (
                "import os, json, importlib, logging\n"
                "from kube_transform import fsutil as fs\n"
                "logging.basicConfig(level=logging.INFO)\n"
                "logging.info('Starting static job')\n"
                "# Load job spec from task_config_path\n"
                "task_config_path = json.loads(os.environ['KT_JOB_ARGS'])['task_config_path']\n"
                "job_spec = json.loads(fs.read(task_config_path))\n"
                "# Use Kubernetes job index to select task\n"
                "index = int(os.environ.get('JOB_INDEX'))\n"
                "task = job_spec['tasks'][index]\n"
                "module_path, func_name = task['function'].rsplit('.', 1)\n"
                "func = getattr(importlib.import_module('kt_functions.' + module_path), func_name)\n"
                "logging.info(f'Running task {index}: {task}')\n"
                "func(**task['args'])\n"
                "logging.info('Success')\n"
            ),
        ],
        memory=memory,
        cpu=cpu,
        task_count=len(job_spec["tasks"]),
    )


def create_dynamic_job(
    job_name: str,
    pipeline_run_id: str,
    job_spec: Dict[str, Any],
    image_path: str,
    namespace: str,
) -> None:
    """
    Creates a Kubernetes Job to run a dynamic job.

    Args:
        job_name (str): The name of the job.
        pipeline_run_id (str): The unique ID for the pipeline run.
        job_spec (Dict[str, Any]): The job specification.
        image_path (str): The path to the container image.
        namespace (str): The Kubernetes namespace to create the job in.
    """

    memory = job_spec.get("memory", "1Gi")
    cpu = job_spec.get("cpu", "500m")

    create_k8s_job(
        job_name=job_name,
        image_path=image_path,
        namespace=namespace,
        config_map_name=None,  # only used for controller job
        data_dir=os.environ["DATA_DIR"],
        pipeline_run_id=pipeline_run_id,
        job_args={"function": job_spec["function"], "args": job_spec["args"]},
        service_account="kt-pod",
        command=[
            "python",
            "-c",
            (
                "import os, json, importlib, logging\n"
                "from kube_transform import fsutil as fs\n"
                "logging.basicConfig(level=logging.INFO)\n"
                "logging.info('Starting dynamic job')\n"
                "# Load job spec from task_config_path\n"
                "job_spec = json.loads(os.environ['KT_JOB_ARGS'])\n"
                "target_func = job_spec['function']\n"
                "target_args = job_spec['args']\n"
                "# Use Kubernetes job index to select task\n"
                "module_path, func_name = target_func.rsplit('.', 1)\n"
                "func = getattr(importlib.import_module('kt_functions.' + module_path), func_name)\n"
                "logging.info('Generating job spec')\n"
                "jobs = func(**target_args)\n"
                "logging.info('Saving job spec')\n"
                "orch_spec_path = os.environ.get('KT_ORCH_SPEC_PATH')\n"
                "fs.write(orch_spec_path, json.dumps(jobs))\n"
                "logging.info('Success')\n"
            ),
        ],
        memory=memory,
        cpu=cpu,
    )


def create_k8s_job(
    job_name: str,
    image_path: str,
    namespace: str,
    config_map_name: str,
    data_dir: str,
    pipeline_run_id: str,
    job_args: Dict[str, Any],
    service_account: str,
    command: list,
    memory: str,
    cpu: str,
    task_count: int = 1,
) -> None:
    """
    Creates a generic Kubernetes Job.

    Args:
        job_name (str): The name of the job.
        image_path (str): The path to the container image.
        namespace (str): The Kubernetes namespace to create the job in.
        config_map_name (str): The name of the ConfigMap (if any).
        data_dir (str): The directory for data storage.
        pipeline_run_id (str): The unique ID for the pipeline run.
        job_args (Dict[str, Any]): Arguments for the job.
        service_account (str): The service account to use.
        command (list): The command to execute in the container.
        memory (str): Memory request for the container.
        cpu (str): CPU request for the container.
        task_count (int, optional): Number of tasks to run. Defaults to 1.
    """
    image_pull_policy = "Always" if "/" in image_path else "Never"

    volumes = [
        client.V1Volume(
            name="data-volume",
            host_path=client.V1HostPathVolumeSource(
                path="/mnt/data",
                type="DirectoryOrCreate",
            ),
        ),
    ]
    volume_mounts = [
        client.V1VolumeMount(name="data-volume", mount_path="/mnt/data"),
    ]

    if config_map_name:
        volumes.append(
            client.V1Volume(
                name="config-volume",
                config_map=client.V1ConfigMapVolumeSource(name=config_map_name),
            )
        )
        volume_mounts.append(
            client.V1VolumeMount(name="config-volume", mount_path="/config")
        )

    orch_spec_path = f"kt-metadata/{pipeline_run_id}/dynamic_job_output/{job_name}.json"
    fully_qualified_job_name = f"{job_name}-{pipeline_run_id}"

    container = client.V1Container(
        name=fully_qualified_job_name,
        image=image_path,
        image_pull_policy=image_pull_policy,
        env=[
            client.V1EnvVar(name="DATA_DIR", value=data_dir),
            client.V1EnvVar(name="pipeline_run_id", value=pipeline_run_id),
            client.V1EnvVar(name="KT_JOB_ARGS", value=json.dumps(job_args)),
            client.V1EnvVar(name="KT_IMAGE_PATH", value=image_path),
            client.V1EnvVar(
                name="KT_ORCH_SPEC_PATH", value=orch_spec_path
            ),  # only used for dynamic jobs
            client.V1EnvVar(
                name="JOB_INDEX",
                value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(
                        field_path="metadata.annotations['batch.kubernetes.io/job-completion-index']"
                    )
                ),
            ),
        ],
        volume_mounts=volume_mounts,
        resources=client.V1ResourceRequirements(
            requests={"memory": memory, "cpu": cpu},
            limits={"memory": _multiply_memory_str(memory, 1.1)},
        ),
        command=command,
    )

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"job-name": fully_qualified_job_name}),
        spec=client.V1PodSpec(
            restart_policy="Never",
            service_account_name=service_account,
            containers=[container],
            volumes=volumes,
        ),
    )

    job_spec = client.V1JobSpec(
        template=template,
        backoff_limit=4,
        completion_mode="Indexed",
        completions=task_count,
        parallelism=task_count,
    )

    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=fully_qualified_job_name),
        spec=job_spec,
    )

    batch_v1 = client.BatchV1Api()
    batch_v1.create_namespaced_job(namespace=namespace, body=job)


def _multiply_memory_str(mem_str: str, factor: float) -> str:
    """
    Multiplies a memory string by a factor and returns the new string.

    Args:
        mem_str (str): The memory string (e.g., "1Gi").
        factor (float): The multiplication factor.

    Returns:
        str: The updated memory string.
    """
    number = int("".join([c for c in mem_str if c.isnumeric()]))
    unit = "".join([c for c in mem_str if not c.isnumeric()])
    return f"{int(number * factor)}{unit}"
