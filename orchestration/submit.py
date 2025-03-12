from kubernetes import client, config
import json
import time
import os
import boto3


def submit_job(job_configs):
    """
    Accepts a single job config or a list of job configs and submits them to Kubernetes.
    k8s_env: "minikube" or "eks" to control behavior.
    """
    k8s_env = os.getenv("K8S_ENV")
    if not isinstance(job_configs, list):
        job_configs = [job_configs]

    for job_config in job_configs:
        _submit_job(job_config, k8s_env)


def _multiply_memory_str(mem_str, factor):
    """
    Multiply a memory string by a factor and return the new string.
    """
    number = int("".join([c for c in mem_str if c.isnumeric()]))
    unit = "".join([c for c in mem_str if not c.isnumeric()])
    return f"{int(number * factor)}{unit}"


def _submit_job(job_config, k8s_env):
    """
    Create a parallel Kubernetes job with tasks passed as structured dictionaries.
    """
    tasks = job_config["tasks"]
    memory = job_config.get("memory")
    cpu = job_config.get("cpu")
    if k8s_env == "minikube":
        image_path = "execute-image:latest"
    elif k8s_env == "eks":
        image_path = f"{os.getenv('AWS_ACCOUNT_ID')}.dkr.ecr.us-east-1.amazonaws.com/kube-transform-ecr-repo:latest"
    else:
        raise ValueError(f"Invalid k8s_env: {k8s_env}")
    namespace = "default"
    config.load_kube_config()
    job_name = job_config["job_name"]
    configmap_name = f"{job_name}-config"

    task_data = json.dumps(tasks)
    if k8s_env == "minikube":
        # Ensure ConfigMap is created when running on Minikube
        config_map = client.V1ConfigMap(
            api_version="v1",
            kind="ConfigMap",
            metadata=client.V1ObjectMeta(name=configmap_name),
            data={"tasks.json": task_data},
        )

        # Submit the ConfigMap to Kubernetes
        core_v1 = client.CoreV1Api()
        try:
            core_v1.create_namespaced_config_map(namespace=namespace, body=config_map)
            print(f"ConfigMap {configmap_name} created")
        except client.exceptions.ApiException as e:
            if e.status == 409:
                print(f"ConfigMap {configmap_name} already exists. Replacing...")
                core_v1.replace_namespaced_config_map(
                    name=configmap_name, namespace=namespace, body=config_map
                )
            else:
                raise e
    elif k8s_env == "eks":
        # Push task config to S3 when running on EKS
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket="kube-transform-config-bucket",
            Key=f"{job_name}/tasks.json",
            Body=task_data,
        )
        print(f"Task Config uploaded to S3: {job_name}/tasks.json")

    # Define volumes & volume mounts correctly
    volumes = []
    volume_mounts = []

    if k8s_env == "minikube":
        volumes.extend(
            [
                client.V1Volume(
                    name="data-volume",
                    host_path=client.V1HostPathVolumeSource(
                        path="/mnt/data",
                        type="DirectoryOrCreate",
                    ),
                ),
                client.V1Volume(
                    name="task-config-volume",
                    config_map=client.V1ConfigMapVolumeSource(name=configmap_name),
                ),
            ]
        )
        volume_mounts.extend(
            [
                client.V1VolumeMount(name="data-volume", mount_path="/app/data"),
                client.V1VolumeMount(
                    name="task-config-volume", mount_path="/config"
                ),  # Mount the ConfigMap here
            ]
        )

    # Ensure the container has correct environment variables & volume mounts
    container_spec = _create_container_spec(
        job_name, image_path, memory, cpu, volume_mounts, k8s_env
    )

    job_spec = client.V1JobSpec(
        parallelism=min(200, len(tasks)),  # Limit parallelism to 200
        completions=len(tasks),
        backoff_limit=10,  # More retries before failing permanently
        ttl_seconds_after_finished=3600,  # Keep job in history for 1 hour
        pod_failure_policy=client.V1PodFailurePolicy(
            rules=[
                client.V1PodFailurePolicyRule(
                    action="Ignore",  # Keep retrying beyond backoffLimit if node dies
                    on_pod_conditions=[
                        client.V1PodFailurePolicyOnPodConditionsPattern(
                            type="DisruptionTarget",  # AWS Auto Mode or Spot eviction
                            status="True",
                        )
                    ],
                )
            ]
        ),
        completion_mode="Indexed",
        template=client.V1PodTemplateSpec(
            metadata=client.V1ObjectMeta(labels={"job-name": job_name}),
            spec=client.V1PodSpec(
                service_account_name="aws-access" if k8s_env == "eks" else "default",
                containers=[container_spec],
                restart_policy="Never",
                volumes=volumes,  # Attach volumes
            ),
        ),
    )

    # Create Kubernetes Job
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=job_name),
        spec=job_spec,
    )

    # Submit job to Kubernetes
    batch_v1 = client.BatchV1Api()
    batch_v1.create_namespaced_job(namespace=namespace, body=job)
    print(f"Job {job_name} created!")

    # Monitor job progress
    completed_tasks = 0
    while completed_tasks < len(tasks):
        job_status = batch_v1.read_namespaced_job_status(job_name, namespace)
        succeeded = job_status.status.succeeded or 0
        if succeeded > completed_tasks:
            completed_tasks = succeeded
        if completed_tasks < len(tasks):
            time.sleep(2)
    print(f"Job {job_name} completed!")


def _create_container_spec(job_name, image_path, memory, cpu, volume_mounts, k8s_env):
    """
    Create the container spec, shared between Minikube and EKS.
    """
    PROJECT_NAME = os.getenv("PROJECT_NAME")
    return client.V1Container(
        name=job_name,
        image=image_path,
        command=[
            "python",
            "-c",
            f"from {PROJECT_NAME} import pod_execution"
            + """
import os, json
import boto3


# Load tasks from mounted ConfigMap or S3
task_path = '/config/tasks.json'
if os.path.exists(task_path):
    print('Loading tasks from ConfigMap...')
    with open(task_path, 'r') as f:
        tasks = json.load(f)
else:
    # Load tasks from S3
    os.makedirs(os.path.dirname(task_path), exist_ok=True)
    print("Downloading tasks from S3...")
    s3 = boto3.client("s3")
    s3_key = f"{os.getenv('JOB_NAME')}/tasks.json"

    # Explicitly download the file and print debugging info
    try:
        s3.download_file("kube-transform-config-bucket", s3_key, task_path)
        print(f"Successfully downloaded {s3_key} to {task_path}")
    except Exception as e:
        print(f"Failed to download {s3_key}: {e}")
        raise

    # Verify the file exists before reading
    if os.path.exists(task_path):
        print(f"File exists: {task_path}")
        with open(task_path, "r") as f:
            tasks = json.load(f)
    else:
        print(f"File not found after download: {task_path}")
        raise FileNotFoundError(f"File {task_path} was not found after download.")

# Get job index
job_index = int(os.getenv('JOB_INDEX', '0'))
task = tasks[job_index]
print(f'Executing task {job_index}: {task}')

# Execute target function
func = getattr(pod_execution, task['function'])
func(**task['args'])

print("Task complete.")
            """,
        ],
        env=[
            client.V1EnvVar(
                name="JOB_INDEX",
                value_from=client.V1EnvVarSource(
                    field_ref=client.V1ObjectFieldSelector(
                        field_path="metadata.annotations['batch.kubernetes.io/job-completion-index']"
                    )
                ),
            ),
            client.V1EnvVar(
                name="JOB_NAME",
                value=job_name,
            ),
            client.V1EnvVar(
                name="DATA_DIR",
                value=(
                    "/app/data"
                    if k8s_env == "minikube"
                    else "s3://kube-transform-data-bucket"
                ),
            ),
            client.V1EnvVar(name="K8S_ENV", value=k8s_env),
        ],
        volume_mounts=volume_mounts,  # Mount ConfigMap properly
        resources=client.V1ResourceRequirements(
            requests={"memory": memory, "cpu": cpu},
            limits={"memory": _multiply_memory_str(memory, 1.1)},
        ),
        image_pull_policy="IfNotPresent",
    )
