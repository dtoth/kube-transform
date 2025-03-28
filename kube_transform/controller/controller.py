import json
import time
import os
import uuid
from kubernetes import client, watch
import kube_transform.fsutil as fs
import logging
from kube_transform.controller.k8s import create_static_job, create_dynamic_job

JOB_STATES = [
    "Running",
    "Pending",
    "AwaitingDescendants",
    "SkippedDueToUpstreamFailure",
    "DescendantFailed",
    "Completed",
    "Failed",
]


class KTController:

    ### Initialization ###
    def __init__(self, dag_uuid, namespace="default", dry_run=False):
        self.namespace = namespace
        self.dag_uuid = dag_uuid
        self.dag_spec_path = "/config/dag_spec.json"
        self.dag_state_path = f"kt-metadata/{dag_uuid}/dag_run_state.json"
        self.dry_run = dry_run
        self.batch_v1 = client.BatchV1Api()
        self.dag = self.load_dag_spec()
        self.state = self.initialize_dag_state()
        self.image = os.getenv("KT_IMAGE_PATH")

    def load_dag_spec(self):
        with open(self.dag_spec_path, "r") as f:
            return json.load(f)

    def initialize_dag_state(self):
        state = {"dag_uuid": self.dag_uuid, "jobs": {}}
        for job in self.dag["jobs"]:
            job_name = job["name"]
            job_type = job.get("type")
            if job_type not in ["dynamic", "static"]:
                raise ValueError(
                    f"Job '{job_name}' must specify type as either 'dynamic' or 'static'."
                )
            # TODO more validation here
            state["jobs"][job_name] = {
                "job_name": job_name,
                "status": "Pending",
                "dependencies": job.get("dependencies", []),
                "tasks": job.get("tasks"),
                "function": job.get("function"),
                "args": job.get("args"),
                "type": job_type,
            }
            if "resources" in job:
                state["jobs"][job_name]["resources"] = job["resources"]
        self.save_dag_state(state)
        return state

    ### State Management ###
    def save_dag_state(self, state):
        fs.write(self.dag_state_path, json.dumps(state, indent=2))

    ### Job Management ###
    def submit_ready_jobs(self):
        """Submit all jobs that are ready to run."""
        for job_name, job in self.state["jobs"].items():
            if job["status"] == "Pending" and all(
                self.state["jobs"][dep]["status"] == "Completed"
                for dep in job["dependencies"]
            ):
                self.submit_job(job_name)

    def submit_job(self, job_name):
        """Submit a job to the Kubernetes cluster."""
        logging.info(f"Submitting job {job_name}...")
        job_spec = self.state["jobs"][job_name]

        if not job_spec:
            raise ValueError(f"Job {job_name} not found in DAG spec!")

        if job_spec["type"] == "static":
            create_static_job(
                job_name,
                self.dag_uuid,
                job_spec,
                self.image,
                self.namespace,
            )
        elif job_spec["type"] == "dynamic":
            create_dynamic_job(
                job_name,
                self.dag_uuid,
                job_spec,
                self.image,
                self.namespace,
            )

        job_spec["status"] = "Running"
        self.save_dag_state(self.state)

    def update_job_states(self):
        """Update job states based on dependencies and descendants."""
        changed = True
        while changed:
            changed = False
            for job_name, job in self.state["jobs"].items():
                if job["status"] not in ["Pending", "AwaitingDescendants"]:
                    continue

                # If any dependencies are marked as Failed or SkippedDueToUpstreamFailure or DescendantFailed,
                # mark as SkippedDueToUpstreamFailure
                if any(
                    self.state["jobs"][dep]["status"]
                    in ["Failed", "SkippedDueToUpstreamFailure", "DescendantFailed"]
                    for dep in job["dependencies"]
                ):
                    job["status"] = "SkippedDueToUpstreamFailure"
                    changed = True
                # If any descendants are marked as Failed or SkippedDueToUpstreamFailure or DescendantFailed,
                # mark as DescendantFailed
                elif any(
                    self.state["jobs"][desc]["status"]
                    in ["Failed", "SkippedDueToUpstreamFailure", "DescendantFailed"]
                    for desc in job.get("direct_descendants", [])
                ):
                    job["status"] = "DescendantFailed"
                    changed = True
                # If AwaitingDescendants and all descendants are Completed,
                # mark as Completed
                elif job["status"] == "AwaitingDescendants" and all(
                    self.state["jobs"][desc]["status"] == "Completed"
                    for desc in job.get("direct_descendants", [])
                ):
                    job["status"] = "Completed"
                    changed = True

    def handle_job_completion(self, job_name, success):
        """Handle job completion or failure by updating state and spawning descendant jobs."""

        logging.info(f"Job {job_name} {'succeeded' if success else 'failed'}")

        job_info = self.state["jobs"][job_name]

        # Set target job to completed or failed or AwaitingDescendants (and register the descendants as pending)
        if not success:
            job_info["status"] = "Failed"

        if success:
            if job_info["type"] == "static":
                job_info["status"] = "Completed"
            elif job_info["type"] == "dynamic":
                # Spawn descendant jobs
                job_info["status"] = "AwaitingDescendants"
                orch_spec_path = (
                    f"kt-metadata/{self.dag_uuid}/dynamic_job_output/{job_name}.json"
                )
                if fs.exists(orch_spec_path):
                    orch_spec = json.loads(fs.read(orch_spec_path))
                    self.dag["jobs"].extend(orch_spec["jobs"])
                    direct_descendants = []
                    for jidx, new_job in enumerate(orch_spec["jobs"]):
                        new_job_name = new_job.get("name", f"{job_name}-spawned-{jidx}")
                        new_job_type = new_job.get("type")
                        # TODO more validation (and consolidate validation)
                        if new_job_type not in ["dynamic", "static"]:
                            raise ValueError(
                                f"Job '{new_job_name}' must specify type as either 'dynamic' or 'static'."
                            )

                        self.state["jobs"][new_job_name] = new_job
                        new_job["status"] = "Pending"
                        new_job["parent_job"] = job_name
                        new_job["dependencies"] = new_job.get("dependencies", [])
                        new_job["tasks"] = new_job.get("tasks", [])
                        direct_descendants.append(new_job_name)
                    job_info["direct_descendants"] = direct_descendants

        # Update state for all jobs, submit ready jobs, and save state
        self.update_job_states()
        self.submit_ready_jobs()
        self.save_dag_state(self.state)

        if job_info["type"] == "dynamic":
            orch_spec_path = (
                f"kt-metadata/{self.dag_uuid}/orchestration/{job_name}.json"
            )
            if fs.exists(orch_spec_path):
                orch_spec = json.loads(fs.read(orch_spec_path))
                self.dag["jobs"].extend(orch_spec["jobs"])
                spawned_jobs = []

                for new_job in orch_spec["jobs"]:
                    new_job_name = new_job["name"]
                    new_job_type = new_job.get("type")
                    if new_job_type not in ["dynamic", "static"]:
                        raise ValueError(
                            f"Job '{new_job_name}' must specify type as either 'dynamic' or 'static'."
                        )

                    self.state["jobs"][new_job_name] = {
                        "status": "Pending",
                        "dependencies": new_job.get("dependencies", []),
                        "type": new_job_type,
                        "parent_job": job_name,
                    }
                    spawned_jobs.append(new_job_name)

                self.state["jobs"][job_name][
                    "status"
                ] = "Awaiting-Descendant-Completion"
                self.state["jobs"][job_name]["spawned_jobs"] = spawned_jobs
                self.save_dag_state(self.state)

        self.submit_ready_jobs()

    def monitor_jobs(self):
        # Submit initial jobs
        self.submit_ready_jobs()

        # Watch for job completions
        watcher = watch.Watch()
        done = False
        while True:
            for event in watcher.stream(
                self.batch_v1.list_namespaced_job, namespace=self.namespace
            ):
                job = event["object"]
                job_name = job.metadata.name

                if self.dag_uuid not in job_name:
                    continue

                job_name = job_name.replace(f"-{self.dag_uuid}", "")
                job_status = job.status.conditions

                if job_name not in self.state["jobs"] or self.state["jobs"][job_name][
                    "status"
                ] not in [
                    "Running",
                    "Pending",
                ]:
                    continue

                if job_status:
                    for condition in job_status:
                        if condition.status != "True":
                            continue
                        if condition.type in ["Complete", "Failed"]:
                            self.handle_job_completion(
                                job_name, condition.type == "Complete"
                            )

                if (
                    sum(
                        job["status"] in ["Running", "Pending", "AwaitingDescendants"]
                        for job in self.state["jobs"].values()
                    )
                    == 0
                ):
                    logging.info("All jobs completed. Exiting monitoring.")
                    done = True
                    break
            if done:
                break
            time.sleep(5)


if __name__ == "__main__":
    from kubernetes import config

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting KT Controller...")
    config.load_incluster_config()
    dag_uuid = os.getenv("DAG_UUID", str(uuid.uuid4()))
    dag_controller = KTController(dag_uuid, namespace="default")
    dag_controller.monitor_jobs()
