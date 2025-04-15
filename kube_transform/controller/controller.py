import json
import time
import os
import uuid
import re
from kubernetes import client, watch
import kube_transform.fsutil as fs
import logging
from kube_transform.controller.k8s import create_static_job, create_dynamic_job
from kube_transform.kube_transform.controller.pipeline_manager import PipelineManager


class KTController:
    """
    Manages a single pipeline run from start through completion or failure
    and then exits.

    KTController handles direct interactions with Kubernetes, and delegates
    scheduling logic to a PipelineManager.
    """

    ### Initialization ###
    def __init__(self, pipeline_run_id, namespace="default"):
        self.namespace = namespace
        self.pipeline_run_id = pipeline_run_id
        self.pipeline_spec_path = "/config/pipeline_spec.json"
        self.pipeline_state_path = (
            f"kt-metadata/{pipeline_run_id}/pipeline_run_state.json"
        )
        self.image = os.getenv("KT_IMAGE_PATH")
        self.batch_v1 = client.BatchV1Api()

        pipeline_spec = self.load_pipeline_spec()
        self.pipeline_manager = PipelineManager(pipeline_spec, pipeline_run_id)

    def load_pipeline_spec(self):
        with open(self.pipeline_spec_path, "r") as f:
            return json.load(f)

    ### State Management ###
    def save_pipeline_state(self):
        state = self.pipeline_manager.get_state()
        fs.write(self.pipeline_state_path, json.dumps(state, indent=2))

    ### Job Management ###
    def submit_ready_jobs(self):
        """Submit all jobs that are ready to run."""
        ready_jobs = self.pipeline_manager.get_ready_jobs()
        for job_name, job_spec in ready_jobs.items():
            self.submit_job(job_name, job_spec)
            self.pipeline_manager.mark_job_running(job_name)
        self.save_pipeline_state()

    def submit_job(self, job_name, job_spec):
        """Submit a job to the Kubernetes cluster."""
        logging.info(f"Submitting job {job_name}...")

        if not job_spec["tasks"] and not job_spec["function"]:
            logging.info(f"Job {job_name} is a no-op.")
            self.pipeline_manager.mark_job_completed(job_name)
        else:
            if job_spec["type"] == "static":
                create_static_job(
                    job_name,
                    self.pipeline_run_id,
                    job_spec,
                    self.image,
                    self.namespace,
                )
            elif job_spec["type"] == "dynamic":
                create_dynamic_job(
                    job_name,
                    self.pipeline_run_id,
                    job_spec,
                    self.image,
                    self.namespace,
                )
        self.pipeline_manager.mark_job_running(job_name)

    def monitor_jobs(self):
        # Submit initial jobs
        self.submit_ready_jobs()

        # Watch for job completions
        watcher = watch.Watch()
        while not self.pipeline_manager.is_done():
            for event in watcher.stream(
                self.batch_v1.list_namespaced_job, namespace=self.namespace
            ):
                job = event["object"]
                job_name = job.metadata.name

                if self.pipeline_run_id not in job_name:
                    continue

                job_name = job_name.replace(f"-{self.pipeline_run_id}", "")
                job_status = job.status.conditions

                if job_status:
                    for condition in job_status:
                        if condition.status != "True":
                            continue
                        if condition.type == "Complete":
                            self.register_dynamically_spawned_jobs(job_name)
                            self.pipeline_manager.mark_job_completed(job_name)
                            self.submit_ready_jobs()
                        elif condition.type == "Failed":
                            self.pipeline_manager.mark_job_failed(job_name)
                            self.save_pipeline_state()
            time.sleep(5)
        logging.info("All jobs completed. Exiting monitoring.")

    def register_dynamically_spawned_jobs(self, job_name):
        """Register dynamically spawned jobs (if any) in the pipeline manager."""
        orch_spec_path = (
            f"kt-metadata/{self.pipeline_run_id}/dynamic_job_output/{job_name}.json"
        )
        if fs.exists(orch_spec_path):
            new_job_list = json.loads(fs.read(orch_spec_path))
            self.pipeline_manager.register_job_list(new_job_list, parent_job=job_name)


if __name__ == "__main__":
    from kubernetes import config

    logging.basicConfig(level=logging.INFO)
    logging.info("Starting KT Controller...")
    config.load_incluster_config()
    pipeline_run_id = os.getenv("pipeline_run_id", str(uuid.uuid4()))
    pipeline_controller = KTController(pipeline_run_id, namespace="default")
    pipeline_controller.monitor_jobs()
