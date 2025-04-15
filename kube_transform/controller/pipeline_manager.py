import logging
from kube_transform.spec import KTPipeline, KTJob
import re
from typing import List, Dict, Optional


JOB_STATES = [
    "Running",  # Submitted to Kubernetes
    "Pending",  # Registered with pipeline manager but not yet submitted (e.g., waiting for dependencies)
    "AwaitingDescendants",  # Descendant jobs have been registered but not yet completed
    "SkippedDueToUpstreamFailure",  # Job was skipped due to upstream failure
    "DescendantFailed",  # Job was marked as failed due to a descendant job failure
    "Completed",  # Job completed successfully
    "Failed",  # Job failed
]


class PipelineManager:
    """
    Manages the state of a pipeline run.

    This includes registering jobs, updating job statuses (including ancestors and dependents),
    and deciding when jobs are ready to be submitted to Kubernetes.

    PipelineManager does not interact with Kubernetes directly.
    """

    def __init__(self, pipeline_spec: Dict, pipeline_run_id: str) -> None:
        """
        Initializes the PipelineManager.

        Args:
            pipeline_spec (Dict): The pipeline specification containing jobs and metadata.
            pipeline_run_id (str): The unique identifier for the pipeline run.
        """
        pipeline_spec = self._validate_pipeline(pipeline_spec)
        self.pipeline_run_id = pipeline_run_id
        self.state = {
            "pipeline_run_id": self.pipeline_run_id,
            "pipeline_name": pipeline_spec["name"],
            "jobs": {},
        }
        job_list = pipeline_spec["jobs"]
        self.register_job_list(job_list)
        self.pipeline_done = len(self.state["jobs"]) == 0

    def register_job_list(
        self, job_list: List[Dict], parent_job_name: Optional[str] = None
    ) -> None:
        """
        Registers a job list with the pipeline manager.

        Args:
            job_list (List[Dict]): A list of job specifications to register.
            parent_job_name (Optional[str]): The name of the parent job, if any.

        Raises:
            ValueError: If a duplicate job name is encountered.
        """
        base_name = (
            parent_job_name
            if parent_job_name
            else f"pipeline-{self.state['pipeline_name']}"
        )
        job_list = self._validate_job_list(job_list, base_name)
        for job in job_list:
            job_name = job["name"]
            if job_name in self.state["jobs"]:
                raise ValueError(f"Duplicate job name: {job_name}")
            job["status"] = "Pending"
            self.state["jobs"][job_name] = job

            if parent_job_name:
                if "direct_descendants" not in self.state["jobs"][parent_job_name]:
                    self.state["jobs"][parent_job_name]["direct_descendants"] = []
                self.state["jobs"][parent_job_name]["direct_descendants"].append(
                    job_name
                )
                job["parent_job"] = parent_job_name

    def _job_can_run(self, job_name: str) -> bool:
        """
        Checks if a job can run based on its dependencies.

        Args:
            job_name (str): The name of the job to check.

        Returns:
            bool: True if the job can run, False otherwise.
        """
        job = self.state["jobs"][job_name]
        return job["status"] == "Pending" and all(
            self.state["jobs"][dep]["status"] == "Completed"
            for dep in job["dependencies"]
        )

    def get_ready_jobs(self) -> Dict[str, Dict]:
        """
        Retrieves all jobs that are ready to be submitted.

        Returns:
            Dict[str, Dict]: A dictionary of job names and their specifications for jobs that are ready to run.
        """
        return {
            job_name: job_spec
            for job_name, job_spec in self.state["jobs"].items()
            if self._job_can_run(job_name)
        }

    def mark_job_running(self, job_name: str) -> None:
        """
        Marks a job as running.

        Args:
            job_name (str): The name of the job to mark as running.
        """
        self.state["jobs"][job_name]["status"] = "Running"

    def mark_job_completed(self, job_name: str) -> None:
        """
        Marks a job as completed or awaiting descendants.

        Args:
            job_name (str): The name of the job to mark as completed.
        """
        # Update job state
        job = self.state["jobs"][job_name]
        descendants = job.get("direct_descendants", [])
        if all(
            self.state["jobs"][descendant]["status"] == "Completed"
            for descendant in descendants
        ):
            job["status"] = "Completed"
        else:
            job["status"] = "AwaitingDescendants"

        # Update parent job state
        if job["status"] == "Completed":
            parent_job_name = job.get("parent_job")
            if parent_job_name:
                # This will trigger a re-check of the parent
                # job's status and mark it as Completed if all
                # dependencies are completed
                self.mark_job_completed(parent_job_name)

        self.check_and_set_is_done()

    def mark_job_failed(self, job_name: str, failure_type: str = "Failed") -> None:
        """
        Marks a job as failed and updates ancestors and dependents.

        Args:
            job_name (str): The name of the job to mark as failed.
            failure_type (str): The type of failure. Must be one of "Failed", "SkippedDueToUpstreamFailure", or "DescendantFailed".

        Raises:
            ValueError: If an invalid failure type is provided.
        """
        if failure_type not in [
            "Failed",
            "SkippedDueToUpstreamFailure",
            "DescendantFailed",
        ]:
            raise ValueError(f"Invalid failure type: {failure_type}")
        job = self.state["jobs"][job_name]
        job["status"] = failure_type

        # Update parent job state
        if failure_type in ["Failed", "DescendantFailed"]:
            parent_job_name = job.get("parent_job")
            if parent_job_name:
                self.mark_job_failed(parent_job_name, failure_type="DescendantFailed")

        # Update dependents
        direct_dependents = [
            other_job_name
            for other_job_name, other_job in self.state["jobs"].items()
            if job_name in other_job.get("dependencies", [])
        ]
        for dependent_job_name in direct_dependents:
            self.mark_job_failed(
                dependent_job_name, failure_type="SkippedDueToUpstreamFailure"
            )

        if failure_type == "Failed":
            # only need to do this once after all recursive updates
            self.check_and_set_is_done()

    def get_state(self) -> Dict:
        """
        Retrieves the current state of the pipeline.

        Returns:
            Dict: The current state of the pipeline.
        """
        return self.state

    def check_and_set_is_done(self) -> None:
        """
        Checks if all jobs are in their final state and sets the pipeline_done flag if true.
        """
        if not any(
            job["status"] in ["Pending", "Running"]
            for job in self.state["jobs"].values()
        ):
            self.pipeline_done = True

    def is_done(self) -> bool:
        """
        Checks if the pipeline is done.

        Returns:
            bool: True if the pipeline is done, False otherwise.
        """
        return self.pipeline_done

    ### Private Methods ###

    def _sanitize_k8s_job_name(self, s: str) -> str:
        """
        Sanitizes a string to be a valid Kubernetes job name.

        Args:
            s (str): The string to sanitize.

        Returns:
            str: The sanitized string.
        """
        s = s.lower()
        s = s.replace("_", "-")
        s = re.sub(r"[^a-z0-9\-]", "", s)
        return s[:63]

    def _validate_job_list(self, job_list: List[Dict], base_name: str) -> List[Dict]:
        """
        Validates the job list and ensures job names are Kubernetes-compliant.

        Args:
            job_list (List[Dict]): The list of job specifications to validate.
            base_name (str): The base name to use for unnamed jobs.

        Returns:
            List[Dict]: The validated and sanitized job list.
        """
        jobs = []
        for idx, job in enumerate(job_list):
            job_obj = KTJob(**job)
            job = job_obj.model_dump()
            if not job.get("name"):
                job["name"] = f"{base_name}-job-{idx}"
            job["name"] = self._sanitize_k8s_job_name(job["name"])
            jobs.append(job)
        return jobs

    def _validate_pipeline(self, pipeline: Dict) -> Dict:
        """
        Validates the pipeline specification.

        Args:
            pipeline (Dict): The pipeline specification to validate.

        Returns:
            Dict: The validated pipeline specification.
        """
        pipeline_obj = KTPipeline(**pipeline)
        pipeline = pipeline_obj.model_dump()
        return pipeline
