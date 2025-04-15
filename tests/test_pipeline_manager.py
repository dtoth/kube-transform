import unittest
import json
from kube_transform.controller.pipeline_manager import PipelineManager

PIPELINE_SPEC_JSON = """
{
  "name": "test-pipeline",
  "jobs": [
    {"name": "job-a", "type": "dynamic", "dependencies": [], "tasks": []},
    {"name": "job-b", "type": "static", "dependencies": ["job-a"], "tasks": []},
    {"name": "job-c", "type": "static", "dependencies": ["job-b"], "tasks": []}
  ]
}
"""

JOB_A_DIRECT_DESCENDANTS = """
[
    {"name": "job-a2", "type": "static", "dependencies": [], "tasks": []}
]
"""


class TestPipelineManager(unittest.TestCase):
    """Unit tests for the PipelineManager class."""

    def test_initial_jobs(self) -> None:
        """Test that only jobs with no dependencies are ready initially."""
        pipeline_manager = PipelineManager(
            json.loads(PIPELINE_SPEC_JSON), "test-pipeline-run-id"
        )
        self.assertEqual(set(pipeline_manager.get_ready_jobs().keys()), {"job-a"})

    def test_job_order(self) -> None:
        """Test that only jobs with no dependencies are ready initially."""
        pipeline_manager = PipelineManager(
            json.loads(PIPELINE_SPEC_JSON), "test-pipeline-run-id"
        )
        self.assertEqual(set(pipeline_manager.get_ready_jobs().keys()), {"job-a"})
        pipeline_manager.mark_job_completed("job-a")
        self.assertEqual(set(pipeline_manager.get_ready_jobs().keys()), {"job-b"})
        pipeline_manager.mark_job_completed("job-b")
        self.assertEqual(set(pipeline_manager.get_ready_jobs().keys()), {"job-c"})
        self.assertEqual(pipeline_manager.is_done(), False)
        pipeline_manager.mark_job_completed("job-c")
        self.assertEqual(set(pipeline_manager.get_ready_jobs().keys()), set())
        self.assertEqual(pipeline_manager.is_done(), True)

    def test_job_failed(self) -> None:
        """Test that a job can be marked as failed."""
        pipeline_manager = PipelineManager(
            json.loads(PIPELINE_SPEC_JSON), "test-pipeline-run-id"
        )
        pipeline_manager.mark_job_failed("job-a")
        self.assertEqual(
            pipeline_manager.get_state()["jobs"]["job-a"]["status"], "Failed"
        )
        self.assertEqual(
            pipeline_manager.get_state()["jobs"]["job-b"]["status"],
            "SkippedDueToUpstreamFailure",
        )
        self.assertEqual(
            pipeline_manager.get_state()["jobs"]["job-c"]["status"],
            "SkippedDueToUpstreamFailure",
        )
        self.assertEqual(pipeline_manager.is_done(), True)

    def test_job_failed_with_descendants(self) -> None:
        """Test that a job can be marked as failed with descendants."""
        pipeline_manager = PipelineManager(
            json.loads(PIPELINE_SPEC_JSON), "test-pipeline-run-id"
        )
        pipeline_manager.mark_job_running("job-a")
        pipeline_manager.register_job_list(
            json.loads(JOB_A_DIRECT_DESCENDANTS), parent_job_name="job-a"
        )
        pipeline_manager.mark_job_completed("job-a")

        self.assertEqual(
            pipeline_manager.get_state()["jobs"]["job-a"]["status"],
            "AwaitingDescendants",
        )

        pipeline_manager.mark_job_running("job-a2")
        pipeline_manager.mark_job_failed("job-a2")

        self.assertEqual(
            pipeline_manager.get_state()["jobs"]["job-a"]["status"], "DescendantFailed"
        )
        self.assertEqual(
            pipeline_manager.get_state()["jobs"]["job-a2"]["status"], "Failed"
        )
        self.assertEqual(
            pipeline_manager.get_state()["jobs"]["job-b"]["status"],
            "SkippedDueToUpstreamFailure",
        )
        self.assertEqual(
            pipeline_manager.get_state()["jobs"]["job-c"]["status"],
            "SkippedDueToUpstreamFailure",
        )
        self.assertEqual(pipeline_manager.is_done(), True)


if __name__ == "__main__":
    unittest.main()
