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


class TestPipelineManager(unittest.TestCase):
    """Unit tests for the PipelineManager class."""

    def test_initial_jobs(self) -> None:
        """Test that only jobs with no dependencies are ready initially."""
        pipeline_manager = PipelineManager(
            json.loads(PIPELINE_SPEC_JSON), "test-pipeline-run-id"
        )
        set(pipeline_manager.get_ready_jobs().keys()) == {"job-a"}


if __name__ == "__main__":
    unittest.main()
