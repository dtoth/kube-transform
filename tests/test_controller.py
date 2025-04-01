import unittest
from unittest.mock import patch, mock_open, MagicMock
from kube_transform.controller.controller import KTController

# Basic pipeline spec for mocking
PIPELINE_SPEC_JSON = """
{
  "name": "test-pipeline",
  "jobs": [
    {"name": "job-a", "type": "static", "dependencies": [], "tasks": []},
    {"name": "job-b", "type": "static", "dependencies": ["job-a"], "tasks": []}
  ]
}
"""


class TestKTControllerBasic(unittest.TestCase):

    @patch("kube_transform.fsutil.write")
    @patch("builtins.open", new_callable=mock_open, read_data=PIPELINE_SPEC_JSON)
    def test_load_pipeline_spec(self, mock_file, mock_write):
        controller = KTController("test-run")
        spec = controller.load_pipeline_spec()
        self.assertEqual(len(spec["jobs"]), 2)
        self.assertEqual(spec["jobs"][1]["name"], "job-b")

    @patch("kube_transform.fsutil.write")
    @patch("builtins.open", new_callable=mock_open, read_data=PIPELINE_SPEC_JSON)
    def test_initialize_pipeline_state(self, mock_file, mock_write):
        controller = KTController("run-123")
        state = controller.initialize_pipeline_state()
        self.assertEqual(state["jobs"]["job-a"]["status"], "Pending")
        self.assertEqual(state["jobs"]["job-b"]["status"], "Pending")
        self.assertEqual(state["jobs"]["job-b"]["dependencies"], ["job-a"])

    @patch("kube_transform.fsutil.write")
    @patch("builtins.open", new_callable=mock_open, read_data=PIPELINE_SPEC_JSON)
    def test_job_can_run_no_deps(self, mock_file, mock_write):
        controller = KTController("run-x")
        state = {
            "jobs": {
                "job-a": {"status": "Pending", "dependencies": []},
            }
        }
        controller.state = state
        self.assertTrue(controller.job_can_run(state["jobs"]["job-a"]))

    @patch("kube_transform.fsutil.write")
    @patch("builtins.open", new_callable=mock_open, read_data=PIPELINE_SPEC_JSON)
    def test_job_can_run_with_met_deps(self, mock_file, mock_write):
        controller = KTController("run-x")
        state = {
            "jobs": {
                "job-a": {"status": "Completed"},
                "job-b": {"status": "Pending", "dependencies": ["job-a"]},
            }
        }
        controller.state = state
        self.assertTrue(controller.job_can_run(state["jobs"]["job-b"]))

    @patch("kube_transform.fsutil.write")
    @patch("builtins.open", new_callable=mock_open, read_data=PIPELINE_SPEC_JSON)
    def test_job_cannot_run_with_unmet_deps(self, mock_file, mock_write):
        controller = KTController("run-x")
        state = {
            "jobs": {
                "job-a": {"status": "Running"},
                "job-b": {"status": "Pending", "dependencies": ["job-a"]},
            }
        }
        controller.state = state
        self.assertFalse(controller.job_can_run(state["jobs"]["job-b"]))


if __name__ == "__main__":
    unittest.main()
