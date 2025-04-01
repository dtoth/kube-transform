from __future__ import annotations
from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field, model_validator, ConfigDict


class KTTask(BaseModel):
    """
    A single function call to be executed as part of a static job.
    This will run as a single Kubernetes Pod.
    """

    model_config = ConfigDict(extra="forbid")

    function: Optional[str] = Field(
        default=None,
        description="The name of the function to call from within kt_functions"
        " (e.g., 'my_module.my_function' will map to 'kt_functions.my_module.my_function')."
        " If missing, the task will be a no-op.",
    )
    args: Optional[Dict] = Field(
        default_factory=dict, description="Keyword arguments to pass to the function"
    )


class KTJob(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["static", "dynamic"] = Field(
        ...,
        title="Job Type",
        description="The type of job. Static jobs are pre-defined with a list of tasks."
        " Dynamic jobs are defined by a function that returns a list of KTJob.",
    )
    name: Optional[str] = Field(
        default=None,
        title="Job Name",
        description="Unique job name. Will be auto-generated if not specified.",
    )
    memory: Optional[str] = Field(
        default="1Gi",
        description="Memory request/limit (e.g., '1Gi'). Optional, defaults to 1Gi.",
    )
    cpu: Optional[str] = Field(
        default="500m",
        description="CPU request/limit (e.g., '500m'). Optional, defaults to 500m.",
    )
    dependencies: Optional[List[str]] = Field(
        default_factory=list,
        description="List of job names this job depends on."
        " This job will not execute until all dependencies have completed."
        " Dependent jobs should coordinate via shared data in the data directory"
        " or via external resources.",
    )

    # Dynamic Job Fields
    function: Optional[str] = Field(
        default=None,
        description="The name of the function to call from within kt_functions."
        " The specified function is expected to return a List[KTJob]."
        " If missing, the job will be a no-op.",
    )
    args: Optional[Dict] = Field(
        default_factory=dict, description="Keyword arguments to pass to the function"
    )

    # Static Job Fields
    tasks: Optional[List[KTTask]] = Field(
        default_factory=list, description="List of tasks to run in this job"
    )

    @model_validator(mode="after")
    def validate_type_specific_fields(self) -> "KTJob":
        if self.type == "static":
            if self.function or self.args:
                raise ValueError("Static jobs must not define 'function' or 'args'")
        elif self.type == "dynamic":
            if self.tasks:
                raise ValueError("Dynamic jobs must not define 'tasks'")
        return self


class KTPipeline(BaseModel):
    """
    A pipeline consisting of jobs and their dependencies.
    """

    model_config = ConfigDict(extra="forbid")
    name: str = Field(
        ...,
        title="Pipeline Name",
        description="The pipeline name. E.g. my-pipeline",
    )
    jobs: Optional[List[KTJob]] = Field(
        default_factory=list, description="List of jobs to run"
    )
