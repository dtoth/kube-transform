from execution.generic import file_system_util as fs
import uuid


def _get_task_config(function, args):
    return {"function": function, "args": args}


def _get_job_name(function_name):
    base_name = function_name.replace("_", "-").lower()
    random_suffix = uuid.uuid4().hex[:8]
    job_name = f"{base_name}-{random_suffix}"
    return job_name


def _get_job_config(tasks, memory, cpu, job_name=None):
    if job_name is None:
        function_name = tasks[0]["function"]
        job_name = _get_job_name(function_name)
    return {
        "tasks": tasks,
        "memory": memory,
        "cpu": cpu,
        "job_name": job_name,
    }


def hello_world_orchestration(
    input_directory,
    output_directory,
):
    filenames = fs.get_filenames_in_directory(input_directory)
    filenames = [
        fn for fn in filenames if fn.startswith("part_") and fn.endswith(".txt")
    ]
    input_paths = [fs.join(input_directory, fn) for fn in filenames]
    output_paths = [fs.join(output_directory, fn) for fn in filenames]
    task_configs = [
        _get_task_config(
            "hello_world_execution",
            {"input_path": input_paths[i], "output_path": output_paths[i]},
        )
        for i in range(len(filenames))
    ]
    job_config = _get_job_config(task_configs, "250Mi", "200m")
    return job_config
