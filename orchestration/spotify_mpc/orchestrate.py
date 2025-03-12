from execution.generic import file_system_util as fs
import uuid
import math
import numpy as np
from orchestration.submit import _multiply_memory_str


def _list_of_lists(lst, group_size):
    out = []
    current = []
    for i in lst:
        if len(current) == group_size:
            out.append(current)
            current = []
        current.append(i)
    if len(current) > 0:
        out.append(current)
    return out


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


def _combine_job_configs(job_configs):
    """Each item in the input list is a list of job configs
    or a single job config. This function combines them into
    a single list of job configs.
    """
    out = []
    for job_config in job_configs:
        if isinstance(job_config, list):
            out.extend(job_config)
        else:
            out.append(job_config)
    return out


### Contest Functions ###


def standardize_data(input_directory, output_directory, parts=None):
    """Merge the raw playlist files (1000 files, 1k playlists each)
    into 100 files of 10k playlists each, and remove unnecessary
    fields. Create a track dataframe for each of the 100 files.
    """
    if parts is None:
        # Standardize all data if parts are not specified
        parts = list(range(100))

    filenames = fs.get_filenames_in_directory(input_directory)
    filenames = [
        f for f in filenames if f.startswith("mpd.slice") and f.endswith(".json")
    ]
    filenames = sorted(filenames, key=lambda x: int(x.split(".")[2].split("-")[0]))
    pod_input_paths = [fs.join(input_directory, filename) for filename in filenames]
    FILES_PER_TASK = 10
    pod_input_paths = _list_of_lists(pod_input_paths, FILES_PER_TASK)
    pod_output_paths = [
        fs.join(output_directory, f"part_{i}_playlists.parquet")
        for i in range(len(pod_input_paths))
    ]
    track_df_output_paths = [
        fs.join(output_directory, f"part_{i}_tracks.parquet")
        for i in range(len(pod_input_paths))
    ]

    task_configs = [
        _get_task_config(
            "standardize_data",
            {
                "input_paths": pod_input_paths[i],
                "playlist_output_path": pod_output_paths[i],
                "track_df_output_path": track_df_output_paths[i],
            },
        )
        for i in parts
    ]

    # Resources should be coordinated with FILES_PER_TASK
    job_config = _get_job_config(task_configs, "1500Mi", "250m")

    return job_config


def _get_iterative_reduction_job_configs(
    reduction_function,
    input_paths,
    output_path,
    max_files_per_task,
    memory,
    cpu,
    factor,
):
    temporary_directory = fs.join("tmp", uuid.uuid4().hex)
    job_configs = []
    iteration = 0
    extension = input_paths[0].split(".")[-1]
    while iteration == 0 or len(input_paths) > 1:
        input_paths = _list_of_lists(input_paths, max_files_per_task)
        output_paths = [
            fs.join(temporary_directory, f"iter_{iteration}_part_{i}.{extension}")
            for i in range(len(input_paths))
        ]
        if len(output_paths) == 1:
            output_paths[0] = output_path

        task_configs = [
            _get_task_config(
                reduction_function,
                {
                    "input_paths": input_paths[i],
                    "output_path": output_paths[i],
                },
            )
            for i in range(len(input_paths))
        ]
        job_configs.append(
            _get_job_config(
                task_configs,
                _multiply_memory_str(memory, (1 + factor) ** min(iteration, 2)),
                cpu,
            )
        )
        input_paths = output_paths
        iteration += 1
    return job_configs


def reduce_track_dfs(input_directory, parts, output_path):
    """Identify the track dataframes in input_directory,
    filter to include those specified in parts, reduce
    them into a single track dataframe, and store it at
    output_path.
    """
    # Reduce the list of track dataframes into a single track dataframe
    filenames = fs.get_filenames_in_directory(input_directory)
    filenames = [
        f
        for f in filenames
        if f.startswith("part")
        and f.endswith("tracks.parquet")
        and int(f.split("_")[1]) in parts
    ]
    filenames = sorted(filenames, key=lambda f: int(f.split("_")[1]))
    input_paths = [fs.join(input_directory, filename) for filename in filenames]
    job_configs = _get_iterative_reduction_job_configs(
        "reduce_track_dfs",
        input_paths,
        output_path,
        max_files_per_task=10,
        memory="3Gi",
        cpu="250m",
        factor=1,  # 100% more mem per iteration (max 3 iterations)
    )
    return job_configs


def renumber_track_ids(input_directory, parts, track_df_path, output_directory):
    """Identify the playlist dataframes in input_directory,
    filter to include those specified in parts, renumber the
    track IDs based on the track dataframe at track_df_path,
    and store the result at output_path. Output one file per input file.
    """
    filenames = fs.get_filenames_in_directory(input_directory)
    filenames = [
        f
        for f in filenames
        if f.startswith("part")
        and f.endswith("playlists.parquet")
        and int(f.split("_")[1]) in parts
    ]
    filenames = sorted(filenames, key=lambda f: int(f.split("_")[1]))
    playlist_input_paths = [
        fs.join(input_directory, filename) for filename in filenames
    ]
    playlist_output_paths = [
        fs.join(output_directory, filename) for filename in filenames
    ]
    task_configs = [
        _get_task_config(
            "renumber_track_ids",
            {
                "input_path": playlist_input_paths[i],
                "track_df_path": track_df_path,
                "output_path": playlist_output_paths[i],
            },
        )
        for i in range(len(playlist_input_paths))
    ]

    job_config = _get_job_config(task_configs, "1.25Gi", "500m")
    return job_config


def create_challenge_set(
    standardized_input_directory, test_part, track_df_path, n_test_cases, output_path
):
    """Take a single standardized playlist df, and create a challenge set
    with n_test_cases test cases, split evenly amongst the 10 challenge types.
    Any tracks that do not appear in the track_df are removed and accounted for
    via unseen_track_count.
    """
    filenames = fs.get_filenames_in_directory(standardized_input_directory)
    filenames = [
        f
        for f in filenames
        if f.startswith("part")
        and f.endswith("playlists.parquet")
        and int(f.split("_")[1]) == test_part
    ]
    standardized_playlist_path = fs.join(standardized_input_directory, filenames[0])
    task_config = _get_task_config(
        "create_challenge_set",
        {
            "standardized_playlist_path": standardized_playlist_path,
            "track_df_path": track_df_path,
            "n_test_cases": n_test_cases,
            "output_path": output_path,
        },
    )
    job_config = _get_job_config([task_config], "2Gi", "1")
    return job_config


def renumber_existing_challenge_set(
    challenge_set_json_path, track_df_path, output_path
):
    task_config = _get_task_config(
        "renumber_existing_challenge_set",
        {
            "challenge_set_json_path": challenge_set_json_path,
            "track_df_path": track_df_path,
            "output_path": output_path,
        },
    )
    job_config = _get_job_config([task_config], "2Gi", "1")
    return job_config


def create_contest(
    standardized_input_directory,
    train_parts,
    track_df_output_path,
    train_output_directory,
    test_part,
    n_test_cases,
    challenge_set_output_path,
):

    return _combine_job_configs(
        [
            reduce_track_dfs(
                standardized_input_directory, train_parts, track_df_output_path
            ),
            renumber_track_ids(
                standardized_input_directory,
                train_parts,
                track_df_output_path,
                train_output_directory,
            ),
            create_challenge_set(
                standardized_input_directory,
                test_part,
                track_df_output_path,
                n_test_cases,
                challenge_set_output_path,
            ),
        ]
    )


### Co-occurrence Matrix Functions ###


def generate_co_dicts(
    train_directory, challenge_df_paths, partial_co_dict_output_directory
):
    NUM_SHARDS = 10

    # List files in the training directory
    filenames = fs.get_filenames_in_directory(train_directory)
    filenames = [
        f for f in filenames if f.startswith("part") and f.endswith("playlists.parquet")
    ]
    playlist_paths = [fs.join(train_directory, filename) for filename in filenames]
    part_ids = [int(f.split("_")[1]) for f in filenames]

    # Generate a task per part per co-occurrence type
    task_configs = []
    for co_dict_type, count_type in [
        ("total", "track"),
        ("forward", None),
        ("forward_10", None),
        ("forward_10_pair", "pair"),
        ("forward_10_triple", "triple"),
        ("ngram_track", "ngram_count"),
    ]:
        output_directory = fs.join(partial_co_dict_output_directory, co_dict_type)
        output_directory_count = (
            fs.join(partial_co_dict_output_directory, count_type)
            if count_type
            else None
        )
        for task_idx in range(len(playlist_paths)):
            tc = _get_task_config(
                "generate_co_dicts",
                {
                    "playlist_path": playlist_paths[task_idx],
                    "challenge_df_paths": challenge_df_paths,
                    "part_id": part_ids[task_idx],
                    "co_dict_type": co_dict_type,
                    "num_shards": NUM_SHARDS,
                    "output_directory": output_directory,
                    "output_directory_count": output_directory_count,
                },
            )
            task_configs.append(tc)

    job_config = _get_job_config(task_configs, "4500Mi", "1")
    return job_config


def reduce_co_partials(partial_co_dict_directory, co_dict_output_directory):
    subfolders = fs.get_filenames_in_directory(partial_co_dict_directory)
    job_configs = []
    for subfolder in subfolders:
        task_configs = []
        shards = {}
        for f in fs.get_filenames_in_directory(
            fs.join(partial_co_dict_directory, subfolder)
        ):
            if "shard" in f:
                shard_id = int(f.split("_")[1])
            else:
                shard_id = "all"

            if shard_id not in shards:
                shards[shard_id] = []
            shards[shard_id].append(f)
            ext = f.split(".")[-1]
        task_configs.extend(
            [
                _get_task_config(
                    "reduce_co_partials",
                    {
                        "input_paths": [
                            fs.join(partial_co_dict_directory, subfolder, f)
                            for f in shard_files
                        ],
                        "output_path": fs.join(
                            co_dict_output_directory,
                            subfolder,
                            f"shard_{shard_id}.{ext}",
                        ),
                    },
                )
                for shard_id, shard_files in shards.items()
            ]
        )
        mem = "2Gi"
        n_parts = len(task_configs[0]["args"]["input_paths"])
        if n_parts > 10:
            if subfolder == "total":
                mem = "14.7Gi"
            elif subfolder == "forward":
                mem = "9.5Gi"
            elif subfolder == "forward_10":
                mem = "4.5Gi"
            elif subfolder == "ngram_track":
                mem = "4.5Gi"
        else:
            if subfolder == "total":
                mem = "6.5Gi"
            elif subfolder == "forward":
                mem = "6.5Gi"
            elif subfolder == "forward_10":
                mem = "4.5Gi"
            elif subfolder == "ngram_track":
                mem = "4.5Gi"
        job_config = _get_job_config(task_configs, mem, "1")
        job_configs.append(job_config)

    return job_configs


def evaluate_submission(
    challenge_df_path, track_df_path, submission_directory, output_directory
):
    # Make a single task to evaluate the submission
    task_config = _get_task_config(
        "evaluate_submission",
        {
            "challenge_df_path": challenge_df_path,
            "track_df_path": track_df_path,
            "submission_directory": submission_directory,
            "output_directory": output_directory,
        },
    )
    job_config = _get_job_config([task_config], "4Gi", "1")
    return job_config


def identify_artist_playlists(challenge_df_path, track_df_path, output_path):
    task_config = _get_task_config(
        "identify_artist_playlists",
        {
            "challenge_df_path": challenge_df_path,
            "track_df_path": track_df_path,
            "output_path": output_path,
        },
    )
    job_config = _get_job_config([task_config], "4Gi", "1")
    return job_config


def generate_generic_features_fcnn_mfe(
    challenge_df_path,
    track_df_path,
    artist_playlists_path,
    co_dict_directory,
    output_directory,
):
    SHARD_SIZE = 5000
    challenge_df = fs.load_data(challenge_df_path)
    NUM_SHARDS = math.ceil(len(challenge_df) / SHARD_SIZE)
    shard_ranges = [
        [i * SHARD_SIZE, (i + 1) * SHARD_SIZE - 1] for i in range(NUM_SHARDS)
    ]
    shard_ranges[-1][1] = len(challenge_df) - 1
    output_paths = [
        fs.join(output_directory, f"part_{i}_generic_features.parquet")
        for i in range(len(shard_ranges))
    ]
    ngrams_total_path = fs.join(co_dict_directory, "ngram_count", "shard_all.dnl")

    task_configs = [
        _get_task_config(
            "generate_generic_features_fcnn_mfe",
            {
                "challenge_df_path": challenge_df_path,
                "track_df_path": track_df_path,
                "artist_playlists_path": artist_playlists_path,
                "ngrams_total_path": ngrams_total_path,
                "output_path": output_paths[i],
                "shard_range": sr,
            },
        )
        for i, sr in enumerate(shard_ranges)
    ]
    job_config = _get_job_config(task_configs, "4Gi", "1")
    return job_config


def generate_track_features_fcnn_mfe(
    challenge_df_path,
    track_df_path,
    challenge_df_generic_features_directory,
    co_dict_directory,
    output_directory,
):
    # Create random shards, since the challenge df is not shuffled
    # and some challenge types use much more memory than others
    SHARD_SIZE = 1000
    challenge_df = fs.load_data(challenge_df_path)
    row_order = np.random.choice(
        range(len(challenge_df)), size=len(challenge_df), replace=False
    ).tolist()
    shards = _list_of_lists(row_order, SHARD_SIZE)

    output_paths = [
        fs.join(output_directory, f"part_{i}_track_features.parquet")
        for i in range(len(shards))
    ]
    co_paths = {}
    for co_type in [
        "total",
        "forward",
        "forward_10",
        "forward_10_pair",
        "forward_10_triple",
        "ngram_track",
        "track",
        "ngram_count",
        "pair",
        "triple",
    ]:
        directory = fs.join(co_dict_directory, co_type)
        fns = fs.get_filenames_in_directory(directory)
        paths = [fs.join(directory, fn) for fn in fns]
        co_paths[co_type] = paths

    challenge_df_generic_features_filenames = fs.get_filenames_in_directory(
        challenge_df_generic_features_directory
    )
    challenge_df_generic_features_filenames = [
        f
        for f in challenge_df_generic_features_filenames
        if f.startswith("part") and f.endswith("generic_features.parquet")
    ]
    challenge_df_generic_features_paths = [
        fs.join(challenge_df_generic_features_directory, f)
        for f in challenge_df_generic_features_filenames
    ]

    task_configs = [
        _get_task_config(
            "generate_track_features_fcnn_mfe",
            {
                "challenge_df_path": challenge_df_path,
                "track_df_path": track_df_path,
                "challenge_df_generic_features_paths": challenge_df_generic_features_paths,
                "co_paths": co_paths,
                "output_path": output_paths[i],
                "shard": shard,
            },
        )
        for i, shard in enumerate(shards)
    ]
    track_df = fs.load_data(track_df_path)
    n_tracks = len(track_df)
    mem = "6.5Gi"
    if n_tracks > 1_000_000:
        mem = "9.5Gi"
    job_config = _get_job_config(task_configs, mem, "1")
    return job_config
