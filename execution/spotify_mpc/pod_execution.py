import pandas as pd

from generic import file_system_util

from spotify_mpc.logic import contest_data
from spotify_mpc.logic import cooccurrence
from spotify_mpc.logic import lexicon
from spotify_mpc.logic import fcnn_mfe


def standardize_data(input_paths, playlist_output_path, track_df_output_path):
    print("Executing standardize_data...")
    dfs = []
    for path in input_paths:
        input_data = file_system_util.load_data(path)
        input_data = input_data["playlists"]
        input_data = pd.DataFrame(input_data)
        dfs.append(input_data)
    playlist_df, track_df = contest_data.standardize_data(dfs)
    file_system_util.save_data(playlist_df, playlist_output_path)
    file_system_util.save_data(track_df, track_df_output_path)
    print("standardize_data complete.")


def reduce_track_dfs(input_paths, output_path):
    print("Executing reduce_track_dfs...")
    track_dfs = [file_system_util.load_data(path) for path in input_paths]
    track_df = contest_data.reduce_track_dfs(track_dfs)
    file_system_util.save_data(track_df, output_path)
    print("reduce_track_dfs complete.")


def renumber_track_ids(input_path, track_df_path, output_path):
    print("Executing renumber_track_ids...")
    playlist_df = file_system_util.load_data(input_path)
    track_df = file_system_util.load_data(track_df_path, columns=["track_uri"])
    playlist_df = contest_data.renumber_track_ids(playlist_df, track_df)
    file_system_util.save_data(playlist_df, output_path)
    print("renumber_track_ids complete.")


def create_challenge_set(
    standardized_playlist_path, track_df_path, n_test_cases, output_path
):
    print("Executing create_challenge_set...")
    playlist_df = file_system_util.load_data(standardized_playlist_path)
    track_df = file_system_util.load_data(track_df_path)
    challenge_set = contest_data.create_challenge_set(
        playlist_df, track_df, n_test_cases
    )
    file_system_util.save_data(challenge_set, output_path)
    print("create_challenge_set complete.")


def renumber_existing_challenge_set(
    challenge_set_json_path, track_df_path, output_path
):
    print("Executing renumber_existing_challenge_set...")
    challenge_set_json = file_system_util.load_data(challenge_set_json_path)
    track_df = file_system_util.load_data(track_df_path)
    challenge_set = contest_data.renumber_existing_challenge_set(
        challenge_set_json, track_df
    )
    file_system_util.save_data(challenge_set, output_path)
    print("renumber_existing_challenge_set complete.")


def generate_co_dicts(
    playlist_path,
    challenge_df_paths,
    part_id,
    co_dict_type,
    num_shards,
    output_directory,
    output_directory_count,
):
    # Takes in N files, outputs 10N files
    print(f"Executing generate_co_dicts for type {co_dict_type} part {part_id}...")
    playlist_df = file_system_util.load_data(playlist_path)
    challenge_df = pd.concat(
        file_system_util.data_generator(
            challenge_df_paths, columns=["tracks", "title", "challenge_type"]
        )
    )
    co_dicts, count_dict = cooccurrence.generate_co_dicts(
        playlist_df, challenge_df, num_shards, co_dict_type
    )
    for shard_id, co_dict in enumerate(co_dicts):
        file_system_util.save_data(
            co_dict,
            file_system_util.join(
                output_directory,
                f"shard_{shard_id}_part_{part_id}.ddnl",
            ),
        )
    if output_directory_count is not None:
        file_system_util.save_data(
            count_dict,
            file_system_util.join(output_directory_count, f"part_{part_id}.dnl"),
        )
    print("generate_co_partial complete.")


def reduce_co_partials(input_paths, output_path):
    print(f"Executing generate_co_dicts for {output_path}...")
    print("Executing reduce_co_partials...")
    partials = file_system_util.data_generator(input_paths)
    if output_path.endswith(".ddnl"):
        out = cooccurrence.reduce_co_dicts(partials)
    elif output_path.endswith(".dnl"):
        out = cooccurrence.reduce_count_dicts(partials)
    file_system_util.save_data(out, output_path)
    print("reduce_co_partials complete.")


def evaluate_submission(
    challenge_df_path, track_df_path, submission_directory, output_directory
):
    print("Executing evaluate_submission...")
    challenge_df = file_system_util.load_data(challenge_df_path)
    track_df = file_system_util.load_data(track_df_path)
    submission_paths = [
        file_system_util.join(submission_directory, fn)
        for fn in file_system_util.get_filenames_in_directory(submission_directory)
    ]
    submission_generator = file_system_util.data_generator(submission_paths)
    detail_df, summary_df = contest_data.evaluate_submission(
        challenge_df, track_df, submission_generator
    )
    file_system_util.save_data(
        detail_df, file_system_util.join(output_directory, "detail.parquet")
    )
    file_system_util.save_data(
        summary_df, file_system_util.join(output_directory, "summary.parquet")
    )
    summary_md = summary_df.to_markdown(index=False)
    file_system_util.save_data(
        summary_md, file_system_util.join(output_directory, "summary.md")
    )
    print("evaluate_submission complete.")


def identify_artist_playlists(challenge_df_path, track_df_path, output_path):
    print("Executing identify_artist_playlists...")
    challenge_df = file_system_util.load_data(challenge_df_path)
    track_df = file_system_util.load_data(track_df_path)
    artist_playlists = lexicon.identify_artist_playlists(challenge_df, track_df)
    file_system_util.save_data(artist_playlists, output_path)
    print("identify_artist_playlists complete.")


def generate_generic_features_fcnn_mfe(
    challenge_df_path,
    track_df_path,
    artist_playlists_path,
    ngrams_total_path,
    output_path,
    shard_range,
):
    print("Executing generate_generic_features_fcnn_mfe...")
    challenge_df = file_system_util.load_data(challenge_df_path)
    challenge_df = challenge_df.iloc[shard_range[0] : shard_range[1] + 1]
    track_df = file_system_util.load_data(track_df_path)
    artist_playlists = file_system_util.load_data(artist_playlists_path)
    ngrams_total = file_system_util.load_data(ngrams_total_path)
    generic_features = fcnn_mfe.generate_generic_features_fcnn_mfe(
        challenge_df, track_df, artist_playlists, ngrams_total
    )
    file_system_util.save_data(generic_features, output_path)
    print("generate_generic_features_fcnn_mfe complete.")


def generate_track_features_fcnn_mfe(
    challenge_df_path,
    track_df_path,
    challenge_df_generic_features_paths,
    co_paths,
    output_path,
    shard,
):
    print("Executing generate_track_features_fcnn_mfe...")
    challenge_df = file_system_util.load_data(challenge_df_path)
    challenge_df = challenge_df.iloc[shard]
    track_df = file_system_util.load_data(track_df_path)
    generic_feature_gen = file_system_util.data_generator(
        challenge_df_generic_features_paths
    )

    co_gens = {
        co_type: file_system_util.data_generator(paths)
        for co_type, paths in co_paths.items()
    }
    track_features = fcnn_mfe.generate_track_features_fcnn_mfe(
        challenge_df, track_df, generic_feature_gen, co_gens
    )
    file_system_util.save_data(track_features, output_path)
    print("generate_track_features_fcnn_mfe complete.")
