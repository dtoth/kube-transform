import pandas as pd
import numpy as np
import random


def standardize_data(playlist_dfs):
    """
    Standardize the data by removing uneccessary fields, but keep the track URIs.
    Return a playlist dataframe and a track dataframe.
    """
    # Concatenate the list of dataframes
    playlist_df = pd.concat(playlist_dfs)
    playlist_dfs = None

    # Drop unnecessary fields
    playlist_df = playlist_df[["pid", "name", "modified_at", "tracks"]].reset_index(
        drop=True
    )

    # Create a dataframe of unique tracks
    track_df = (
        pd.DataFrame([t for lst in playlist_df.tracks.values for t in lst])
        .drop_duplicates(subset="track_uri")
        .reset_index(drop=True)[
            [
                "track_name",
                "artist_name",
                "album_name",
                "duration_ms",
                "track_uri",
                "artist_uri",
                "album_uri",
            ]
        ]
    )

    # Convert the 'tracks' column to a list of track URIs
    playlist_df["tracks"] = playlist_df["tracks"].apply(
        lambda x: [t["track_uri"] for t in x]
    )

    # Add popularity data to the track dataframe
    pids, counts = np.unique(
        np.concatenate(playlist_df.tracks.values), return_counts=True
    )
    count_dict = dict(zip(pids, counts))
    track_df["occurrence_count"] = track_df.track_uri.map(count_dict)

    # Add last modified data to the playlist dataframe
    modified_at_map = (
        playlist_df.explode("tracks").groupby("tracks").modified_at.mean().to_dict()
    )
    track_df["mean_modified_at"] = track_df.track_uri.map(modified_at_map)
    track_df["mean_modified_at"] = (track_df.mean_modified_at / 3600 / 24 / 365.25) - 39

    return playlist_df, track_df


def reduce_track_dfs(track_dfs):
    """
    Reduce the list of track dataframes into a single track dataframe.
    """
    # Concatenate the dataframes
    combined_df = pd.concat(track_dfs)
    combined_df["weighted_mean_modified_at"] = (
        combined_df["occurrence_count"] * combined_df["mean_modified_at"]
    )

    # Identify the columns to keep and those to aggregate
    columns_to_aggregate = ["occurrence_count", "weighted_mean_modified_at"]
    columns_to_keep = combined_df.columns.difference(columns_to_aggregate)

    # Group by 'track_uri' and sum the 'occurrence_count' while keeping the first values for other columns
    result_df = combined_df.groupby("track_uri", as_index=False).agg(
        {
            "occurrence_count": "sum",
            "weighted_mean_modified_at": "sum",
            **{
                col: "first" for col in columns_to_keep
            },  # Apply 'first' to all non-aggregated columns
        }
    )
    result_df["mean_modified_at"] = (
        result_df["weighted_mean_modified_at"] / result_df["occurrence_count"]
    )
    result_df = result_df.drop(columns=["weighted_mean_modified_at"])

    # Reset the index for a clean dataframe
    result_df = result_df.reset_index(drop=True)
    return result_df


def renumber_track_ids(playlist_df, track_df, allow_missing_tracks=False):
    """
    Renumber the track IDs in the playlist dataframe using the track
    dataframe. This function assumes that all track URIs in the
    playlist dataframe are present in the track dataframe.
    """
    # Create a mapping from track_uri to track_id
    track_id_mapping = {uri: i for i, uri in enumerate(track_df.track_uri.values)}
    track_df = None

    # Replace the track URIs with track IDs in the playlist dataframe
    def get_track_number_arr(tracks):
        unseen_count = sum([1 for uri in tracks if uri not in track_id_mapping])
        track_ids = [track_id_mapping[uri] for uri in tracks if uri in track_id_mapping]
        return track_ids, unseen_count

    playlist_df[["tracks", "unseen_track_count"]] = (
        playlist_df["tracks"].apply(get_track_number_arr).apply(pd.Series)
    )

    if not allow_missing_tracks:
        assert playlist_df["unseen_track_count"].sum() == 0
        del playlist_df["unseen_track_count"]

    return playlist_df


def create_challenge_set(playlist_df, track_df, n_test_cases):
    """
    Create a challenge set by selecting n_test_cases playlists from the
    playlist dataframe and removing the tracks from the track dataframe.
    """
    playlist_df = renumber_track_ids(playlist_df, track_df, allow_missing_tracks=True)
    challenge_types = [  # (Include Title, Use First Tracks, Number of Tracks)
        (True, False, 0),  # 0
        (True, True, 1),  # 1
        (True, True, 5),  # 2
        (False, True, 5),  # 3
        (True, True, 10),  # 4
        (False, True, 10),  # 5
        (True, True, 25),  # 6
        (True, False, 25),  # 7
        (True, True, 100),  # 8
        (True, False, 100),  # 9
    ]
    # Target playlists per challenge type
    playlists_per_case = int(n_test_cases / len(challenge_types))

    # Prepare challenge rows
    challenge_rows = []

    # Iterate over challenge types
    for challenge_type, (include_title, use_first_tracks, num_tracks) in enumerate(
        challenge_types
    ):
        # Filter playlists with enough tracks for the current challenge
        filtered_playlists = playlist_df[playlist_df["tracks"].apply(len) > num_tracks]

        # Randomly sample playlists for the challenge
        sampled_playlists = filtered_playlists.sample(
            min(playlists_per_case, len(filtered_playlists)), random_state=42
        )

        # Generate challenge rows
        for idx, playlist in sampled_playlists.iterrows():
            if use_first_tracks:
                # Use the first `num_tracks` tracks
                tracks = playlist["tracks"][:num_tracks]
            else:
                # Randomly select `num_tracks` tracks (without replacement)
                tracks = (
                    random.sample(playlist["tracks"], num_tracks)
                    if num_tracks > 0
                    else []
                )

            challenge_row = {
                "title": playlist["name"] if include_title else None,
                "tracks": tracks,  # Seed tracks
                "hidden_tracks": [
                    t for t in playlist["tracks"] if t not in tracks
                ],  # Remaining tracks
                "unseen_track_count": playlist["unseen_track_count"],
                "challenge_type": challenge_type,
            }
            challenge_rows.append(challenge_row)

    # Convert challenge rows to a DataFrame for easier manipulation
    challenge_df = pd.DataFrame(challenge_rows)
    return challenge_df


def renumber_existing_challenge_set(challenge_df_json, track_df):
    print("started...")
    track_id_dict = {v: k for k, v in track_df["track_uri"].to_dict().items()}

    df = pd.DataFrame(challenge_df_json["playlists"])
    df["include_title"] = pd.notnull(df.name)
    df["use_first_tracks"] = df.apply(
        lambda r: [t["pos"] for t in r.tracks] != list(range(len(r.tracks))), axis=1
    )
    df.loc[df.num_samples == 0, "challenge_type"] = 0
    df.loc[df.num_samples == 1, "challenge_type"] = 1
    df.loc[(df.num_samples == 5) & (df.include_title == True), "challenge_type"] = 2
    df.loc[(df.num_samples == 5) & (df.include_title == False), "challenge_type"] = 3
    df.loc[(df.num_samples == 10) & (df.include_title == True), "challenge_type"] = 4
    df.loc[(df.num_samples == 10) & (df.include_title == False), "challenge_type"] = 5
    df.loc[(df.num_samples == 25) & (df.use_first_tracks == True), "challenge_type"] = 6
    df.loc[
        (df.num_samples == 25) & (df.use_first_tracks == False), "challenge_type"
    ] = 7
    df.loc[
        (df.num_samples == 100) & (df.use_first_tracks == True), "challenge_type"
    ] = 8
    df.loc[
        (df.num_samples == 100) & (df.use_first_tracks == False), "challenge_type"
    ] = 9
    df["challenge_type"] = df["challenge_type"].astype(int)

    df["tracks"] = df.tracks.apply(
        lambda ts: [
            track_id_dict.get(t["track_uri"])
            for t in ts
            if t["track_uri"] in track_id_dict
        ]
    )
    df["acutal_num_samples"] = df.tracks.apply(len)
    df["unseen_track_count"] = df.num_samples - df.acutal_num_samples

    df = df[["pid", "name", "tracks", "unseen_track_count", "challenge_type"]].rename(
        columns={"name": "title"}
    )
    df["hidden_tracks"] = df.apply(lambda r: [], axis=1)
    print("finished...")
    return df


def evaluate_submission(challenge_df, track_df, submission_generator):
    # Add submission details to the challenge dataframe
    submission_dfs = []
    for submission_df in submission_generator:
        submission_dfs.append(submission_df)
    submission_df = pd.concat(submission_dfs)
    challenge_df = pd.merge(
        challenge_df, submission_df, left_index=True, right_index=True, how="left"
    )

    # TO OPTIMIZE: Fix contest creation so this doesn't happen
    challenge_df = challenge_df[challenge_df.hidden_tracks.apply(lambda x: len(x) > 0)]

    # Evaluate the submission
    def get_artists(track_ids):
        artists = track_df.iloc[track_ids].artist_uri.values
        return artists

    challenge_df["artist_uris_hidden"] = challenge_df.hidden_tracks.apply(get_artists)
    challenge_df["artist_uris_suggested"] = challenge_df.suggested.apply(get_artists)

    def track_hit_count(row):
        hidden = row.hidden_tracks
        hidden_count = row.unseen_track_count + len(hidden)
        predicted = row.suggested[:hidden_count]
        return len(set([t for t in hidden if t in predicted]))

    def artist_hit_count(row):
        hidden = row.artist_uris_hidden
        hidden_count = row.unseen_track_count + len(hidden)
        predicted = row.artist_uris_suggested[:hidden_count]
        return len(set([t for t in hidden if t in predicted]))

    challenge_df["track_hit_count"] = challenge_df.apply(track_hit_count, axis=1)
    challenge_df["artist_hit_count"] = challenge_df.apply(artist_hit_count, axis=1)
    challenge_df["hidden_count"] = challenge_df.hidden_tracks.apply(
        lambda x: len(set(x))
    )
    challenge_df["r_prec"] = (
        challenge_df.track_hit_count + 0.25 * challenge_df.artist_hit_count
    ) / (challenge_df.hidden_count + challenge_df.unseen_track_count)

    def ndcg(row):
        hidden = row.hidden_tracks
        predicted = row.suggested
        dcg = 0
        idcg = 0
        for i, track in enumerate(predicted):
            if track in hidden:
                dcg += 1 / np.log2(i + 2)
        for i in range(len(hidden)):
            idcg += 1 / np.log2(i + 2)
        return dcg / idcg

    challenge_df["ndcg"] = challenge_df.apply(ndcg, axis=1)

    def click_count(row):
        hidden = row.hidden_tracks
        predicted = row.suggested
        for i, track in enumerate(predicted):
            if track in hidden:
                return i // 10
        return 51

    challenge_df["click_count"] = challenge_df.apply(click_count, axis=1)

    # Summarize
    score_df = challenge_df.groupby("challenge_type")[
        ["r_prec", "ndcg", "click_count"]
    ].mean()
    all_mean = challenge_df[["r_prec", "ndcg", "click_count"]].mean()
    all_row = pd.DataFrame([all_mean], index=["all"])
    score_df = pd.concat([score_df, all_row])
    score_df = score_df.reset_index(names="challenge_type")
    score_df["challenge_type"] = score_df["challenge_type"].astype("str")

    return challenge_df, score_df
