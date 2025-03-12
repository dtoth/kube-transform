import pandas as pd
import heapq
import numpy as np
from numpy.lib.stride_tricks import as_strided
import hashlib
from functools import reduce
from collections import defaultdict
from spotify_mpc.logic.ngram import generate_char_ngrams


def generate_co_dicts(df, challenge_df, n_shards, co_dict_type="total"):
    """
    Create a sparse co-occurrence matrix from a `df` of playlists.
    The matrix will be split into multiple parts based on track ranges.
    Only tracks/pairs/triples/ngrams in the challenge df will be considered
    (to reduce computation time).
    Returns:
        - co_dicts:
            - A list of dictionaries, where each dictionary
              contains the co-occurrences for a given track shard.
        - count_dict:
            - A dictionary containing the total number of
              occurrences of each seed track.
    The following co-occurrence types are supported:
        - total:
            - Total co-occurrences
        - forward:
            - Only consider occurrences where the candidate track
              follows the seed track.
        - forward_10:
            - Only consider occurrences where the candidate track
              follows the seed track within 10 tracks.
        - forward_10_pair:
            - Only consider occurrences where the candidate track
              follows the seed pair within 10 tracks.
              Will be computed for all sequential pairs of seed tracks
              as represented by _hash_array([first_seed, second_seed]).
        - forward_10_triple:
            - Only consider occurrences where the candidate track
              follows the seed triple within 10 tracks.
              Will be computed for all sequential triples of seed tracks
              as represented by _hash_array([first_seed, second_seed, third_seed]).
        - ngram_track:
            - Co-occurrences between an ngram in a playlist title, and a track.
    """

    # Prep work
    print("Starting initial prep...")
    df["tracks"] = df.tracks.apply(_remove_duplicates)
    df["track_id"] = df.tracks

    # Select IDs based on co-occurrence type
    if co_dict_type == "total":
        id_col = "track_id"
        others_col = "tracks"
    elif co_dict_type == "forward":
        id_col = "track_id"
        others_col = "forward_tracks"
    elif co_dict_type == "forward_10":
        id_col = "track_id"
        others_col = "forward_10_tracks"
    elif co_dict_type == "forward_10_pair":
        id_col = "track_pair_hash"
        others_col = "forward_10_tracks"
    elif co_dict_type == "forward_10_triple":
        id_col = "track_triple_hash"
        others_col = "forward_10_tracks"
    elif co_dict_type == "ngram_track":
        id_col = "ngram"
        others_col = "tracks"
    else:
        raise ValueError(f"Invalid co_dict_type: {co_dict_type}")

    # Create pair/triple hash and ngram columns if necessary, and identify relevant IDs
    if co_dict_type == "forward_10":
        challenge_df = challenge_df[~challenge_df.challenge_type.isin([0, 7, 9])]
        relevant_ids = np.unique(np.concatenate(challenge_df.tracks.values))
    elif co_dict_type == "forward_10_pair":
        df["track_pair_hash"] = df.tracks.apply(lambda x: _subsequences(x, 2))
        # Exclude challenge types with randomized track order or <2 tracks
        challenge_df = challenge_df[~challenge_df.challenge_type.isin([0, 1, 7, 9])]
        challenge_df["track_pair_hash"] = challenge_df.tracks.apply(
            lambda x: _hash_array(x[-2:])
        )
        relevant_ids = np.unique(challenge_df.track_pair_hash.values)
    elif co_dict_type == "forward_10_triple":
        df["track_triple_hash"] = df.tracks.apply(lambda x: _subsequences(x, 3))
        challenge_df = challenge_df[~challenge_df.challenge_type.isin([0, 1, 7, 9])]
        challenge_df["track_triple_hash"] = challenge_df.tracks.apply(
            lambda x: _hash_array(x[-3:])
        )
        relevant_ids = np.unique(challenge_df.track_triple_hash.values)
    elif co_dict_type == "ngram_track":
        df["ngram"] = df.name.apply(generate_char_ngrams)
        challenge_df["ngram"] = challenge_df.title.apply(generate_char_ngrams)
        relevant_ids = np.unique(np.concatenate(challenge_df.ngram.values))
    else:
        relevant_ids = np.unique(np.concatenate(challenge_df.tracks.values))

    # Explode the df and filter out irrelevant tracks
    print("Exploding and filtering the dataframe...")
    df = _explode_with_position_and_length(df, id_col)
    df = df[df[id_col].isin(relevant_ids)]

    # Additional prep work based on co-occurrence type
    print("Adding additional columns if needed...")
    if co_dict_type == "forward":
        df["forward_tracks"] = df.apply(_get_forward_tracks, axis=1)
    elif co_dict_type == "forward_10":
        df["forward_10_tracks"] = df.apply(_get_forward_10_tracks, axis=1)
    elif co_dict_type == "forward_10_pair":
        df["forward_10_tracks"] = df.apply(_get_forward_10_tracks_after_pair, axis=1)
    elif co_dict_type == "forward_10_triple":
        df["forward_10_tracks"] = df.apply(_get_forward_10_tracks_after_triple, axis=1)

    # Count the total number of occurrences of each seed
    # This is small, so doesn't need to be sharded
    print("Counting tokens...")
    count_dict = dict(df[id_col].value_counts())

    # Split the df into multiple parts based on track ranges
    print("Splitting DF into shards...")
    df["shard"] = df[id_col].apply(_get_shard, n_shards=n_shards)
    dfs = [df[df.shard == i] for i in range(n_shards)]

    # Compute co-occurrence dictionaries
    print("Computing co-occurrence dicts...")
    co_dicts = [_get_cooccurrence_dict(d, id_col, others_col=others_col) for d in dfs]
    print("Done computing co-occurrence dicts.")
    return (co_dicts, count_dict)


def reduce_co_dicts(co_dicts_for_given_track_shard, n=1000):
    """
    Sum an iterable of COO matrices for a given track shard.
    Then, identify the top N co-occurring tracks for each seed track.
    """
    print("Reducing co-occurrence matrices.")
    out = reduce(_merge_nested_dicts, co_dicts_for_given_track_shard)
    print("Done reducing co-occurrence matrices.")
    print("Identifying top co-occurring tracks...")
    for idx, item in enumerate(out.items()):
        track_id, other_tracks = item
        top_n_items = heapq.nlargest(n, other_tracks.items(), key=lambda x: x[1])
        out[track_id] = dict(top_n_items)
        if idx % 1000 == 0:
            print(f"Processed {idx}/{len(out)} keys.")
    print("Done identifying top co-occurring tracks.")
    return out


def reduce_count_dicts(count_dicts):
    """
    Sum an iterable of count dicts.
    """
    print("Reducing count matrices.")
    out = reduce(_merge_dicts, count_dicts)
    print("Done reducing count matrices.")
    return out


### Helper Functions ###


def _get_shard(key, n_shards):
    if isinstance(key, str):
        key = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return key % n_shards


def _hash_array(arr, hash_function="md5"):
    hasher = hashlib.new(hash_function)
    hasher.update(arr.tobytes())
    return hasher.hexdigest()[:16]


def _subsequences(lst, m):
    if type(lst) == list:
        arr = np.array(lst)
    elif type(lst) == np.ndarray:
        arr = lst
    n = arr.size - m + 1
    s = arr.itemsize
    strides = as_strided(arr, shape=(m, n), strides=(s, s)).T
    return [_hash_array(i) for i in strides]


def _explode_with_position_and_length(df, column):
    # Reset index to preserve original row identifiers
    df_reset = df.reset_index()

    # Compute original list length
    df_reset["original_length"] = df_reset[column].apply(len)

    # Explode the specified column while keeping track of the original row
    df_exploded = df_reset.explode(
        column, ignore_index=False
    )  # Keep the original index

    # Compute position within each original list
    df_exploded["position"] = df_exploded.groupby(df_exploded.index).cumcount()

    # Reset index to a sequential index
    df_exploded = df_exploded.reset_index(drop=True)

    return df_exploded


def _remove_duplicates(lst):
    seen = set()
    return [x for x in lst if not (x in seen or seen.add(x))]


def _get_forward_tracks(row):
    return row.tracks[row.position + 1 :]


def _get_forward_10_tracks(row):
    return row.tracks[row.position + 1 : row.position + 11]


def _get_forward_10_tracks_after_pair(row):
    return row.tracks[row.position + 2 : row.position + 12]


def _get_forward_10_tracks_after_triple(row):
    return row.tracks[row.position + 3 : row.position + 13]


def _merge_nested_dicts(d1, d2):
    """Merge two nested dicts by summing values for matching keys."""
    merged = defaultdict(dict, d1)  # Start with d1

    for key, subdict in d2.items():
        if key in merged:
            for subkey, value in subdict.items():
                merged[key][subkey] = merged[key].get(subkey, 0) + value
        else:
            merged[key] = subdict  # Add new key from d2

    return dict(merged)  # Convert back to a regular dict


def _merge_dicts(d1, d2):
    """Merge two dicts by summing values for matching keys."""
    merged = d1.copy()

    for key, value in d2.items():
        if key in merged:
            merged[key] += value
        else:
            merged[key] = value

    return merged


def _get_cooccurrence_dict(df, id_col, others_col):
    out = {}
    for t, grp in df.groupby(id_col):
        out[t] = dict(
            zip(*np.unique(np.concatenate(grp[others_col].values), return_counts=True))
        )
    return out
