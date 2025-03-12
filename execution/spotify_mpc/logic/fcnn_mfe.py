"""This module contains the logic for generating features for the FCNN MFE model.
This stands for "Fully Connected Neural Network - Manual Feature Engineering".
"""

import pandas as pd
from spotify_mpc.logic.lexicon import _normalize_text
from spotify_mpc.logic.ngram import generate_char_ngrams
import numpy as np
from spotify_mpc.logic.cooccurrence import _hash_array


def generate_generic_features_fcnn_mfe(
    challenge_df, track_df, artist_playlists, ngrams_total
):
    """Generate generic features for each row in the challenge_df,
    to be fed into the FCNN MFE model. They are generic in that they
    are not track-specific.
    """
    track_df["artist_name_normalized"] = track_df.artist_name.apply(_normalize_text)

    challenge_df["num_seed_tracks"] = challenge_df.tracks.apply(len)
    challenge_df["title_available"] = challenge_df.challenge_type.apply(
        lambda x: x not in [3, 5]
    )
    challenge_df["seed_tracks_randomized"] = challenge_df.challenge_type.apply(
        lambda x: x in [7, 9]
    )

    challenge_df["ngrams"] = challenge_df.title.apply(generate_char_ngrams)
    challenge_df["num_ngrams"] = challenge_df.ngrams.apply(len)

    challenge_df["last_seed"] = challenge_df.tracks.apply(
        lambda x: x[-1] if len(x) > 0 else None
    )
    challenge_df["last_seed_pair"] = challenge_df.tracks.apply(
        lambda x: _hash_array(x[-2:]) if len(x) > 1 else None
    )
    challenge_df["last_seed_triple"] = challenge_df.tracks.apply(
        lambda x: _hash_array(x[-3:]) if len(x) > 2 else None
    )

    challenge_df["mean_seed_track_occurrences"] = challenge_df.tracks.apply(
        lambda x: (np.mean(track_df.loc[x].occurrence_count) if len(x) > 0 else 0)
    )

    challenge_df["mean_ngram_occurrences"] = challenge_df.ngrams.apply(
        lambda ngram_list: (
            np.mean([ngrams_total.get(ng, 0) for ng in ngram_list])
            if len(ngram_list) > 0
            else 0
        )
    )

    challenge_df["is_artist_playlist"] = challenge_df.index.isin(artist_playlists)
    return challenge_df


def _get_fast_access(co_dicts, co_type, normalizer_type):
    BUFFER = 5
    co_ids = {k: np.array(list(v.keys())) for k, v in co_dicts[co_type].items()}
    co_scores = {
        k: np.array(list(v.values())) / (co_dicts[normalizer_type][k] + BUFFER)
        for k, v in co_dicts[co_type].items()
    }
    return co_ids, co_scores


def _get_score_dict(fast_access, co_type, seeds):
    co_ids = fast_access[co_type][0]
    co_scores = fast_access[co_type][1]
    seed_co_ids = (
        np.concatenate([co_ids.get(st, np.array([])) for st in seeds])
        if len(seeds) > 0
        else np.array([])
    )
    seed_co_scores = (
        np.concatenate([co_scores.get(st, np.array([])) for st in seeds])
        if len(seeds) > 0
        else np.array([])
    )

    unique_co_ids, indices = np.unique(seed_co_ids, return_inverse=True)

    sum_score = np.zeros_like(unique_co_ids, dtype=float)
    np.add.at(sum_score, indices, seed_co_scores)
    score_dct = {int(k): float(v) for k, v in zip(unique_co_ids, sum_score)}
    return score_dct


def generate_track_features_fcnn_mfe(
    challenge_df,
    track_df,
    generic_feature_gen,
    co_gens,
):
    """Generate candidate tracks for each row in the challenge_df.
    Then, generate features for every <playlist, candidate track> pair.
    """

    # Calculate basic popularity features
    artist_popularity_dict_unnorn = (
        track_df.groupby("artist_uri").occurrence_count.sum()
    ).to_dict()
    track_df["artist_popularity"] = track_df.artist_uri.map(
        artist_popularity_dict_unnorn
    )
    track_id_to_artist_popularity_dict = track_df.artist_popularity.to_dict()
    track_df["artist_name_normalized"] = track_df.artist_name.apply(_normalize_text)
    challenge_df["title_normalized"] = challenge_df.title.apply(_normalize_text)
    popular_1k = list(
        track_df.sort_values("occurrence_count", ascending=False).index[:1000]
    )
    track_id_to_ann = track_df.artist_name_normalized.to_dict()

    # Concatenate all generic features
    full_generic_feature_df = None
    for generic_feature_df in generic_feature_gen:
        if full_generic_feature_df is None:
            full_generic_feature_df = generic_feature_df
        else:
            full_generic_feature_df = pd.concat(
                [full_generic_feature_df, generic_feature_df]
            )

    new_columns = full_generic_feature_df.columns.difference(challenge_df.columns)
    challenge_df = challenge_df.join(full_generic_feature_df[new_columns], how="left")

    print("Creating relevant tracks dicts...")
    # Create a dict containing co-occurrence data for all seed tracks
    relevant_tracks = set(np.concatenate(challenge_df.tracks.values).astype(object))
    relevant_ngrams = set(np.concatenate(challenge_df.ngrams.values).astype(object))
    relevant_last_seeds = set(challenge_df.last_seed.dropna().astype(object))
    relevant_last_seed_pairs = set(challenge_df.last_seed_pair.dropna().astype(object))
    relevant_last_seed_triples = set(
        challenge_df.last_seed_triple.dropna().astype(object)
    )
    co_dicts = {}
    for co_dict_type, co_dict_gen in co_gens.items():
        if co_dict_type in ["ngram_track", "ngram_count"]:
            relevant_ids = relevant_ngrams
        elif co_dict_type in ["total", "forward", "track"]:
            relevant_ids = relevant_tracks
        elif co_dict_type == "forward_10":
            relevant_ids = relevant_last_seeds
        elif co_dict_type in ["forward_10_pair", "pair"]:
            relevant_ids = relevant_last_seed_pairs
        elif co_dict_type in ["forward_10_triple", "triple"]:
            relevant_ids = relevant_last_seed_triples
        else:
            raise ValueError(f"Invalid co_dict_type: {co_dict_type}")
        co_dicts[co_dict_type] = {}
        for co_dict in co_dict_gen:
            co_dict_keys = list(co_dict.keys())
            co_dict_values = list(co_dict.values())
            if co_dict_type in [
                "pair",
                "triple",
                "forward_10_pair",
                "forward_10_triple",
            ]:
                co_dict_keys = np.array([o for o in co_dict_keys]).astype(str)
            co_dicts[co_dict_type].update(
                {
                    k: v
                    for k, v in zip(co_dict_keys, co_dict_values)
                    if k in relevant_ids
                }
            )
    # For the sake of identifying candidates, we'll use the sum of the
    # total co-occurrence ratios.  Let's create an efficient data structure
    # for accessing these values.
    fast_access = {
        co_type: _get_fast_access(co_dicts, co_type, normalizer_type)
        for co_type, normalizer_type in [
            ("total", "track"),
            ("ngram_track", "ngram_count"),
            ("forward", "track"),
            ("forward_10", "track"),
            ("forward_10_pair", "pair"),
            (
                "forward_10_triple",
                "triple",
            ),
        ]
    }

    # Prepare seed track co-occurrence data
    print("Preparing seed track co-occurrence data...")

    # Define a function to generate track-specific features efficiently
    def get_candidate_tracks(row):
        # Popular Tracks
        candidates = set(popular_1k)

        # Artist Tracks (for targeted playlists)
        if row.is_artist_playlist == True:
            artist_tracks = track_df[
                track_df.artist_name_normalized == row.title_normalized
            ]
            candidates.update(list(artist_tracks.index))

        # Get co-occurrence score dicts for the row
        score_dicts = {}
        for co_type in fast_access.keys():
            if co_type in ["total", "forward"]:
                seeds = row.tracks
            elif co_type in ["ngram_track"]:
                seeds = row.ngrams
            elif co_type in ["forward_10"]:
                seeds = [row.last_seed]
            elif co_type in ["forward_10_pair"]:
                seeds = [row.last_seed_pair]
            elif co_type in ["forward_10_triple"]:
                seeds = [row.last_seed_triple]

            score_dicts[co_type] = _get_score_dict(fast_access, co_type, seeds)

        # Co-occurring w/ Seed Tracks
        MAX_CO_SEED_CANDIDATES = 3000
        track_score_dict = score_dicts["total"]
        sorted_track_candidates = sorted(
            track_score_dict, key=track_score_dict.get, reverse=True
        )[:MAX_CO_SEED_CANDIDATES]
        candidates.update(set(sorted_track_candidates))

        # Co-occurring w/ Ngrams
        MAX_CO_NGRAM_CANDIDATES = 3000
        ngram_score_dict = score_dicts["ngram_track"]
        sorted_ngram_candidates = sorted(
            ngram_score_dict, key=ngram_score_dict.get, reverse=True
        )[:MAX_CO_NGRAM_CANDIDATES]
        candidates.update(set(sorted_ngram_candidates))

        # Exclude known tracks
        candidates.difference_update(set(row.tracks))

        cand_scores_tracks_total = [
            score_dicts["total"].get(ct, 0) for ct in candidates
        ]
        cand_scores_tracks_forward = [
            score_dicts["forward"].get(ct, 0) for ct in candidates
        ]
        cand_scores_tracks_forward_10 = [
            score_dicts["forward_10"].get(ct, 0) for ct in candidates
        ]
        cand_scores_tracks_forward_10_pair = [
            score_dicts["forward_10_pair"].get(ct, 0) for ct in candidates
        ]
        cand_scores_tracks_forward_10_triple = [
            score_dicts["forward_10_triple"].get(ct, 0) for ct in candidates
        ]
        cand_scores_ngrams = [
            score_dicts["ngram_track"].get(ct, 0) for ct in candidates
        ]

        cand_track_popularity = [co_dicts["track"].get(ct, 0) for ct in candidates]
        cand_artist_popularity = [
            track_id_to_artist_popularity_dict.get(ct, 0) for ct in candidates
        ]
        cand_is_target_artist = [
            int(
                row.is_artist_playlist == True
                and track_id_to_ann[ct] == row.title_normalized
            )
            for ct in candidates
        ]
        cand_other_is_target_artist = [
            int(
                row.is_artist_playlist == True
                and track_id_to_ann[ct] != row.title_normalized
            )
            for ct in candidates
        ]

        cand_is_hidden_track = [int(ct in row.hidden_tracks) for ct in candidates]

        return zip(
            candidates,
            cand_scores_tracks_total,
            cand_scores_tracks_forward,
            cand_scores_tracks_forward_10,
            cand_scores_tracks_forward_10_pair,
            cand_scores_tracks_forward_10_triple,
            cand_scores_ngrams,
            cand_track_popularity,
            cand_artist_popularity,
            cand_is_target_artist,
            cand_other_is_target_artist,
            cand_is_hidden_track,
        )

    print("Generating candidate tracks...")
    # Apply the function to each row in the challenge dataframe
    challenge_df["candidate_tracks"] = challenge_df.apply(get_candidate_tracks, axis=1)

    # Thin out the challenge_df
    challenge_df = challenge_df[
        [
            c
            for c in challenge_df.columns
            if c
            not in [
                "tracks",
                "hidden_tracks",
                "ngrams",
                "title",
                "unseen_track_count",
                "is_artist_playlist",
                "title_normalized",
            ]
        ]
    ]
    challenge_df["candidate_tracks"] = challenge_df["candidate_tracks"].apply(list)

    # Explode the df to create one column per candidate track
    print("Exploding candidate tracks...")
    challenge_df = challenge_df.explode("candidate_tracks")

    print("Concatenating DFs...")

    column_names = [
        "candidate_track_id",
        "track_cooccurrence_score",
        "track_forward_cooccurrence_score",
        "track_forward_10_cooccurrence_score",
        "track_forward_10_pair_cooccurrence_score",
        "track_forward_10_triple_cooccurrence_score",
        "ngram_cooccurrence_score",
        "candidate_track_popularity",
        "candidate_artist_popularity",
        "is_target_artist",
        "other_is_target_artist",
        "is_hidden_track",
    ]
    candidate_tracks_df = pd.DataFrame(
        challenge_df["candidate_tracks"].tolist(), columns=column_names
    )

    if "pid" not in challenge_df.columns:
        challenge_df.reset_index(drop=False, inplace=True, names="pid")
    else:
        challenge_df.reset_index(drop=True, inplace=True)

    challenge_df = pd.concat(
        [challenge_df.drop(columns=["candidate_tracks"]), candidate_tracks_df], axis=1
    )

    return challenge_df
