import string
import unicodedata
import nltk
import string
from nltk.corpus import wordnet
import unicodedata


def identify_artist_playlists(challenge_df, track_df):
    """Identify playlists that are likely to be artist playlists."""
    # Download necessary NLTK data
    nltk.download("wordnet")

    # PID to normalized playlist name
    playlists = challenge_df.title.apply(_normalize_text).to_dict()

    # Get relevant artist names
    total_track_occurrences = track_df.occurrence_count.sum()
    track_df["artist_name_normalized"] = track_df.artist_name.apply(_normalize_text)
    artist_pop_dict = track_df.groupby("artist_uri").occurrence_count.sum().to_dict()
    track_df["artist_popularity_ratio"] = track_df.artist_uri.apply(
        lambda x: artist_pop_dict[x] / total_track_occurrences
    )
    THREHSOLD = 1 / 2000 / 100
    relevant_artist_names = track_df[
        track_df.artist_popularity_ratio > THREHSOLD
    ].artist_name_normalized.unique()

    # Classify playlists
    artist_playlists = []
    for pid, playlist_name in playlists.items():
        if len(playlist_name) < 4:
            continue
        if _is_common_phrase(playlist_name):
            continue
        elif _is_artist_playlist(playlist_name, relevant_artist_names):
            artist_playlists.append(pid)  # It's an artist playlist
        else:
            pass

    return artist_playlists


### Helper Functions ###


def _normalize_text(text):
    if text is None:
        return ""

    # Normalize the text (decompose characters and remove diacritics)
    text = unicodedata.normalize("NFD", text)
    text = "".join(
        [c for c in text if unicodedata.category(c) != "Mn"]
    )  # Remove accent marks

    # Remove punctuation (anything that's not a letter or space)
    text = "".join([c for c in text if c not in string.punctuation])

    # Further normalize: lowercase and replace spaces with underscores
    text = text.lower().strip().replace(" ", "_")  # Simple normalization for comparison
    return text


# Check if all words in a phrase are common (have synsets in WordNet)
def _is_common_phrase(phrase):
    words = phrase.split()  # Split the phrase into individual words
    return all(bool(wordnet.synsets(word)) for word in words)


# Identify if a playlist name matches an artist name (from a known list)
def _is_artist_playlist(playlist, relevant_artist_names):
    return playlist in relevant_artist_names
