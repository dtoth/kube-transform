from nltk.util import ngrams as nltk_ngrams
from spotify_mpc.logic.lexicon import _normalize_text


def generate_char_ngrams(text, N=[2, 3]):
    """
    Generate character ngrams from a given text.
    """
    if text is None:
        return []
    # Normalize the text (lowercase, remove punctuation)
    normalized_text = _normalize_text(text)
    # Generate ngrams of characters (2 and 3 grams in this case)
    char_ngrams = list(nltk_ngrams(normalized_text, N[0]))
    for n in N[1:]:
        char_ngrams.extend(list(nltk_ngrams(normalized_text, n)))

    char_ngrams = ["".join(cn) for cn in char_ngrams]  # Convert tuples to strings

    return list(set(char_ngrams))  # Remove duplicates
