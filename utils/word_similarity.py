from gensim.models import KeyedVectors
from itertools import combinations
import pandas as pd
import numpy as np


def load_wv_model(wv_model_path):
    if 'glove' in wv_model_path:
        model = KeyedVectors.load(wv_model_path)
    else:
        model = KeyedVectors.load(wv_model_path).wv
    print("Successfully loaded the word vector model.")
    return model


def batch_cosine_similarity(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norm_matrix = matrix / np.where(norms == 0, 1, norms)
    return np.dot(norm_matrix, norm_matrix.T)


def calculate_batch_similarity(words, model):
    vectors = np.array([model[word] for word in words])
    similarity_matrix = batch_cosine_similarity(vectors)
    return similarity_matrix


def get_word_pairs_and_similarities(words, model, option='without_self'):
    '''
    Calculate the similarities for the word pairs.
    words: a list of words.
    model: a word vector model.
    option: 'without_self' or 'with_self'.
    If 'without_self', return the word pairs and similarities without the self-pair.
    If 'with_self', return the word pairs and similarities with the self-pair similarity.
    '''
    similarity_matrix = calculate_batch_similarity(words, model)
    if option == 'without_self':
        pairs = list(combinations(words, 2))
        similarities = [similarity_matrix[i, j] for i, j in combinations(range(len(words)), 2)]
    
    elif option == 'with_self':
        pairs = list(combinations(words, 2))
        similarities = [similarity_matrix[i, j] for i, j in combinations(range(len(words)), 2)]
        pairs_self = [(word, word) for word in words]
        similarities_self = np.diag(similarity_matrix)
        pairs += pairs_self
        similarities += list(similarities_self)

    return pairs, similarities


def get_dataframe_of_similarity_matrix(words, model, option='undiagonal'):
    '''
    Calculate the similarity matrix of the words.
    words: a list of words.
    model: a word vector model.
    option: 'complete', 'undiagonal', or 'triangle'.
    If 'complete', return the complete similarity matrix.
    If 'undiagonal', set the diagonal elements to NaN.
    If 'triangle', return the lower triangle of the similarity matrix without the diagonal elements.
    '''
    similarity_matrix = calculate_batch_similarity(words, model)

    if option == 'complete':
        df_similarity_matrix = pd.DataFrame(similarity_matrix, index=words, columns=words)
    elif option == 'undiagonal':
        np.fill_diagonal(similarity_matrix, np.nan)
        df_similarity_matrix = pd.DataFrame(similarity_matrix, index=words, columns=words)
    elif option == 'lower_triangle':
        similarity_matrix = np.tril(similarity_matrix, k=-1)
        # Fill the upper triangle and the diagonal with NaN
        similarity_matrix[np.triu_indices(similarity_matrix.shape[0], k=1)] = np.nan
        np.fill_diagonal(similarity_matrix, np.nan)
        df_similarity_matrix = pd.DataFrame(similarity_matrix, index=words, columns=words)
    elif option == 'upper_triangle':
        similarity_matrix = np.triu(similarity_matrix, k=1)
        # Fill the lower triangle and the diagonal with NaN
        similarity_matrix[np.tril_indices(similarity_matrix.shape[0], k=-1)] = np.nan
        np.fill_diagonal(similarity_matrix, np.nan)
        df_similarity_matrix = pd.DataFrame(similarity_matrix, index=words, columns=words)

    return df_similarity_matrix