import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from .word_similarity import load_wv_model
from .util import word_segmentation, get_text_vectors_wv, get_text_vectors_sbert


def encode_column(model, df, col_name, model_type):
    df = df.dropna(subset=[col_name])
    if model_type == 'sbert':
        df["embedding"] = list(model.encode(df[col_name].tolist(), normalize_embeddings=True, show_progress_bar=True))
    elif model_type == 'wv':
        df["embedding"] = get_text_vectors_wv(df[col_name].tolist(), model)
    return df


def item_level_ccr(text_encoded_df, questionnaire_encoded_df):
    
    q_embeddings = questionnaire_encoded_df.embedding
    d_embeddings = text_encoded_df.embedding

    q_embeddings_array = np.stack(q_embeddings)
    d_embeddings_array = np.stack(d_embeddings)
    q_embeddings_tensor = torch.tensor(q_embeddings_array).double()
    d_embeddings_tensor = torch.tensor(d_embeddings_array).double()

    similarities = util.pytorch_cos_sim(d_embeddings_tensor, q_embeddings_tensor)
    for i in range(1, len(questionnaire_encoded_df) + 1):
        text_encoded_df["sim_item_{}".format(i)] = similarities[:, i - 1]
    mean_similarity = similarities.mean(axis=1)
    text_encoded_df["mean_similarity"] = mean_similarity
    return text_encoded_df


def ccr_wrapper(df_text, text_col, text_embedding_file, df_questionnaire, q_col, model, model_type='sbert'):
    """
    Returns a Dataframe that is the content of data_file with one additional column for CCR value per question

    Parameters:
        df_text (str): dataframe containing user text
        text_col (str): column that includes user text
        df_questionnaire (str): dataframe containing questionnaires
        q_col (str): column that includes questions
        model (str): name of the SBERT model to use for CCR see https://www.sbert.net/docs/pretrained_models.html for full list
    """
    if model_type == 'sbert':
        model = SentenceTransformer(model)

    q_encoded_df = encode_column(model, df_questionnaire, q_col, model_type)
    print("Questionnaires were encoded Successfully.")

    if text_embedding_file == '':
        text_encoded_df = encode_column(model, df_text, text_col, model_type)
        print("Texts were encoded Successfully.")
    else:
        try:
            d_embeddings = np.load(text_embedding_file, allow_pickle=True)
            print("Text Embedding file found. Texts have been encoded previously.")
            text_encoded_df = df_text.copy()
            text_encoded_df["embedding"] = d_embeddings
        except FileNotFoundError:
            print("Text Embedding file not found, creating new text embedding file...")
            text_encoded_df = encode_column(model, df_text, text_col, model_type)
            np.save(text_embedding_file, text_encoded_df.embedding)
        print("Texts were encoded Successfully.")

    ccr_df = item_level_ccr(text_encoded_df, q_encoded_df)

    return ccr_df







