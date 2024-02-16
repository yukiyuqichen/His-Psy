import hanlp
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.cross_encoder import CrossEncoder
from collections import Counter


def word_segmentation(text):
    text = remove_punctuation(text)
    tok_coarse = hanlp.load(hanlp.pretrained.tok.MSR_TOK_ELECTRA_BASE_CRF)
    words = tok_coarse(text)
    return words


def remove_punctuation(text):
    remove_chars = "[·,，;；.。……/\\\[\]'\〡【】:：@＠＆Х％≮≯︿﹀︵︶""”“?？!！、+*━═=<>《》「」〔〕『』〖〗_#︹︺□■◎○〇●◇◆△∈╩()（）~]+"
    remove_en = '[a-zA-Z]'
    remove_num = r'[0-9]+'
    remove_num_full = r'[０１２３４５６７８９]'
    text = text.replace(" ", "")
    text = re.sub('-', '', text)
    text = re.sub(remove_chars, '', text)
    text = re.sub(remove_en, '', text)
    text = re.sub(remove_num, '', text)
    text = re.sub(remove_num_full, '', text)
    return text


def remove_quotaion_marks(texts):
    quotation_marks = ['“', '”', '‘', '’', '\'', '\"', '「', '」', '『', '』']
    text_cleaned = []
    for text in texts:
        for quotation_mark in quotation_marks:
            text = text.replace(quotation_mark, '')
        text_cleaned.append(text)
    return text_cleaned


def identical_punctuation(texts):
    identical_punctuation = [
        [',', '，'],
        ['.', '。'],
        [';', '；'],
        ['?', '？'],
        ['!', '！'],
        [':', '：'],
        ['(', '（'],
        [')', '）'],
        ['[', '【'],
        [']', '】'],
        ['‘', '『'],
        ['’', '』'],
        ['“', '「'],
        ['”', '」'],
    ]
    texts_cleaned = []
    for text in texts:
        for pair in identical_punctuation:
            text = text.replace(pair[0], pair[1])
        texts_cleaned.append(text)
    return texts_cleaned    


def text_cleaning(texts):
    # english and other  special cases(〔オーフン5ve)
    text_cleaned = []
    patterns = [r'\【.*?\】',
                r'\〔.*?\〕',
                '〔', 
                '〕', 
                '●',
                '○', 
                '□',
                '■',
                '\u3000',
                r'[0-9a-zA-Z]']
    remove_num_full = r'[０１２３４５６７８９]'
    for text in texts:
        for pattern in patterns:
            text = re.sub(pattern, '', text)
        # other special cases
        text = re.sub('。.?.?.?。', '。', text)
        text = re.sub('。+', '。', text)
        text = re.sub('！+', '！', text)
        text = re.sub('？+', '？', text)
        text = re.sub(remove_num_full, '', text)
        text_cleaned.append(text)
    return text_cleaned


def get_word_vectors_wv(words, model):
    word_vectors = []
    for word in words:
        try:
            word_vector = model[word]
        except Exception as e:
            print(e)
            print(word + ' not in model. Use zero vector instead.')
            word_vector = np.zeros(300)
        word_vectors.append(word_vector)

    # # Another way to get word vectors
    # word_vectors = model[words].tolist()

    return word_vectors


def get_text_vectors_wv(texts, model, option):
    text_vectors = []
    for text in texts:
        # if 'word_unsegmented', the text should be string
        if option == 'word_unsegmented':
            words = word_segmentation(text)
        # if 'word_segmented', the text should be a list of words
        elif option == 'word_segmented':
            words = text
        word_vectors = get_word_vectors_wv(words, model)
        # Remove zero vectors
        word_vectors = [word_vector for word_vector in word_vectors if not np.all(word_vector == 0)]
        stacked_word_vectors = np.stack(word_vectors, axis=0)
        average_word_vector = np.mean(stacked_word_vectors, axis=0)
        text_vectors.append(average_word_vector)
    return text_vectors


def get_text_vectors_sbert(texts, model, device):
    try:
        text_vectors = model.encode(texts, convert_to_tensor=True, device=device)
    except:
        model = SentenceTransformer(model)
        # Move model to GPU
        model.to(device)
        model.eval()
        text_vectors = model.encode(texts, convert_to_tensor=True, device=device)
    
    text_vectors = [vec.cpu() for vec in text_vectors]

    # Convert to numpy array
    text_vectors = [v.numpy() if hasattr(v, 'numpy') else v for v in text_vectors]
    text_vectors = np.array(text_vectors)

    return text_vectors


# text_array_1[i] and text_array_2[i] has already been a sentence pair
def load_cross_encoder_and_get_scores_matched(model_path, text_array_1, text_array_2, device):
    text_pairs =  list(zip(text_array_1, text_array_2))
    print('Length of text pairs: ', len(text_pairs))
    cross_model = CrossEncoder(model_path, device=device)
    scores = cross_model.predict(text_pairs)
    print('Scores successfully computed.')
    return scores


# texts from text_array_row and text_array_2 need to be combined into sentence pairs
def load_cross_encoder_and_get_scores_unmatched(model_path, text_array_row, text_array_col, device):
    row_length = len(text_array_row)
    col_length = len(text_array_col)
    text_pairs = [[a, b] for b in text_array_row for a in text_array_col]
    cross_model = CrossEncoder(model_path, device=device)
    scores = cross_model.predict(text_pairs)
    print('Scores successfully computed.')
    count = Counter(type(item) for item in scores)
    print(count)
    scores_matrix = np.array(scores).reshape(row_length, col_length)
    scores = np.mean(scores_matrix, axis=1).tolist()

    return scores


def cosine_similarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


def deduplicate_texts(texts, labels):
    text_dict = {}
    duplicate_text = []
    for i in range(len(texts)):
        label = labels[i]
        text = texts[i]
        if text in text_dict:
            # Delete words with multiple labels
            duplicate_text.append(text)
            del text_dict[text]
        elif text in duplicate_text:
            continue
        else:
            text_dict[text] = label
    return list(text_dict.keys()), list(text_dict.values())


def encode_column(model, df, col_name):
    model = SentenceTransformer(model)
    df = df.dropna(subset=[col_name])
    df["embedding"] = list(model.encode(df[col_name].tolist(), batch_size=64, normalize_embeddings=True, show_progress_bar=True))
    return df


def text_embedding(text_embedding_file, model, df_text, text_col):
    if text_embedding_file == '':
        text_encoded_df = encode_column(model, df_text, text_col)
        print("Texts were encoded Successfully.")
    else:
        try:
            d_embeddings = np.load(text_embedding_file, allow_pickle=True)
            print("Text Embedding file found. Texts have been encoded previously.")
            text_encoded_df = df_text.copy()
            text_encoded_df["embedding"] = d_embeddings
        except FileNotFoundError:
            print("Text Embedding file not found, creating new text embedding file...")
            text_encoded_df = encode_column(model, df_text, text_col)
            np.save(text_embedding_file, text_encoded_df.embedding)
        print("Texts were encoded Successfully.")
    return text_encoded_df