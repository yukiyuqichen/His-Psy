from hanlp_restful import HanLPClient
import hanlp
import string
import numpy as np
import torch
import re
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer, models
from sentence_transformers.cross_encoder import CrossEncoder
from collections import Counter


def word_segmentation(texts):
    texts = [remove_punctuation(i) for i in texts]
    try:
        tok_coarse = hanlp.load('./models/tokenizer/coarse_electra_small')
        print('Using local tokenizer model.')
    except:
        tok_coarse = hanlp.load(hanlp.pretrained.tok.MSR_TOK_ELECTRA_BASE_CRF)
    words = tok_coarse(texts)
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
            print(word, ' not in model. Use zero vector instead.')
            word_vector = np.zeros(300)
        word_vectors.append(word_vector)

    # # Another way to get word vectors
    # word_vectors = model[words].tolist()

    return word_vectors


def get_text_vectors_wv(texts, model, option='word_unsegmented'):
    text_vectors = []

    if option == 'word_unsegmented':
        # Assuming 'word_segmentation' is a function that segments unsegmented text
        texts_seg = word_segmentation(texts)
    elif option == 'word_segmented':
        texts_seg = texts

    for words in texts_seg:
        word_vectors = get_word_vectors_wv(words, model)
        if len(word_vectors) > 0:  # Ensure there are word vectors to average
            stacked_word_vectors = np.vstack(word_vectors)  # Stack vectors for averaging
            average_word_vector = np.mean(stacked_word_vectors, axis=0)  # Calculate the average vector
            text_vectors.append(average_word_vector)
        else:
            # Handle case with no word vectors (e.g., all words were out of vocabulary)
            text_vectors.append(np.zeros(model.vector_size))  # Append a zero vector of the same size as the model's vectors

    return text_vectors


def get_text_vectors_bert(texts, tokenizer, model, option, device):
    # Encode the entire batch of texts at once
    encoded_input = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    encoded_input.to(device)

    # Forward pass the entire batch
    with torch.no_grad():
        model_output = model(**encoded_input)
        last_hidden_state = model_output.last_hidden_state

    if option == 'cls':
        text_vectors = last_hidden_state[:, 0, :]
    elif option == 'meanpooling':
        text_vectors = torch.mean(last_hidden_state[:, 1:-1, :], dim=1)
    elif option == 'maxpooling':
        text_vectors, _ = torch.max(last_hidden_state[:, 1:-1, :], dim=1)
    else:
        raise ValueError(f"Unsupported option: {option}")

    # Convert the entire batch to CPU at once
    text_vectors = text_vectors.cpu()

    return text_vectors


def get_text_vectors_roberta(texts, tokenizer, model, option, device):
    # Encode the entire batch of texts at once
    encoded_input = tokenizer(texts, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    encoded_input.to(device)

    # Forward pass the entire batch
    with torch.no_grad():
        model_output = model(**encoded_input)
        hidden_states = model_output.hidden_states
        last_hidden_state = hidden_states[-1]

    if option == 'cls':
        text_vectors = last_hidden_state[:, 0, :]
    elif option == 'meanpooling':
        text_vectors = torch.mean(last_hidden_state[:, 1:-1, :], dim=1)
    elif option == 'maxpooling':
        text_vectors, _ = torch.max(last_hidden_state[:, 1:-1, :], dim=1)
    else:
        raise ValueError(f"Unsupported option: {option}")

    # Convert the entire batch to CPU at once
    text_vectors = text_vectors.cpu()

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


def load_transformer_and_get_vectors(model_path, model_type, pooling_option, device, texts):
    if model_type in ['bert', 'bert-ancient']:
        tf_model = AutoModel.from_pretrained(model_path)
        # Move model to GPU
        tf_model.to(device)
        tf_model.eval()
        tf_tokenizer = AutoTokenizer.from_pretrained(model_path)
        vectors = get_text_vectors_bert(texts, tf_tokenizer, tf_model, pooling_option, device)
    elif model_type in ['roberta', 'roberta-ancient', 'ernie-consent']:
        tf_model = AutoModelForMaskedLM.from_pretrained(model_path, output_hidden_states=True)
        # Move model to GPU
        tf_model.to(device)
        tf_model.eval()
        tf_tokenizer = AutoTokenizer.from_pretrained(model_path)
        vectors = get_text_vectors_roberta(texts, tf_tokenizer, tf_model, pooling_option, device)
    elif model_type in ['sbert', 'sbert-ancient']:
        tf_model = SentenceTransformer(model_path)
        # Move model to GPU
        tf_model.to(device)
        tf_model.eval()
        vectors = get_text_vectors_sbert(texts, tf_model, device)
    return vectors


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


def descriptive_statistics(data):
    mean = np.mean(data)
    median = np.median(data)
    minimum = np.min(data)
    maximum = np.max(data)
    std_dev = np.std(data)
    variance = np.var(data)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)

    statsdict = {
        "Mean": float(mean),
        "Median": float(median),
        "Minimum": float(minimum),
        "Maximum": float(maximum),
        "Standard Deviation": float(std_dev),
        "Variance": float(variance),
        "1st Quartile (Q1)": float(q1),
        "3rd Quartile (Q3)": float(q3)
    }

    return statsdict





