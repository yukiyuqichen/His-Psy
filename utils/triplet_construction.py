import pandas as pd
import os
from utils.word_similarity import load_wv_model, get_word_pairs_and_similarities, get_dataframe_of_similarity_matrix
from utils.hard_sample_mining import get_thresholds, get_text_embedding, get_n_positive_samples_for_each_text, get_n_negative_samples_for_each_text


def get_text_from_id(df_paragraph, text_id, text_col, id_col):
    return df_paragraph[df_paragraph[id_col] == text_id][text_col].values[0]

def get_topic_from_id(df_paragraph, text_id, id_col):
    return df_paragraph[df_paragraph[id_col] == text_id].index[0]


def construct_triplets(df_all_file, quantiles, df_paragraph_file, wv_model, tf_model, option, n, seed, text_col='text', id_col='text_id', topic_col='topic'):
    
    df_all = pd.read_csv(df_all_file, encoding='utf-8-sig')
    titles = list(set(df_all['topic'].tolist()))
    print(f'There are {len(titles)} unique topics in the entire corpus.')
    _, title_pair_similarities = get_word_pairs_and_similarities(titles, wv_model, option='with_self')
    df_similarity_matrix_complete = get_dataframe_of_similarity_matrix(titles, wv_model, option='complete')

    thresholds = get_thresholds(title_pair_similarities, df_similarity_matrix_complete, option='quantile', quantiles=quantiles)

    df_paragraph = pd.read_csv(df_paragraph_file, encoding='utf-8-sig')
    df_paragraph.reset_index(drop=True, inplace=True)
    df_paragraph[id_col] = df_paragraph.index
    df_paragraph.set_index(topic_col, inplace=True)

    df_paragraph_dir, df_paragraph_name = os.path.dirname(df_paragraph_file), os.path.basename(df_paragraph_file)
    quantiles_str = str(round(quantiles[0]*1000)) + '-' + str(round((quantiles[1]*1000)))
    df_triplet_file = df_paragraph_dir + '/triplets/' + df_paragraph_name.replace('.csv', '_triplet_' + 'quantiles=' + quantiles_str + '.csv')
    output_path = df_triplet_file

    if option == 'random':
        df_paragraph_encoded = df_paragraph.copy()
        output_path = output_path.replace('.csv', '_random_seed=' + str(seed) + '.csv')
    elif option == 'top':
        df_paragraph_encoded = get_text_embedding(df_paragraph_file, df_paragraph, tf_model, text_col)
        output_path = output_path.replace('.csv', '_top_' + tf_model.split('/')[-1] + '.csv')

    titles = df_paragraph.index.unique().tolist()
    df_similarity_matrix_complete = get_dataframe_of_similarity_matrix(titles, wv_model, option='complete')

    df_n_positive_samples_for_each_text = get_n_positive_samples_for_each_text(df_similarity_matrix_complete, thresholds, df_paragraph_encoded, option, n, seed, id_col)

    df_n_negative_samples_for_each_text = get_n_negative_samples_for_each_text(df_similarity_matrix_complete, thresholds, df_paragraph_encoded, option, n, seed, id_col)

    triplets = []
    for index, row in df_n_positive_samples_for_each_text.iterrows():
        text_id = row[id_col]
        positive_samples = row['positive_samples']
        if text_id in df_n_negative_samples_for_each_text[id_col].values:
            negative_samples = df_n_negative_samples_for_each_text[df_n_negative_samples_for_each_text[id_col] == text_id]['negative_samples'].values[0]
            for positive_sample in positive_samples:
                for negative_sample in negative_samples:
                    triplets.append((text_id, positive_sample, negative_sample))

    df_triplets = pd.DataFrame(columns=['target_id', 'positive_id', 'negative_id'])
    df_triplets['target_id'] = [triplet[0] for triplet in triplets]
    df_triplets['positive_id'] = [triplet[1] for triplet in triplets]
    df_triplets['negative_id'] = [triplet[2] for triplet in triplets]

    df_triplets['target'] = df_triplets['target_id'].apply(lambda x: get_text_from_id(df_paragraph, x, text_col, id_col))
    df_triplets['positive'] = df_triplets['positive_id'].apply(lambda x: get_text_from_id(df_paragraph, x, text_col, id_col))
    df_triplets['negative'] = df_triplets['negative_id'].apply(lambda x: get_text_from_id(df_paragraph, x, text_col, id_col))

    df_triplets['target_topic'] = df_triplets['target_id'].apply(lambda x: get_topic_from_id(df_paragraph, x, id_col))
    df_triplets['positive_topic'] = df_triplets['positive_id'].apply(lambda x: get_topic_from_id(df_paragraph, x, id_col))
    df_triplets['negative_topic'] = df_triplets['negative_id'].apply(lambda x: get_topic_from_id(df_paragraph, x, id_col))

    df_triplets.drop(columns=['target_id', 'positive_id', 'negative_id'], inplace=True)

    # Shuffle and save the triplets
    df_triplets = df_triplets.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_triplets.to_csv(output_path, encoding='utf-8-sig', index=False)


    if option == 'top':
        df_n_positive_samples_for_each_text.drop(columns=['embedding'], inplace=True)
        df_n_negative_samples_for_each_text.drop(columns=['embedding'], inplace=True)

    df_n_positive_samples_for_each_text['positive_samples_topics'] = df_n_positive_samples_for_each_text['positive_samples'].apply(lambda x: [get_topic_from_id(df_paragraph, y, id_col) for y in x])
    df_n_negative_samples_for_each_text['negative_samples_topics'] = df_n_negative_samples_for_each_text['negative_samples'].apply(lambda x: [get_topic_from_id(df_paragraph, y, id_col) for y in x])

    df_n_positive_samples_for_each_text['positive_samples'] = df_n_positive_samples_for_each_text['positive_samples'].apply(lambda x: [get_text_from_id(df_paragraph, y, text_col, id_col) for y in x])
    df_n_negative_samples_for_each_text['negative_samples'] = df_n_negative_samples_for_each_text['negative_samples'].apply(lambda x: [get_text_from_id(df_paragraph, y, text_col, id_col) for y in x])

    df_n_positive_samples_for_each_text.to_csv(output_path.replace('_triplet_', '_positive_'), encoding='utf-8-sig', index=True)
    df_n_negative_samples_for_each_text.to_csv(output_path.replace('_triplet_', '_negative_'), encoding='utf-8-sig', index=True)
