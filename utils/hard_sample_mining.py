import numpy as np
import pandas as pd
from scipy.stats import normaltest
import os
from joblib import Parallel, delayed
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from .text_similarity import text_embedding


def get_thresholds(similarities, df_similarity_matrix, option='quantile', n_sigmas=(2, 2), quantiles=(0.95, 0.95)):
    '''
    Calculate the thresholds for the similarities by setting the number of sigma or quantile.
    Similarities: a list of similarities.
    df_similarity: a dataframe of the similarity matrix.
    Option: 'sigma' or 'quantile'.
    n_sigmas: a tuple of the number of sigma for the lower and upper thresholds.
    quantiles: a tuple of the quantiles for the lower and upper thresholds.
    '''

    similarities = np.array(similarities)

    # test if it is normal distribution
    print(normaltest(similarities))
    if normaltest(similarities).pvalue < 0.05:
        print('The similarities are not normally distributed.')
    else:
        print('The similarities are normally distributed.')

    # Calculate the thresholds by setting the number of sigma
    if option == 'sigma':
        # Calculate the mean and standard deviation of the similarities
        mean = similarities.mean()
        std = similarities.std()

        # Calculate the thresholds by mean ± n_sigma * std
        thresholds = (mean - n_sigmas[0] * std, mean + n_sigmas[1] * std)
        print(f'Thresholds calculated by mean - {n_sigmas[0]} * std and mean + {n_sigmas[1]} * std:', thresholds)
        
    # Calculate the thresholds by quantile
    elif option == 'quantile':
        thresholds = (np.quantile(similarities, quantiles[0]), np.quantile(similarities, quantiles[1]))
        print(f'Thresholds calculated by {quantiles[0] * 100}% and {quantiles[1] * 100}% quantiles:', thresholds)

    num_outliers_high = sum(similarities >= thresholds[1])
    num_outliers_low = sum(similarities <= thresholds[0])
    print('Number of outliers (high):', num_outliers_high)
    print('Number of outliers (low):', num_outliers_low)

    # Calculate the number of negative and positive counterparts for each topic/title
    count_negative = {}
    count_positive = {}

    for topic in df_similarity_matrix.columns:
        df_negative = df_similarity_matrix[df_similarity_matrix[topic] <= thresholds[0]]
        df_positive = df_similarity_matrix[df_similarity_matrix[topic] >= thresholds[1]]

        count_negative[topic] = len(df_negative)
        count_positive[topic] = len(df_positive)

    # number of topics with more than 0 negative similarities
    print('Number of topics with more than 0 negative similarities:',
        sum([1 for k, v in count_negative.items() if v > 0]),
        '/', len(count_negative))

    # number of topic with more than 0 positive similarities
    print('Number of topics with more than 0 positive similarities:',
        sum([1 for k, v in count_positive.items() if v > 0]),
        '/', len(count_positive))

    return thresholds


def get_positive_topic_pairs(df_similarity_matrix_triangle, thresholds):
    '''
    Get the topic pairs with similarities higher than the upper threshold.
    df_similarity_matrix_triangle: a dataframe of the triangular similarity matrix without diagonal elements.
    thresholds: a tuple of thresholds for the similarities.
    '''
    df_positive = df_similarity_matrix_triangle[df_similarity_matrix_triangle >= thresholds[1]]
    positive_topic_pairs = df_positive.stack().reset_index()
    positive_topic_pairs.columns = ['topic1', 'topic2', 'similarity']
    positive_topic_pairs = positive_topic_pairs[positive_topic_pairs['topic1'] != positive_topic_pairs['topic2']]
    print(f"Successfully get {len(positive_topic_pairs)} positive topic pairs.")
    return positive_topic_pairs


def get_negative_topic_pairs(df_similarity_matrix_triangle, thresholds):
    '''
    Get the topic pairs with similarities lower than the lower threshold.
    df_similarity_matrix_triangle: a dataframe of the triangular similarity matrix without diagonal elements, which can avoid the duplication of the topic pairs.
    thresholds: a tuple of thresholds for the similarities.
    '''
    df_negative = df_similarity_matrix_triangle[df_similarity_matrix_triangle <= thresholds[0]]
    negative_topic_pairs = df_negative.stack().reset_index()
    negative_topic_pairs.columns = ['topic1', 'topic2', 'similarity']
    negative_topic_pairs = negative_topic_pairs[negative_topic_pairs['topic1'] != negative_topic_pairs['topic2']]
    print(f"Successfully get {len(negative_topic_pairs)} negative topic pairs.")
    return negative_topic_pairs


def get_positive_topics_for_each_topic(df_similarity_matrix_complete, thresholds):
    '''
    Get the topics with higher than the upper threshold of similarity for each topic.
    df_similarity_matrix_complete: a dataframe of the similarity matrix with diagonal elements (including the topic itself).
    thresholds: a tuple of thresholds for the similarities.
    '''
    n_jobs = os.cpu_count()
    n_jobs = max(1, round(n_jobs / 8) if n_jobs is not None else 1)

    def process_column(col):
        return col, df_similarity_matrix_complete[df_similarity_matrix_complete[col] >= thresholds[1]][col]

    results = Parallel(n_jobs=n_jobs)(delayed(process_column)(col) for col in df_similarity_matrix_complete.columns)
    positive_topics = {col: data for col, data in results}
    return positive_topics


def get_negative_topics_for_each_topic(df_similarity_matrix_complete, thresholds):
    '''
    Get the topics with lower than the lower threshold of similarity for each topic.
    df_similarity_matrix_complete: a dataframe of the similarity matrix with diagonal elements (including the topic itself).
    thresholds: a tuple of thresholds for the similarities.
    '''
    n_jobs = os.cpu_count()
    n_jobs = max(1, round(n_jobs / 8) if n_jobs is not None else 1)

    def process_column(col):
        return col, df_similarity_matrix_complete[df_similarity_matrix_complete[col] <= thresholds[0]][col]

    results = Parallel(n_jobs=n_jobs)(delayed(process_column)(col) for col in df_similarity_matrix_complete.columns)
    negative_topics = {col: data for col, data in results}
    return negative_topics


def get_positive_samples_for_each_topic(df_similarity_matrix_complete, thresholds, df_paragraph, id_col):
    '''
    Get the positive samples for each topic.
    df_paragraph: a dataframe of the paragraphs with the corresponding topics.
    positive_topics: a dictionary of the topics with higher than the upper threshold of similarity for each topic.
    '''
    positive_topics = get_positive_topics_for_each_topic(df_similarity_matrix_complete, thresholds)

    positive_samples = {}
    for topic, similar_topics in positive_topics.items():
        positive_samples[topic] = df_paragraph.loc[similar_topics.index, id_col]
    return positive_samples


def get_negative_samples_for_each_topic(df_similarity_matrix_complete, thresholds, df_paragraph, id_col):
    '''
    Get the negative samples for each topic.
    df_paragraph: a dataframe of the paragraphs with the corresponding topics.
    negative_topics: a dictionary of the topics with lower than the lower threshold of similarity for each topic.
    '''
    negative_topics = get_negative_topics_for_each_topic(df_similarity_matrix_complete, thresholds)

    negative_samples = {}
    for topic, similar_topics in negative_topics.items():
        negative_samples[topic] = df_paragraph.loc[similar_topics.index, id_col]
    return negative_samples


def get_text_embedding(df_paragraph_file, df_paragraph, tf_model, text_col='text'):
    df_paragraph_dir = os.path.dirname(df_paragraph_file)
    df_paragraph_name = os.path.basename(df_paragraph_file)
    text_embedding_dir = df_paragraph_dir + '/embedding/'
    if not os.path.exists(text_embedding_dir):
        os.makedirs(text_embedding_dir)
    tf_model_name = tf_model.split('/')[-1]

    text_embedding_file = text_embedding_dir + df_paragraph_name.replace('.csv', '_embedding_' + tf_model_name + '.npy')
    print('Load', text_embedding_file)

    df_paragraph_encoded = text_embedding(text_embedding_file, tf_model, df_paragraph, text_col)
    return df_paragraph_encoded




def process_n_samples_positive(row, df_paragraph_encoded, positive_samples_for_each_topic, option, n, seed, id_col):
        topic = row.name
        text_id = row[id_col]

        positive_topics = positive_samples_for_each_topic[topic]
        positive_texts = df_paragraph_encoded.loc[positive_topics.index, id_col]

        if option == 'random':
            if len(positive_texts) > 0:
                
                np.random.seed(seed)    ##############
                random_n_indices = np.random.choice(len(positive_texts), n, replace=False,)
                random_n_samples = [positive_texts.iloc[i] for i in random_n_indices]

                return random_n_samples
            else:
                return None
        

        elif option == 'top':

            #########
            # df_paragraph_encoded['if_used'] = 0

            if len(positive_texts) > 0:

                df_positive_samples = pd.DataFrame(positive_texts, columns=[id_col])

                df_positive_samples = df_positive_samples.merge(df_paragraph_encoded, on=id_col, how='left')

                ######## # Remove the positive samples that have been used
                # df_positive_samples = df_positive_samples[df_paragraph_encoded['if_used'] == 0]

                positive_sample_vectors = np.stack(df_positive_samples['embedding'].values)
                text_vector = np.stack(df_paragraph_encoded[df_paragraph_encoded[id_col] == text_id]['embedding'].values[0])

                similarity = np.dot(positive_sample_vectors, text_vector.T)
                # print(f"List of similarities for potential positive samples: {similarity}")
                top_n_indices = np.argsort(similarity, axis=0)[:n].flatten()
                top_n_samples = [positive_texts.iloc[i] for i in top_n_indices]
                top_n_similarity = similarity[top_n_indices].tolist()
                if not isinstance(top_n_similarity, list):
                    top_n_similarity = [top_n_similarity]
                # print(f"Top {top_n} similarities: {top_n_similarity}")
                
                # ########### Mark the positive samples that have been used
                # for sample in top_n_samples:
                #     df_paragraph_encoded.loc[df_paragraph_encoded[id_col] == sample, 'if_used'] = 1
            
                return top_n_samples, top_n_similarity
            else:
                return None, None


def apply_func_n_positive(df_row, df_paragraph_encoded, positive_samples_for_each_topic, option, n, seed, id_col):
    return process_n_samples_positive(df_row, df_paragraph_encoded, positive_samples_for_each_topic, option, n, seed, id_col)


def worker_func_n_positive(params):
    df_row, df_paragraph_encoded, positive_samples_for_each_topic, option, n, seed, id_col = params
    return apply_func_n_positive(df_row, df_paragraph_encoded, positive_samples_for_each_topic, option, n, seed, id_col)


def parallel_apply_n_positive(df, func, args=(), executor=ProcessPoolExecutor):
    n_jobs = os.cpu_count()
    n_jobs = max(1, round(n_jobs / 8) if n_jobs is not None else 1)
    with executor(max_workers=n_jobs) as exec:
        tasks = [(row, *args) for _, row in df.iterrows()]
        results = list(tqdm(exec.map(worker_func_n_positive, tasks), total=len(df)))
        return results


def get_n_positive_samples_for_each_text(df_similarity_matrix_complete, thresholds, df_paragraph_encoded, option, n, seed, id_col):
    '''
    Get the top n positive samples with the lowest similarity for each text.
    '''
    positive_samples_for_each_topic = get_positive_samples_for_each_topic(df_similarity_matrix_complete, thresholds, df_paragraph_encoded, id_col)

    df_n_positive_samples_for_each_text = df_paragraph_encoded.loc[positive_samples_for_each_topic.keys()]
    
    if option == 'random':
        print(f"Getting random {n} positive samples for each text...")
        results = parallel_apply_n_positive(df_n_positive_samples_for_each_text, apply_func_n_positive, args=(df_paragraph_encoded, positive_samples_for_each_topic, option, n, seed, id_col))

        df_n_positive_samples_for_each_text['positive_samples'] = results
        df_n_positive_samples_for_each_text = df_n_positive_samples_for_each_text.dropna(subset=['positive_samples'])
        print(f"Succesfully get random {n} positive samples for {len(df_n_positive_samples_for_each_text)} texts.")

    elif option == 'top':
        print(f"Getting top {n} positive samples for each text...")
        results = parallel_apply_n_positive(df_n_positive_samples_for_each_text, apply_func_n_positive, args=(df_paragraph_encoded, positive_samples_for_each_topic, option, n, seed, id_col))

        top_n_positive_samples, top_n_positive_similarity = zip(*results)
        df_n_positive_samples_for_each_text['positive_samples'] = top_n_positive_samples
        df_n_positive_samples_for_each_text['positive_similarities'] = top_n_positive_similarity

        df_n_positive_samples_for_each_text = df_n_positive_samples_for_each_text.dropna(subset=['positive_samples'])

        print(f"Succesfully get top {n} positive samples for {len(df_n_positive_samples_for_each_text)} texts.")

    return df_n_positive_samples_for_each_text



def apply_func_n_negative(df_row, df_paragraph_encoded, negative_samples_for_each_topic, option, n, seed, id_col):
    return process_n_samples_negative(df_row, df_paragraph_encoded, negative_samples_for_each_topic, option, n, seed, id_col)


def worker_func_n_negative(params):
    df_row, df_paragraph_encoded, negative_samples_for_each_topic, option, n, seed, id_col = params
    return apply_func_n_negative(df_row, df_paragraph_encoded, negative_samples_for_each_topic, option, n, seed, id_col)


def parallel_apply_n_negative(df, func, args=(), executor=ProcessPoolExecutor):
    n_jobs = os.cpu_count()
    n_jobs = max(1, round(n_jobs / 8) if n_jobs is not None else 1)

    with executor(max_workers=n_jobs) as exec:
        tasks = [(row, *args) for _, row in df.iterrows()]
        results = list(tqdm(exec.map(worker_func_n_negative, tasks), total=len(df)))
    return results


def process_n_samples_negative(row, df_paragraph_encoded, negative_samples_for_each_topic, option, n, seed, id_col):
        topic = row.name
        text_id = row[id_col]

        negative_topics = negative_samples_for_each_topic[topic]
        negative_texts = df_paragraph_encoded.loc[negative_topics.index, id_col]

        if option == 'random':
            if len(negative_texts) > 0:
                np.random.seed(seed)    ##############
                random_n_indices = np.random.choice(len(negative_texts), n, replace=False)
                random_n_samples = [negative_texts.iloc[i] for i in random_n_indices]
                return random_n_samples
            else:
                return None

        elif option == 'top':
            if len(negative_texts) > 0:
                df_negative_samples = pd.DataFrame(negative_texts, columns=[id_col])
                df_negative_samples = df_negative_samples.merge(df_paragraph_encoded, on=id_col, how='left')

                negative_sample_vectors = np.stack(df_negative_samples['embedding'].values)
                text_vector = np.stack(df_paragraph_encoded[df_paragraph_encoded[id_col] == text_id]['embedding'].values[0])

                similarity = np.dot(negative_sample_vectors, text_vector.T)
                # print(f"List of similarities for potential negative samples:: {similarity}")
                top_n_indices = np.argsort(similarity, axis=0)[-n:].flatten()
                top_n_samples = [negative_texts.iloc[i] for i in top_n_indices]
                top_n_similarity = similarity[top_n_indices].tolist()
                if not isinstance(top_n_similarity, list):
                    top_n_similarity = [top_n_similarity]
                # print(f"Top {top_n} similarities: {top_n_similarity}")
            
                return top_n_samples, top_n_similarity
            else:
                return None, None

 
def get_n_negative_samples_for_each_text(df_similarity_matrix_complete, thresholds, df_paragraph_encoded, option, n, seed, id_col):
    '''
    Get the top n negative samples with the lowest similarity for each text.
    '''
    negative_samples_for_each_topic = get_negative_samples_for_each_topic(df_similarity_matrix_complete, thresholds, df_paragraph_encoded, id_col)
    
    df_n_negative_samples_for_each_text = df_paragraph_encoded.loc[negative_samples_for_each_topic.keys()]

    args = (df_paragraph_encoded, negative_samples_for_each_topic, option, n, seed, id_col)

    
    if option == 'random':
        print(f"Getting random {n} negative samples for each text...")
        results = parallel_apply_n_negative(df_n_negative_samples_for_each_text, apply_func_n_negative, args=args)
        df_n_negative_samples_for_each_text['negative_samples'] = results
        df_n_negative_samples_for_each_text = df_n_negative_samples_for_each_text.dropna(subset=['negative_samples'])

        print(f"Succesfully get random {n} negative samples for {len(df_n_negative_samples_for_each_text)} texts.")

    elif option == 'top':
        print(f"Getting top {n} negative samples for each text...")
        results = parallel_apply_n_negative(df_n_negative_samples_for_each_text, apply_func_n_negative, args=args)
        top_n_negative_samples, top_n_negative_similarity = zip(*results)
        df_n_negative_samples_for_each_text['negative_samples'] = top_n_negative_samples
        df_n_negative_samples_for_each_text['negative_similarities'] = top_n_negative_similarity

        df_n_negative_samples_for_each_text = df_n_negative_samples_for_each_text.dropna(subset=['negative_samples'])

        print(f"Succesfully get top {n} negative samples for {len(df_n_negative_samples_for_each_text)} texts.")
    
    return df_n_negative_samples_for_each_text






