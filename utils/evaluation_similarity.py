import pandas as pd
import numpy as np
import os
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from gensim.models import KeyedVectors
import ast
import yaml
import json
import torch
# from opencc import OpenCC
# import matplotlib_config
from .ccr import ccr_wrapper
from .util import text_cleaning, word_segmentation, get_word_vectors_wv, get_text_vectors_wv, cosine_similarity, remove_punctuation, remove_quotaion_marks, load_cross_encoder_and_get_scores_unmatched, load_transformer_and_get_vectors, load_cross_encoder_and_get_scores_matched, get_text_vectors_sbert, descriptive_statistics
from .word_similarity import load_wv_model
from .pair_construction import construct_pairs


def linear_regression_and_plot(dependent_var, independent_var, labels, topic, output_dir):
    slope, intercept, r_value, p_value, std_err = stats.linregress(dependent_var, independent_var)
    print(f"Slope: {slope}")
    print(f"Intercept: {intercept}")
    print(f"R-squared: {r_value**2}")
    print(f"P-value: {p_value}")
    print(f"Standard error: {std_err}")

    # Set the font and solve the problem of messy code
    plt.rcParams["font.sans-serif"]=["SimHei"] 
    plt.rcParams["axes.unicode_minus"]=False 
    plt.figure(figsize=(16, 16), dpi=300)

    plt.scatter(dependent_var, independent_var, color='#6D91D1', label='Data points', s=100, edgecolor='black', linewidth=0.6)
    # plt.plot(dependent_var, [slope*xi + intercept for xi in dependent_var], color='blue', label='Regression line')
    regression_line = [slope * xi + intercept for xi in dependent_var]
    plt.plot(dependent_var, regression_line, color='black', label='Regression line')

    # Add labels of topics
    for i, label in enumerate(labels):
        plt.annotate(label, (dependent_var[i], independent_var[i]), textcoords="offset points", xytext=(0,10), ha='center', color='black', alpha=0.6, fontproperties=FontProperties(fname="C:/Windows/Fonts/msjhl.ttc", size=12))
    
    plt.title(topic, fontsize=35, pad=15)
    plt.xlabel("Similarity between text and questionnaire", fontsize=35, labelpad=15)
    plt.ylabel("Similarity between title and dictionaries", fontsize=35, labelpad=15)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.legend(fancybox=False, framealpha=1, fontsize=30, edgecolor='black')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    output = output_dir + '_' + topic + '.svg'
    plt.savefig(output, format='svg', dpi=300)
    
    return slope, r_value**2, p_value, std_err


def compare_models_topic_similarity(method, wv_model, wv_model_paths, tf_model_names, tf_cross_model_names, encoder_type, text_file, text_embedding_file, questionnaire_file, dict_file, output_dir, device, topic_col='topic'):

    df_text = pd.read_csv(text_file, encoding='utf-8-sig')
    labels = df_text[topic_col].tolist()
    
    df_questionnaire = pd.read_csv(questionnaire_file, encoding='utf-8-sig')
    q_topics = df_questionnaire['topic'].tolist()

    df_dict = pd.read_csv(dict_file, encoding='utf-8-sig')
    words = df_dict['word'].tolist()
    d_topics = df_dict['topic'].tolist()

    if set(q_topics) != set(d_topics):
        print('Error: topics in questionnaire and dictionary are not the same.')
        return
    else:
        print('Topics in questionnaire and dictionary are the same, including: ', sorted(set(q_topics)))

    # Compute dependent variables: similarity between labels and mean vector of topic words
    word_vectors = get_word_vectors_wv(words, wv_model)
    df_dict['vector'] = word_vectors
    topic_mean_word_vectors = {}
    for topic in d_topics:
        topic_mean_word_vectors[topic] = np.mean(df_dict[df_dict['topic'] == topic]['vector'].tolist(), axis=0)

    topic_label_similarities = {}
    for topic in set(d_topics):
        topic_label_similarities[topic] = dict()
        for label in set(labels):
            label_vector = wv_model[label]
            topic_label_similarity = cosine_similarity(topic_mean_word_vectors[topic], label_vector)
            topic_label_similarities[topic][label] = topic_label_similarity


    # Compute independent variables: scores computed by ccr-based method
    df_scores_regression = pd.DataFrame(columns=['encoder', 'model', 'topic', 'slope', 'r_squared', 'p_value', 'std_err'])
    df_scores_correlation = pd.DataFrame(columns=['encoder', 'model', 'topic', 'pearson_correlation', 'pearson_p_value', 'spearman_correlation', 'spearman_p_value'])
    df_text_with_similarities_dict = {}
    
    if 'bi_encoder' in encoder_type:
        ## Calculate scores for Bi-Encoder BERT, Roberta, and SBERT model
        df_text_with_similarities_dict['bi-encoder'] = {}
        for tf_model_name in tf_model_names.keys():
            print(tf_model_name)
            model = tf_model_name
            
            df_text_with_similarities = df_text.copy()

            for topic in set(q_topics):
                df_questionnaire_topic = df_questionnaire[df_questionnaire['topic'] == topic]
                ccr_result_dataframe = ccr_wrapper(df_text, 'text', text_embedding_file, df_questionnaire_topic, 'text', model)
                df_text_with_similarities[topic] = ccr_result_dataframe['mean_similarity']
            print('Scores successfully computed.')
            df_text_with_similarities_dict['bi-encoder'][tf_model_name] = df_text_with_similarities
    
    if 'word_vector' in encoder_type:
        ## Calculate scores for Word Vector model
        df_text_with_similarities_dict['word-vector'] = {}
        for wv_model_path in wv_model_paths.values():
            print(wv_model_path)
            model = load_wv_model(wv_model_path)

            df_text_with_similarities = df_text.copy()

            for topic in set(q_topics):
                df_questionnaire_topic = df_questionnaire[df_questionnaire['topic'] == topic]
                ccr_result_dataframe = ccr_wrapper(df_text, 'text', '', df_questionnaire_topic, 'text', model, model_type='wv')
                df_text_with_similarities[topic] = ccr_result_dataframe['mean_similarity']
            print('Scores successfully computed.')
            df_text_with_similarities_dict['word-vector'][wv_model_path] = df_text_with_similarities

    if 'cross_encoder' in encoder_type:
        ## Calculate scores for Cross-Encoder model
        df_text_with_similarities_dict['cross-encoder'] = {}
        for tf_cross_model_name in tf_cross_model_names.keys():
            print(tf_cross_model_name)
            model = tf_cross_model_name
            model_type = tf_cross_model_names[tf_cross_model_name]

            df_text_with_similarities = df_text.copy()

            for topic in set(q_topics):
                df_questionnaire_topic = df_questionnaire[df_questionnaire['topic'] == topic]
                text_array_col = df_questionnaire_topic['text'].values
                text_array_row = df_text['text'].values
                mean_similarities = load_cross_encoder_and_get_scores_unmatched(model, text_array_row, text_array_col, device)
                df_text_with_similarities[topic] = mean_similarities
            print('Scores successfully computed.')
            df_text_with_similarities_dict['cross-encoder'][tf_cross_model_name] = df_text_with_similarities


    for encoder in df_text_with_similarities_dict.keys():
        for model_name in df_text_with_similarities_dict[encoder].keys():
            df_text_with_similarities = df_text_with_similarities_dict[encoder][model_name]

            topic_text_similarities = {}
            for topic in set(d_topics):
                topic_text_similarities[topic] = dict()
                for label in set(labels):
                    topic_text_similarity = df_text_with_similarities[df_text_with_similarities[topic_col] == label][topic].mean()
                    topic_text_similarities[topic][label] = topic_text_similarity

            img_dir = output_dir + model_name.split('/')[-1]
            for topic in set(d_topics):
                print('Topic: ', topic)
                dependent_var = np.array(list(topic_label_similarities[topic].values()))
                independent_var = np.array(list(topic_text_similarities[topic].values()))
                topics = np.array(list(topic_label_similarities[topic].keys()))
                # if method == 'regression':
                slope, r_squared, p_value, std_err = linear_regression_and_plot(independent_var, dependent_var, topics, topic, img_dir)
                new_row = {'encoder':encoder, 'model':model_name, 'topic':topic, 'slope':slope, 'r_squared':r_squared, 'p_value':p_value, 'std_err':std_err}
                df_scores_regression.loc[len(df_scores_regression)] = new_row
                df_scores_regression.to_csv(output_dir + 'topic_regression_scores.csv', encoding='utf-8-sig', index=False)
                # elif method == 'correlation':
                pearson_correlation, pearson_p_value = stats.pearsonr(dependent_var, independent_var)
                spearman_correlation, spearman_p_value = stats.spearmanr(dependent_var, independent_var)
                new_row = {'encoder':encoder, 'model':model_name, 'topic':topic, 'pearson_correlation':pearson_correlation, 'pearson_p_value':pearson_p_value, 'spearman_correlation':spearman_correlation, 'spearman_p_value':spearman_p_value}
                df_scores_correlation.loc[len(df_scores_correlation)] = new_row
                df_scores_correlation.to_csv(output_dir + 'topic_correlation_scores.csv', encoding='utf-8-sig', index=False)


def compare_models_title(option, method, wv_model, wv_model_paths, tf_model_names, tf_cross_model_names, encoder_type, df_all_file, text_file, pair_file, pair_option, pair_quantiles, pair_n, seed, output_dir, device):
    try:
        df = pd.read_csv(pair_file, encoding='utf-8-sig')
    except:
        print("Pair file not found, creating new pair file...")
        construct_pairs(df_all_file, wv_model, text_file, pair_quantiles, pair_option, pair_n, seed, text_col='text', topic_col='topic')
        print("Pair file created:", pair_file)
        df = pd.read_csv(pair_file, encoding='utf-8-sig')
    
    text1 = df['text1'].tolist()
    text2 = df['text2'].tolist()

    if option == 'title_similarity':
        topic_similarity = df['topic_similarity'].tolist()
    elif option == 'title_binary':
        topic_binary = df['topic_binary'].tolist()

    text_similarities_dict = {}

    similarities_statsdicts = {}

    if 'bi_encoder' in encoder_type:
        text_similarities_dict['bi-encoder'] = {}
        for tf_model_name in tf_model_names.keys():
            print(tf_model_name)
            model = tf_model_name

            text1_vectors = get_text_vectors_sbert(text1, model, device)
            text2_vectors = get_text_vectors_sbert(text2, model, device)
                
            text_similarities = [cosine_similarity(text1_vectors[i], text2_vectors[i]) for i in range(len(text1))]
            text_similarities_dict['bi-encoder'][tf_model_name] = text_similarities

            similarities_statsdict = descriptive_statistics(text_similarities)
            similarities_statsdicts[tf_model_name] = similarities_statsdict

    if 'word_vector' in encoder_type:
        text_similarities_dict['word-vector'] = {}
        for wv_model_path in wv_model_paths.values():
            print(wv_model_path)
            model = load_wv_model(wv_model_path)

            text1_vectors = get_text_vectors_wv(text1, model)
            text2_vectors = get_text_vectors_wv(text2, model)

            text_similarities = [cosine_similarity(text1_vectors[i], text2_vectors[i]) for i in range(len(text1))]
            text_similarities_dict['word-vector'][wv_model_path] = text_similarities

            similarities_statsdict = descriptive_statistics(text_similarities)
            similarities_statsdicts[wv_model_path] = similarities_statsdict

    elif 'cross_encoder' in encoder_type:
        text_similarities_dict['cross-encoder'] = {}
        for tf_cross_model_name in tf_cross_model_names.keys():
            print(tf_cross_model_name)
            model = tf_cross_model_name
 
            text_similarities = load_cross_encoder_and_get_scores_matched(model, text1, text2, device)

            text_similarities_dict['cross-encoder'][tf_cross_model_name] = text_similarities
    
            similarities_statsdict = descriptive_statistics(text_similarities)
            similarities_statsdicts[tf_cross_model_name] = similarities_statsdict

    if option == 'title_similarity':
        if method == 'regression':
            df_scores = pd.DataFrame(columns=['encoder', 'model', 'slope', 'intercept', 'r_squared', 'p_value', 'std_err'])
            for encoder in text_similarities_dict.keys():
                for model_name in text_similarities_dict[encoder].keys():
                    text_similarities = text_similarities_dict[encoder][model_name]
                    df['similarity_' + model_name] = text_similarities
                    slope, intercept, r_value, p_value, std_err = stats.linregress(topic_similarity, text_similarities)
                    print(f"Slope: {slope}")
                    print(f"Intercept: {intercept}")
                    print(f"R-squared: {r_value**2}")
                    print(f"P-value: {p_value}")
                    print(f"Standard error: {std_err}")
                    new_row = {'encoder':encoder, 'model':model_name, 'slope':slope, 'intercept':intercept, 'r_squared':r_value**2, 'p_value':p_value, 'std_err':std_err}
                    df_scores.loc[len(df_scores)] = new_row
                    df_scores.to_csv(output_dir + 'title_linear_scores.csv', encoding='utf-8-sig', index=False)
        elif method == 'correlation':
            df_scores = pd.DataFrame(columns=['encoder', 'model', 'pearson_correlation', 'pearson_p_value', 'spearman_correlation', 'spearman_p_value'])
            for encoder in text_similarities_dict.keys():
                for model_name in text_similarities_dict[encoder].keys():
                    text_similarities = text_similarities_dict[encoder][model_name]
                    df['similarity_' + model_name] = text_similarities
                    pearson_correlation, pearson_p_value = stats.pearsonr(topic_similarity, text_similarities)
                    spearman_correlation, spearman_p_value = stats.spearmanr(topic_similarity, text_similarities)
                    print(f"Pearson Correlation: {pearson_correlation}")
                    print(f"P-value: {pearson_p_value}")
                    print('--------------------------------')
                    print(f"Spearman Correlation: {spearman_correlation}")
                    print(f"P-value: {spearman_p_value}")
                    new_row = {'encoder':encoder, 'model':model_name, 
                               'pearson_correlation':pearson_correlation, 'pearson_p_value':pearson_p_value,
                               'spearman_correlation':spearman_correlation, 'spearman_p_value':spearman_p_value}
                    df_scores.loc[len(df_scores)] = new_row
                    df_scores.to_csv(output_dir + 'title_pearsoncorrelation_scores.csv', encoding='utf-8-sig', index=False)
    
    elif option == 'title_binary':
        if method == 'regression':
            clf = LogisticRegression()
            df_scores = pd.DataFrame(columns=['encoder', 'model', 'accuracy', 'cm', 'precision', 'recall', 'f1', 'fpr', 'tpr', 'thresholds', 'auc'])
            for encoder in text_similarities_dict.keys():
                for model_name in text_similarities_dict[encoder].keys():
                    text_similarities = text_similarities_dict[encoder][model_name]
                    text_similarities = np.array(text_similarities).reshape(-1, 1)
                    topic_binary = np.array(topic_binary)
                    clf.fit(text_similarities, topic_binary)
                    predictions = clf.predict(text_similarities)
                    accuracy = accuracy_score(topic_binary, predictions)
                    cm = confusion_matrix(topic_binary, predictions)
                    precision = precision_score(topic_binary, predictions)
                    recall = recall_score(topic_binary, predictions)
                    f1 = f1_score(topic_binary, predictions)
                    fpr, tpr, thresholds = roc_curve(topic_binary, predictions)
                    auc = roc_auc_score(topic_binary, predictions)
                    new_row = {'encoder':encoder, 'model':model_name, 'accuracy':accuracy, 'cm':cm, 'precision':precision, 'recall':recall, 'f1':f1, 'fpr':fpr, 'tpr':tpr, 'thresholds':thresholds, 'auc':auc}
                    df_scores.loc[len(df_scores)] = new_row
                    df_scores.to_csv(output_dir + 'title_logistic_scores.csv', encoding='utf-8-sig', index=False)
        elif method == 'correlation':
            df_scores = pd.DataFrame(columns=['encoder', 'model', 'pointbiserialr_correlation', 'p_value'])
            for encoder in text_similarities_dict.keys():
                for model_name in text_similarities_dict[encoder].keys():
                    text_similarities = text_similarities_dict[encoder][model_name]
                    correlation, p_value = stats.pointbiserialr(topic_binary, text_similarities)
                    print(f"Correlation: {correlation}")
                    print(f"P-value: {p_value}")
                    new_row = {'encoder':encoder, 'model':model_name, 'pointbiserialr_correlation':correlation, 'p_value':p_value}
                    df_scores.loc[len(df_scores)] = new_row
                    df_scores.to_csv(output_dir + 'title_pointbiserialrcorrelation_scores.csv', encoding='utf-8-sig', index=False)

    with open(output_dir + 'output_similarities_statsdicts.json', 'w') as json_file:
        json.dump(similarities_statsdicts, json_file)

    df.to_csv(output_dir + 'output_similarities.csv', encoding='utf-8-sig', index=False)


def evaluate_similarity(option, method, encoder_type, df_all_file, text_file, pair_file, pair_option, pair_quantiles, pair_n, seed, questionnaire_file, dict_file, output_dir, wv_model):

    # Load config file
    config = yaml.safe_load(open("config.yml", 'r'))
    wv_model_paths = config['wv_model_paths']
    tf_model_names = config['tf_model_names']
    tf_cross_model_names = config['tf_cross_model_names']

    # Load word vector model
    try:
        wv_model = wv_model.wv
    except:
        pass

    # Set text embedding file
    text_dir, text_name = os.path.dirname(text_file), os.path.basename(text_file)
    text_embedding_file_base = text_dir + '/embedding/' + text_name.replace('.csv', '_embedding.npy')

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device: ', device)

    if option == 'topic_similarity':
        compare_models_topic_similarity(method, wv_model, wv_model_paths, tf_model_names, tf_cross_model_names, encoder_type, text_file, text_embedding_file_base, questionnaire_file, dict_file, output_dir, device)
    elif option == 'title_similarity':
        compare_models_title(option, method, wv_model, wv_model_paths, tf_model_names, tf_cross_model_names, encoder_type, df_all_file, text_file, pair_file, pair_option, pair_quantiles, pair_n, seed, output_dir, device)
    elif option == 'title_binary':
        compare_models_title(option, method, wv_model, wv_model_paths, tf_model_names, tf_cross_model_names, encoder_type, df_all_file, text_file, pair_file, pair_option, pair_quantiles, pair_n, seed, output_dir, device)