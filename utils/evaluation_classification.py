from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from gensim.models import KeyedVectors
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
from sentence_transformers import SentenceTransformer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from util import word_segmentation, get_word_vectors_wv, get_text_vectors_wv, get_text_vectors_bert, get_text_vectors_roberta, get_text_vectors_sbert, deduplicate_texts, text_cleaning, load_transformer_and_get_vectors
import yaml
import ast
import os


def cross_validation(vectors, labels, score_option, classifier):

    if classifier == 'svm':
        clf = svm.SVC()
    elif classifier == 'mlp':
        clf = MLPClassifier(random_state=42, max_iter=1000)

    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    scores = cross_val_score(clf, vectors, labels, cv=skf, scoring=score_option)
    
    labels_pred = cross_val_predict(clf, vectors, labels, cv=skf)
    cm = confusion_matrix(labels, labels_pred)

    print('Cross-validation scores: ', scores)
    print('Average score: ', scores.mean())
    
    return cm, scores[0], scores[1], scores[2], scores[3], scores.mean()


# Confusion Matrix Heatmap
def plot_heatmap(cm, label_names, img_save_dir, title):
    # Set the font and solve the problem of messy code
    plt.rcParams["font.sans-serif"]=["SimHei"] 
    plt.rcParams["axes.unicode_minus"]=False 
    plt.figure(figsize=(12, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
    # plt.xlabel('Predicted Labels')
    # plt.ylabel('True Labels')
    plt.title(title)
    img_save_path = img_save_dir + 'confusion_matrix_' + title + '.png'
    plt.savefig(img_save_path, dpi=300)


def plot_barchart(model_names, model_types, avg_scores, img_save_dir, score_option):
    # Set the font and solve the problem of messy code
    plt.rcParams["font.sans-serif"]=["SimHei"] 
    plt.rcParams["axes.unicode_minus"]=False 
    plt.figure(figsize=(16, 16))
    # Use different colors for types of models
    color_dict = {'wordvector':'steelblue', 'transformer':'forestgreen', 'llm':'orange'}
    colors = [color_dict[model_type] for model_type in model_types]
    plt.bar(model_names, avg_scores, color=colors, edgecolor='black')
    for i, score in enumerate(avg_scores):
        plt.text(i, score + 0.01, f"{score:.2f}", ha='center', fontsize=20)
    plt.title('Average Cross-Validation Scores for Different Models', fontsize=30)
    plt.xlabel('Models', fontsize=20)
    plt.ylabel('Average Accuracy', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # let the x-axis labels rotate 45 degrees
    plt.xticks(rotation=45, ha='right')
    img_save_path = img_save_dir + 'average_' + score_option + '.png'
    plt.savefig(img_save_path, dpi=300)


def compare_models(df_text, wv_model_paths, tf_model_names, score_option, classifier, text_type, img_save_dir, score_output, device):

    # Create a dataframe to store scores
    df_score = pd.DataFrame(columns=['model', 'type', 'score_1', 'score_2', 'score_3', 'score_4', 'score_avg'])

    # Get texts and labels
    texts, labels = df_text['text'].tolist(), df_text['topic'].tolist()
    # Remove duplicate texts
    texts, labels = deduplicate_texts(texts, labels)
    

    # Calculate scores for each model of word vectors
    # segment words and save them for reuse
    if text_type == 'text' or 'sentence':
        texts = text_cleaning(texts)
        print("Conducting word segmentation...")
        texts_word_segmented = word_segmentation(texts)

    for key, value in wv_model_paths.items():
        model_name = key
        model_path = value
        model = KeyedVectors.load(model_path)
        if model_name != 'glove':
            model = model.wv
        print(model_name)
        if text_type == 'word':
            vectors = get_word_vectors_wv(texts, model)
        elif text_type == 'text' or 'sentence':
            vectors = get_text_vectors_wv(texts_word_segmented, model, 'word_segmented')
        cm, score_1, score_2, score_3, score_4, score_avg = cross_validation(vectors, labels, score_option, classifier)
        new_row = {'model':model_name, 'type':'wordvector', 'score_1':score_1, 'score_2':score_2, 'score_3':score_3, 'score_4':score_4, 'score_avg':score_avg}
        
        # Save scores
        df_score.to_csv(score_output, index=False)
        df_score.loc[len(df_score)] = new_row
        # Plot confusion matrix heatmap
        # label_names = np.unique(labels)
        # title = model_name
        # plot_heatmap(cm, label_names, img_save_dir, title)


    # Calculate scores for BERT, Roberta, and SBERT model
    for tf_model_name in tf_model_names.keys():
        print(tf_model_name)
        model_path = tf_model_name
        model_type = tf_model_names[tf_model_name]
        
        vectors = get_text_vectors_sbert(texts, tf_model_name, device)
        # Calculate cross validation scores
        cm, score_1, score_2, score_3, score_4, score_avg = cross_validation(vectors, labels, score_option, classifier)
        new_row = {'model':tf_model_name, 'type':'transformer', 'score_1':score_1, 'score_2':score_2, 'score_3':score_3, 'score_4':score_4, 'score_avg':score_avg}
        
        # Save scores
        df_score.loc[len(df_score)] = new_row
        df_score.to_csv(score_output, index=False)
        # Plot confusion matrix heatmap
        # label_names = np.unique(labels)
        # title = model_type + '_' + tf_model_name.split('/')[1]
        # plot_heatmap(cm, label_names, img_save_dir, title)

    # Plot barchart
    # plot_barchart(df_score['model'], df_score['type'], df_score['score_avg'], img_save_dir, score_option)
    # Save scores
    df_score.to_csv(score_output, index=False)


def classification(score_option, classifier, text_type, input, output_dir):

    # Load text list
    df_text = pd.read_csv(input, encoding='utf-8-sig')

    # Load config file
    config = yaml.safe_load(open("config.yml", 'r'))
    wv_model_paths = config['wv_model_paths']
    tf_model_names = config['tf_model_names']

    # Set output paths
    output_subdir = output_dir + text_type + '_classification/'
    if not os.path.exists(output_subdir):
        os.makedirs(output_subdir)
    score_output = output_subdir + 'cross_validation_scores' + '_' + score_option + '.csv'
    img_save_dir = output_subdir

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device: ', device)
    
    # Run cross validation
    compare_models(df_text, wv_model_paths, tf_model_names, score_option, classifier, text_type, img_save_dir, score_output, device)


if __name__ == '__main__':

    score_option = 'accuracy'   # 'accuracy', 'f1_macro'
    classifier = 'svm'    # 'svm', 'mlp'
    text_type = 'text'    # 'word', 'text', 'sentence'
    input = './evaluation/questionnaires_CITL.csv'
    output_dir = './evaluation/ccr_results/classification'

    classification(score_option, classifier, text_type, input, output_dir)