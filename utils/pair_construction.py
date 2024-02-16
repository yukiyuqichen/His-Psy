import pandas as pd
import random
import os
from .triplet_construction import construct_triplets


def random_match(lst, seed):
    '''
    lst: list of texts
    return: two lists of texts
    Match texts in the list to form pairs. Each text can only be used once.
    '''
    random.seed(seed) 
    random.shuffle(lst)
    middle_index = len(lst) // 2
    lst_1 = lst[:middle_index]
    if len(lst) % 2 == 0:
        lst_2 = lst[middle_index:]
    else:
        lst_2 = lst[middle_index:]
        n = random.randint(0, len(lst_2)-1)
        lst_1.append(lst_2[n])
    return lst_1, lst_2


def construct_pairs(df_all_file, wv_model, text_file, quantiles, option, n, seed, text_col='text', topic_col='topic'):
    '''
    If option is 'random', match texts randomly to form pairs.
    If option is 'from_triplet', match texts from triplets to form pairs.
    n: number of positive/negative samples for each text
    '''

    text_dir, text_name = os.path.dirname(text_file), os.path.basename(text_file)
    pair_file = text_dir + '/pairs/' + text_name.replace('.csv', '_pair.csv')

    if option == 'random':
        df_text = pd.read_csv(text_file, encoding='utf-8-sig')
        texts = df_text[text_col].tolist()
        topics = df_text[topic_col].tolist()
        # Duplicate texts and topics to form as many pairs as the original number of texts
        texts_topics = list(zip(texts, topics)) * 2
        # If n is greater than 1, duplicate texts_topics n times
        texts_topics = texts_topics * n
        texts_topics_1, texts_topics_2 = random_match(texts_topics, seed)
        texts_1 = [text_topic_1[0] for text_topic_1 in texts_topics_1]
        texts_2 = [text_topic_2[0] for text_topic_2 in texts_topics_2]
        topics_1 = [text_topic_1[1] for text_topic_1 in texts_topics_1]
        topics_2 = [text_topic_2[1] for text_topic_2 in texts_topics_2]
        topic_similarities = [wv_model.similarity(topic_1, topic_2) for topic_1, topic_2 in zip(topics_1, topics_2)]
        df_topic = pd.DataFrame(columns=['text1', 'text2', 'topic1', 'topic2', 'topic_similarity'])

        df_topic['text1'] = texts_1
        df_topic['text2'] = texts_2
        df_topic['topic1'] = topics_1
        df_topic['topic2'] = topics_2
        df_topic['topic_similarity'] = topic_similarities

        pair_file = pair_file.replace('.csv', '_random_seed=' + str(seed) + '.csv')
        df_topic.to_csv(pair_file, encoding='utf-8-sig', index=False)


    elif option == 'from_triplet':
    
        # Load triplets
        quantiles_str = str(round(quantiles[0]*1000)) + '-' + str(round((quantiles[1]*1000)))
        triplet_file = text_dir + '/triplets/' + text_name.replace('.csv', '_triplet_' + 'quantiles=' + quantiles_str + '_random_seed=' + str(seed) + '.csv')

        try:
            df_triplet = pd.read_csv(triplet_file, encoding='utf-8-sig')
            triplets = list(zip(df_triplet['target'], df_triplet['positive'], df_triplet['negative']))
            topics = list(zip(df_triplet['target_topic'], df_triplet['positive_topic'], df_triplet['negative_topic']))

        except:
            print("Triplet file not found, creating new triplet file...")
            tf_model = ''
            # Construct triplets
            construct_triplets(df_all_file, quantiles, text_file, wv_model, tf_model, 'random', n, seed, text_col='text', id_col='text_id', topic_col='topic')
            print("Triplet file created.")
            df_triplet = pd.read_csv(triplet_file, encoding='utf-8-sig')
            triplets = list(zip(df_triplet['target'], df_triplet['positive'], df_triplet['negative']))
            topics = list(zip(df_triplet['target_topic'], df_triplet['positive_topic'], df_triplet['negative_topic']))

        # Construct anchor-positive pairs/anchor-negative pairs and their binary and continuous similarity scores
        df_topic = pd.DataFrame(columns=['text1', 'text2', 'topic1', 'topic2', 'topic_binary', 'topic_similarity'])
        df_topic['text1'] = [triplet[0] for triplet in triplets] + [triplet[0] for triplet in triplets]
        df_topic['text2'] = [triplet[1] for triplet in triplets] + [triplet[2] for triplet in triplets]
        df_topic['topic1'] = [topic[0] for topic in topics] + [topic[0] for topic in topics]
        df_topic['topic2'] = [topic[1] for topic in topics] + [topic[2] for topic in topics]
        df_topic['topic_binary'] = [1 for i in range(len(triplets))] + [0 for i in range(len(triplets))]
        df_topic['topic_similarity'] = [wv_model.similarity(topic[0], topic[1]) for topic in topics] + [wv_model.similarity(topic[0], topic[2]) for topic in topics]

        # Shuffle and pick half data
        df_topic = df_topic.sample(frac=0.5, random_state=seed)
        
        pair_file = pair_file.replace('.csv', '_from_triplet_' + 'quantiles=' + quantiles_str + '_random_seed=' + str(seed) + '.csv')
        df_topic.to_csv(pair_file, encoding='utf-8-sig', index=False)
    