import pandas as pd
import itertools
from .util import word_segmentation


def split_long_sentence_from_middle(sentence, max_length):
    temp_separators = ['：', '，', '；']

    all_sentences = [sentence]
    while max(len(s) for s in all_sentences) > max_length:
        new_sentences = []
        for sentence in all_sentences:
            if len(sentence) <= max_length:
                new_sentences.append(sentence)
            else:
                middle_index = len(sentence) // 2
                left_pointer = middle_index
                right_pointer = middle_index
                while left_pointer >= 0 and right_pointer < len(sentence):
                    # move the left pointer until it finds a punctuation or reaches the beginning of the sentence
                    if sentence[left_pointer] in temp_separators:
                        break
                    left_pointer -= 1
                    # move the right pointer until it finds a punctuation or reaches the end of the sentence
                    if sentence[right_pointer] in temp_separators:
                        break
                    right_pointer += 1

                # If there is no punctuation in the middle of sentence, split the sentence in the middle
                if middle_index - left_pointer > 10 and right_pointer - middle_index > 10:
                    split_index = middle_index
                # If there is punctuation in the sentence, split the sentence at the punctuation
                else:
                    if middle_index - left_pointer <= right_pointer - middle_index:
                        split_index = left_pointer
                    else:
                        split_index = right_pointer
                new_sentences.append(sentence[:split_index])
                new_sentences.append(sentence[split_index+1:])
            all_sentences = new_sentences

    return all_sentences


# Split text into sentences by punctuation, but also consider the length
def sentence_segmentation_by_punc_and_length(text, max_length, min_length):

    quote_separators = ['」', '』']
    other_separators = ['。', '？', '！']
    temp_separators = ['：', '，', '；']
    all_separators = quote_separators + other_separators
    product1 = [item[0]+item[1] for item in itertools.product(quote_separators, other_separators)]
    product2 = [item[1]+item[0] for item in itertools.product(quote_separators, other_separators)]
    product3 = [item[0]+item[1] for item in itertools.product(quote_separators, temp_separators)]
    product4 = [item[1]+item[0] for item in itertools.product(quote_separators, temp_separators)]
    separators_product = product1 + product2 + product3 + product4
    

    # quote_patterns = [r'「.+?」', r'『.+?』']
    # # Find text between quotes
    # for pattern in quote_patterns:
    #     quotes = re.findall(pattern, text)
    #     for quote in quotes:
    #         new_quote = quote
    #         # Replace other separators in quotes with temporary marks
    #         for i in range(len(other_separators)):
    #             new_quote = new_quote.replace(other_separators[i], '[temp]' + str(i))
    #         text = text.replace(quote, new_quote)

    # Replace separators with separators + temporary marks
    for separator in separators_product + all_separators:
        # text = text.replace('\n', '')
        text = text.replace(separator, separator + '[split]')

    # # Replace temporary marks with original separators
    # for i in range(len(other_separators)):
    #     text = text.replace('[temp]' + str(i), other_separators[i])
    
    # Split text into sentences
    sentences = text.split('[split]')

    # new_sentences = []
    # current_sentence = sentences[0]
    # for i in range(1, len(sentences)):
    #     sentence = sentences[i]
    #     if len(current_sentence) >= min_length:
    #         new_sentences.append(current_sentence)
    #         current_sentence = sentence
    #     else:
    #         if len(sentence) + len(current_sentence) < max_length:
    #             current_sentence += sentence
    #         else:
    #             new_sentences.append(current_sentence)
    #             current_sentence = sentence

    # new_sentences.append(current_sentence)

    # return new_sentences

    # Combine sentences shorter than min_length
    sentences_longer_than_min = [sentences[0]]
    for sentence in sentences[1:]:
        if len(sentence) >= min_length:
            sentences_longer_than_min.append(sentence)
        else:
            sentences_longer_than_min[-1] += sentence

    # Split sentences longer than max_length
    sentences_shorter_than_max = []
    for sentence in sentences_longer_than_min:
        if len(sentence) <= max_length:
            sentences_shorter_than_max.append(sentence)
        else:
            new_sentences = split_long_sentence_from_middle(sentence, max_length)
            for new_sentence in new_sentences:
                sentences_shorter_than_max.append(new_sentence)

    sentences = sentences_shorter_than_max

    return sentences


# Split text into segments shorter than max_length, but also consider the punctuation
def text_segmentation_by_punc_and_length(text, max_length, text_min_length, sentence_min_length):
    sentences = sentence_segmentation_by_punc_and_length(text, max_length, sentence_min_length)
    segments = []
    current_segment = ''
    for sentence in sentences:
        if len(current_segment) + len(sentence) < max_length:
            current_segment += sentence
        else:
            segments.append(current_segment)
            current_segment = sentence
    segments.append(current_segment)

    new_segments = [segments[0]]
    for i in range(1, len(segments)):
        segment = segments[i]
        if len(segment) >= text_min_length:
            new_segments.append(segment)
        else:
            new_segments[-1] += segment

    return new_segments


def text_segmentation_by_length(text, max_length):
    segments = []
    total_length = len(text)
    start = 0
    while start < total_length:
        end = start + max_length
        if end > total_length:
            end = total_length
        segments.append(text[start:end])
        start = end
    return segments


def text_merge(input, output, text_column, label_column, min_length):
    try:
        df = pd.read_csv(input, encoding='utf-8-sig', dtype=str)
    except:
        df = pd.read_excel(input)
    columns = df.columns
    new_df = pd.DataFrame(columns=columns)

    previous_label = df[label_column][0]
    current_text = df[text_column][0]
    for i in range(1, len(df)):
        print(str(i+1) + '/' + str(len(df)))
        label = df[label_column][i]
        text = df[text_column][i]

        if label == previous_label and len(text) < min_length:
            current_text += text
        elif label == previous_label and len(current_text) < min_length:
            current_text += text
        else:
            new_row = df.iloc[i-1].copy()
            new_row[text_column] = current_text
            new_row[label_column] = previous_label
            new_df.loc[len(new_df)] = new_row
            previous_label = label
            current_text = text
    new_row = df.iloc[-1].copy()
    new_row[text_column] = current_text
    new_row[label_column] = previous_label
    new_df.loc[len(new_df)] = new_row

    new_df['text_length'] = new_df[text_column].apply(lambda x: len(x))
    new_df.to_csv(output, encoding='utf-8-sig', index=False)

    return new_df


def text_segmentation(input, output, text_column, max_length, text_min_length, sentence_min_length, option, if_word_segmentation):
    
    df = pd.read_csv(input, encoding='utf-8-sig')
    columns = df.columns
    new_df = pd.DataFrame(columns=columns)

    for i in range(len(df)):
        print(str(i+1) + '/' + str(len(df)))
        text = df[text_column][i]
        if option == 'sentence':
            segments = sentence_segmentation_by_punc_and_length(text, max_length, sentence_min_length)
        if option == 'text_end':
            segments = text_segmentation_by_punc_and_length(text, max_length, text_min_length, sentence_min_length)
        elif option == 'text_middle':
            if len(text) <= max_length:
                segments = [text]
            else:
                n = (len(text) // max_length) + 1
                middle_length = len(text) // n
                segments = text_segmentation_by_punc_and_length(text, middle_length, text_min_length, sentence_min_length)
        for segment in segments:
            if if_word_segmentation == True:
                words = word_segmentation(segment)
                segment = '[w]'.join(words)
            new_row = df.iloc[i].copy()
            new_row[text_column] = segment
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    
    new_df['text_length'] = new_df[text_column].apply(lambda x: len(x))
    new_df = new_df[new_df['text_length'] > 0]
    new_df.to_csv(output, encoding='utf-8-sig', index=False)

    return new_df