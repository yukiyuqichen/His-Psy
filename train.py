import pandas as pd
import numpy as np
import math
import torch
from torchinfo import summary
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import sentence_transformers
from sentence_transformers import SentenceTransformer, CrossEncoder, models, InputExample, losses, evaluation, SentencesDataset
from gensim.models import KeyedVectors
from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
import gc


def train(pretrained_model_name, train_file, val_file, model_save_path,
          batch_size, optimizer_class, optimizer_params, epochs, 
          warmup_steps, evaluation_steps, checkpoint_save_steps):

    # Load the pre-trained model
    model = SentenceTransformer('./models/sbert/pretrained/' + pretrained_model_name)
    print("Model", pretrained_model_name, "loaded...")

    # Load the training data
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    target = df_train['target']
    positive = df_train['positive']
    negative = df_train['negative']
    print("Training on", len(target), "triplets...")

    # Load the validation data
    df_val = pd.read_csv(val_file, encoding='utf-8-sig')
    val_text_1 = df_val['text1']
    val_text_2 = df_val['text2']
    scores = df_val['topic_similarity']
    print("Validating on", len(val_text_1), "pairs...")

    # Define the evaluator
    evaluator = evaluation.EmbeddingSimilarityEvaluator(val_text_1,
                                                        val_text_2,
                                                        scores,
                                                        batch_size=batch_size,
                                                        name='similarity',
                                                        show_progress_bar=True,
                                                        write_csv=True
                                                        )

    # Define the loss function and Send training data to InputExample and Dataloader
    train_loss = losses.TripletLoss(model)
    data = {
        'target': target,
        'positive': positive,
        'negative': negative
    }
    df = pd.DataFrame(data)

    train_examples = df.apply(lambda row: InputExample(texts=[row['target'], row['positive'], row['negative']]), axis=1).tolist()
    train_dataset = SentencesDataset(train_examples, model)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    # Delete the previous data to save memory
    del df_train
    del df_val
    del df
    del data
    del target
    del positive
    del negative

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
            optimizer_class=optimizer_class,
            optimizer_params=optimizer_params,
            epochs=epochs,
            warmup_steps=warmup_steps,
            evaluator=evaluator,
            evaluation_steps=evaluation_steps,
            output_path=model_save_path,
            save_best_model=True,
            show_progress_bar=True,
            use_amp=True,
            checkpoint_path=checkpoint_path,
            checkpoint_save_steps=checkpoint_save_steps
            )

if __name__ == '__main__':

    # Set the pretrained model
    pretrained_model_name = 'bert-ancient-chinese'

    # Set parameters for training set
    quantiles = (0.01, 0.99)
    quantiles_str = str(round(quantiles[0]*1000)) + '-' + str(round((quantiles[1]*1000)))
    triplet_option = 'random'   # 'top', 'random'
    train_file = './dataset/triplets/train_triplet_' + 'quantiles=' + quantiles_str + '_' + triplet_option + '.csv'
    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    train_length = len(df_train)
    
    # Set parameters for validation set
    pair_option = 'random'  # 'from_triplet', 'random'
    if pair_option == 'random':
        val_file = './dataset/pairs/val_pair_random.csv'
    elif pair_option == 'from_triplet':
        val_file = './dataset/pairs/val_pair_from_triplet_' + 'quantiles=' + quantiles_str + '.csv'

    # Set hyperparameters for training
    batch_size = 32
    optimizer_class = torch.optim.Adam
    optimizer_params = {'lr': 1e-5}
    epochs = 5
    evaluation_steps = 10
    warmup_n_epoch = 3

    steps_per_epoch = round(train_length / batch_size)
    warmup_steps = steps_per_epoch * warmup_n_epoch
    checkpoint_save_steps = steps_per_epoch

    # Set paths
    model_save_path = './models/sbert/finetuned/' + triplet_option + '_n=1_' + '_quantiles=' + quantiles_str + '/'
    model_save_path += pretrained_model_name + '_batch=' + str(batch_size) + '_warmepoch=' + str(warmup_n_epoch) + '_lr=' + str(optimizer_params['lr']) + '/'                                                                                                                       
    checkpoint_path = model_save_path + 'checkpoint/'
    
    # Training
    train(pretrained_model_name, train_file, val_file, model_save_path,
          batch_size, optimizer_class, optimizer_params, epochs, 
          warmup_steps, evaluation_steps, checkpoint_save_steps)


