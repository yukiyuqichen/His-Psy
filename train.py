import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, SentencesDataset
import sys
import ast
import os


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

    
    pretrained_model_name = sys.argv[1]

    quantiles = sys.argv[2]
    quantiles = ast.literal_eval(quantiles)
    
    triplet_option = sys.argv[3]

    learning_rate = float(sys.argv[4])
    warmup_n_epoch = int(sys.argv[5])

    # Set parameters for training set
    quantiles_str = str(round(quantiles[0]*1000)) + '-' + str(round((quantiles[1]*1000)))
    seed = 42
    if triplet_option == 'random':
        train_file = './dataset/triplets/train_triplet_' + 'quantiles=' + quantiles_str + '_random_seed=' + str(seed) + '.csv'
    elif triplet_option == 'top':
        train_file = './dataset/triplets/train_triplet_' + 'quantiles=' + quantiles_str + '_top_' + pretrained_model_name + '.csv'

    df_train = pd.read_csv(train_file, encoding='utf-8-sig')
    train_length = len(df_train)
    
    # Set parameters for validation set
    pair_option = 'random'  # 'from_triplet', 'random'
    if pair_option == 'random':
        val_file = './dataset/pairs/val_pair_random.csv'
    elif pair_option == 'from_triplet':
        val_file = './dataset/pairs/val_pair_from_triplet_' + 'quantiles=' + quantiles_str + '.csv'

    # Set hyperparameters for training
    optimizer_params = {'lr': learning_rate}
    batch_size = 32
    epochs = 3
    evaluation_steps = 10
    steps_per_epoch = round(train_length / batch_size)
    warmup_steps = steps_per_epoch * warmup_n_epoch
    checkpoint_save_steps = steps_per_epoch

    optimizer_class = torch.optim.Adam

    # Set paths
    model_save_path = './models/sbert/finetuned/' + triplet_option + '_n=1' + '_quantiles=' + quantiles_str + '/'
    model_save_path += pretrained_model_name + '_batch=' + str(batch_size) + '_warmepoch=' + str(warmup_n_epoch) + '_lr=' + str(optimizer_params['lr']) + '/'                                                                                                                       
    print(f"Model saved to {model_save_path}")
    checkpoint_path = model_save_path + 'checkpoint/'

    # Check if path has already existed
    if os.path.exists(model_save_path):
        print("Model has already been trained with these parameters.")
    else:
        # Training
        train(pretrained_model_name, train_file, val_file, model_save_path,
            batch_size, optimizer_class, optimizer_params, epochs, 
            warmup_steps, evaluation_steps, checkpoint_save_steps)


