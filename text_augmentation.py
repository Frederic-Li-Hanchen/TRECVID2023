import nltk
from pdb import set_trace as st
import pandas as pd
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from textaugment import EDA, Word2vec, Wordnet, Translate
import gensim


### Function to generate augmented questions using the textaugment EDA synonym generative approach
### [str] csv_path: path to the csv file containing the dataset
### [str] res_path: path to the csv file of results to be saved
### [int] nb_sentences: nb of sentences to be generated
### [str] augmentation: type of approach for augmentation. Has to be in ['eda','word2vec','wordnet','translation','double_translation','hybrid']
def dataset_augmentation(csv_path, res_path, nb_sentences=2, augmentation='eda'):

    # Load the dataset
    data = pd.read_csv(csv_path)
    # Initialise the result panda frame to be saved as csv
    res = pd.DataFrame(columns=data.columns)
    # Initialise data augmenters
    if augmentation == 'eda':
        augmenter = EDA()
    elif augmentation == 'word2vec':
        model = gensim.models.KeyedVectors.load_word2vec_format("./models/GoogleNews-vectors-negative300.bin.gz", binary=True)
        augmenter = Word2vec(model=model)
    elif augmentation == 'wordnet':
        augmenter = Wordnet()
    elif augmentation == 'translation':
        augmenter = Translate(src='en',to='fr')
    elif augmentation == 'double_translation':
        augmenter = Translate(src='en',to='fr')
        augmenter2 = Translate(src='en',to='de')
    elif augmentation == 'hybrid':
        augmenter = Wordnet()
        augmenter2 = EDA()
    else:
        print('ERROR: improper augmentation technique! Must take one of the following values:')
        print(['eda','word2vec','wordnet','translation','double_translation','hybrid'])
        return

    running_idx = 0
    # Loop on the examples of the dataset
    for idx in range(len(data)):
        print('Augmenting example %d/%d ...' % (idx+1,len(data)))
        # Add current example to augmented dataset
        res.loc[running_idx] = data.iloc[idx]
        running_idx += 1
        # Copy meta-data
        tmp_sample_id = data['sample_id'].iloc[idx]
        tmp_answer_start = data['answer_start'].iloc[idx]
        tmp_answer_end = data['answer_end'].iloc[idx]
        tmp_answer_start_second = data['answer_start_second'].iloc[idx]
        tmp_answer_end_second = data['answer_end_second'].iloc[idx]
        tmp_video_length = data['video_length'].iloc[idx]
        tmp_video_id = data['video_id'].iloc[idx]
        tmp_video_url = data['question'].iloc[idx]
        tmp_transcript = data['transcript'].iloc[idx]
        # Get the question
        question = data['question'].iloc[idx]
        # Perform augmentation on the question as many times as requested
        if augmentation == 'hybrid':
            for augment_idx in range(nb_sentences):
                new_sentence = augmenter.augment(question)
                new_sentence2 = augmenter2.synonym_replacement(question)
                res.loc[running_idx] = [tmp_sample_id+0.1*(augment_idx+1),
                            new_sentence,
                            tmp_answer_start,
                            tmp_answer_end,
                            tmp_answer_start_second,
                            tmp_answer_end_second,
                            tmp_video_length,
                            tmp_video_id,
                            tmp_video_url,
                            tmp_transcript]
                res.loc[running_idx+1] = [tmp_sample_id+0.1+0.1*(augment_idx+2),
                            new_sentence2,
                            tmp_answer_start,
                            tmp_answer_end,
                            tmp_answer_start_second,
                            tmp_answer_end_second,
                            tmp_video_length,
                            tmp_video_id,
                            tmp_video_url,
                            tmp_transcript]
                running_idx += 2
        else:
            for augment_idx in range(nb_sentences):
                if augmentation == 'eda':
                    new_sentence = augmenter.synonym_replacement(question)
                elif augmentation == 'double_translation':
                    tmp_new_sentence = augmenter.augment(question)
                    new_sentence = augmenter2.augment(tmp_new_sentence)
                else:
                    new_sentence = augmenter.augment(question)
                res.loc[running_idx] = [tmp_sample_id+0.1*(augment_idx+1),
                            new_sentence,
                            tmp_answer_start,
                            tmp_answer_end,
                            tmp_answer_start_second,
                            tmp_answer_end_second,
                            tmp_video_length,
                            tmp_video_id,
                            tmp_video_url,
                            tmp_transcript]
                running_idx += 1
    
    # Save the new result frame
    res.to_csv(res_path,index=False)


### Main function
if __name__ == '__main__':
    # # Synonym augmentation
    #dataset_augmentation(csv_path='./results/val_set.csv', res_path='./results/augmented_val_set_synonym.csv', nb_sentences=3, augmentation='eda')

    # # Word2Vec augmentation
    #dataset_augmentation(csv_path='./results/val_set.csv', res_path='./results/augmented_val_set_word2vec.csv', nb_sentences=3, augmentation='word2vec')

    # # WordNet augmentation
    # dataset_augmentation(csv_path='./results/val_set.csv', res_path='./results/augmented_val_set_wordnet.csv', nb_sentences=3, augmentation='wordnet')

    # # Translate augmentation
    # dataset_augmentation(csv_path='./results/val_set.csv', res_path='./results/augmented_val_set_translate.csv', nb_sentences=3, augmentation='translation')

    # Double translate augmentation
    #dataset_augmentation(csv_path='./results/val_set.csv', res_path='./results/augmented_val_set_double_translate.csv', nb_sentences=3, augmentation='double_translation')

    # Augmentation with both WordNet and Synonyms
    dataset_augmentation(csv_path='./data/train_set.csv', res_path='./results/augmented_train_set_hybrid.csv', nb_sentences=2, augmentation='hybrid')