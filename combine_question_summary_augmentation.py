import pandas as pd
from pdb import set_trace as st
from copy import copy
import numpy as np

### Script to combine both the augmented questions and summaries

# Load the csv of augmented questions
augmented_questions = pd.read_csv('./results/augmented_train_set_hybrid.csv')

# Load the csv of augmented summaries (+ keywords)
augmented_texts = pd.read_csv('./results/train_summary_10600_keywords.csv')
augmented_texts.drop(columns=augmented_texts.columns[0], axis=1,  inplace=True) # Drop the index column

# Determine the size of the final augmented dataset
nb_summaries = len(augmented_texts)
total_nb_ex = nb_summaries*5 # 1 original question + 4 augmented versions
res = pd.DataFrame(columns=['sample_id','question','summary','answer_start','answer_end','answer_start_second','answer_end_second','video_length','video_id','video_url','transcript','keywords'], index=np.arange(total_nb_ex))

# Merge the augmentations
current_idx = 0
for idx in range(nb_summaries):
#for idx in range(50): # DEBUG
    print('Processing sample %d/%d and its augmented versions' % (idx+1,nb_summaries))
    # Get the orginal example
    original_sample = augmented_texts.iloc[idx]
    current_sample_id = original_sample['sample_id']
    # Extract and save the sample information
    res['sample_id'].loc[current_idx] = current_sample_id
    res['summary'].loc[current_idx] = original_sample['summary']
    res['question'].loc[current_idx] = original_sample['question']
    res['answer_start'].loc[current_idx] = original_sample['answer_start']
    res['answer_end'].loc[current_idx] = original_sample['answer_end']
    res['answer_start_second'].loc[current_idx] = original_sample['answer_start_second']
    res['answer_end_second'].loc[current_idx] = original_sample['answer_end_second']
    res['video_length'].loc[current_idx] = original_sample['video_length']
    res['video_id'].loc[current_idx] = original_sample['video_id']
    res['video_url'].loc[current_idx] = original_sample['video_url']
    res['transcript'].loc[current_idx] = original_sample['transcript']
    res['keywords'].loc[current_idx] = original_sample['topical_page_rank']
    current_idx += 1
    # Save the question augmented versions corresponding to this transcript
    associated_questions = augmented_questions[round(augmented_questions['sample_id'])==current_sample_id]
    # Remove the sample corresponding to the original question
    associated_questions = associated_questions[associated_questions['sample_id']!=current_sample_id]
    # Save the augmented questions with the new question
    for idx2 in range(len(associated_questions)):
        res['sample_id'].loc[current_idx] = associated_questions['sample_id'].iloc[idx2]
        res['summary'].loc[current_idx] = original_sample['summary']
        res['question'].loc[current_idx] = associated_questions['question'].iloc[idx2]
        res['answer_start'].loc[current_idx] = original_sample['answer_start']
        res['answer_end'].loc[current_idx] = original_sample['answer_end']
        res['answer_start_second'].loc[current_idx] = original_sample['answer_start_second']
        res['answer_end_second'].loc[current_idx] = original_sample['answer_end_second']
        res['video_length'].loc[current_idx] = original_sample['video_length']
        res['video_id'].loc[current_idx] = original_sample['video_id']
        res['video_url'].loc[current_idx] = original_sample['video_url']
        res['transcript'].loc[current_idx] = original_sample['transcript']
        res['keywords'].loc[current_idx] = original_sample['topical_page_rank']
        current_idx += 1
    
# Remove duplicates
print('Removing duplicates ...')
tmp_res = copy(res)
tmp_res.drop(columns=['sample_id'], axis=1,  inplace=True) # Drop the sample column
#tmp_res = tmp_res[:50] # DEBUG

# Lower case the questions
for idx in range(len(tmp_res)):
    tmp_res['question'].iloc[idx] = tmp_res['question'].iloc[idx].lower()

# Duplicate indices
duplicate_id = tmp_res.duplicated()

# Remove entries with duplicate indices
res = res[~duplicate_id]

# Save the full augmented dataset
res.to_csv('./results/full_augmented_train_set.csv',index=False)
