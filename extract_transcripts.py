from youtube_transcript_api import YouTubeTranscriptApi
import numpy as np
import json
from pdb import set_trace as st
import pandas as pd


### Function to find the closest value in a list
def find_closest(element,list):
    diff_list = [abs(e-element) for e in list]
    idx = np.argmin(diff_list)
    return list[idx], idx


### Function to extract the video transcripts from Youtube videos and save them in a csv file
def extract_transcripts(json_path,save_path):
    print('')
    print('Loading the dataset contained in %s ...' % json_path)
    with open(json_path,'r') as f:
        dataset = json.load(f)
    nb_examples = len(dataset)
    print('%d samples found.' % nb_examples)

    # Prepare dataframe to be saved as csv
    result = pd.DataFrame(columns=['sample_id','question','answer_start','answer_end','answer_start_second','answer_end_second','video_length','video_id','video_url','transcript'])

    # Loop on the videos
    print('')
    print('Extracting transcripts ...')
    for idx in range(nb_examples):

        print('    Sample %d/%d' % (idx+1,nb_examples))

        # Save meta-data in the resulting data frame
        video_id = dataset[idx]['video_id']
        result.at[idx,'sample_id'] = dataset[idx]['sample_id']
        result.at[idx,'question'] = dataset[idx]['question']
        result.at[idx,'answer_start'] = dataset[idx]['answer_start']
        result.at[idx,'answer_end'] = dataset[idx]['answer_end']
        result.at[idx,'answer_start_second'] = dataset[idx]['answer_start_second']
        result.at[idx,'answer_end_second'] = dataset[idx]['answer_end_second']
        result.at[idx,'video_length'] = dataset[idx]['video_length']
        result.at[idx,'video_id'] = dataset[idx]['video_id']
        result.at[idx,'video_url'] = dataset[idx]['video_url']

        # Extract the full transcript of the video
        # First check if the English transcript exists
        try:
            full_transcript = YouTubeTranscriptApi.get_transcript(video_id)
        except:
            print('      ERROR: no English transcript found for video "%s"!' % (dataset[idx]['video_id']))
            print('      Skipping transcript extraction for this video.')
            result.at[idx,'transcript'] = ''
        else:
            # Get all the starting timestamps for the transcripts
            start_ts = dataset[idx]['answer_start_second']
            end_ts = dataset[idx]['answer_end_second']
            all_start_ts = [e['start'] for e in full_transcript]

            # Find the closest timestamps to the start and end ones
            closest_start_ts, closest_start_id = find_closest(start_ts,all_start_ts)
            #closest_end_ts, closest_end_id = find_closest(end_ts,all_end_ts)
            closest_end_ts, closest_end_id = find_closest(end_ts,all_start_ts)

            # Extract and concatenate the transcripts corresponding to the period of time delimitated by the found indices
            tmp_str = ''
            if closest_end_ts > end_ts:
                for transcript_idx in range(closest_start_id,closest_end_id):
                    tmp_str += full_transcript[transcript_idx]['text'] + ' '
            else:
                for transcript_idx in range(closest_start_id,closest_end_id+1):
                    tmp_str += full_transcript[transcript_idx]['text'] + ' '
            # Clean formatting of text (e.g. newlines)
            tmp_str = ' '.join(tmp_str.splitlines())
            result.at[idx,'transcript'] = tmp_str

    # Save the csv file at the specified path
    result.to_csv(save_path,index=False)


### Main
if __name__ == '__main__':
    extract_transcripts(json_path='./data/train.json',save_path='./results/train_set.csv')
    extract_transcripts(json_path='./data/test.json',save_path='./results/test_set.csv')
    extract_transcripts(json_path='./data/val.json',save_path='./results/val_set.csv')