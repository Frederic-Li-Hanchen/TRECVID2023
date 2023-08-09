import json
import pandas as pd
from pdb import set_trace as st


### Function to generate the .json file containing the generated questions according to the TRECVID 2023 MIQG challenge requirements
### [str] csv_path: path to the csv file containing the meta data
### [str] text_path: path to the text file containing the generated questions
### [str] res_path: path where to save the json file
### [bool] beam: if True, select the best result with beam search. Otherwise with nucleus outputs 
def generate_json(csv_path, text_path, res_path, beam=True):
    # Output list to be converted to json
    output_list = []
    # Load the meta data
    meta_data = pd.read_csv(csv_path)
    # Load the text file
    with open(text_path,'r') as f:
        lines = [line.rstrip() for line in f]
    # Find the position of the sample indices in the text file
    id_positions = []
    sample_id = meta_data['sample_id']
    for idx in range(len(sample_id)):
        id_positions += [lines.index(sample_id[idx])]
    # Add full number of lines to the id_position list
    id_positions += [len(lines)-1]
    # Loop on the samples
    for idx in range(len(meta_data)):
        tmp_dict = {}
        tmp_dict['sample_id'] = meta_data['sample_id'][idx]
        # Extract the correct generated question
        relevant_text = lines[id_positions[idx]:id_positions[idx+1]]
        if beam:
            pos = relevant_text.index('Beam Outputs:')
        else:
            pos = relevant_text.index('Nucleus Outputs:')
        tmp_dict['question'] = relevant_text[pos+1][3:] # Remove the 3 first characters that are '1: '
        tmp_dict['answer_start'] = meta_data['answer_start'][idx]
        tmp_dict['answer_end'] = meta_data['answer_end'][idx]
        tmp_dict['video_id'] = meta_data['video_id'][idx]
        output_list += [tmp_dict]

    # Generate the json and save it
    with open(res_path,'w',encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=False, indent=4)
       

### Main function
if __name__ == '__main__':
    generate_json(csv_path='./results/miqg_test_set.csv', text_path='./results/summary_epoch4model_generation_results.txt', res_path='./results/generated_test_questions_beam.json', beam=True)
    generate_json(csv_path='./results/miqg_test_set.csv', text_path='./results/summary_epoch4model_generation_results.txt', res_path='./results/generated_test_questions_nucleus.json', beam=False)