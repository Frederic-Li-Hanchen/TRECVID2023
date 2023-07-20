from pdb import set_trace as st
from collections import Counter
import yake
import pke
import pandas as pd
from copy import copy

### Function to find the most frequent words
def return_most_frequent_words(input_text, nb_keywords=3):
    # Remove punctuation and formatting
    input_text = input_text.replace(',','')
    input_text = input_text.replace('.','')
    input_text = input_text.replace(';','')
    input_text = input_text.replace(':','')
    input_text = input_text.replace('?','')
    input_text = input_text.replace('!','')
    input_text = input_text.replace('"','')
    input_text = input_text.replace("'",'')
    input_text = input_text = input_text.lower()

    # Remove most common pronouns/prepositions/articles
    words_to_remove = [' the ',' a ',' of ',' i ',' an ',' and ',' you ',' to ',' is ',' are ',' this ',' in ',' we ',' he ',' she ',' it ',' they ',' this ',' that ',' these ',' those ',' on ',' for ',' as ',' so ',' by ',' or ',' from ',' but ',' your ',' be ',' if ',' any ',r'\n']
    for e in words_to_remove:
        input_text = input_text.replace(e,' ')

    # Get list of words
    word_list = input_text.split()

    # Count number of occurences in list
    count = Counter(word_list)
    count = count.most_common()
    
    # Return the most frequent keywords
    keywords = [key for key,_ in count[:nb_keywords]]
    return keywords


### Function to print keywords with different keyword extraction methods
# NOTE: unclean becuase the hyper-parameters of YAKE! are hard-coded in the function, while the hyper-parameters of the other functions are not coded
def print_keywords(input_text,nb_keywords):
    print('')
    print('#######################################################################################################################################################')
    print('# Input text:')
    print('#######################################################################################################################################################')
    print('"'+input_text+'"')
    print('')
    print('Extracting the top %d keywords from the input text ...' % nb_keywords)
    print('')

    print('#######################################################################################################################################################')
    print('# Tests with YAKE! (lower score = more relevant):')
    print('#######################################################################################################################################################')
    language = 'en'
    max_ngram_size = 2
    deduplication_threshold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 3

    kw_extractor = yake.KeywordExtractor(lan=language, 
                                        n=max_ngram_size, 
                                        dedupLim=deduplication_threshold, 
                                        dedupFunc=deduplication_algo, 
                                        windowsSize=windowSize, 
                                        top=nb_keywords)
                                                
    keywords = kw_extractor.extract_keywords(input_text)

    for kw in keywords:
        print(kw)
        

    print('')
    print('#######################################################################################################################################################')
    print('# Tests with TextRank:')
    print('#######################################################################################################################################################')

    # initialize keyphrase extraction model, here TopicRank
    extractor = pke.unsupervised.TextRank()

    # load the content of the document, here document is expected to be a simple 
    # test string and preprocessing is carried out using spacy
    extractor.load_document(input=input_text, language='en')

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    extractor.candidate_selection()

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 3 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=nb_keywords)
    print(keyphrases)
        

    print('')
    print('###########################################################################################################################################################################')
    print('# Tests with SingleRank:')
    print('###########################################################################################################################################################################')

    # initialize keyphrase extraction model
    extractor = pke.unsupervised.SingleRank()

    # load the content of the document, here document is expected to be a simple 
    # test string and preprocessing is carried out using spacy
    extractor.load_document(input=input_text, language='en')

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    extractor.candidate_selection()

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 3 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=nb_keywords)
    print(keyphrases)


    print('')
    print('#######################################################################################################################################################')
    print('# Tests with TopicRank:')
    print('#######################################################################################################################################################')

    # initialize keyphrase extraction model, here TopicRank
    extractor = pke.unsupervised.TopicRank()

    # load the content of the document, here document is expected to be a simple 
    # test string and preprocessing is carried out using spacy
    extractor.load_document(input=input_text, language='en')

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    extractor.candidate_selection()

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 3 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=nb_keywords)
    print(keyphrases)


    print('')
    print('###########################################################################################################################################################################')
    print('# Tests with TopicalPageRank:')
    print('###########################################################################################################################################################################')

    # initialize keyphrase extraction model
    extractor = pke.unsupervised.TopicalPageRank()

    # load the content of the document, here document is expected to be a simple 
    # test string and preprocessing is carried out using spacy
    extractor.load_document(input=input_text, language='en')

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    extractor.candidate_selection()

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 3 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=nb_keywords)
    print(keyphrases)


    print('')
    print('###########################################################################################################################################################################')
    print('# Tests with PositionRank:')
    print('###########################################################################################################################################################################')

    # initialize keyphrase extraction model
    extractor = pke.unsupervised.PositionRank()

    # load the content of the document, here document is expected to be a simple 
    # test string and preprocessing is carried out using spacy
    extractor.load_document(input=input_text, language='en')

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    extractor.candidate_selection()

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 3 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=nb_keywords)
    print(keyphrases)

    print('')
    print('#######################################################################################################################################################')
    print('# Tests with MultipartiteRank:')
    print('#######################################################################################################################################################')

    # initialize keyphrase extraction model
    extractor = pke.unsupervised.MultipartiteRank()

    # load the content of the document, here document is expected to be a simple 
    # test string and preprocessing is carried out using spacy
    extractor.load_document(input=input_text, language='en')

    # keyphrase candidate selection, in the case of TopicRank: sequences of nouns
    # and adjectives (i.e. `(Noun|Adj)*`)
    extractor.candidate_selection()

    # candidate weighting, in the case of TopicRank: using a random walk algorithm
    extractor.candidate_weighting()

    # N-best selection, keyphrases contains the 3 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=nb_keywords)
    print(keyphrases)
    print('')


### Function to save keywords in a csv file given a txt file of transcripts
def save_keywords_in_csv(transcript_path, save_path, nb_keywords=3):
    print('')
    print('Extracting transcripts ...')
    # Load the transcript file
    with open(transcript_path, "r", encoding='utf-8') as file:
        transcripts = file.readlines()
    nb_transcripts = len(transcripts)

    print('')
    print('Extracting keywords for %d transcripts ...' % nb_transcripts)
    results = pd.DataFrame(columns=['transcript','most_common','YAKE!','TextRank','SingleRank','TopicRank','TopicalPageRank','PositionRank','MultipartiteRank'])
    # YAKE! parameters
    language = 'en'
    max_ngram_size = 2
    deduplication_threshold = 0.9
    deduplication_algo = 'seqm'
    windowSize = 3
    # Loop on the transcripts
    for idx in range(nb_transcripts):
        print('    Processing transcript %d/%d ...' % (idx+1,nb_transcripts))
        current_text = transcripts[idx]
        results.at[idx,'transcript'] = current_text
        # Most common
        results.at[idx,'most_common'] = return_most_frequent_words(current_text,nb_keywords)
        # YAKE!
        kw_extractor = yake.KeywordExtractor(lan=language, 
                                        n=max_ngram_size, 
                                        dedupLim=deduplication_threshold, 
                                        dedupFunc=deduplication_algo, 
                                        windowsSize=windowSize, 
                                        top=nb_keywords)                    
        results.at[idx,'YAKE!'] = [e[0] for e in kw_extractor.extract_keywords(current_text)]
        # TextRank
        extractor = pke.unsupervised.TextRank()
        extractor.load_document(input=current_text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        results.at[idx,'TextRank'] = [e[0] for e in extractor.get_n_best(n=nb_keywords)]
        # SingleRank
        extractor = pke.unsupervised.SingleRank()
        extractor.load_document(input=current_text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        results.at[idx,'SingleRank'] = [e[0] for e in extractor.get_n_best(n=nb_keywords)]
        # TopicRank
        extractor = pke.unsupervised.TopicRank()
        extractor.load_document(input=current_text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        results.at[idx,'TopicRank'] = [e[0] for e in extractor.get_n_best(n=nb_keywords)]
        # TopicalPageRank
        extractor = pke.unsupervised.TopicalPageRank()
        extractor.load_document(input=current_text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        results.at[idx,'TopicalPageRank'] = [e[0] for e in extractor.get_n_best(n=nb_keywords)]
        # PositionRank
        extractor = pke.unsupervised.PositionRank()
        extractor.load_document(input=current_text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        results.at[idx,'PositionRank'] = [e[0] for e in extractor.get_n_best(n=nb_keywords)]
        # MultipartiteRank
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=current_text, language='en')
        extractor.candidate_selection()
        extractor.candidate_weighting()
        results.at[idx,'MultipartiteRank'] = [e[0] for e in extractor.get_n_best(n=nb_keywords)]
    
    # Save the results in a csv file
    results.to_csv(save_path,index=False)


### Main
if __name__ == '__main__':
    # # Print keywords
    # input_text = 'Carefully palpate the individual lymph node stations. To facilitate differentiation between lymph nodes and muscles, the area that is palpated should be as relaxed as possible. Every palpable lymph node is considered enlarged. If there is enlargement, pay attention to consistency, tenderness, mobility, the number of enlarged lymph nodes, and any erythema in the affected area.'
    # nb_keywords = 3
    # print_keywords(input_text,nb_keywords)

    # Save keywords in csv file
    transcript_file = './data/transcripts.txt'
    save_keywords_in_csv(transcript_file,'./results/extracted_keywords.csv')

    #return_most_frequent_words('Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.')

