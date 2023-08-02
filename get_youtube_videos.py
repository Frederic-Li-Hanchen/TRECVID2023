from pytube import YouTube
from moviepy.editor import VideoFileClip
import pandas as pd
import os
from time import time, sleep
from pdb import set_trace as st
from extract_transcripts import convert_ts


### Function to download a Youtube video given its url
### [str] url: Youtube url
### [str] save_path: path to the folder where the video should be saved
### [str] file_name: name of the file to be saved, without extension
### [bool] login: use authentification information to obtain videos behind age verification
def download_video(url,save_path,file_name,login=False):
    # Get the video
    yt = YouTube(url,use_oauth=login,allow_oauth_cache=login)
    # List streams per quality
    stream_list = yt.streams.filter(adaptive=True)
    # Get tag of the first stream in the list
    stream = yt.streams.get_by_itag(stream_list[0].itag)
    # Find extension of file
    extension = stream.mime_type
    pos = extension.find('/')
    extension = '.'+extension[pos+1:]
    # Download the video
    stream.download(save_path,str(file_name).zfill(4)+extension)
    # Return the path and filename to the saved video
    return save_path, str(file_name).zfill(4)+extension


### Function to download all videos given a list of videos in csv format
### [str] csv_file: path to the csv file containing url, sample IDs and start/end timestamps
### [str] save_path: path to the folder where videos should be saved
### [int list] sample_list: list of samples to process (optional)
### [bool] login: use authentification information to obtain videos behind age verification (default: False)
### [bool] verbose: shows the progress for the clipping of the videos (default: False)
def download_all_videos(csv_file,save_path,sample_list=[],login=False,verbose=False):
    # Prepare save folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Read the csv file of examples
    file_list_all = pd.read_csv(csv_file)
    # Filter the list of samples to keep only the selected ones
    if len(sample_list) > 0:
        file_list = file_list_all[file_list_all['sample_id'].isin(sample_list)]
    else:
        file_list = file_list_all
    nb_examples = len(file_list)
    # Loop on the videos
    if verbose:
        log = 'bar'
    else:
        log = None

    columns = list(file_list.columns) # NOTE: assumption that columns contain at least 'video_id'
    if 'answer_start_second' in columns:
        start_ts_name = 'answer_start_second'
    else:
        start_ts_name = 'answer_start'
    if 'answer_end_second' in columns:
        end_ts_name = 'answer_end_second'
    else:
        end_ts_name = 'answer_end'

    for idx in range(5): # DEBUG
    #for idx in range(nb_examples):
        current_sample = file_list.iloc[idx]
        if 'video_url' in columns:
            url = current_sample['video_url']
        else: 
            url = r'https://www.youtube.com/watch?v=' + current_sample['video_id']
        file_name = current_sample['sample_id']
        start_ts = current_sample[start_ts_name]
        end_ts = current_sample[end_ts_name]
        if type(start_ts) is str:
            start_ts = convert_ts(start_ts)
        if type(end_ts) is str:
            end_ts = convert_ts(end_ts)
        start = time()
        try:
            video_path, video_name = download_video(url,save_path,str(file_name)+'_tmp',login)
        except Exception as error:
            print('ERROR: video download failed for sample %d!'%file_name)
            print(error)
        else:
            # Extract clip
            clip = VideoFileClip(os.path.join(video_path,video_name)).subclip(start_ts,end_ts)
            # Prepare name of video to save
            new_video_name = video_name.replace('_tmp','')
            # Save video clip
            clip.write_videofile(os.path.join(video_path,new_video_name),logger=log)
            # Delete old video
            clip.close()
            os.remove(os.path.join(video_path,video_name))
            end = time()
            print('Sample %s downloaded and clipped in %.2f seconds' % (str(file_name),end-start))
            print('')


### Main function
if __name__ == '__main__':
    # # Test of video download
    # download_video('https://www.youtube.com/watch?v=h5MvX50zTLM','./results','video')

    # Download all videos
    #download_all_videos('./results/val_set.csv','./results/val_videos')
    #download_all_videos('./results/train_set.csv','./results/train_videos',[1566,1567,1568,1569],verbose=True,login=True)
    #download_all_videos('./results/test_set.csv','./results/test_videos',[2718,2719,2720,2721,2722],login=True)
    download_all_videos('./results/miqg_test_set.csv','./results/test_videos',verbose=True,login=False)