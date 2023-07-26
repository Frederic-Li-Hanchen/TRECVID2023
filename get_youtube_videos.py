from pytube import YouTube
from moviepy.editor import VideoFileClip
import pandas as pd
import os
from time import time, sleep
from pdb import set_trace as st


### Function to download a Youtube video given its url
### [str] url: Youtube url
### [str] save_path: path to the folder where the video should be saved
### [str] file_name: name of the file to be saved, without extension
def download_video(url,save_path,file_name):
    # Get the video
    yt = YouTube(url)
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
def download_all_videos(csv_file,save_path):
    # Prepare save folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Read the csv file of examples
    file_list = pd.read_csv(csv_file)
    nb_examples = len(file_list)
    # Loop on the videos
    for idx in range(nb_examples):
    #for idx in range(5): # DEBUG
        url = file_list.iloc[idx]['video_url']
        file_name = file_list.iloc[idx]['sample_id']
        start_ts = file_list.iloc[idx]['answer_start_second']
        end_ts = file_list.iloc[idx]['answer_end_second']
        try:
            start = time()
            video_path, video_name = download_video(url,save_path,str(file_name)+'_tmp')
        except:
            print('ERROR: video download failed for sample %d!'%file_name)
        else:
            # Extract clip
            clip = VideoFileClip(os.path.join(video_path,video_name)).subclip(start_ts,end_ts)
            # Prepare name of video to save
            new_video_name = video_name.replace('_tmp','')
            # Save video clip
            clip.write_videofile(os.path.join(video_path,new_video_name))
            # Delete old video
            clip.close()
            os.remove(os.path.join(video_path,video_name))
            end = time()
            print('Sample %d downloaded and clipped in %.2f seconds' % (file_name,end-start))
            print('')


### Main function
if __name__ == '__main__':
    # # Test of video download
    # download_video('https://www.youtube.com/watch?v=h5MvX50zTLM','./results','video')

    # Download all videos
    download_all_videos('./results/val_set.csv','./results/val_videos')
    #download_all_videos('./results/train_set.csv','./results/train_videos')
    #download_all_videos('./results/test_set.csv','./results/test_videos')