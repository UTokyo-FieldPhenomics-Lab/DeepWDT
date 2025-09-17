def make_new_video(path_df, path_videos, filtered_runs, path_new_video):
    '''
    Makes an annotated video from the annotation dataframe.
    Args:
        path_df (Path): path to the annotation dataframe, containing columns: frame_id, x0, y0, x1, y1, confidence, run_id, angle
        path_videos (Path): path to the video folder
        filtered_runs (list): id of the runs to keep
        path_new_video (Path): path and filename to the newly made video

    '''
    df_result = pd.read_csv(path_df)

    # remove all lines whose column id is not in the filtered_run list

    # make the new video with annotations on it for each frame (bbox + label id + vector from the center of bbox and
    # the angle from the annotation, keep in mind that the angle is 0 upward and become positive clockwise and negative
    # anticlockwise)

if __name__ == '__main__':
    path_videos = Path(r'C:\Users\CREST\Documents\GitHub\DeepWDT\data\marunouchi_2025_05\videos')
    path_results = Path('C:\Users\CREST\Documents\GitHub\DeepWDT\runs\infer\250522-115133-marunouchi_2025_05')
    video_name =  'marunouchi202505-1-1'
    path_one_result = path_results / Path(video_name)
    path_df = path_one_result / Path('results.csv')
    path_geopackage = path_results / Path('run.gpkg')
    path_save = path_one_result / Path('filtered_results')

    filtered_runs = {
        'marunouchi202505-1-1': [26, 36, 43]
    }

    make_new_video(path_df, path_videos, filtered_runs[video_name], path_save / Path(video_name+'.mp4'))
