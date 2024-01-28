#!pip install roboflow


#from roboflow import Roboflow
#rf = Roboflow(api_key="htpcxp3XQh7SsgMfjJns")
#project = rf.workspace("ai-79z1a").project("basketball_child")
#dataset = project.version(6).download("coco-segmentation")


from roboflow import Roboflow
import torch
import requests
import yt_dlp
import os

def download_videos_from_youtube(video_urls, output_path):
    """
    Downloads videos from YouTube.

    Args:
    video_urls (list): List of YouTube video URLs.
    output_path (str): Directory where videos will be saved.

    Returns:
    tuple: A tuple containing lists of successful and failed downloads.
    """

    ydl_opts = {
        'format': 'best',
        'outtmpl': output_path + '/%(title)s.%(ext)s',
        'quiet': True
    }

    failed_downloads = []
    successful_downloads = []

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in video_urls:
            try:
                ydl.download([url])
                print(f"Successfully downloaded {url}")
                successful_downloads.append(url)
            except Exception as e:
                print(f"Failed to download {url}: {e}")
                failed_downloads.append(url)

    return successful_downloads, failed_downloads


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_project(api_key, workspace, project_name, version):
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    dataset = project.version(version).download("coco-segmentation")
    return dataset

def download_files(urls):
    for url in urls:
        response = requests.get(url)
        if response.status_code == 200:
            with open(url.split("/")[-1], 'wb') as file:
                file.write(response.content)
        else:
            print(f"Failed to download {url}. Status code: {response.status_code}")


def construct_dataset_paths(project_name, version):
    base_path = f"{project_name}-{version}"
    train_annotation_path = f"{base_path}/train/_annotations.coco.json"
    valid_annotation_path = f"{base_path}/valid/_annotations.coco.json"
    test_annotation_path = f"{base_path}/test/_annotations.coco.json"

    train_root_dir = f"{base_path}/train"
    valid_root_dir = f"{base_path}/valid"
    test_root_dir = f"{base_path}/test"

    return train_annotation_path, valid_annotation_path, test_annotation_path, train_root_dir, valid_root_dir, test_root_dir

def create_directory(dir_path):
    """Create a directory if it does not exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
