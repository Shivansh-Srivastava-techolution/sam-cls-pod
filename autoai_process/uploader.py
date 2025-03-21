import os
from concurrent.futures import ThreadPoolExecutor
import requests

def convert_video(input_path, output_path):
    os.system(f"ffmpeg -i '{input_path}' -c:v libx264 '{output_path}'")

def sending_videos(label, filename, model_id, tag, csv):

    rlef_path = 'rlef_videos'
    os.makedirs(rlef_path, exist_ok=True)

    file_name = os.path.basename(filename)
    rlef_video_path = os.path.join(rlef_path, file_name)
    convert_video(filename, rlef_video_path)

    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
    payload = {
        'model': model_id,
        'status': 'backlog',
        'csv': csv,
        'label': label,
        'tag': tag,
        'model_type': 'video',
        'prediction': "predicted",
        'confidence_score': '100',
        'appShouldNotUploadResourceFileToGCS': 'true',
        'resourceFileName': rlef_video_path,
        'resourceContentType': "video/mp4"
    }
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload)

    headers = {'Content-Type': 'video/mp4'}

    print(response.status_code)

    api_url_upload = response.json()["resourceFileSignedUrlForUpload"]

    response = requests.request("PUT", api_url_upload, headers=headers, data=open(os.path.join(rlef_video_path), 'rb'))
    os.remove(rlef_video_path)
    print(response.text)
    if response.status_code == 200:
        print("Video sent to RLEF sucessfully")

def send_tracking_videos_in_threads(model_id = "67dc28d05e236c564cdde2e3", folder="sam2_results", max_workers=4):
    videos = os.listdir(folder)

    # Optional: define a small helper function for clarity
    def send_video(video_filename):
        video_path = os.path.join(folder, video_filename)
        sending_videos(
            label="short_hanging",
            filename=video_path,
            model_id=model_id,
            tag="Tracking",
            csv="csv"
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(send_video, videos)

    # Once we exit the 'with' block, all tasks have completed
    print("All videos have been submitted.")