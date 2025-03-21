import os
import uuid
import requests

def convert_video(input_path, output_path):
    os.system(f"ffmpeg -i '{input_path}' -c:v libx264 '{output_path}'")

def sending_videos(label, filename, model_id, tag, csv):

    # rlef_path = 'rlef_videos'
    # os.makedirs(rlef_path, exist_ok=True)

    # rlef_video_path = f"{rlef_path}/{uuid.uuid4()}.mp4"
    # convert_video(filename, rlef_video_path)

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
        'resourceFileName': filename,
        'resourceContentType': "video/mp4"
    }
    headers = {}

    response = requests.request("POST", url, headers=headers, data=payload)

    headers = {'Content-Type': 'video/mp4'}

    print(response.status_code)

    api_url_upload = response.json()["resourceFileSignedUrlForUpload"]

    response = requests.request("PUT", api_url_upload, headers=headers, data=open(os.path.join(filename), 'rb'))
    # os.remove(filename)
    print(response.text)
    if response.status_code == 200:
        print("Video sent to RLEF sucessfully")