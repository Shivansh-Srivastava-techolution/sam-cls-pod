import requests
from google.cloud import storage
import pandas as pd
import time
import os
from threading import Thread
from os import listdir
from autoai_process import Config

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'faceopen_key.json'
storage_client = storage.Client()


def get_files(folder_path):
    files = [f for f in listdir(folder_path)]
    files = sorted(files)
    return files


def download_gcp_file(source, destination):
    if 'gs://' in source:
        gcs = source.replace('gs://', '')
        bucket = gcs.split('/')[0]
        source = gcs.replace(f'{bucket}/', '')
        bucket = storage_client.bucket(bucket)
    else:
        bucket = storage_client.bucket(Config.AUTOAI_BUCKET)
    blob = bucket.blob(source)
    blob.download_to_filename(destination)
    return True


def upload_gcp_file(source, destination, bucket):
    """Uploads a file to the bucket."""

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(destination)

    blob.upload_from_filename(source)

    print(
        f"File {source} uploaded to {destination}."
    )


def download_bulk_files(filepath, destination):
    os.system('cat %s | gsutil -m cp -I %s' % (filepath, destination))


def upload_files(url, files_to_be_sent, default_index, id):
    print("Uploading to AutoAI API call")
    payload = {}
    files = []
    payload['id'] = id

    for index, file in enumerate(files_to_be_sent):
        payload['files[%s][description]' % str(index)] = 'None'
        files.append(('filesToUpload', (os.path.basename(file),
                     open(file, 'rb'), 'application/octet-stream')))
    payload['files[%s][isDefaultCheckpoint' % str(default_index)] = 'true'
    headers = {}

    response = requests.request(
        "PUT", url, headers=headers, data=payload, files=files, verify=False)
    return response


def download_blob(bucket, source_blob_name, destination_file_name):
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    return True


def thread_download_from_autoai(max_number_threads, input_csv_path, output_folder_path):
    df = pd.read_csv(input_csv_path)

    names = df['name']
    labels = df['label']
    statuses = df['status']
    gcp_paths = df['GCStorage_file_path']

    index = 0
    start = time.time()
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)

    # Downloading using threads
    while index < len(names):
        print(index)
        try:
            num_threads = max_number_threads
            threads = []
            for i in range(num_threads):
                threads.append(Thread(target=download_gcp_file,
                                      args=(gcp_paths[index + i], os.path.join(output_folder_path, names[index + i]),)))
            for i in threads:
                i.start()
            for i in threads:
                i.join()
            index = index + num_threads
        except KeyError or IndexError:
            num_threads = 1
            threads = []
            for i in range(num_threads):
                threads.append(Thread(target=download_gcp_file,
                                      args=(gcp_paths[index + i], os.path.join(output_folder_path, names[index + i]),)))
            for i in threads:
                i.start()
            for i in threads:
                i.join()
            index = index + num_threads
        except Exception as e:
            print('Unknown error : %s' % str(e))
            exit()

    # Arranging all the files after downloading
    print('Time taken : %s' % str(time.time() - start))


def data_preprocessing():
    pass
