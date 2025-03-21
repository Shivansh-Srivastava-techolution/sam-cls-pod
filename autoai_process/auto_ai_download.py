# Native Library Import
import os
import time
import json
from threading import Thread
import random
# Pip or Installed Library Import
import yaml
import numpy as np
import pandas as pd
from google.cloud import storage
import requests
# Custom file Import
from autoai_process import Config


BUCKET_NAME = Config.AUTOAI_BUCKET
MAX_NUMBER_THREADS = 100

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'faceopen_key.json'

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)

def sending_images(label,filename,model, tag):
    
    url = "https://autoai-backend-exjsxe2nda-uc.a.run.app/resource"
    try:
        payload = {'status': "backlog",
                  'csv': "",
                  'model': model,
                  'label': label,
                  'tag': tag,
                  'confidence_score': "100",
                  'prediction': "rgb",
                  'imageAnnotations': {},
                  'model_type': "image/png"}
        files = [('resource', (filename, open(filename, 'rb'),"image/png"))]
        headers = {}
        response = requests.request(
          'POST', url, headers=headers, data=payload, files=files, verify=False)
      # print(response.text)
        if response.status_code == 200:
            print('Successfully sent to AutoAI', end="\r")
            return True
        else:
            print('Error while sending to AutoAI')
            return False
    except:
        print("failed")


def get_files(folder_path):
    files = [f for f in os.listdir(folder_path)]
    files = sorted(files)
    return files


def download_blob(output_folder_path, blob_path):
    # for v in range(10):
    filename = os.path.basename(blob_path)
    output_file_path = os.path.join(output_folder_path, f"{filename}")

    blob = bucket.blob('/'.join(blob_path.split('/')[3:]))

    blob.download_to_filename(f"{output_file_path}")

def img_annot_txt(image_annotations):
    # annotations_csv = image_annotations.replace("\'", "\"")
    # objects = json.loads(str(annotations_csv))
    objects = image_annotations
    for object in objects:
        try:
            px = [vertex['x'] for vertex in object['vertices']]
            py = [vertex['y'] for vertex in object['vertices']]
            
            x_min = int(np.min(px))
            x_max = int(np.max(px))
            y_min = int(np.min(py))
            y_max = int(np.max(py))

            # bbox = [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
            bbox = []
            bbox.appned([x_min, y_min, x_max, y_max])
            
            print("Got bbox")
            print(bbox)
            return bbox
        except Exception as e:
            print("================")
            print(f"Failed to get img annots {e}")
            print(image_annotations)
            print(objects)
            print("================")
            return None
    print("BBox not annotated")

def get_image_annotation(im_id):
    response = requests.get(f'https://autoai-backend-exjsxe2nda-uc.a.run.app/resource/{im_id}')
    image_annotations = response.json()['imageAnnotations']
    # print(image_annotations)
    return image_annotations

def convert_image_annots_to_json(im_id, output_folder_path, label, blob_path):
    image_annotations = get_image_annotation(im_id)
    bbox = img_annot_txt(image_annotations)
    if bbox:
        filename = os.path.basename(blob_path)
        json_filename = filename.split('.')[0] + '.json'
        json_file_path =os.path.join(output_folder_path, label, json_filename)
        print("json_file_path", json_file_path)
        with open(json_file_path, 'w') as f:
            json.dump(bbox, f)

def download_files(input_csv_path, output_folder_path):
    df = pd.read_csv(input_csv_path)
    gcp_paths = df['GCStorage_file_path']
    labels = df['label']
    image_ids = df['csv']

    threads = []
    start = time.time()

    idx = 0

    for blob_path, label, im_id in zip(gcp_paths, labels, image_ids):
        label = str(label)
        # for v in range(10):
        if not os.path.exists(os.path.join(output_folder_path, label)):
            os.mkdir(os.path.join(output_folder_path, label))

        # output_folder_path = os.path.join(output_folder_path, label)
        
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)

        threads.append(Thread(target=download_blob,
                              # args=(os.path.join(f"{output_folder_path}", f"{label}"), f"{blob_path}")))
                                      args=(f"{os.path.join(output_folder_path, label)}", f"{blob_path}")))
        
        convert_image_annots_to_json(im_id, output_folder_path, label, blob_path)

        if idx % MAX_NUMBER_THREADS == 0:
            for th in threads:
                th.start()
            for th in threads:
                th.join()

            print(f"Data Download Status : {idx}/{df.shape[0]}")
            threads = []

        idx = idx + 1

    if len(threads) > 0:
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        print(f"Data Download Status : {df.shape[0]}/{df.shape[0]}")

    print('Time taken : %s' % str(time.time() - start))

def csv_to_json_convert(csv_file, json_file_name):
    input_csv = csv_file

    images = []

    image_dict = {}

    annotations = []
    categories = []

    catagory_dict = {}

    df = pd.read_csv(input_csv)
    categories_set = set()

    for index, row in (df.iterrows()):
        try:
            annotations_csv = row['imageAnnotations']
            annotations_csv = annotations_csv.replace("\'", "\"")
            objects = json.loads(str(annotations_csv))
            for object in objects:
                label = object['selectedOptions'][1]['value']
                label = label.lower()
                categories_set.add(label)
        except:
            continue

    categories_set = Config.CATEGORIES

    Total_annotation = 0

    for i, label in enumerate(categories_set):
        catagory_dict[label] = i
        categories.append({
            "id": catagory_dict[label],
            "name": label,
            "supercategory": "none"
        })

    annotation_count = 0

    for index, row in (df.iterrows()):
        annotations_csv = row['imageAnnotations']

        image_dict[row['name']] = len(image_dict.keys())
        image = {
            "id": image_dict[row['name']],
            "file_name": row['name'],
            "path": None,
            "width": 640,
            "height": 480,
            "depth": 3
        }
        images.append(image)

        # JSON does not read single quotes. So replacing single quotes with double
        annotations_csv = annotations_csv.replace("\'", "\"")
        objects = json.loads(str(annotations_csv))
        for object in objects:
            Total_annotation += 1
            try:
                label = object['selectedOptions'][1]['value']
                px = [vertex['x'] for vertex in object['vertices']]
                py = [vertex['y'] for vertex in object['vertices']]
                poly = [(vertex['x'], vertex['y'])
                        for vertex in object['vertices']]
                poly = [p for x in poly for p in x]

                obj = {
                    "id": annotation_count,
                    "image_id": image_dict[row['name']],
                    "bbox": [float(np.min(px)), float(np.min(py)), float(np.max(px)) - float(np.min(px)),
                             float(np.max(py)) - float(np.min(py))],
                    "segmentation": [poly],
                    "category_id": catagory_dict[label],
                    "iscrowd": 0
                }
                annotation_count += 1
                annotations.append(obj)
            except:
                print("================")
                print(image)
                pass

    print("Total Annotations", Total_annotation)
    print("Registered Annotations", annotation_count)
    print('\n')

    json_file = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    out_file = open(json_file_name, "w")

    json.dump(json_file, out_file, indent=6)
    out_file.close()



def json_to_txt_convert(json_file, training_id, dataset_dir):
    jsfile = json.load(open(json_file, "r"))

    image_id = {}

    for image in jsfile["images"]:
        image_id[image['id']] = image['file_name']

    for ann in jsfile["annotations"]:
        poly = ann["segmentation"][0]

        xmin = 999
        ymin = 999
        xmax = -1
        ymax = -1

        for i in range(len(poly) // 2):
            xmin = min(xmin, poly[2 * i])
            xmax = max(xmax, poly[2 * i])
            ymin = min(ymin, poly[2 * i + 1])
            ymax = max(ymax, poly[2 * i + 1])

        bbox = [ann["category_id"], (xmax + xmin) / (2 * 1920), (ymax + ymin) / (2 * 1080), (xmax - xmin) / 1920,
                (ymax - ymin) / 1080]

        label_dir = os.path.join(dataset_dir, "labels")

        file = open(os.path.join(
            label_dir, image_id[ann["image_id"]][:-4] + ".txt"), "a")

        file.write(" ".join(map(str, bbox)) + "\n")
        file.close()

        classes = [i["name"] for i in jsfile["categories"]]

        yaml_file = {
            "path": f"../data/{training_id}",
            "train": "images",
            "val": "images",
            "nc": len(classes),
            "names": classes

        }

        yaml_file_path = os.path.join(dataset_dir, f"{training_id}.yaml")
        file = open(yaml_file_path, "w")
        yaml.dump(yaml_file, file)
