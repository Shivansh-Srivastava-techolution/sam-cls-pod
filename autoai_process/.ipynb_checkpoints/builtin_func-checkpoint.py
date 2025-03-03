# Native Library Import
import os
import requests
import shutil
from datetime import timedelta
import glob

# Pip or Installed Library Import
from colorama import Fore, Style
from halo import Halo

# Custom file Import
from autoai_process import Config
from autoai_process import gcp_train_utils

class auto_ai_connect(object):

    def __init__(self, training_id):
        self.training_id = training_id

    def reset():
        """
        Reset Directories and status

        Returns
        -------
        None.

        """

        global pod_status, model_status
        if os.path.exists(Config.MODELS_PATH):
            shutil.rmtree(Config.MODELS_PATH)
            
        if os.path.exists(Config.DATA_PATH):
            shutil.rmtree(Config.DATA_PATH)
            
        if os.path.exists(os.path.join("results")):
            os.system("rm -r results/*")
            
        pod_status = "Available"
        model_status = "None"

    # About this Function ??
    def signedurl(self):  # ???Check Buckets are Hardcoded
        global storage_client
        bucket = storage_client.get_bucket('detectron2_train_data')
        blob = bucket.blob('{}/{}/{}'.format('yolo_output',
                           self.training_id, 'model_details.zip'))
        url = blob.generate_signed_url(
            version="v4", expiration=timedelta(minutes=15), method="GET")
        return url

    # About this Function ??
    def upload_the_model_file(self, output_dir):  # ??? Check Buckets are Hardcoded
        global storage_client
        bucket = storage_client.get_bucket('detectron2_train_data')
        blob = bucket.blob('{}/{}/{}'.format('yolo_output',
                           self.training_id, 'model_details.zip'))
        blob.upload_from_filename(output_dir)

    def send_accuracy(self, model_id, acc):
        """
        URL: https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/model
        method: PUT

        Sample Request Payload:
        {
            "id": "6215e9cfa609e5940f0529e2",  // modelCollectionID
            "accuracy":23
        }
        """

        URL = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/model/accuracy"
        payload = {
            "id": model_id,
            "accuracy": acc["Total"]
        }
        headers = {'Content-Type': 'application/json'}
        response = requests.request(
            "PUT", url=URL, headers=headers, json=payload, verify=False)
        if response.status_code == 200:
            return response

        print("Sending status , ", response.text)

    def autoai_upload_files_test(self, url, files, parentCheckpointFileId, accuracy):
        print("Uploading files to AutoAI")

        payload = {'id': self.training_id,
                   'parentCheckpointFileId': parentCheckpointFileId,
                   'description': f'Analytic files for {parentCheckpointFileId}',
                   'accuracy': accuracy}

        files_to_send = []

        for file in files:
            files_to_send.append(('filesToUpload', (os.path.basename(
                file), open(file, 'rb'), 'application/octet-stream')))

        headers = {}

        print(files_to_send)
        response = requests.request("PUT", url + '/collection/test/files', headers=headers, data=payload,
                                    files=files_to_send, verify=False)

        if response.status_code == 200:
            print(Fore.GREEN + "Files Uploaded Successfully", Style.RESET_ALL)
        else:
            print(Fore.RED + "Files Uploading Failed", Style.RESET_ALL)
            print(response.text)
        return response

    def delete(training_id):
        print("delete")
        try:
            shutil.rmtree(f"train_{training_id}")
        except Exception as e:
            print("Error: %s" % (e))
            print("delete error")

    def autoai_upload_additional(file_paths, id, parent_checkpoint_id):
        url = Config.ADDITIONAL_FILE_UPLOAD

        payload = {'modelCollectionId': id,
                   'parentCheckpointFileId': parent_checkpoint_id,
                   'description': f'Additional files for {parent_checkpoint_id}'}
        files = []

        for file_path in file_paths:
            files.append(
                (
                    'additionalFiles',
                    (
                        os.path.basename(file_path),
                        open(file_path, 'rb'),
                        ('application/octet-stream' if file_path.split('.')
                         [-1] != 'csv' else 'text/csv')
                    )
                )
            )

        headers = {}

        response = requests.request(
            "POST", url, headers=headers, data=payload, files=files, verify=False)

        if response.status_code == 200:
            print(Fore.GREEN + 'Response', response.text)
            print(Style.RESET_ALL)
        else:
            print(Fore.LIGHTRED_EX + 'Response', response.text)
            print(Style.RESET_ALL)

    def autoai_upload_analytics(file_paths, id, parent_checkpoint_id, test_detail):
        url = Config.ANALYTIC_FILE_UPLOAD

        payload = {'modelCollectionId': id,
                   'parentCheckpointFileId': parent_checkpoint_id,
                   'description': f'Analytic file for {parent_checkpoint_id}',
                   'defaultDataSetCollectionId': test_detail['defaultDataSetCollectionId'],
                   'defaultDataSetCollectionFileName': test_detail['defaultDataSetCollectionFileName'],
                   'accuracy': test_detail['accuracy']}
        files = []

        for file_path in file_paths:
            files.append(
                (
                    'analysisFiles',
                    (
                        os.path.basename(file_path),
                        open(file_path, 'rb'),
                        'application/octet-stream'
                    )
                )
            )

        headers = {}

        response = requests.request(
            "POST", url, headers=headers, data=payload, files=files)

        if response.status_code == 200:
            print(Fore.GREEN + 'Response', response.text)
            print(Style.RESET_ALL)
        else:
            print(Fore.LIGHTRED_EX + 'Response', response.text)
            print(Style.RESET_ALL)

    def autoai_upload_parent_large(file_path, id, description="Model File", isDefault=False):
        print("it came here too")
        
        print(f"id is {id}")
        
        demo_file = open("temp.empty", "w")
        demo_file.close()

        url = Config.PARENT_FILE_UPLOAD

        payload = {'modelCollectionId': id,
                   'isDefaultCheckpoint': ('true' if isDefault else 'false'),
                   'description': description,
                   'appShouldNotUploadFileToGCS': 'true'}
        files = [
            ('parentCheckpointFile',
             (os.path.basename(file_path), open(demo_file.name, 'rb'), 'application/octet-stream'))
        ]
        headers = {}

        response = requests.request(
            "POST", url, headers=headers, data=payload, files=files)

        if response.status_code == 200:
            print(Fore.GREEN + 'Response', response.text)
            print(Style.RESET_ALL)
            data_json = response.json()
            # print the json response

            # Sending file to GCP
            gcp_train_utils.upload_gcp_file(
                source=file_path, destination=data_json['parentCheckpointFileGCSPath'], bucket=data_json['gcsBucketName'])

            return data_json['parentCheckpointFileId']
        else:
            print(Fore.LIGHTRED_EX + 'Response', response.text)
            print(Style.RESET_ALL)
            return False

    def autoai_upload_parent(file_path, id, description="Model File", isDefault=False):
        print("uploading the model file")
        try:
            print("uploading the model file")
            url = Config.PARENT_FILE_UPLOAD

            payload = {'modelCollectionId': id,
                       'isDefaultCheckpoint': ('true' if isDefault else 'false'),
                       'description': description}
            files = [
                ('parentCheckpointFile',
                 (os.path.basename(file_path), open(file_path, 'rb'), 'application/octet-stream'))
            ]
            headers = {}

            response = requests.request(
                "POST", url, headers=headers, data=payload, files=files)

            if response.status_code == 200:
                print(Fore.GREEN + 'Response', response.text)
                print(Style.RESET_ALL)
                data_json = response.json()
                # print the json response
                return data_json['parentCheckpointFileId']
            
        except Exception as e:
            print(e)
            print("RETRYING...........")
            return auto_ai_connect.autoai_upload_parent_large(file_path, id, description="Model File", isDefault=False)

    def autoai_upload_models(file_path, id, parent_checkpoint_id, description=""):
        url = Config.MODEL_FILE_UPLOAD

        payload = {'modelCollectionId': id,
                   'parentCheckpointFileId': parent_checkpoint_id,
                   'description': f'Additional files for {parent_checkpoint_id}'}
        files = [(
            'modelFile',
            (
                os.path.basename(file_path),
                open(file_path, 'rb'),
                'application/octet-stream'
            )
        )]

        headers = {}

        response = requests.request(
            "POST", url, headers=headers, data=payload, files=files)

        if response.status_code == 200:
            print(Fore.GREEN + 'Response', response.text)
            print(Style.RESET_ALL)
        else:
            print(Fore.LIGHTRED_EX + 'Response', response.text)
            print(Style.RESET_ALL)

    def autoai_upload_files(url, files, isDefault, id, test_detail):
        spinner = Halo(text='Parent file upload ', spinner='dots')
        spinner.start()
        # parent_checkpoint_id = None
        print(f" =========={id}===========")
        if len(files['parentfile']) > 0:
            parent_checkpoint_id = auto_ai_connect.autoai_upload_parent(
                file_path=files['parentfile'][0], id=id, isDefault=isDefault, description="Model File"
            )
            
            print(f"parent_checkpoint_id =========== {parent_checkpoint_id}")

        spinner.stop()

        # if not parent_checkpoint_id:
        #     return

        spinner = Halo(text='Analytic file upload ', spinner='dots')
        spinner.start()

        if len(files['analyticfile']) > 0:
            auto_ai_connect.autoai_upload_analytics(
                files['analyticfile'], id, parent_checkpoint_id, test_detail
            )

        spinner.stop()

        spinner = Halo(text='Additional file upload ', spinner='dots')
        spinner.start()

        if len(files['additionalfile']) > 0:
            print()
            auto_ai_connect.autoai_upload_additional(
                files['additionalfile'], id, parent_checkpoint_id
            )

        spinner.stop()

        spinner = Halo(text='Model file upload ', spinner='dots')
        spinner.start()

        if len(files['modelfile']) > 0:
            for file in files['modelfile']:
                auto_ai_connect.autoai_upload_models(
                    file, id, parent_checkpoint_id
                )

        spinner.stop()


class testCollection:
    def __init__(self, id):
        self.testCollectionId = None
        self.parentCheckpointFileId = None
        self.description = None
        self.analysisFiles = []
        self.accuracy = 0
        self.best_ckpt = None
        self._id = id

    def filter_ckpt(self):
        file_list = glob.glob(os.path.join(Config.MODELS_PATH, self._id, 'checkpoints', '*ckpt-*'))
        file_list.sort()
        best_file = file_list[-1]
        self.best_ckpt = ".".join(best_file.split(".")[:-1])
        return self.best_ckpt

    def upload_to_autoai(self):
        url = Config.TEST_COLLECTION_ANALYTIC_FILE_UPLOAD
        payload = {'testCollectionId': self.testCollectionId,
                   'parentCheckpointFileId': self.parentCheckpointFileId,
                   'description': f'Additional files for {self.parentCheckpointFileId}',
                   'accuracy': self.accuracy}
        files = []

        for file_path in self.analysisFiles:
            files.append(
                (
                    'analysisFiles',
                    (
                        os.path.basename(file_path),
                        open(file_path, 'rb'),
                        ('application/octet-stream' if file_path.split('.')[-1] != 'csv' else 'text/csv')
                    )
                )
            )

        headers = {}

        response = requests.request("POST", url, headers=headers, data=payload, files=files, verify=False)

        if response.status_code == 200:
            print(Fore.GREEN + 'Response', response.text)
            print(Style.RESET_ALL)
        else:
            print(Fore.LIGHTRED_EX + 'Response', response.text)
            print(Style.RESET_ALL)

    def __str__(self):
        print(self.testCollectionId, '\n', self.parentCheckpointFileId, '\n', self.description, '\n',
              self.analysisFiles, '\n', self.accuracy)
        return ""

