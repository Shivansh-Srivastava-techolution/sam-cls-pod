# Native Library Import
import json
import os
import sys
import traceback

import random

# Pip or Installed Library Import
from colorama import Fore, Style

# Custom file Import
from autoai_process import Config
from autoai_process import uploader
from autoai_process.auto_ai_download import csv_to_json_convert, download_files
from autoai_process.builtin_func import auto_ai_connect
from autoai_process.gcp_train_utils import data_preprocessing
from training import test_model, train_model, data_creator

class training_function():
    def __init__(self, model_details, retrain):
        self.model_details = model_details
        self.retrain = retrain

        # Paths can be updated, added, or removed based on developer requirements #

        # This is where training data would be stored
        self.dataset_path = os.path.join(
            Config.DATA_PATH, self.model_details['_id'], "train_data")
        # This is where the weights of the trained model would be stored
        self.model_save_path = os.path.join(
            Config.MODELS_PATH, self.model_details['_id'], "cnn1d_model.pth")
        # Path to csv file which contains info about the training data
        self.train_csv_path = os.path.join(
            Config.DATA_PATH, self.model_details['_id'], self.model_details['resourcesFileName'])

        # This is where testing data would be stored
        self.test_dataset_path = os.path.join(
            Config.DATA_PATH, f"test_{self.model_details['_id']}", "test_data")
        # Path to csv file which contains info about the testing data
        self.test_csv_path = os.path.join(
            Config.DATA_PATH, f"test_{self.model_details['_id']}",
            self.model_details['defaultDataSetCollectionResourcesFileName'])

    def train(self):
        print("============== model_details ==============")
        print(self.model_details)

        Config.MODEL_STATUS = 'Downloading Data'

        # Downloading the data from the csv file for training
        download_files(self.train_csv_path, self.dataset_path)
        download_files(self.test_csv_path, self.test_dataset_path)

        # creating samurai dataset
        data_creator.main(self.dataset_path, "Train")
        data_creator.main(self.test_dataset_path, "Test")

        Config.MODEL_STATUS = "Training"

        print('Training started')

        # The call to start the training of the model should be made here #
        # train_dataset_path, test_dataset_path, models_path, model_details, train_csv_path, test_csv_path
        cls_map = train_model.main(train_dataset_path=self.dataset_path,
                        test_dataset_path=self.test_dataset_path,
                        model_details=self.model_details,
                        model_save_path = self.model_save_path)
        
        with open("cls_map.json", 'w') as f:
            json.dump(cls_map, f)

        print('Training Completed')

        files_to_send = {
            'parentfile': [],
            'modelfile': [],
            'analyticfile': [],
            'additionalfile': []
        }
        
        files_to_send['parentfile'].append(self.model_save_path)
        files_to_send['additionalfile'].append("cls_map.json")

        # =============== Starting the test =============== #

        print("Testing Started")
        Config.MODEL_STATUS = 'Testing'

        test_detail = dict()
        try:
            print(self.model_details)
            print(self.model_details['defaultDataSetCollectionId'])
        except:
            self.model_details['defaultDataSetCollectionId'] = f"{random.randint(1111,10000000)}"

        if 'defaultDataSetCollectionResourcesFileGCStoragePath' in self.model_details.keys():
            test_detail['defaultDataSetCollectionId'] = self.model_details['defaultDataSetCollectionId']
            test_detail['defaultDataSetCollectionFileName'] = os.path.basename(
                self.model_details['defaultDataSetCollectionResourcesFileGCStoragePath']
            )
            try:
                # This is where the predictions of the model will be stored
                # output_file = os.path.join(Config.DATA_PATH, f"test_{self.model_details['_id']}", "results.csv")
                output_file = os.path.join(Config.DATA_PATH,
                                           self.model_details['_id'], f"test_{self.model_details['_id']}.csv")
                print(Fore.RED + "From training/train_func.py Line 139")
                print(Fore.RED + "The main() function present in the file training/test_model.py has to be updated based on developer requirements")

                # The call to start the testing of the model should be made here
########################################################################################################################
                acc, confusion_matrix = test_model.main(test_csv_path=self.test_csv_path,
                                                          dataset_path=self.test_dataset_path,
                                                          output_file=output_file,
                                                          class_map = cls_map,
                                                          model_save_path = self.model_save_path)
            
                test_detail['accuracy'] = acc['Total']

                # Adding the test CSV
                files_to_send['analyticfile'].append(output_file)

                # Adding the test JSON
                with open(os.path.join(Config.DATA_PATH, f"test_{self.model_details['_id']}", f"test_{self.model_details['_id']}.json"), "w") as outfile:
                    json.dump(acc, outfile)

                files_to_send['analyticfile'].append(os.path.join(Config.DATA_PATH,
                                                                  f"test_{self.model_details['_id']}",
                                                                  f"test_{self.model_details['_id']}.json"
                                                                  )
                                                     )

                # Adding the confusion JSON
                with open(os.path.join(Config.DATA_PATH, f"test_{self.model_details['_id']}", f"confusion_{self.model_details['_id']}.json"), "w") as outfile:
                    json.dump(confusion_matrix, outfile)

                files_to_send['analyticfile'].append(os.path.join(Config.DATA_PATH,
                                                                  f"test_{self.model_details['_id']}",
                                                                  f"confusion_{self.model_details['_id']}.json"
                                                                  )
                                                     )

            except Exception:
                print(Fore.RED + "Test Failed")
                traceback.print_exc(file=sys.stdout)
                print(Style.RESET_ALL)

            # =============== Testing Completed =============== #

        print("Testing Completed")
        
        print("Uploading Files")
        Config.MODEL_STATUS = 'Uploading Files'

        # Sending trained data to AutoAI        
        auto_ai_connect.autoai_upload_files(
            url=Config.AUTOAI_URL_SEND_FILES,
            files=files_to_send,
            isDefault=True,
            id=self.model_details['_id'],
            test_detail=test_detail
        )
        
        print("Files Upoloaded Successfully")
        # os.remove("inference_model.zip")
        
        # for file in os.listdir(saved_model_path):
        #     os.remove(os.path.join(saved_model_path, file))

        # uploading all tracking videos to AutoAI
        for video in os.listdir("sam2_results"):
            video_path = os.path.join("sam2_results", video)
            uploader.sending_videos(label="short_hanging", 
                                    filename=video_path, 
                                    model_id="67dc28d05e236c564cdde2e3", 
                                    tag="Tracking",
                                    csv="csv")

        Config.MODEL_STATUS = 'Training Completed'
        Config.POD_STATUS = "Available"

        print(Fore.GREEN + "Training Completed")
        print(Style.RESET_ALL)



