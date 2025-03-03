# Native Library Import
import os
import sys
import traceback
import zipfile
import random

# Pip or Installed Library Import
from colorama import Fore, Style


# Custom file Import
from autoai_process import Config, gcp_train_utils
from autoai_process.builtin_func import auto_ai_connect
from training.test_func import testing_function
from training.train_func import training_function


# Training function
def start_train(model_details):
    try:
        # Reset everything
        auto_ai_connect.reset()

        retrain = False
        if 'startCheckpointFileGCStoragePath' in model_details:
            retrain = True
            model_details['startCheckpointFileName'] = os.path.basename(
                model_details['startCheckpointFileGCStoragePath']
            )

        Config.POD_STATUS = 'Busy'
        Config.MODEL_STATUS = 'Starting'

        # Creating models and data directory
        if not os.path.exists(Config.MODELS_PATH):
            os.mkdir(Config.MODELS_PATH)
        if not os.path.exists(Config.DATA_PATH):
            os.mkdir(Config.DATA_PATH)

        # Creating subdirectories
        os.mkdir(os.path.join(Config.MODELS_PATH, model_details['_id']))
        os.mkdir(os.path.join(Config.DATA_PATH, model_details['_id']))
        os.mkdir(os.path.join(Config.DATA_PATH,
                 f"test_{model_details['_id']}")
                 )
        os.mkdir(os.path.join(Config.DATA_PATH,
                 model_details['_id'], "train_data")
                 )
        os.mkdir(os.path.join(Config.DATA_PATH,
                 f"test_{model_details['_id']}", "test_data")
                 )

        # if retrain == True download the weights
        if retrain:
            os.mkdir(os.path.join(Config.MODELS_PATH,
                     model_details['_id'], "checkpoint"))

            gcp_train_utils.download_gcp_file(os.path.join(model_details['startCheckpointFileGCStoragePath']),
                                              os.path.join(Config.MODELS_PATH, model_details['_id'],
                                                           "checkpoint", model_details['startCheckpointFileName']
                                                           )
                                              )

            path_to_zip_file = os.path.join(Config.MODELS_PATH, model_details['_id'],
                                            "checkpoint", model_details['startCheckpointFileName'])

            directory_to_extract_to = os.path.join(
                Config.MODELS_PATH, model_details['_id'], "checkpoint", 'modelDir')

            with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
                zip_ref.extractall(directory_to_extract_to)

        # Downloading csv file for the test dataset
        if 'defaultDataSetCollectionResourcesFileGCStoragePath' in model_details.keys():
            model_details['defaultDataSetCollectionResourcesFileName'] = os.path.basename(
                model_details['defaultDataSetCollectionResourcesFileGCStoragePath']
            )

            gcp_train_utils.download_gcp_file(
                os.path.join(
                    model_details['defaultDataSetCollectionResourcesFileGCStoragePath']),
                os.path.join(Config.DATA_PATH,
                             f"test_{model_details['_id']}",
                             model_details['defaultDataSetCollectionResourcesFileName']
                             )
            )

        # Downloading csv file for the train dataset
        try:
            print( model_details['resourcesFileName'])
        except:
             model_details['resourcesFileName'] = f"res_{random.randint(1111,10000000)}.csv"
            
        try:
            gcp_train_utils.download_gcp_file(
                os.path.join(model_details["startCheckpointFileArray"][0]['resourcesFileGCStoragePath']),
                                          os.path.join(Config.DATA_PATH, model_details['_id'],
                                                       model_details['resourcesFileName']
                                                       # f"{random.randint(1111,10000000)}.png"
                                                       )
                                          )

        # Setting up hyperparameters
        except:
            gcp_train_utils.download_gcp_file(
                os.path.join(model_details['resourcesFileGCStoragePath']),
                                          os.path.join(Config.DATA_PATH, model_details['_id'],
                                                       model_details['resourcesFileName']
                                                       # f"{random.randint(1111,10000000)}.png"
                                                       )
                                          )
        # Setting up hyperparameters
        if 'hyperParameter' not in model_details:
            model_details['hyperParameter'] = dict()

        if 'startCheckpointFileModelCollectionHyperparameter' in model_details:
            for key, ele in model_details['startCheckpointFileModelCollectionHyperparameter'].items():
                if key not in model_details['hyperParameter']:
                    model_details['hyperParameter'][key] = ele

        # Training Function Calling
        training_function(model_details, retrain).train()

    except Exception:
        print(Fore.RED + 'Model Testing failed')
        traceback.print_exc(file=sys.stdout)
        print(Style.RESET_ALL)
        Config.MODEL_STATUS = "Failed"
        Config.POD_STATUS = "Available"


# Testing Function
def start_test(model_details):
    try:
        # Reset everything
        auto_ai_connect.reset()

        Config.POD_STATUS = 'Busy'
        Config.MODEL_STATUS = 'Starting'

        # Creating models and data directory
        if not os.path.exists(Config.MODELS_PATH):
            os.mkdir(Config.MODELS_PATH)
        if not os.path.exists(Config.DATA_PATH):
            os.mkdir(Config.DATA_PATH)

        # Starting the Download
        Config.MODEL_STATUS = 'Downloading Data'

        # making sub dir for main test dataset
        os.mkdir(os.path.join(Config.DATA_PATH, model_details["_id"]))
        os.mkdir(os.path.join(Config.DATA_PATH,
                 model_details['_id'], "test data")
                 )

        # Main test dataset
        model_details['defaultDataSetCollectionResourcesFileName'] = os.path.basename(
            model_details['defaultDataSetCollectionResourcesFileGCStoragePath']
        )
        # Downloading the test resource csv
        gcp_train_utils.download_gcp_file(model_details["defaultDataSetCollectionResourcesFileGCStoragePath"],
                                          os.path.join(Config.DATA_PATH,
                                                       model_details["_id"],
                                                       model_details['defaultDataSetCollectionResourcesFileName']
                                                       )
                                          )

        # Downloading the Weights and resources for each model
        for checkpoint in model_details["startCheckpointFileArray"]:
            # sub dir for each model
            os.mkdir(os.path.join(Config.MODELS_PATH, checkpoint["_id"]))

            # sub dir for each model csv
            os.mkdir(os.path.join(Config.DATA_PATH, checkpoint["_id"]))

            file_name = os.path.basename(
                checkpoint["resourcesFileGCStoragePath"])

            # Downloading the csv file
            gcp_train_utils.download_gcp_file(checkpoint["resourcesFileGCStoragePath"],
                                              os.path.join(Config.DATA_PATH,
                                                           checkpoint["_id"],
                                                           file_name
                                                           )
                                              )

            checkpoint["csv_path"] = os.path.join(Config.DATA_PATH,
                                                  checkpoint["_id"],
                                                  file_name
                                                  )

            # Downloading the weights
            file_name = os.path.basename(
                checkpoint["startCheckpointFileGCStoragePath"])

            gcp_train_utils.download_gcp_file(os.path.join(checkpoint['startCheckpointFileGCStoragePath']),
                                              os.path.join(Config.MODELS_PATH,
                                                           checkpoint["_id"],
                                                           file_name
                                                           )
                                              )

            path_to_zip_file = os.path.join(
                Config.MODELS_PATH, checkpoint['_id'], file_name)

            directory_to_extract_to = os.path.join(
                Config.MODELS_PATH, checkpoint['_id'], "inference_model")
            
            os.system(f"unzip {path_to_zip_file} -d {directory_to_extract_to}")
            print(f"unzipped using this command: unzip {path_to_zip_file} -d {directory_to_extract_to}")

        # Starting the test
        Config.MODEL_STATUS = 'Testing'
        testing_function(model_details).test()

    except Exception:
        print(Fore.RED + 'Model Testing failed')
        traceback.print_exc(file=sys.stdout)
        print(Style.RESET_ALL)
        Config.MODEL_STATUS = "Failed"
        Config.POD_STATUS = "Available"
