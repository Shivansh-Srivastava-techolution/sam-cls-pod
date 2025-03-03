# Native Library Import
import os
import json
import sys
import traceback

# Pip or Installed Library Import
from colorama import Fore, Style


# Custom file Import
from autoai_process import Config
from training import test_model
from autoai_process.auto_ai_download import download_files, csv_to_json_convert
from autoai_process.builtin_func import auto_ai_connect, testCollection
from autoai_process.gcp_train_utils import data_preprocessing


class testing_function():
    def __init__(self, model_details):
        self.model_details = model_details

        # Paths can be updated, added, or removed based on developer requirements #

        # This is where testing data would be stored
        self.test_dataset_path = os.path.join(
            Config.DATA_PATH, self.model_details["_id"], "test data")

        # Path to csv file which contains info about the testing data
        self.test_csv_path = os.path.join(
            Config.DATA_PATH, self.model_details['_id'],
            self.model_details['defaultDataSetCollectionResourcesFileName'])

    def test(self):
        print("============== model_details ==============")
        print(self.model_details)

        Config.MODEL_STATUS = 'Downloading Data'

        # Downloading the data from the csv file
        download_files(self.test_csv_path, self.test_dataset_path)

        files_to_send = list()

        print("Testing Started")
        Config.MODEL_STATUS = 'Testing'

        # Iterating Through all models and running the test
        for checkpoint in self.model_details["startCheckpointFileArray"]:
            try:
                # This is where the predictions of the model will be stored
                output_file = os.path.join(Config.DATA_PATH,
                                           checkpoint['_id'], f"test_{checkpoint['_id']}_results.csv")

                # Path to the weights of a particular checkpoint model
                weight_path = os.path.join(
                    Config.MODELS_PATH,
                    checkpoint['_id'],
                    'best_model.h5'
                )
                
                try:
                    print("*"*80)
                    print("Searching for best model")
                    print("*"*80)
                    saved_model_dir = os.path.join(Config.MODELS_PATH, checkpoint['_id'], "inference_model")
                    # os.system(f"unzip {saved_model_dir}.zip")
                    # The call to start the testing of the model should be made here #
                    acc, confusion_matrix = test_model.main(test_csv_path=self.test_csv_path,
                                          dataset_path=self.test_dataset_path,
                                          output_file=output_file,
                                          model_details=self.model_details,
                                          saved_model_dir = saved_model_dir)
                except Exception as e:
                    print(e)
                    print("testing failed")
#                     try:
#                         print("*"*80)
#                         print("Trying to search latest model")
#                         print("*"*80)
#                         saved_model_dir = os.path.join(Config.MODELS_PATH, checkpoint['_id'], "inference_model", "latest")
#                         # The call to start the testing of the model should be made here #
#                         acc, confusion_matrix = test_model.main(test_csv_path=self.test_csv_path,
#                                               dataset_path=self.test_dataset_path,
#                                               models_path=weight_path,
#                                               output_file=output_file,
#                                               model_details=self.model_details,
#                                               saved_model_dir = saved_model_dir)
#                     except Exception as e:
#                         print(e)
#                         print("*"*80)
#                         print("Trying to run on the base model since no checkpoint path found")
#                         print("*"*80)
#                         acc, confusion_matrix = test_model.main(test_csv_path=self.test_csv_path,
#                                               dataset_path=self.test_dataset_path,
#                                               models_path=weight_path,
#                                               output_file=output_file,
#                                               model_details=self.model_details,
#                                               config_path = './default_config.yml',
#                                               saved_model_dir = saved_model_dir)
                        

                # Adding the confusion JSON
                with open(os.path.join(Config.DATA_PATH, checkpoint["_id"], f"confusion_{checkpoint['_id']}.json"),
                          "w") as outfile:
                    json.dump(confusion_matrix, outfile)

                with open(os.path.join(Config.DATA_PATH, checkpoint["_id"], f"test_{checkpoint['_id']}_results.json"), "w") as outfile:
                    json.dump(acc, outfile)

                test_results = testCollection(checkpoint['_id'])
                test_results.parentCheckpointFileId = checkpoint['startCheckpointId']
                test_results.testCollectionId = self.model_details['_id']
                test_results.accuracy = acc['Total']
                test_results.analysisFiles.append(output_file)
                test_results.analysisFiles.append(os.path.join(Config.DATA_PATH, checkpoint["_id"], f"test_{checkpoint['_id']}_results.json"))

                test_results.analysisFiles.append(os.path.join(Config.DATA_PATH, checkpoint["_id"], f"confusion_{checkpoint['_id']}.json"))
                
                files_to_send.append(test_results)

            except Exception as e:
                print(Fore.RED + "Test Failed", checkpoint['_id'])
                traceback.print_exc(file=sys.stdout)
                print(Style.RESET_ALL)

        for test_results in files_to_send:
            print(test_results)
            test_results.upload_to_autoai()
            print('\n\n')

        Config.MODEL_STATUS = "Testing Completed"
        Config.POD_STATUS = "Available"

        print(Fore.GREEN + "Testing Completed")
        print(Style.RESET_ALL)