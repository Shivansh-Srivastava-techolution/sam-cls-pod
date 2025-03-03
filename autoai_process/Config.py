import os


if 'MODELS_PATH' in os.environ:
    MODELS_PATH = os.environ['MODELS_PATH']
else:
    MODELS_PATH = 'models'

if 'DATA_PATH' in os.environ:
    DATA_PATH = os.environ['DATA_PATH']
else:
    DATA_PATH = 'data'

if 'AUTOAI_BUCKET' in os.environ:
    AUTOAI_BUCKET = os.environ['AUTOAI_BUCKET']
else:
    AUTOAI_BUCKET = 'auto-ai_resources_fo'

if 'AUTOAI_URL_SEND_FILES' in os.environ:
    AUTOAI_URL_SEND_FILES = os.environ['AUTOAI_URL_SEND_FILES']
else:
    AUTOAI_URL_SEND_FILES = 'http://34.132.219.249:3000'

if 'HYPERVISOR_URL' in os.environ:
    HYPERVISOR_URL = os.environ['HYPERVISOR_URL']
else:
    HYPERVISOR_URL = 'http://localhost:5000'

if 'VM_NAME' in os.environ:
    VM_NAME = os.environ['VM_NAME']
else:
    VM_NAME = 'detectron-1'

if 'EXTERNAL_IP' in os.environ:
    EXTERNAL_IP = os.environ['EXTERNAL_IP']
else:
    EXTERNAL_IP = 'http://34.93.125.222:5001'

if 'BACKUP_BUCKET_LOCATION' in os.environ:
    BACKUP_BUCKET_LOCATION = os.environ['BACKUP_BUCKET_LOCATION']
else:
    BACKUP_BUCKET_LOCATION = 'ml-faceopen-bucket'

if 'CATEGORIES' in os.environ:
    CATEGORIES = os.environ['CATEGORIES']
else:
    CATEGORIES = ["box"]

if 'ENVIROMENT' in os.environ:
    ENV = os.environ['ENVIROMENT']
else:
    ENV = 'prod'

MODEL_STATUS = "None"
POD_STATUS = "Available"

PARENT_FILE_UPLOAD = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/model/addParentCheckpointFile"
ANALYTIC_FILE_UPLOAD = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/model/addAnalysisFileObjectInParentCheckpointFile"
ADDITIONAL_FILE_UPLOAD = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/model/addAdditionalFileObjectInParentCheckpointFile"
MODEL_FILE_UPLOAD = 'https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/model/addChildCheckpointFileInModels'
TEST_COLLECTION_ANALYTIC_FILE_UPLOAD = "https://autoai-backend-exjsxe2nda-uc.a.run.app/collection/test/addAnalysisFileObjectInParentCheckpointFile"
