import sagemaker
#from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from sagemaker.sklearn.estimator import SKLearn
import boto3
import argparse
import os

sm_boto3 = boto3.client("sagemaker")
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = 'dndfsagemaker'
print("Using bucket: "+ bucket)

##SageMaker
    
print("[INFO] Extracting arguments..")
parser = argparse.ArgumentParser()
    
parser.add_argument("--depth",type=int,default=10)   
parser.add_argument("--used_features",type=float,default=1.0)
parser.add_argument("--num_classes",type=int,default=2)
     
   
parser.add_argument("--batch_size",type=int,default=265)
parser.add_argument("--epochs",type=int,default=10)
    
parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
parser.add_argument("--data_dir_train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
parser.add_argument("--data_dir_test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
parser.add_argument("--train-file", type=str, default="train_data.csv")
parser.add_argument("--test-file", type=str, default="test_data.csv")
    
args, _ = parser.parse_known_args()
    
    
print("[INFO] Reading data..")
print()
    

sk_prefix = "sagemaker/customer_attrition/sklearncontainer"
trainpath = sess.upload_data(path = "train_data.csv",bucket = bucket , key_prefix= sk_prefix)
testpath = sess.upload_data(path = "test_data.csv",bucket = bucket , key_prefix= sk_prefix)

print(trainpath)
print(testpath)

FRAMEWORK_VERSION  = "0.23-1"

sklearn_estimator = TensorFlow(
    entry_point="train.py",  # Path to your Keras training script
    #role=role,
    instance_count=1,
    instance_type="ml.p3.2xlarge",  # Choose an appropriate instance type
    hyperparameters={"epochs": 10,"depth": 10, "used_features": 1.0,"num_classes":2,"batch_size":265,"learning_rate":0.01},    
    use_spot_instances = True,
    framework_version=FRAMEWORK_VERSION,
    max_wait = 7200,
    max_run = 3600
    
    # Specify hyperparameters
)

sklearn_estimator.fit({"train":trainpath, "test":testpath},wait = True)

sklearn_estimator.latest_training_job.wait(logs= "None")
artifact = sm_boto3.describe_training_job(TrainingJobName = sklearn_estimator.latest_training_job)["ModelArtifacts"]["S3ModelArtifacts"]

print("Model artifacts persisted at : ", artifact)



