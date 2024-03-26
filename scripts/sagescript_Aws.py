import sagemaker
from sagemaker import get_execution_role
from sagemaker.tensorflow import TensorFlow
from sagemaker.sklearn.estimator import SKLearn
import boto3
import argparse
import os

sm_boto3 = boto3.client("sagemaker")
sess = sagemaker.Session()
region = sess.boto_session.region_name
bucket = 'mysagey'
print("Using bucket: "+ bucket)
print("region: "+ region)

current_directory = os.getcwd()
##SageMaker
    
print("[INFO] Extracting arguments..")
parser = argparse.ArgumentParser()
    
parser.add_argument("--depth",type=int,default=10)   
parser.add_argument("--used_features",type=float,default=1.0)
parser.add_argument("--num_classes",type=int,default=2)
     
   
parser.add_argument("--batch_size",type=int,default=265)
parser.add_argument("--epochs",type=int,default=10)
    
parser.add_argument("--model_dir", type=str, default=os.path.join(current_directory, "model"))
parser.add_argument("--data_dir_train", type=str, default=os.path.join(current_directory, "//customerChurn//data"))
parser.add_argument("--data_dir_test", type=str, default=os.path.join(current_directory, "//cstomerChurn//data"))
parser.add_argument("--train_file", type=str, default="train_data.csv")
parser.add_argument("--test_file", type=str, default="test_data.csv")
    
args, _ = parser.parse_known_args()
    
    
print("[INFO] Reading data..")
print()
    

#sk_prefix = "customer_attrition/"
#trainpath = sess.upload_data(path = args.data_dir_train+"//"+args.train_file,bucket = bucket , key_prefix= sk_prefix)
#testpath = sess.upload_data(path = args.data_dir_train+"//"+args.test_file,bucket = bucket , key_prefix= sk_prefix)

trainpath = "s3://mysagey/test/train_data.csv"
testpath = "s3://mysagey/test/test_data.csv"

#print(trainpath)
#print(testpath)

FRAMEWORK_VERSION  = "2.5"

temp_py = os.path.join(current_directory+"/customerChurn/scripts")
script_py = os.path.join(temp_py+"/train.py")
                          
print(script_py)

role_sg = get_execution_role()

sklearn_estimator = TensorFlow(
    entry_point=script_py,  # Path to your Keras training script - model training scriot
    role= "arn:aws:iam::031335268209:role/service-role/AmazonSageMaker-ExecutionRole-20230930T155612", #role_sg,
    instance_count=1,
    #instance_type="ml.p3.2xlarge",  # Choose an appropriate instance type
    instance_type= "ml.m4.xlarge",
    hyperparameters={"epochs": 10,"depth": 10, "used_features": 1.0,"num_classes":2,"batch_size":265,"learning_rate":0.01},    
    use_spot_instances = True,
    framework_version=FRAMEWORK_VERSION,
    py_version="py37",
    max_wait = 7200,
    max_run = 3600
    
    #image_uri='your-container-image-uri',
    # Specify your custom code and data locations
    #code_uri='s3://your-s3-bucket/code',
    #data_uri='s3://your-s3-bucket/data',
    
    # Specify hyperparameters
)

#sklearn_estimator.fit({"train":trainpath, "test":testpath},wait = True)
sklearn_estimator.fit()

sklearn_estimator.latest_training_job.wait(logs= "None")
artifact = sm_boto3.describe_training_job(TrainingJobName = sklearn_estimator.latest_training_job)["ModelArtifacts"]["S3ModelArtifacts"]

print("Model artifacts persisted at : ", artifact)



