""" Module for training the churn prediction model with text. 

Training parameters are stored in 'params.yaml'.

Run in CLI example:
    'python train.py'


##How to handle integer type values column
"""



import sys
import json
import yaml
import joblib
import logging
import argparse
import math
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import precision_recall_curve
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import StringLookup,IntegerLookup
import os
import sklearn
import pandas as pd
import sklearn.metrics
import boto3
#import datetime

#def model_fn(model_dir):
 #   clf = joblib.load(os.path.join(model_dir,"model.joblib"))
  #  return clf


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#sm_boto3 = boto3.client("sagemaker")
#sess = sagemaker.Session()
#region = sess.boto_session.region_name

 
##Create model Inputs 
def create_model_inputs(FEATURE_NAMES,NUMERIC_FEATURE_NAMES):
    inputs = {}
    for feature_name in FEATURE_NAMES:
        if feature_name in NUMERIC_FEATURE_NAMES:
            inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype=tf.float32)
        else:
            inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype=tf.string)
    return inputs

    
""""
used_features_rate : The used_features_rate variable is a hyperparameter that you can set when configuring your DNDF model. It controls what proportion of the available features should be considered for each decision tree. This is a value between 0 and 1, where:

If used_features_rate is set to 1, it means that all available features will be considered for each tree.
If used_features_rate is set to a value less than 1, it means that only a fraction of the available features will be considered for each tree. The specific features to include are typically chosen randomly.
"""

class NeuralDecisionTree(keras.Model):
    def __init__(self, depth, num_features, used_features_rate, num_classes):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2**depth  #2 to the Power of 10 is equal to 1024
        self.num_classes = num_classes
        print(self.num_classes, ": No of Target classes")
        print(self.depth,": Tree Depth")
        print(self.num_leaves,": No of Leaves")

        # Create a mask for the randomly selected features.
        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)  #np.eye(3) creates a 3x3 identity matrix.
        sampled_feature_indicies = np.random.choice(  #shuffles all the features random order
            np.arange(num_features), num_used_features, replace=False
        )
        self.used_features_mask = one_hot[sampled_feature_indicies]

        # Initialize the weights of the classes in leaves.
        self.pi = tf.Variable( 
            initial_value=tf.random_normal_initializer()(
                shape=[self.num_leaves, self.num_classes]  # Creates a tensorflow matrix 1024 leaves by 2 target
            ),
            dtype="float32",
            trainable=True,
        )

        # Initialize the stochastic routing layer.
        self.decision_fn = layers.Dense(
            units=self.num_leaves, activation="sigmoid", name="decision"
        )

    
    def call(self, features):
        batch_size = tf.shape(features)[0]

        # Apply the feature mask to the input features.
        features = tf.matmul(
            features, self.used_features_mask, transpose_b=True
        )  # [batch_size, num_used_features]
        # Compute the routing probabilities.
        decisions = tf.expand_dims(
            self.decision_fn(features), axis=2
        )  # [batch_size, num_leaves, 1]
        # Concatenate the routing probabilities with their complements.
        decisions = layers.concatenate(
            [decisions, 1 - decisions], axis=2
        )  # [batch_size, num_leaves, 2]

        mu = tf.ones([batch_size, 1, 1])
              
        
        begin_idx = 1
        end_idx = 2
        # Traverse the tree in breadth-first order.
        for level in range(self.depth):
            mu = tf.reshape(mu, [batch_size, -1, 1])  # [batch_size, 2 ** level, 1]
            mu = tf.tile(mu, (1, 1, 2))  # [batch_size, 2 ** level, 2]
            level_decisions = decisions[
                :, begin_idx:end_idx, :
            ]  # [batch_size, 2 ** level, 2]
            mu = mu * level_decisions  # [batch_size, 2**level, 2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (level + 1)

        mu = tf.reshape(mu, [batch_size, self.num_leaves])  # [batch_size, num_leaves]
        probabilities = keras.activations.softmax(self.pi)  # [num_leaves, num_classes]
        
        print(mu, ": Value of mu")
        print(probabilities,": Value of Probabilities")
        outputs = tf.matmul(mu, probabilities)  # [batch_size, num_classes]
        return outputs

def get_featureType(x):
 #### DNDF Code starts #################################################################
    Headers = ["CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited"]  
    # A list of the numerical feature names.
    NUMERIC_FEATURE_NAMES = [
    "CreditScore",
    "Age",
    "Balance",
    "EstimatedSalary",
    "NumOfProducts",
    "Tenure"
    
    ]
# A dictionary of the categorical features and their vocabulary.
    CATEGORICAL_FEATURES_WITH_VOCABULARY = {
        "Geography": sorted(list(x["Geography"].unique())),
        "Gender": sorted(list(x["Gender"].unique()))
        #,"HasCrCard": sorted(list(x["HasCrCard"].unique())),
        #"IsActiveMember": sorted(list(x["IsActiveMember"].unique()))        
    }
    
    
    # A list of the categorical feature names.
    CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
    FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
    
    return FEATURE_NAMES,CATEGORICAL_FEATURE_NAMES,NUMERIC_FEATURE_NAMES,Headers,CATEGORICAL_FEATURES_WITH_VOCABULARY


def encode_inputs(inputs,CATEGORICAL_FEATURE_NAMES,CATEGORICAL_FEATURES_WITH_VOCABULARY):
    encoded_features = []
    for feature_name in inputs:
        #Convert string feature names into categories
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            print("Feature Name : ",feature_name)
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
            # Since we are not using a mask token, nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and num_oov_indices to 0.If we set num_oov_indices to 1 it will take any other values
            #outside the vocabulary
            #if feature_name.find("Gender") !=-1 or feature_name.find("Geography") !=-1:
            lookup = StringLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)  #num_oov_indices = 1
            #else:
             #   lookup = IntegerLookup(vocabulary=vocabulary, mask_token=None, num_oov_indices=0)  #num_oov_indices = 1
            
            # Convert the string input values into integer indices.
            value_index = lookup(inputs[feature_name])
            embedding_dims = int(math.sqrt(lookup.vocabulary_size()))
            # Create an embedding layer with the specified dimensions.
            embedding = layers.Embedding(
                input_dim=lookup.vocabulary_size(), output_dim=embedding_dims
            )
            # Convert the index values to embedding representations.
            encoded_feature = embedding(value_index)
        else:
            # Use the numerical features as-is.
            encoded_feature = inputs[feature_name]
            if inputs[feature_name].shape[-1] is None:
                encoded_feature = tf.expand_dims(encoded_feature, -1)

        encoded_features.append(encoded_feature)

    encoded_features = layers.concatenate(encoded_features)
    return encoded_features

def run_train_experiment(model,train_file,Headers,TARGET_FEATURE_NAME,NUMERIC_FEATURE_NAMES):
    learning_rate = 0.01
    batch_size = 265
    num_epochs = 10

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    #Converts dataset into Tensorflow data set wth Batch
    print("Start training the model...")
    train_dataset = get_dataset_from_csv(train_file,Headers,TARGET_FEATURE_NAME,NUMERIC_FEATURE_NAMES,shuffle=True, batch_size=batch_size)

    model.fit(train_dataset, epochs=num_epochs)
    print("Model training finished")
    
    # Generate a timestamp
  #  timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

# Define the filename with the timestamp
#model_filename = f"my_model_{timestamp}.h5"

    model.save("s3://mysagey/model_forest/model.pkl")
    #sess.upload_data(path = args.data_dir_train+"//"+args.train_file,bucket = bucket , key_prefix= sk_prefix)
    model_path = "s3://mysagey/model_forest"
    #joblib.dump(model,"s3://mysagey/model") 
    print("Model persisted at " + model_path)
    print()
    
    


def run_test_experiment(model,test_file,Headers,TARGET_FEATURE_NAME,NUMERIC_FEATURE_NAMES,X_test):
    batch_size = 265
       
    print("Evaluating the model on the test data...")
    test_dataset = get_dataset_from_csv(test_file,Headers,TARGET_FEATURE_NAME,NUMERIC_FEATURE_NAMES,shuffle=True, batch_size=batch_size)

    _, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    
    #y_pred_test = model.predict(X_test)
    #test_acc = accuracy_score(y_pred_test,y_test)
    #test_rep = classification_report(y_pred_test,y_test)
    
"""
## Create `tf.data.Dataset` objects for training and validation

We create an input function to read and parse the file, and convert features and labels
into a [`tf.data.Dataset`](https://www.tensorflow.org/guide/datasets)
for training and validation. We also preprocess the input by mapping the target label
to an index.

The tf.data.experimental.make_csv_dataset function in TensorFlow is a utility for creating datasets directly from CSV files. However, it might not always infer the data types of columns correctly. 
If you need to explicitly specify the data types of columns when using this function, you can use the column_defaults argument
"""

def get_dataset_from_csv(csv_file_path,headers,TARGET_FEATURE_NAME,NUMERIC_FEATURE_NAMES,shuffle=False, batch_size=128):
    
    TARGET_LABELS = ["1","0"]
    COLUMN_DEFAULTS = [[0.0] if feature_name in NUMERIC_FEATURE_NAMES else ["NA"]
    for feature_name in Headers]

    target_label_lookup = StringLookup(
        vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0
    )
    
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=headers,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=True,
        na_value="?",
        shuffle=shuffle,
    ).map(lambda features, target: (features, target_label_lookup(target)))
    return dataset.cache()



def create_tree_model(TARGET_LABELS,x):    
    depth = 10
    used_features_rate = 1.0
    num_classes = len(TARGET_LABELS)
    FEATURE_NAMES,CATEGORICAL_FEATURE_NAMES,NUMERIC_FEATURE_NAMES,Headers,CATEGORICAL_FEATURES_WITH_VOCABULARY = get_featureType(x)
        
    inputs = create_model_inputs(FEATURE_NAMES,NUMERIC_FEATURE_NAMES)
       
    features = encode_inputs(inputs,CATEGORICAL_FEATURE_NAMES,CATEGORICAL_FEATURES_WITH_VOCABULARY)       
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]
    
    print("**********  TRAINING NEURAL DECISION TREE *************")
    tree = NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)

    outputs = tree(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model,Headers,NUMERIC_FEATURE_NAMES

class NeuralDecisionForest(keras.Model):
    def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.ensemble = []        
        # Initialize the ensemble by adding NeuralDecisionTree instances.
        # Each tree will have its own randomly selected input features to use.
        for _ in range(num_trees):
            self.ensemble.append(
                NeuralDecisionTree(depth, num_features, used_features_rate, self.num_classes)
            )

    def call(self, inputs):
        # Initialize the outputs: a [batch_size, num_classes] matrix of zeros.
        batch_size = tf.shape(inputs)[0]
        outputs = tf.zeros([batch_size, self.num_classes])

        # Aggregate the outputs of trees in the ensemble.
        for tree in self.ensemble:
            outputs += tree(inputs)
        # Divide the outputs by the ensemble size to get the average.
        outputs /= len(self.ensemble)
        return outputs
    
    

def create_forest_model(TARGET_LABELS,x):

    
    num_classes = len(TARGET_LABELS)
    num_trees = 25
    depth = 5
    used_features_rate = 0.5

    FEATURE_NAMES,CATEGORICAL_FEATURE_NAMES,NUMERIC_FEATURE_NAMES,Headers,CATEGORICAL_FEATURES_WITH_VOCABULARY = get_featureType(x)
    
    inputs = create_model_inputs(FEATURE_NAMES,NUMERIC_FEATURE_NAMES)
    features = encode_inputs(inputs,CATEGORICAL_FEATURE_NAMES,CATEGORICAL_FEATURES_WITH_VOCABULARY)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]
    
    print("********** TRAINING NEURAL DECISION FOREST *************")
    forest_model = NeuralDecisionForest(
        num_trees, depth, num_features, used_features_rate, num_classes
    )

    outputs = forest_model(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model,Headers,NUMERIC_FEATURE_NAMES

if __name__ == "__main__":

       
    
    train_path =  "s3://mysagey/test/train_data.csv"
    test_path = "s3://mysagey/test/test_data.csv"
   
    
    TARGET_FEATURE_NAME = "Exited"
    # A list of the labels of the target features.
    TARGET_LABELS = [1, 0]
    
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    #tree_model,Headers,NUMERIC_FEATURE_NAMES = create_tree_model(TARGET_LABELS,train_data)
    #run_train_experiment(tree_model,train_path,Headers,TARGET_FEATURE_NAME,NUMERIC_FEATURE_NAMES)
    
    #tree_model,Headers,NUMERIC_FEATURE_NAMES = create_tree_model(TARGET_LABELS,test_data)
    #run_test_experiment(tree_model,test_path,Headers,TARGET_FEATURE_NAME,NUMERIC_FEATURE_NAMES,test_data)
   
    forest_model,Headers,NUMERIC_FEATURE_NAMES = create_forest_model(TARGET_LABELS,train_data)
    run_train_experiment(forest_model,train_path,Headers,TARGET_FEATURE_NAME,NUMERIC_FEATURE_NAMES)
    
    #forest_model,Headers,NUMERIC_FEATURE_NAMES = create_forest_model(TARGET_LABELS,test_data)
    #run_test_experiment(forest_model,test_path,Headers,TARGET_FEATURE_NAME,NUMERIC_FEATURE_NAMES,test_data)
   
