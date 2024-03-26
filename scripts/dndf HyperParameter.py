
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
print(kt.__version__)
print(tf.__version__)

from tensorflow.keras import layers
from tensorflow.keras.layers import IntegerLookup

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



#change num-oov-incides =0
#change string to integerlookup
#change string type to categorical type
#skip encoding based on cat/num column in create_model and column_defaults
#change target datatype to int
#

CSV_HEADER = ['RiskAdjustedReturn', 'TransactionFrequency', 'TransactionFees', 'EstimatedAvgReturnPrivateAsset', 'EstimatedReturnPublicAsset', 'EstimatedAvgReturnCommodities', 'FuturePricesNaturalGas', 'SPVolatilityIndex', 'BloombergHedgeFundIndex', 'RealEstateIndex', 'SPPrivateEquityIndex','CreditScore', 'AccountType', 'RiskType', 'FinancialGoalUpdated', 'CustomerRevenueFall70', 'CustomerRevenueRecovery40', 'SentimentAnalysis','Churn']

NUMERIC_FEATURE_NAMES = [
    "RiskAdjustedReturn",
    "TransactionFrequency",
    "TransactionFees",
    "EstimatedAvgReturnPrivateAsset",
    "EstimatedReturnPublicAsset",
    "EstimatedAvgReturnCommodities",
    "FuturePricesNaturalGas",
    "SPVolatilityIndex",
    "BloombergHedgeFundIndex",
    "RealEstateIndex",
    "SPPrivateEquityIndex"
     ]

CATEGORICAL_FEATURE_NAMES = ['CreditScore', 'AccountType', 'RiskType', 'FinancialGoalUpdated', 'CustomerRevenueFall70', 'CustomerRevenueRecovery40', 'SentimentAnalysis']

FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES

X_train_file = "D:\\MLProject\\customerRealtime\\data\\X_train_csv.csv"
X_test_file = "D:\\MLProject\\customerRealtime\\data\\X_test_csv.csv"


Y_train_file = "D:\\MLProject\\customerRealtime\\data\\Y_train_csv.csv"
Y_test_file = "D:\\MLProject\\customerRealtime\\data\\Y_test_csv.csv"


X_train = pd.read_csv(X_train_file)
y_train = pd.read_csv(Y_train_file)


smote = SMOTE(k_neighbors=1, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

df = pd.DataFrame(X_resampled, columns=FEATURE_NAMES)  # Create DataFrame from features
df['Churn'] = y_resampled 

print(df.head())

print("X resampled datatype : ", X_resampled.dtypes)

print(f"Train dataset shape before SMOTE : {X_train.shape}")
print(f"Train dataset shape after SMOTE : {df.shape}")

df.to_csv('D:\\MLProject\\customerRealtime\\data\\X_train_combined.csv', index=False)

train_data_file = "D:\\MLProject\\customerRealtime\\data\\X_train_combined.csv"
test_data_file = "D:\\MLProject\\customerRealtime\\data\\X_test_csv.csv"

train_data = pd.read_csv(train_data_file)
test_data = pd.read_csv(test_data_file)

train_data.head(5)

print(f"Train dataset shape: {train_data.shape}")
print(f"Test dataset shape: {test_data.shape}")



train_data['RiskAdjustedReturn'] = train_data['RiskAdjustedReturn'].astype(np.float32)
test_data['RiskAdjustedReturn'] = test_data['RiskAdjustedReturn'].astype(np.float32)

train_data['TransactionFrequency'] = train_data['TransactionFrequency'].astype(np.float32)
test_data['TransactionFrequency'] = test_data['TransactionFrequency'].astype(np.float32)

train_data['TransactionFees'] = train_data['TransactionFees'].astype(np.float32)
test_data['TransactionFees'] = test_data['TransactionFees'].astype(np.float32)

train_data['EstimatedAvgReturnPrivateAsset'] = train_data['EstimatedAvgReturnPrivateAsset'].astype(np.float32)
test_data['EstimatedAvgReturnPrivateAsset'] = test_data['EstimatedAvgReturnPrivateAsset'].astype(np.float32)

train_data['EstimatedReturnPublicAsset'] = train_data['EstimatedReturnPublicAsset'].astype(np.float32)
test_data['EstimatedReturnPublicAsset'] = test_data['EstimatedReturnPublicAsset'].astype(np.float32)

train_data['EstimatedAvgReturnCommodities'] = train_data['EstimatedAvgReturnCommodities'].astype(np.float32)
test_data['EstimatedAvgReturnCommodities'] = test_data['EstimatedAvgReturnCommodities'].astype(np.float32)

train_data['FuturePricesNaturalGas'] = train_data['FuturePricesNaturalGas'].astype(np.float32)
test_data['FuturePricesNaturalGas'] = test_data['FuturePricesNaturalGas'].astype(np.float32)

train_data['SPVolatilityIndex'] = train_data['SPVolatilityIndex'].astype(np.float32)
test_data['SPVolatilityIndex'] = test_data['SPVolatilityIndex'].astype(np.float32)

train_data['BloombergHedgeFundIndex'] = train_data['BloombergHedgeFundIndex'].astype(np.float32)
test_data['BloombergHedgeFundIndex'] = test_data['BloombergHedgeFundIndex'].astype(np.float32)

train_data['RealEstateIndex'] = train_data['RealEstateIndex'].astype(np.float32)
test_data['RealEstateIndex'] = test_data['RealEstateIndex'].astype(np.float32)

train_data['SPPrivateEquityIndex'] = train_data['SPPrivateEquityIndex'].astype(np.float32)
test_data['SPPrivateEquityIndex'] = test_data['SPPrivateEquityIndex'].astype(np.float32)


# A dictionary of the categorical features and their vocabulary.
CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "CreditScore": sorted(list(train_data["CreditScore"].unique())),
    "AccountType": sorted(list(train_data["AccountType"].unique())),
    "RiskType": sorted(list(train_data["RiskType"].unique())),
    "FinancialGoalUpdated": sorted(list(train_data["FinancialGoalUpdated"].unique())),
    "CustomerRevenueFall70": sorted(list(train_data["CustomerRevenueFall70"].unique())),
    "CustomerRevenueRecovery40": sorted(list(train_data["CustomerRevenueRecovery40"].unique())),
    "SentimentAnalysis": sorted(list(train_data["SentimentAnalysis"].unique()))
}


# A list of the categorical feature names.
CATEGORICAL_FEATURE_NAMES = list(CATEGORICAL_FEATURES_WITH_VOCABULARY.keys())
# A list of all the input features.
#FEATURE_NAMES = NUMERIC_FEATURE_NAMES + CATEGORICAL_FEATURE_NAMES
# A list of column default values for each feature.
COLUMN_DEFAULTS = [
    [0.0] #if feature_name in NUMERIC_FEATURE_NAMES else ["NA"]
    for feature_name in CSV_HEADER
]
# The name of the target feature.
TARGET_FEATURE_NAME = "Churn"
# A list of the labels of the target features.
TARGET_LABELS = [0, 1]





def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
         inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype=tf.float32)
    return inputs

#Treat them as categorical columns: handle categorical columns with integer-encoded indices
#This is preferred when the order of the integer values doesn't have inherent meaning. For example, if the indices represent different categories of products (e.g., 0: Electronics, 1: Clothing, 2: Furniture), order doesn't matter.
#Use methods like:
#tf.keras.layers.StringLookup: Converts integer indices to string representations of the categories.
#Embedding layers: These learn vector representations for each category, capturing semantic relationships even though the indices are numerica



target_label_lookup = IntegerLookup(
    vocabulary=TARGET_LABELS, mask_token=None, num_oov_indices=0
)


def get_dataset_from_csv(csv_file_path, shuffle=False, batch_size=128):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path,
        batch_size=batch_size,
        column_names=CSV_HEADER,
        column_defaults=COLUMN_DEFAULTS,
        label_name=TARGET_FEATURE_NAME,
        num_epochs=1,
        header=True,
        na_value="?",
        shuffle=shuffle,
    ).map(lambda features, target: (features, target_label_lookup(target)))      
    return dataset.cache()


def encode_inputs(inputs):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in CATEGORICAL_FEATURE_NAMES:
            print(feature_name, ": Cat Feature name encode_inputs")
            vocabulary = CATEGORICAL_FEATURES_WITH_VOCABULARY[feature_name]
            # Create a lookup to convert a string values to an integer indices.
            # Since we are not using a mask token, nor expecting any out of vocabulary
            # (oov) token, we set mask_token to None and num_oov_indices to 0.
            lookup = IntegerLookup(
                vocabulary=vocabulary, mask_token=None, num_oov_indices=0
            )
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
            print(feature_name, ": Numeric Feature name encode_inputs")
            encoded_feature = inputs[feature_name]
            if inputs[feature_name].shape[-1] is None:
                encoded_feature = tf.expand_dims(encoded_feature, -1)

        encoded_features.append(encoded_feature)

    encoded_features = layers.concatenate(encoded_features)
    return encoded_features


class NeuralDecisionTree(keras.Model):
    def __init__(self, depth, num_features, used_features_rate, num_classes):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_classes = num_classes

        print(self.depth)
        print(self.num_leaves)
        print(self.num_classes)

        # Create a mask for the randomly selected features.
        num_used_features = int(num_features * used_features_rate)
        one_hot = np.eye(num_features)
        sampled_feature_indicies = np.random.choice(
            np.arange(num_features), num_used_features, replace=False
        )
        self.used_features_mask = one_hot[sampled_feature_indicies]
        print(num_used_features)

        # Initialize the weights of the classes in leaves.
        self.pi = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=[self.num_leaves, self.num_classes]
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
        outputs = tf.matmul(mu, probabilities)  # [batch_size, num_classes]
        print(outputs)
        return outputs
    
    
    
class NeuralDecisionForest(keras.Model):
    def __init__(self, num_trees, depth, num_features, used_features_rate, num_classes):
        super().__init__()
        self.ensemble = []
        # Initialize the ensemble by adding NeuralDecisionTree instances.
        # Each tree will have its own randomly selected input features to use.
        for _ in range(num_trees):
            self.ensemble.append(
                NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)
            )

    def call(self, inputs):
        # Initialize the outputs: a [batch_size, num_classes] matrix of zeros.
        batch_size = tf.shape(inputs)[0]
        outputs = tf.zeros([batch_size, num_classes])

        # Aggregate the outputs of trees in the ensemble.
        for tree in self.ensemble:
            outputs += tree(inputs)
        # Divide the outputs by the ensemble size to get the average.
        outputs /= len(self.ensemble)
        return outputs
    
    #SparseCategoricalCrossentropy Vs CategoricalCrossEntropy
   #crossentropy loss function when there are two or more label classes. We expect labels to be provided in a one_hot representation.
   # If you want to provide labels as integers, please use SparseCategoricalCrossentropy loss.
 
def run_experiment(model):
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(), 
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

    print("Start training the model...")
    train_dataset = get_dataset_from_csv(
        train_data_file, shuffle=True, batch_size=batch_size
    )

    model.fit(train_dataset, epochs=num_epochs)
    print("Model training finished")
    
    print("Evaluating the model on the test data...")
    test_dataset = get_dataset_from_csv(test_data_file, batch_size=batch_size)

    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test loss: {round(loss * 100, 2)}%")
    
    y_classes = model.predict(test_dataset)
    y_pred_numpy = y_classes.argmax(axis=1)
    
    y_pred = pd.DataFrame(y_pred_numpy,columns = ['Churn_Prediction'])
    
    y_test = test_data['Churn']
    
    print("y_pred : ", y_pred)
    
    print("y_test : ", y_test)
    
    #It is about regularization. model.predict() returns the final output of the model, i.e. answer. While model.evaluate() returns the loss.
    # The loss is used to train the model (via backpropagation).
    
    accuracyscore = accuracy_score(y_test, y_pred)
    print(f"Test accuracy score:" ,accuracyscore)

    cm = confusion_matrix(y_test, y_pred)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])

    cm_display.plot()
    plt.show()
    
   
def create_tree_model(depth, used_features_rate, num_classes):
    inputs = create_model_inputs()
    features = encode_inputs(inputs)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]

    tree = NeuralDecisionTree(depth, num_features, used_features_rate, num_classes)

    outputs = tree(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_forest_model(num_trees, depth, used_features_rate, num_classes):
    inputs = create_model_inputs()
    features = encode_inputs(inputs)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]

    forest_model = NeuralDecisionForest(
        num_trees, depth, num_features, used_features_rate, num_classes)

    outputs = forest_model(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def create_forest_model_hyper(hp):
    
    inputs = create_model_inputs()
    features = encode_inputs(inputs)
    features = layers.BatchNormalization()(features)
    num_features = features.shape[1]
    
    num_trees = hp.Int('num_trees', min_value=10, max_value=150)
    batch_size = hp.Int('batch_size', min_value=64, max_value=512)
    depth = hp.Int('depth', min_value=5, max_value=15)
    used_features_rate = hp.Float('used_features_rate', min_value=0.3,max_value=1.0)
    num_epochs = hp.Int('num_epochs', min_value = 10,max_value=20)
    
    
    forest_model = NeuralDecisionForest(
        num_trees, depth, num_features, used_features_rate, num_classes)

    outputs = forest_model(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(), 
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        )

    print("Start training the model...")
    train_dataset = get_dataset_from_csv(
        train_data_file, shuffle=True, batch_size=batch_size
    )

    model.fit(train_dataset, epochs=num_epochs)
    print("Model training finished")
    
    print("Evaluating the model on the test data...")
    test_dataset = get_dataset_from_csv(test_data_file, batch_size=batch_size)

    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test loss: {round(loss * 100, 2)}%")
    
    return model
    


if __name__ == "__main__":
    
    #### **************** MAIN PROGRAM ****************
    learning_rate = 0.01
    batch_size = 265
    num_epochs = 10
    num_trees = 10
    depth = 5
    used_features_rate = 1.0
    num_classes = len(TARGET_LABELS)
    
    #THE ABOVE PARAMETERS ARE AUTOMATICALLY PASSED ACROSS METHODS PRESENT IN SAME CLASS SIMILAR TO GLOBAL PARAMETERS
    
    # tree_model = create_tree_model(depth, used_features_rate, num_classes)
    # run_experiment(tree_model)

    # num_trees = 36
    # depth = 5
    # used_features_rate = 0.5
    # forest_model = create_forest_model(num_trees, depth, used_features_rate, num_classes)
    # run_experiment(forest_model)
    
    
    ## *****************  HYPERPARAMETER TUNING ********************************
    # Define search space for each hyperparameter
    X_train_file = "D:\\MLProject\\customerRealtime\\data\\X_train_csv.csv"
    Y_train_file = "D:\\MLProject\\customerRealtime\\data\\Y_train_csv.csv"
    
    X_train = pd.read_csv(X_train_file)
    y_train = pd.read_csv(Y_train_file)
    
    hyperparameters =[ 
                       {
                        "num_trees" : [50, 100, 150],  # List of values to try
                        "depth" :[4, 6, 8],
                        "used_features_rate" : [0.5, 0.7, 0.9],
                        "epochs" :[50, 100, 150],
                        "batch_size" : [32, 64, 128]
                        }
                    ]
    tuner = kt.RandomSearch(
        objective='val_accuracy',  # Objective metric to optimize (e.g., val_loss)
        max_trials=10,  # Number of random hyperparameter combinations to try
        directory='D:\\MLProject\\customerRealtime\\my_tuner_results',  # Optional: Directory to save search results
        project_name='dndf_tuning'  # Optional: Project name for saving results
    )
    
    for hp_dict in hyperparameters:
  # Update tuner with current hyperparameter set
        tuner.hyperparameters.update(hp_dict)
  
  # Call the build_dndf_model function with hp object
        best_model = tuner.search(create_forest_model_hyper, epochs=10, validation_data=(X_train, y_train))

  # Evaluate the best model (optional)
  # ...  # Reset tuner for next hyperparameter set (optional)
        
        best_model = tuner.get_best_models()[0]
        print("Best Model : ",best_model)
                
        tuner.reset()  # This might be helpful depending on your tuner
        print("Tuning finished!")


