import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import math
import shap

from tensorflow.keras.layers import IntegerLookup,StringLookup
from tensorflow.keras.saving import register_keras_serializable
from imblearn.over_sampling import SMOTE
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import lime
from lime import lime_tabular

#tf.debugging.set_log_device_placement(True)
#from tensorflow.keras import tuners
#from tensorflow.keras.tuners import RandomSearch

#change num-oov-incides =0
#change string to integerlookup
#change string type to categorical type
#skip encoding based on cat/num column in create_model and column_defaults
#change target datatype to int
#
print("SHAP version is:", shap.__version__)
print("Tensorflow version is:", tf.__version__)
    
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
COLUMN_DEFAULTS = [[0.0] if feature_name in CSV_HEADER  else ["NA"] for feature_name in CSV_HEADER ]

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
    
    if model is not None:
      model.save("D:\\MLProject\\customerRealtime\\data\\DNDF_model.keras")
      print("Model saved successfully")
    
    print("Evaluating the model on the test data...")
    test_dataset = get_dataset_from_csv(test_data_file, batch_size=batch_size)
    print("Test dataset - ",test_dataset)
    
    loss, accuracy = model.evaluate(test_dataset)
    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test loss: {round(loss * 100, 2)}%")
    
    # Preprocess data if needed
    
  

  #  y_classes = model.predict(test_dataset)
  #  y_pred_numpy = y_classes.argmax(axis=1)
    
    # y_pred = pd.DataFrame(y_pred_numpy,columns = ['Churn_Prediction'])
    # y_test = test_data['Churn']
    # print("y_pred : ", y_pred)
    # print("y_test : ", y_test)
    
    # #It is about regularization. model.predict() returns the final output of the model, i.e. answer. While model.evaluate() returns the loss.
    # # The loss is used to train the model (via backpropagation).
    
    # accuracyscore = accuracy_score(y_test, y_pred)
    # print(f"Test accuracy score:" ,accuracyscore)

    # cm = confusion_matrix(y_test, y_pred)
    # print("Classification Report:\n", classification_report(y_test, y_pred))

    # cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [False, True])
    # cm_display.plot()
    # plt.show() 
    
    pred_data_file = "D:\\MLProject\\customerRealtime\\data\\X_pred_csv.csv"
    pred_data = pd.read_csv(pred_data_file)
    
    pred_dataset = get_dataset_from_csv(pred_data_file, batch_size=batch_size)
    
    y_prediction = model.predict(pred_dataset)
    y_prediction_numpy = y_prediction.argmax(axis=1)
    print(y_prediction)
    print(y_prediction_numpy[0])
    
    
    print("Model Summary- ",model.summary())
  
  ####################COde for SHAP ########################333
    
    
    pred_data_file_shap = "D:\\MLProject\\customerRealtime\\data\\X_pred_csv_Shap.csv"
    pred_df_shap = pd.read_csv(pred_data_file_shap)
    
    pred_dataset_shap = get_dataset_from_csv(pred_data_file_shap, batch_size=batch_size)
    
     #Convert and use with model
#TF 2.3.0 and SHAP 0.35.0
    shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough #this solves the "shap_ADDV2" problem but another one will appear
   # shap.explainers._deep.deep_tf.op_handlers["FusedBatchNormV3"] = shap.explainers._deep.deep_tf.passthrough
   # tf.experimental.numpy.experimental_enable_numpy_behavior()
    
    single_instance_batch = next(iter(pred_dataset_shap))  # Assuming features are at index 1 nad 0 has column headers
    
    
    input_tensors = {}
    for feature_name, tensor in single_instance_batch[0].items():
        input_tensors[feature_name] = tensor
    
    values = tf.nest.flatten(list(input_tensors.values()))
    
    print("Input Tensor valuess - ", values)
    
    shap_explainer = shap.DeepExplainer(model,values)
    print(shap_explainer)
    #tf.Tensor([[0.7732629  0.22673711]], shape=(1, 2), dtype=float32)
    output = [tf.convert_to_tensor(np.array([0.6091264, 0.39087358])),tf.convert_to_tensor(np.array([0.2703359, 0.72966415]))]
  
    # value_input  = list(tf.cast(input_tensors,tf.float32))
    # print(len(value_input))
    # print(value_input)
    shap_values = shap_explainer.shap_values(output)
   # Get SHAP explanation for the chosen customer
   
# Explain the model's prediction (assuming binary classification)
    shap.force_plot( shap_values, feat_names=FEATURE_NAMES)
    
    #######################LIME##############################
#     explainer = lime_tabular.LimeTabularExplainer(
#         input_tensors.__getitem__,
#         feature_names=FEATURE_NAMES, 
#         class_names=["0", "1"],
#         mode="classification"
#     )
     
# # # Get the explanation for the chosen customer
#     explanation = explainer.explain_instance(model) 

# # # Print the explanation
#     print(explanation.as_text())

# # # Alternatively, visualize the explanation with a bar chart
#     explanation.as_pyplot_bar(label="Churn")
    
    
    
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

    forest_model = NeuralDecisionForest(num_trees, depth, num_features, used_features_rate, num_classes)

    outputs = forest_model(features)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def limeEvaluation():   
# Assuming you have your DNDF model loaded as 'model'
# Assuming your test data is stored in a pandas dataframe named 'df_test'

   loaded_model = keras.models.load_model("D:\\MLProject\\customerRealtime\\data\\DNDF_model.keras",custom_objects={'NeuralDecisionForest': NeuralDecisionForest})
   if loaded_model is not None:
       print("Model loaded successfully")

   pred_data_file = "D:\\MLProject\\customerRealtime\\data\\X_pred_csv.csv"
   pred_data = pd.read_csv(pred_data_file)
   
   target_name = pred_data['Churn']
    # Exclude the last column (target)
   feature_names = list(pred_data.columns)
  
   inputs ={}
   # Potential correction:
# Create a plain tensor for data storage
   pred_data[feature_names] = tf.constant(pred_data[feature_names].values, dtype=tf.float32)  # Assuming you have feature values

# Use input layers for model architecture only
   for feature_name,feature_data in pred_data.items():
        inputs[feature_name] = layers.Input(data= feature_data,name=feature_name, shape=(), dtype=tf.float32)
   print(inputs)
   print("\n\n\n")
   encoded_features = encode_inputs(inputs)
   print(encoded_features)
        
# # Create a LIME explainer for tabular data
   explainer = lime_tabular.LimeTabularExplainer(
        pred_data[feature_names].values, 
        feature_names=feature_names, 
        class_names=["0", "1"],
        mode="classification"
    )

   predict_fn = lambda x :loaded_model.predict(x)  
   
# # Get the explanation for the chosen customer
   explanation = explainer.explain_instance(pred_data.iloc[0][feature_names].values,predict_fn) 

# # Print the explanation
   print(explanation.as_text())

# # Alternatively, visualize the explanation with a bar chart
   explanation.as_pyplot_bar(label=target_name)


def shapEvaluation():
    # Assuming you have your DNDF model loaded as 'model'
# Assuming your test data is stored in a NumPy array named 'X_test'
   loaded_model = keras.models.load_model("D:\\MLProject\\customerRealtime\\data\\DNDF_model.keras",custom_objects={'NeuralDecisionForest': NeuralDecisionForest})
   if loaded_model is not None:
       print("Model loaded successfully")

   pred_data_file = "D:\\MLProject\\customerRealtime\\data\\X_pred_csv.csv"
   pred_data = pd.read_csv(pred_data_file)
   target_name = pred_data['Churn']
#     # Exclude the last column (target)
   feature_names = list(pred_data.columns)
   
   for col in feature_names:
        pred_data[col] = pred_data[col].astype(np.float32)
 
     
   inputs = create_model_inputs()
   encoded_features = encode_inputs(inputs)
   encoded_features = layers.BatchNormalization()(encoded_features)
   
  
   print("Encoded features - ",encoded_features)
#    # Potential correction:
# # Create a plain tensor for data storage
#    pred_data[feature_names] = tf.constant(pred_data[feature_names], dtype=tf.float32)  # Assuming you have feature values

# # Use input layers for model architecture only
#    for feature_name in pred_data.items():
#         inputs[feature_name] = layers.Input(name=feature_name, shape=(), dtype=tf.float32)
#    print(inputs)
#    print("\n\n\n")
#    encoded_features = encode_inputs(inputs)
#    print(encoded_features)
    
   
# Create a SHAP explainer for your DNDF model
   explainer = shap.DeepExplainer(loaded_model, encoded_features[feature_names].values)

  # Get SHAP explanation for the chosen customer
   shap_values = explainer.shap_values(pred_data[0,:])
# Explain the model's prediction (assuming binary classification)
   shap.force_plot(explainer.base_value, shap_values[0], encoded_features[feature_names].values, feat_names=feature_names)

if __name__ == "__main__":
    
    learning_rate = 0.01
    batch_size = 265
    num_epochs = 10
    num_trees = 10
    depth = 5
    used_features_rate = 1.0
    num_classes = len(TARGET_LABELS)
     
    #tree_model = create_tree_model(depth, used_features_rate, num_classes)
    #run_experiment(tree_model)

    num_trees = 2
    depth = 5
    used_features_rate = 0.5
    forest_model = create_forest_model(num_trees, depth, used_features_rate, num_classes)
    run_experiment(forest_model)
    #
    # limeEvaluation()
    #shapEvaluation()