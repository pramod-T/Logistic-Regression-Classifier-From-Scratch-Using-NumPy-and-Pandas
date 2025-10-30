import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

    
#Funtion for calculating 
def replace_zeros_with_mean(df):
    """
    Replace zero values in each column with the mean of that column.

    Parameters:
    - data (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The DataFrame with zero values replaced by the mean.
    """
    col_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in col_to_replace:
        df[col] = df[col].replace(0, df[col].mean())
    return df

#function to scale the data
def scale_data(df):
    """
    Scale data such that mean = 0 and standard deviation = 1.

    Parameters:
    - data (pandas.DataFrame): The input DataFrame.

    Returns:
    pandas.DataFrame: The scaled DataFrame.
    """
    outcome_column = df['Outcome']
    #print(outcome_column)
    data_to_scale = df.drop('Outcome', axis=1)

    mean = np.mean(data_to_scale, axis=0)
    std_dev = np.std(data_to_scale, axis=0)
    scaled_data = (data_to_scale - mean) / std_dev
    scaled_data_with_outcome = pd.concat([scaled_data, outcome_column], axis=1)
    #print(scaled_data_with_outcome.head())
    return scaled_data_with_outcome

def train_test_split_data(df, test_size=0.2,random_seed=42):
    """
    Split the dataset into training and testing sets.

    Parameters:
    - data (pandas.DataFrame): The input DataFrame.
    - test_size (float): how much part of the dataset is to include in the test split.

    Returns:
    (pandas.DataFrame, pandas.DataFrame, pandas.Series, pandas.Series):
    - X_train: The training features.
    - X_test: The testing features.
    - y_train: The training labels.
    - y_test: The testing labels.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    shuffled_indexes = np.random.permutation(len(df))
    test_dataset_length = int(len(df) * test_size)
    test_indexes = shuffled_indexes[:test_dataset_length]
    train_indexes = shuffled_indexes[test_dataset_length:]
    return df.iloc[train_indexes, :-1], df.iloc[test_indexes, :-1], df.iloc[train_indexes, -1], df.iloc[test_indexes, -1]

# function to calculate accuracy
def calculate_accuracy(predictions, y_test):
    """
    Calculate the accuracy of predicted labels.

    Parameters:
    - predictions (numpy.ndarray): Predicted labels.
    - y_test (pandas.Series): True labels.

    Returns:
    float: Accuracy score.
    """
    #print(predictions)
    #print(y_test)
    correct_predictions = np.sum(predictions == y_test)
    total_samples = len(y_test)
    accuracy = correct_predictions / total_samples
    return accuracy

def precision_score(predictions, y_test):
    """
    Calculate the accuracy of predicted labels.

    Parameters:
    - predictions (numpy.ndarray): Predicted labels.
    - y_test (pandas.Series): True labels.


    Returns:
    float: precision score.
    """
    true_positives = sum((prediction == 1 and label == 1) for prediction, label in zip(predictions, y_test))
    predicted_positives = true_positives+sum((prediction == 1 and label == 0) for prediction, label in zip(predictions, y_test))
    #print(true_positives)
    #print(predicted_positives)
    precision = true_positives / predicted_positives
    #print(precision)
    return precision

def recall_score(predictions, y_test):
    """
    Calculate the accuracy of predicted labels.

    Parameters:
    - predictions (numpy.ndarray): Predicted labels.
    - y_test (pandas.Series): True labels.


    Returns:
    float: precision score.
    """
    true_positives = sum((prediction == 1 and label == 1) for prediction, label in zip(predictions, y_test))
    denominator = true_positives+sum((prediction == 0 and label == 1) for prediction, label in zip(predictions, y_test))
    recall = true_positives /  denominator
    return recall

def f1_score(predictions, y_test):
    """
    Calculate the accuracy of predicted labels.

    Parameters:
    - predictions (numpy.ndarray): Predicted labels.
    - y_test (pandas.Series): True labels.


    Returns:
    float: f1 score.
    """
    precision=precision_score(predictions, y_test)
    recall=recall_score(predictions, y_test)
    f1=2*((precision*recall)/(precision+recall))
    return f1

def confusion_matrix(predictions, y_test):
    true_positives = sum((prediction == 1 and label == 1) for prediction, label in zip(predictions, y_test))
    false_positives= sum((prediction == 1 and label == 0) for prediction, label in zip(predictions, y_test))
    false_negatives = sum((prediction == 0 and label == 1) for prediction, label in zip(predictions, y_test))
    true_negatives = sum((prediction == 0 and label == 0) for prediction, label in zip(predictions, y_test))
    
    return  np.array([[true_positives, false_positives], [false_negatives, true_negatives]])

def evaluate(y_test,prediction):
    """
    Evaluate for each K

    Parameters:
    - X_train (pandas.DataFrame): The training features.
    - Y_train (pandas.Series): The training labels.
    - x_test (pandas.DataFrame): The test features.
    - y_test (pandas.Series): The test labels.

    Returns:
    None
    """
    
    #finding accuracy,percision,recall,f1,confusion matrix
    accuracy = calculate_accuracy(prediction, y_test )
    precision = precision_score(prediction, y_test)
    recall = recall_score(prediction, y_test)
    f1 = f1_score(prediction, y_test)
    conf_matrix = confusion_matrix(prediction, y_test)
    print('accuracyl:'+str(accuracy))
    print('precision:'+str(precision))
    print('recall for:'+ str(recall))
    print('f1 score:'+str(f1))
    print('Confusion matrix:'+str(conf_matrix))






#model

class logistic:
    """
    Parameters:
    - X_train (pandas.DataFrame): The training features.
    - Y_train (pandas.Series): The training labels.
    """
    def __init__(self, lr=0.01, epochs=10000):
        self.lr = lr
        self.epochs = epochs
        self.weights = None   

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def predict(self,X_test):
        predictions = []
        for index, row in X_test.iterrows():
            z = np.dot(row, self.weights)
            p = self.sigmoid(z)
            predictions.append(0 if p < 0.5 else 1)
        return np.array(predictions)
    
        
    def fit(self, X, Y):
        m, n = X.shape
        self.weights = np.zeros(n)
        for i in range(self.epochs):
            z = np.dot(X, self.weights)
            p = self.sigmoid(z)
            gradient = X.T.dot(p - Y) / m
            self.weights -= self.lr * gradient
        #print(self.weights)




#load data into data frame
df = pd.read_csv('diabetes.csv')
#print(df.head())

# Clean the data
cleaned_data = replace_zeros_with_mean(df)
#print(cleaned_data.head())

# Scale the data
scaled_data = scale_data(cleaned_data)
#print(scaled_data.head())

#check the datatypes of data
#print(df.info())

#check is null
#print(df.isnull().sum())


# Split the data into training and testing sets
X_train, x_test, Y_train, y_test = train_test_split_data(scaled_data,test_size=0.2, random_seed=42)

l = logistic(0.01,1000)

l.fit(X_train,Y_train)


predictions = l.predict(x_test)
#print(predictions)

evaluate(y_test,predictions)

