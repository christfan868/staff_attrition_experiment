# Import the necessary libraries
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt

# For model building
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

# For model evaluation and selection
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


# 1. Load the Model Architecture and its weights...
# load json and create model
model_json = open("MODEL.json", "r")
loaded_model_json = model_json.read()
model_json.close()

# Load weights into model...
model = model_from_json(loaded_model_json)
model.load_weights("MODEL_WEIGHTS.h5")

# Compile the model...
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# 2. Load Retention Profile
pd.read_pickle('Retention_Profile.pkl')

# Do a simple less than for numerical values...and check to see where things are different for categories.

# 3. Load the Validation Set [In a real scenario, this would be a test set.]
X_test = np.load('X_test.npy')  # Used for testing the model.
Y_test = np.load('Y_test.npy')  # Used for comparison of our results...

num_rows = len(X_test)

for row in X_test:
    output = model.predict(row)


# For each row...

# Create empty frame
output_df = pd.DataFrame(index=[range(num_rows)], columns=[categories])
output_df.fillna(0)  # Set the default value to 0.

# For each row
for index, row in test_df.iterrows():

    test_input = []

    test_input.append(float(row['Month']))
    test_input.append(float(row['Day']))
    test_input.append(float(row['DayOfWeek']))
    test_input.append(float(row['Hour']))
    test_input.append(float(row['X']))
    test_input.append(float(row['Y']))

    # Run predictions for every crime category we have
    for category in categories:
        #     input = np.array(sample_input)

        input = np.array(test_input)
        input = input.reshape(-1, 1)
        # input = input.reshape(-1, 6)
        input = input.swapaxes(0, 1)

        output = models[category].predict(input)
        # prediction = output[0][1]
        prediction = output[0][0]
        output_df.set_value(index, category, prediction)  # Set value for that category [row, col, value]
        print(prediction)

# classes = model.predict_classes(X_test, batch_size=32)
# proba = model.predict_proba(X_test, batch_size=32)
# 0. ==> [1. 0.] ==> prediction[1]
# 1. ==> [0. 1.]
output_df.to_csv("submission.csv", sep=',')


'''

To eliminate supervisors as a cause of turnover, you need to ask:

Has he or she received adequate training?
Does one supervisor have more or less turnover than another?
Is turnover high on one shift or in one location but good in another?
Does the supervisor have performance goals that include retention, turnover, and employee engagement?
To evaluate other potential causes of turnover, ask:

Are employees leaving after three to five years or during their first 12 months?
Are you providing millennials enough opportunities to learn?
Are you providing generation X enough opportunities to advance?
Are your wages and benefits competitive … and do they meet the needs of a multi-generation workforce?


Causes of employee turnover

    - Rude behavior.
    - Work-life imbalance.
    - The job did not meet expectations.
    - Employee misalignment.
    - Feeling undervalued.
    - Coaching and feedback are lacking.
    - Decision-making ability is lacking.
    - People skills are inadequate.
    - Organizational instability.
    - Raises and promotions frozen.
    - Faith and confidence shaken.
    - Growth opportunities not available.

'''






