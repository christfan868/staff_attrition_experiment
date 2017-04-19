# Import the necessary libraries
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import pickle

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
pd.read_pickle('Satisfaction_Factors.pkl')

# 3. Load the Validation Set [In a real scenario, this would be a test set.]
X_test = np.load('X_test.npy')  # Used for testing the model.
Y_test = np.load('Y_test.npy')  # Used for comparison of our results...


# 4. Load other programme variables.
# Save column names for use in the Insight Generation.
# column_names = pickle.load(open('column_names.p', 'rb'))
column_names = np.load('column_names.npy')
num_rows = len(X_test)
# print(column_names)


# Convert the test data into a dataframe
testDF = pd.DataFrame(X_test, columns=column_names)
# testDF.values = X_test
print(testDF.head(5))

# For each record in our validation set.
for row in X_test:

    # Predict whether or not this member of staff is likely to leave the organization.
    row = row.reshape(-1, 9)
    output = model.predict(row)

    # If they are,
    if output == 1:
        #  Create a new data frame which has the scores and rankings of each category
        staff_profile = pd.DataFrame(index=column_names, columns=['score', 'weight'])

        # Remove the satisfaction row
        staff_profile = staff_profile.drop(['satisfaction_level'])

        # Populate the DataFrame
        # retention_profile = staff_profile.append({'satisfaction_level': stay_df['satisfaction_level'].mean(),
        #                                         'last_evaluation': stay_df['last_evaluation'].mean(),
        #                                         'number_project': stay_df['number_project'].mean(),
        #                                         'average_montly_hours': stay_df['average_montly_hours'].mean(),
        #                                               'time_spend_company': stay_df['time_spend_company'].mean(),
        #                                               'Work_accident': stay_df['Work_accident'].mode(),
        #                                               'promotion_last_5years': stay_df['promotion_last_5years'].mode(),
        #                                               'department': stay_df['department'].mode(),
        #                                               'salary': stay_df['salary'].mode()
        #                                               },
        #                                              ignore_index=True)

        # Do a simple less than for numerical values...and check to see where things are different for categories.
        # df.sort_values(by=['coverage', 'reports'], ascending=0)

        print(staff_profile)



# For each row
# for index, row in test_df.iterrows():
#
#     test_input = []
#
#     test_input.append(float(row['Month']))
#     test_input.append(float(row['Day']))
#     test_input.append(float(row['DayOfWeek']))
#     test_input.append(float(row['Hour']))
#     test_input.append(float(row['X']))
#     test_input.append(float(row['Y']))
#
#     # Run predictions for every crime category we have
#     for category in categories:
#         #     input = np.array(sample_input)
#
#         input = np.array(test_input)
#         input = input.reshape(-1, 1)
#         # input = input.reshape(-1, 6)
#         input = input.swapaxes(0, 1)
#
#         output = models[category].predict(input)
#         # prediction = output[0][1]
#         prediction = output[0][0]
#         output_df.set_value(index, category, prediction)  # Set value for that category [row, col, value]
#         print(prediction)



