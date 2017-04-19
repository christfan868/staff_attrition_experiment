# Import the necessary libraries
import numpy as np
import pandas as pd
import pickle
from keras.models import model_from_json


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
retention_profile = pd.read_pickle('Retention_Profile.pkl')
satisfaction_profile = pd.read_pickle('Satisfaction_Factors.pkl')
TestDF = pd.read_pickle('Test_DF.pkl')
TestY = TestDF['left']
TestDF = TestDF.drop('left', 1)  # Drop the 'left' column
# print(TestDF.head(5))


# 3. Load the Validation Set [In a real scenario, this would be a test set.]
X_test = np.load('X_test.npy')  # Used for testing the model.
Y_test = np.load('Y_test.npy')  # Used for comparison of our results...


# unique, counts = np.unique(Y_test, return_counts=True)
# unique, counts = np.unique(Y_test, return_counts=True)
# print("Unique Values")
# print(dict(zip(unique, counts)))
# Unique Values => {0.0: 3794, 1.0: 1156}


# 4. Load other programme variables.
column_names = np.load('column_names.npy')
columns = column_names.tolist()
data_columns = columns[1:]

# Convert the test data into a dataframe
testDF = pd.DataFrame(X_test, columns=column_names)

# For each record in our validation set.
for index, row in testDF.iterrows():

    # Predict whether or not this member of staff is likely to leave the organization.
    row = row.values.reshape(-1, 9)
    output = model.predict_classes(row)

    # Convert row data into to a list
    row_data = list(row[0])


    # Drop the satisfaction_level column []
    drop_index = columns.index('satisfaction_level')
    del row_data[drop_index]

    # Alert the user to what is likely to happen with this staff member...
    leave_txt = "will" if output == 1 else "will not"
    print("This member of staff {} leave. ".format(leave_txt))

    # If they are likely to leave the organization,
    if output[0][0] == 1:

        # Create a new data frame which has the scores and rankings of each category
        staff_profile = pd.DataFrame(index=column_names, columns=['score', 'weight'])

        # Remove the satisfaction row [since its not in the satisfaction factors variable]
        staff_profile = staff_profile.drop(['satisfaction_level'])

        # Add weights and scores to staff profile...
        for name in data_columns:

            index = np.where(column_names == name)
            name_index = data_columns.index(name)
            target = satisfaction_profile[name]
            weight = satisfaction_profile[name]

            # Adding scores and weights to this staff profile..
            staff_profile.loc[name, 'score'] = row_data[name_index]
            staff_profile.loc[name, 'weight'] = weight

        # Current staff profile
        # print(staff_profile)

        # print(type(staff_profile))


        # Sort values of staff profile
        staff_profile = staff_profile.sort_values(by='weight', ascending=0)

        # Select the first row [most pressing issue].
        root = staff_profile.iloc[[0]]
        root_key = (root.index.values).tolist()[0]
        print(type(root_key))
        root_value = staff_profile['score'][root_key]
        target_value = retention_profile[root_key].values[0]

        # Compare the value to the corresponding value in the retention profile.
        if root_value < target_value:

            # Make a recommendation.
            action_obs = "below" if root_value < target_value else "above"
            action_txt = "increasing" if root_value < target_value else "decreasing"
            action_val = target_value - root_value
            insight = "I've noticed that this member of staff's '{}' is {} the norm. " \
                      "You may want to consider {} his/her '{}' by {}".format(root_key, action_obs, action_txt, root_key, action_val)

            print(insight)

        # Do a simple less than for numerical values...and check to see where things are different for categories.
        # print(staff_profile)
        exit()
