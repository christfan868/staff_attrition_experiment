'''
This is a Data Project Development Experiment.
It consists of three (3) core features:
    - Descriptive Analytics which aims to provide observations about the dataset that will form our theories about staff attrition. 
    - Predictive Analytics which predict the likelihood of a staff member's departure. 
    - Prescriptive Analytics which suggest what should be done about staff who are likely to leave the organization. 
'''
# Import the necessary libraries
import numpy as np
import pandas as pd
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sbs
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


# Standard Machine Learning - Answering Questions about our staff...
# 1. Prepare Problem
# a) Load libraries
# b) Load dataset
df = pd.read_csv("HR_comma_sep.csv")
df.rename(columns={'sales': 'department'}, inplace=True)  # Rename 'sales' column to 'department'

# Save column names for use in the Insight Generation.
column_names = df.columns.values
column_names = column_names[column_names!='left']  # Remove the 'left' column
'''
index = np.argwhere(x==3)
y = np.delete(x, index)
'''
# pickle.dump(column_names, open('column_names.p', 'wb'))
np.save('column_names.npy', column_names)


'''
To plot just a selection of your columns you can select the columns of interest by passing a list to the subscript operator:

ax = df[['V1','V2']].plot(kind='bar', title ="V comp", figsize=(15, 10), legend=True, fontsize=12)
What you tried was df['V1','V2'] this will raise a KeyError as correctly no column exists with that label, although it looks funny at first you have to consider that your are passing a list hence the double square brackets [[]].

import matplotlib.pyplot as plt
ax = df[['V1','V2']].plot(kind='bar', title ="V comp", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Hour", fontsize=12)
ax.set_ylabel("V", fontsize=12)
plt.show()

'''


# 2. Summarize Data - Here's where any questions about the data can be answered.
# a) Descriptive statistics
print(df.shape)  # shape
print(df.head(20))  # Print the first 20 rows of the dataset.
print(df.describe())  # Statistical Summary Descriptions


# Univariate Anaylysis: Box and Whisker Plots
df.plot(kind='box', subplots=True, layout=(4,3), sharex=False, sharey=False)
# plt.show()

# Histograms
df.hist()
# plt.show()

# Multivariate Analysis
# scatter plot matrix
scatter_matrix(df)
# plt.show()

# Correlations
correlations = df.corr(method='pearson')
# print(correlations)


# Select the row of correlations for the 'left'
print('\n\nCorrelations to leaving')
turnover_causes_df = correlations['left'].abs().sort_values(ascending=False)
# turnover_causes_df.iloc[1:]
print(turnover_causes_df[1:])
print("\n")
# df.sort(df.columns[0], ascending=False)


indices = np.where(correlations > 0.5)
indices = [(correlations.index[x], correlations.columns[y]) for x, y in zip(*indices)
                                        if x != y and x < y]
# print(indices)
# Apart from correlations, what else do you see?


# correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(df.corr(), vmin=-1, vmax=1, interpolation='none')
fig.colorbar(cax)

# ax.set_xticklabels(list(df))
ax.set_yticklabels(list(df))
# plt.show()


# b) Data visualizations
# Leading Causes of Employee Turnover
# TODO: Use aggregate functions to get the sum of each of the columns selected
# cause_df =
ax = df[['satisfaction_level','Work_accident']].plot(kind='bar', title ="Causes of Employee Turnover", figsize=(15, 10), legend=True, fontsize=12)
ax.set_xlabel("Reason for Leaving", fontsize=12)
ax.set_ylabel("No. of Employees", fontsize=12)
# plt.show()

# Create a dataframe for folks who left.
left_df = df[df['left'] == 1]

# Create a dataframe for folks who stayed.
stay_df = df[df['left'] == 0]


'''

time_spend_company      -0.168791
number_project          -0.092799
last_evaluation          0.086357
average_montly_hours     0.055354


'''

# What does the profile of someone who is likely to stay look like?
print("\nStay Profile\n")
retention_profile = pd.DataFrame(columns=[list(df)])
retention_profile = retention_profile.drop('left', 1)  # Drop the 'left' column
retention_profile = retention_profile.append({'satisfaction_level': stay_df['satisfaction_level'].mean(),
                                              'last_evaluation': stay_df['last_evaluation'].mean(),
                                              'number_project': stay_df['number_project'].mean(),
                                              'average_montly_hours': stay_df['average_montly_hours'].mean(),
                                              'time_spend_company': stay_df['time_spend_company'].mean(),
                                              'Work_accident': stay_df['Work_accident'].mode(),
                                              'promotion_last_5years': stay_df['promotion_last_5years'].mode(),
                                              'department': stay_df['department'].mode(),
                                              'salary': stay_df['salary'].mode()
                                             },
                                             ignore_index=True)


# Examine the difference is job satisfaction between the two groups.
attrition_satisfaction = round(left_df['satisfaction_level'].mean(), 2)
retention_satisfaction = round(stay_df['satisfaction_level'].mean(), 2)

if attrition_satisfaction < retention_satisfaction:
    print("The folks who left were generally less satisfied with their jobs. ")

    satisfaction_diff = round((abs(attrition_satisfaction) - retention_satisfaction), 2)
    satisfaction_change = round(((satisfaction_diff/retention_satisfaction) * 100), 2)
    chg_txt = "decrease" if satisfaction_change < 0 else "increase"
    print("There was a {}% {} in employee satisfaction".format(abs(satisfaction_change), chg_txt))


# Display the characteristics of someone who stays...
print(retention_profile)

# Save this profile for later use...
# retention_profile.to_hdf('Staff_Retention_Profile.h5', 'retention_profile')
retention_profile.to_pickle('Retention_Profile.pkl')

'''

number_project          -0.142970
last_evaluation          0.105021
time_spend_company      -0.100866
average_montly_hours    -0.020048

'''

'''
The above dataframe looks like this: 
last_evaluation          0.105021
number_project          -0.142970
average_montly_hours    -0.020048
time_spend_company      -0.100866
Work_accident            0.058697
left                    -0.388375
promotion_last_5years    0.025605


last_evaluation          0.105021
number_project          -0.142970
average_montly_hours    -0.020048
time_spend_company      -0.100866
Work_accident            0.058697
promotion_last_5years    0.025605


'last_evaluation'
'number_project'
'average_montly_hours'
'time_spend_company'
'Work_accident'
'promotion_last_5years'
'department'???
'salary'???


'''

# list(satisfaction_df[1:])


# Plot the satisfaction and time spent...
# style.use('ggplot')
# group_name=list(range(20))
# x = pd.cut(df['time_spend_company'], 20, labels=group_name)  # [:19]
# y = pd.cut(df['satisfaction_level'], 20, labels=group_name)  # [:19]
# plt.plot(x, y)
# plt.title('Job Satisfaction x Time Spent')
# plt.ylabel('Job Satisfaction')
# plt.xlabel('Time Spent')
# plt.show()


# exit()

# df.sort_values(by=['coverage', 'reports'], ascending=0)


# Deep Learning - Predicting Which Staff Will Leave
# Target Accuracy: 95-97%
# 1. Prepare the Data
# a) Data Transforms: Converting non-numeric to numeric types:
le = LabelEncoder()
# department column
train = df['department'].unique().tolist()
test = df['department']
df['department'] = le.fit(train).transform(test)
# salary column...
train = df['salary'].unique().tolist()
test = df['salary']
df['salary'] = le.fit(train).transform(test)


# Move 'left' column to the end of the DataFrame
df = df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident', 'promotion_last_5years','department','salary','left']]

# Rescale column values for use with our ML Model
df[['number_project','average_montly_hours', 'time_spend_company', 'department', 'salary']] = df[['number_project','average_montly_hours', 'time_spend_company', 'department', 'salary']].apply(lambda x: StandardScaler().fit_transform(x))

# A quick look at the new DataFrame
print(df.head(5))

'''
CORRELATIONS
'''
# Which factor contributed most to turnover?
# What does the profile of someone who is likely to leave look like?
# Factors contributing to satisfaction levels among staff who left.
print("Factors contributing to satisfaction levels among staff who left.")
leftDF2 = df[df['left'] == 1]
print(leftDF2.corr(method='pearson')['satisfaction_level'][1:].sort_values(ascending=False))
'''

time_spend_company       0.446440
number_project          -0.227113
last_evaluation          0.182295
average_montly_hours    -0.084117

# Check for nulls in field. 
df.isnull().any()

'''


stayDF2 = df[df['left'] == 0]
print("Factors contributing to satisfaction levels among staff who stay.")
print(stayDF2.corr(method='pearson')['satisfaction_level'][1:].sort_values(ascending=False))


# What are the factors that are correlated to the satisfaction level?
print('\n\nCorrelations to satisfaction')
correlations2 = df.corr(method='pearson')
satisfaction_df = correlations2['satisfaction_level'][1:].abs().sort_values(ascending=False)
satisfaction_df = satisfaction_df.drop('left')
print(satisfaction_df)
satisfaction_df.to_pickle('Satisfaction_Factors.pkl')


# 2. Define Model.
model = Sequential()
model.add(Dense(15, input_dim=9, init='normal', activation='relu'))
model.add(Dense(12, init='normal', activation='relu'))
model.add(Dense(9, init='normal', activation='relu'))
model.add(Dense(6, init='normal', activation='relu'))
model.add(Dense(3, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))


# 3. Compile Model.
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# Score of this architecture: loss: 0.1028 - acc: 0.9705
# score = model.evaluate(x_test, y_test, batch_size=128)
'''
standard_scalerX = StandardScaler()
standard_scalerX.fit(X_test)
'''


# 4. Fit Model.
# Split Training Data into X and Y values
num_columns = len(df.columns)
array = df.values
X = array[:, 0:(num_columns - 1)].astype(float)
Y = array[:, (num_columns - 1)]

# Split data into training and validation sets...
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# Save the validation set for later.
np.save('X_test.npy', X_test)
np.save('Y_test.npy', Y_test)

TrainDF, TestDF = train_test_split(df, test_size=test_size, random_state=seed)

# Create a test dataframe
TestDF.to_pickle('Test_DF.pkl')

print(TestDF)

# exit()

# 5. Evaluate Model.
# Train the model using the X and Y values created.
model.fit(X_train, Y_train, nb_epoch=150, batch_size=10)  # TODO: find out how to calculate these...


# 6. Save the Model.
model_json = model.to_json()
with open("MODEL.json", "w") as json_file: json_file.write(model_json)  # Save Network Architecture
model.save_weights("MODEL_WEIGHTS.h5")  # Save Weights

# That's all folks!
