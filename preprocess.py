import pandas as pd

# Load data

ncaa = pd.read_csv('Data/ncaa_data.csv')
ncaa.columns = ncaa.columns.str.strip()


# Handling missing values and creating new features

ncaa['3P%'] = ncaa['3P%'].fillna(0)

ncaa['PPG']  = ncaa['PTS'] / ncaa['GP'] # points per game
ncaa['RPG']  = ncaa['TRB'] / ncaa['GP'] # rebounds per game
ncaa['APG']  = ncaa['AST'] / ncaa['GP'] # assists per game

ncaa['SPG']  = ncaa['STL'] / ncaa['GP'] # steals per game
ncaa['BPG']  = ncaa['BLK'] / ncaa['GP'] # blocks per game
ncaa['TOVPG'] = ncaa['TOV'] / ncaa['GP'] # turnovers per game
ncaa['MPG']  = ncaa['MP'] / ncaa['GP'] # minutes per game
ncaa['GSPG'] = ncaa['GS'] / ncaa['GP'] # games started per game

# 3-point attempt rate: fraction of FG attempts that are 3s
ncaa['3PAr'] = ncaa['3PA'] / ncaa['FGA'].replace(0, 1)

# Free-throw attempt rate: FT attempts relative to FG attempts
ncaa['FTr']  = ncaa['FTA'] / ncaa['FGA'].replace(0, 1)


# Encoding categorical variables

# Position mapping: Guard (G), Forward (F), Center (C)
pos_map = {'G': 0, 'F': 1, 'C': 2}
ncaa['POS_enc'] = ncaa['POS'].map(pos_map)

# Class mapping: Freshman, Sophomore, Junior, Senior
ncaa['Class'] = ncaa['Class'].str.strip() # Remove any leading/trailing whitespace
class_map = {'FR': 0, 'SO': 1, 'JR': 2, 'SR': 3}
ncaa['Class_enc'] = ncaa['Class'].map(class_map)


# Selecting relevant features for modeling and target

feature_cols = ['POS_enc', 'Class_enc', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TOVPG', 'MPG', 'GSPG', '3PAr', 'FTr']
X = ncaa[feature_cols]
y = ncaa['Drafted']


# Splitting the dataset into training and testing sets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y) 
# random_state = 42 : ensures reproducibility, getting same split every time
# stratify = y : tells sklearn to preserve original draft rate in both splits


# Saving splits
train_df = X_train.copy()
train_df['Drafted'] = y_train
test_df  = X_test.copy()
test_df['Drafted']  = y_test

train_df.to_csv('Data/train.csv', index = False)
test_df.to_csv('Data/test.csv', index = False)

print("Saved → Data/train.csv and Data/test.csv")