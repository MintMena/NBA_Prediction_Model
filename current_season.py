import pandas as pd
import numpy as np

# Load data

curr = pd.read_csv('Data/current_season.csv')
curr.columns = curr.columns.str.strip() 


# Handling missing values and creating new features

curr['3P%'] = curr['3P%'].fillna(0)

curr['PPG']  = curr['PTS'] / curr['GP'] # points per game
curr['RPG']  = curr['TRB'] / curr['GP'] # rebounds per game
curr['APG']  = curr['AST'] / curr['GP'] # assists per game

curr['SPG']  = curr['STL'] / curr['GP'] # steals per game
curr['BPG']  = curr['BLK'] / curr['GP'] # blocks per game
curr['TOVPG'] = curr['TOV'] / curr['GP'] # turnovers per game
curr['MPG']  = curr['MP'] / curr['GP'] # minutes per game
curr['GSPG'] = curr['GS'] / curr['GP'] # games started per game

# 3-point attempt rate: fraction of FG attempts that are 3s
curr['3PAr'] = curr['3PA'] / curr['FGA'].replace(0, 1)

# Free-throw attempt rate: FT attempts relative to FG attempts
curr['FTr']  = curr['FTA'] / curr['FGA'].replace(0, 1)

# Assist-to-turnover ratio
curr['AST_TOV'] = curr['APG'] / (curr['TOVPG'] + 0.1)

# True shooting % — better efficiency measure than FG%
curr['TS%'] = curr['PTS'] / (2 * (curr['FGA'] + 0.44 * curr['FTA']).replace(0, 1))

# Points + assists per minute 
curr['impact_per_min'] = (curr['PPG'] + curr['APG']) / (curr['MPG'] + 0.1)



# Encoding categorical variables

# Position mapping: Guard (G), Forward (F), Center (C)
pos_map = {'G': 0, 'F': 1, 'C': 2}
curr['POS_enc'] = curr['Pos'].map(pos_map)

# Class mapping: Freshman, Sophomore, Junior, Senior
curr['Class'] = curr['Class'].str.strip() # Remove any leading/trailing whitespace
class_map = {'FR': 0, 'SO': 1, 'JR': 2, 'SR': 3}
curr['Class_enc'] = curr['Class'].map(class_map)


# Selecting relevant features for modeling and target

feature_cols = ['POS_enc', 'Class_enc', 'PPG', 'RPG', 'APG', 'SPG', 'BPG', 'TOVPG', 'MPG', 'GSPG', '3PAr', 'FTr', 'AST_TOV', 'TS%', 'impact_per_min']


# Load the model and predict

import xgboost as xgb

model = xgb.XGBClassifier()
model.load_model('xgb_final.json')

pred_proba = model.predict_proba(curr[feature_cols])[:, 1] # probability of being drafted
curr['Draft_Prob'] = pred_proba

# Top 10 by draft probability

top10 = curr[['Player', 'Team', 'Pos', 'Class', 'PPG', 'RPG', 'APG','Draft_Prob']].sort_values(by='Draft_Prob', ascending=False).head(10)
print("Top 10 NBA Draft Prospects for the Current Season:")
print(top10.to_string(index=False))

# Favorite team : Michigan State Spartans
msu = curr[curr['Team'] == 'Michigan State'][['Player', 'Pos', 'Class', 'PPG', 'RPG', 'APG', 'Draft_Prob']].sort_values(by='Draft_Prob', ascending=False).head(10)
print("\nMichigan State Spartans Draft Prospects:")
print(msu.to_string(index=False))


# Store results in DuckDB

import duckdb

con = duckdb.connect(database = 'curr_season.duckdb')
con.execute("CREATE TABLE IF NOT EXISTS curr AS SELECT * FROM curr")
con.close()

con = duckdb.connect(database = 'curr_season.duckdb')
result_top10 = con.execute("""SELECT Player, Team, Pos, Class, PPG, RPG, APG, ROUND(Draft_Prob, 3) AS Draft_Prob 
                     FROM curr 
                     ORDER BY Draft_Prob DESC 
                     LIMIT 10""").fetchdf()
print(result_top10.to_string(index = False))

result_msu = con.execute("""SELECT Player, Pos, Class, PPG, RPG, APG, ROUND(Draft_Prob, 3) AS Draft_Prob
                        FROM curr 
                        WHERE Team = 'Michigan State' 
                        ORDER BY Draft_Prob DESC""").fetchdf()
print(result_msu.to_string(index = False))
con.close()

# Jeremy Fears (G, SR) has the highest draft probability at 0.639 , definitely has much lower score than top 10
