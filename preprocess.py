
ncaa['3P%'] = ncaa['3P%'].fillna(0)

ncaa['PPG']  = ncaa['PTS'] / ncaa['GP']
ncaa['RPG']  = ncaa['TRB'] / ncaa['GP']
ncaa['APG']  = ncaa['AST'] / ncaa['GP']


