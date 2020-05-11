# This script cleans the data, creates some extra features, and joins all the data that was 
# scraped or downloaded preiously in the GetData.py script.

import pandas as pd
import numpy as np
import datetime



# Creating Games DF which includes the games list with box-score stats for the 1st half
df1 = pd.read_csv('data/games.csv')
df1.drop('Unnamed: 0', axis=1, inplace=True)
df2 = pd.read_csv('data/game_stats.csv')
df2.drop('Unnamed: 0', axis=1, inplace=True)
data = df1.join(df2, rsuffix='2')
data.fillna(0, inplace=True)
data['date'] = data['Link'].apply(lambda x: x[11:19])

# Creating back-to-back-games features
data['date_ts'] = data['date'].apply(lambda x: 
                                     datetime.datetime(int(str(x)[:4]), 
                                                       int(str(x)[4:6]), 
                                                       int(str(x)[6:8])))
home_b2b = []
away_b2b = []
for i, day in enumerate(data['date_ts']):
    home = data.Home.iloc[i]
    away = data.Away.iloc[i]
    yesterday = day - datetime.timedelta(1) 
    played_yesterday = list(data[data.date_ts==yesterday].Away) + list(data[data.date_ts==yesterday].Home)
    if home in played_yesterday:
        home_b2b.append(1)
    else:
        home_b2b.append(0)
    if away in played_yesterday:
        away_b2b.append(1)
    else:
        away_b2b.append(0)
data['away_b2b'] = away_b2b
data['home_b2b'] = home_b2b

# Creating season feature for future filtering
data['season'] = data['date'].apply(lambda x: x[:4] if int(x[4:6])>7 else str(int(x[:4])-1))

# Creating TOV/AST and AST/FG features
data['H-AST/FG'] = data['H-AST']/data['H-FG']
data['A-AST/FG'] = data['A-AST']/data['A-FG']
data['H-TOV/AST'] = data['H-TOV']/data['H-AST']
data['A-TOV/AST'] = data['A-TOV']/data['A-AST']

# Creating statistical feature as differential: Home - Away
to_create = ['A-S-Min', 'A-FG', 'A-FGA', 'A-3P', 'A-3PA', 'A-FT',
       'A-FTA', 'A-ORB', 'A-DRB', 'A-TRB', 'A-AST', 'A-STL', 'A-BLK', 'A-TOV',
       'A-PF', 'A-PTS', 'A-FG%', 'A-3P%', 'A-FT%', 'A-AST/FG', 'A-TOV/AST']
for feat in to_create:
    x = feat[2:]
    name = x + ' H-A'
    data[name] = data['H-' + x] - data[feat]

# Creating Potential Target Variables
data['2-Half H-A'] = (data['H-Pts-F'] - data['H-PTS']) - (data['A-Pts-F'] - data['A-PTS'])
data['2-Half H>A'] = data['2-Half H-A'].apply(lambda x: 1 if x>0 else 0)
data['2-Half A-H'] = (data['A-Pts-F'] - data['A-PTS']) - (data['H-Pts-F'] - data['H-PTS'])
data['2-Half A>H'] = data['2-Half A-H'].apply(lambda x: 1 if x>0 else 0)

# Dropping unnecessary feature after join
data.drop(['Link', 'Link2'], axis=1, inplace=True)

# Creating copy of DF for future editing and join
df_games = data.copy()



# Creating Betting Odds DF which includes 2nd-half Moneyline betting odds (for betting simulation)
df3 = pd.read_csv('data/game_odds.csv')
df3.drop(['Unnamed: 0', 'away_open', 'home_open'], axis=1, inplace=True)

# Dropping Games missing betting odds
for i, line in enumerate(df3['home_ML']):
    if line == '-':
        df3.drop(index=i, inplace=True)

# Removing plus sign from betting odds strings
df3['away_ML'] = df3['away_ML'].apply(lambda x: int(x.strip('+')))
df3['home_ML'] = df3['home_ML'].apply(lambda x: int(x.strip('+')))
df3['tp wins'] = df3['home_ML'].apply(lambda x : (x) if x > 0 else (-100/x)*100)
df3['tn wins'] = df3['away_ML'].apply(lambda x : (x) if x > 0 else (-100/x)*100)

# Removing betting outliers (odds over/under +/-1000)
df3 = df3[df3['away_ML'] >= -1000]
df3 = df3[df3['home_ML'] >= -1000]
df3 = df3[df3['away_ML'] <= 1000]
df3 = df3[df3['away_ML'] <= 1000]

# Creating string date feature for joining purposes
df3['date'] = df3['date'].apply(lambda x : str(x))

# Creating copy of DF for future editing and join
df_odds = df3.copy()



# Creating Point Spreads DF which includes pre-game point spreads to  be used as a feature in prediction
df4 = pd.read_excel('data/game_lines/nba odds 2007-08.xlsx')
df4 = df4[['Date', 'VH', 'Team', 'Open', '2H']]

# Creating date string for joining purposes
df4['Date'] = df4['Date'].apply(lambda x: '2007' + str(x) if len(str(x))>3 else '20080' + str(x))

# Scraping Point Spreads from the stored spreadsheets
for i in range(2008, 2020):
    temp = pd.read_excel('data/game_lines/nba odds ' + str(i) + '-' + str(i+1)[-2:] + '.xlsx')
    temp = temp[['Date', 'VH', 'Team', 'Open']]
    temp['Date'] = temp['Date'].apply(lambda x: str(i) + str(x) if len(str(x))>3 else str(i+1) + '0' + str(x))
    df4 = df4.append(temp)
df4.reset_index(inplace=True)
df4.drop('index', axis=1, inplace=True)

# Creating Final Point Spread DF
df5 = pd.DataFrame(columns=['Date', 'Home', 'Away', 'Home Spread'])

# Temporary DF with home teams
temp1 = df4[::2].reset_index()
temp1.drop('index', axis=1, inplace=True)

# Temporary DF with away teams
temp2 = df4[1::2].reset_index()
temp2.drop('index', axis=1, inplace=True)

# Merging home and away DFs and replacing 'Pk' with 0 point spread.
temp3 = temp1.merge(temp2, left_index=True, right_index=True)
temp3['Open_x'] = temp3['Open_x'].apply(lambda x: 0 if x=='pk' else x)
temp3['Open_x'] = temp3['Open_x'].apply(lambda x: 0 if x=='PK' else x)
temp3['Open_y'] = temp3['Open_y'].apply(lambda x: 0 if x=='pk' else x)
temp3['Open_y'] = temp3['Open_y'].apply(lambda x: 0 if x=='PK' else x)
temp3.drop(index=15863, inplace=True)
temp3.drop(index=15870, inplace=True)

# Convert point spreads to numeric and append into DF
for i, j in temp3.iterrows():
    if float(j.Open_y) > 100:
        spread = float(j.Open_x)
    else:
        spread = -1 * float(j.Open_y)
    temp4 = pd.DataFrame(data={'Date': [j.Date_x], 
                              'Home': [j.Team_y], 
                              'Away': [j.Team_x], 
                              'Home Spread': [spread]})
    df5 = df5.append(temp4)
df5.reset_index(inplace=True)
df5.drop('index', axis=1, inplace=True)

# Creating copy of DF for future editing and join
df_spread = df5.copy()



# Cleaning Team Names from all 3 DFs for Join
#Cleaning team names from betting odds DF
df_odds['away'] = df_odds['away'].apply(lambda x: 'LA Lakers' if x == 'L.A. Lakers' else  x)
df_odds['home'] = df_odds['home'].apply(lambda x: 'LA Lakers' if x == 'L.A. Lakers' else  x)
df_odds['away'] = df_odds['away'].apply(lambda x: 'LA Clippers' if x == 'L.A. Clippers' else  x)
df_odds['home'] = df_odds['home'].apply(lambda x: 'LA Clippers' if x == 'L.A. Clippers' else  x)

# Cleaning team names from games stats DF
df_games['Away'] = df_games['Away'].apply(lambda x: x.split(' ')[0] if len(x.split(' ')) == 2 else  x)
df_games['Home'] = df_games['Home'].apply(lambda x: x.split(' ')[0] if len(x.split(' ')) == 2 else  x)
df_games['Away'] = df_games['Away'].apply(lambda x: 'LA Clippers' if x == 'Los Angeles Clippers' else  x)
df_games['Home'] = df_games['Home'].apply(lambda x: 'LA Clippers' if x == 'Los Angeles Clippers' else  x)
df_games['Away'] = df_games['Away'].apply(lambda x: 'LA Lakers' if x == 'Los Angeles Lakers' else  x)
df_games['Home'] = df_games['Home'].apply(lambda x: 'LA Lakers' if x == 'Los Angeles Lakers' else  x)
df_games['Away'] = df_games['Away'].apply(lambda x: 'Portland' if x == 'Portland Trail Blazers' else  x)
df_games['Home'] = df_games['Home'].apply(lambda x: 'Portland' if x == 'Portland Trail Blazers' else  x)
df_games['Away'] = df_games['Away'].apply(lambda x: 'New Orleans' if x == 'New Orleans/Oklahoma City Hornets' else  x)
df_games['Home'] = df_games['Home'].apply(lambda x: 'New Orleans' if x == 'New Orleans/Oklahoma City Hornets' else  x)
df_games['Away'] = df_games['Away'].apply(lambda x: ' '.join(x.split(' ')[:2]) if len(x.split(' ')) == 3 else  x)
df_games['Home'] = df_games['Home'].apply(lambda x: ' '.join(x.split(' ')[:2]) if len(x.split(' ')) == 3 else  x)

# Cleaning team names from point spreads DF
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'San Antonio' if x == 'SanAntonio' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'San Antonio' if x == 'SanAntonio' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'New York' if x == 'NewYork' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'New York' if x == 'NewYork' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'New Jersey' if x == 'NewJersey' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'New Jersey' if x == 'NewJersey' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Miami' if x == 'MiamiHeat' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Miami' if x == 'MiamiHeat' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'LA Lakers' if x == 'LALakers' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'LA Lakers' if x == 'LALakers' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Golden State' if x == 'GoldenState' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Golden State' if x == 'GoldenState' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'New Orleans' if x == 'NewOrleans' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'New Orleans' if x == 'NewOrleans' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'LA Clippers' if x == 'LAClippers' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'LA Clippers' if x == 'LAClippers' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Oklahoma City' if x == 'OklahomaCity' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Oklahoma City' if x == 'OklahomaCity' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Milwaukee' if x == 'MilwaukeeBucks' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Milwaukee' if x == 'MilwaukeeBucks' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Charlotte' if x == 'CharlotteHornets' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Charlotte' if x == 'CharlotteHornets' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Brooklyn' if x == 'BrooklynNets' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Brooklyn' if x == 'BrooklynNets' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Memphis' if x == 'MemphisGrizzlies' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Memphis' if x == 'MemphisGrizzlies' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Houston' if x == 'HoustonRockets' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Houston' if x == 'HoustonRockets' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Detroit' if x == 'DetroitPistons' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Detroit' if x == 'DetroitPistons' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Atlanta' if x == 'AtlantaHawks' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Atlanta' if x == 'AtlantaHawks' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Chicago' if x == 'ChicagoBulls' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Chicago' if x == 'ChicagoBulls' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Philadelphia' if x == 'Philadelphia76ers' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Philadelphia' if x == 'Philadelphia76ers' else  x)
df_spread['Away'] = df_spread['Away'].apply(lambda x: 'Sacramento' if x == 'SacramentoKings' else  x)
df_spread['Home'] = df_spread['Home'].apply(lambda x: 'Sacramento' if x == 'SacramentoKings' else  x)



# Joining Tables into Final DF
df6 =  pd.merge(df_games, df_odds,  how='left', 
                left_on=['date','Home', 'Away'], 
                right_on = ['date','home', 'away'])
df7 =  pd.merge(df6, df_spread,  how='left', 
                left_on=['date','Home', 'Away'], 
                right_on = ['Date','Home', 'Away'])

# Creating feature for How many points favored team is ahead by at halftime.
conditions = [
    (df7['Home Spread'] < 0),
    (df7['Home Spread'] > 0)]

choices = [df7['PTS H-A'], -1 * df7['PTS H-A']]
df7['Favored Ahead By'] = np.select(conditions, choices)

# Removing Outliers
df7 = df7[df7['Home Spread'].notna()]
df7 = df7[df7['Home Spread'].abs() <= 30]
df7 = df7[df7['TOV/AST H-A'].abs() <= 6]
df7 = df7[df7['PF H-A'].abs() <= 15]
df7 = df7[df7['PTS H-A'].abs() <= 40]
df7 = df7[df7['BLK H-A'].abs() <= 12]
df7 = df7[df7['AST H-A'].abs() <= 20]

# Storing Final DF
df7.to_csv('data/data.csv')

