# This script scrapes the games list, game box scores, and betting odds for 
# the 2nd half moneyline bet. 
# Pre-game point spread were also acquired by manual download from 
# https://www.sportsbookreviewsonline.com/scoresoddsarchives/nba/nbaoddsarchives.htm
# beginning with the 2007-08 season. The downloaded spreadsheets were stored in
# 'data/game_lines/'

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pymongo import MongoClient
import numpy as np

# Defining Function to use in parsing and storing box score data
def build_temp_df(temp_df, rows, home=True):
    '''
    Takes a list of table rows and creates features for starters minutes played,
    3-Point %, Field Goal %, and Free Throw % while filling the supplied temporary 
    dataframe with the home and away teams' data for that game.
    ----------
    Parameters
    ----------
    temp_df : Pandas DataFrame
        A DF containing the game box-score link 
    rows : list
        A list of rows containing the Home or away teams' box-score data from the game.
    home : Boolean
        A Boolean value indicating weather the rows list is the Home team' data or the
        Away team's data.
    ----------
    Returns 
    ----------
    None
    '''
    rows = [row for row in rows if (len(row)>1 and row[0]!='')]
    df_new = pd.DataFrame(rows[1:], columns=rows[0])  
    if home==True:
        pre = 'H-'
    else:
        pre = 'A-'    
    for i, MP in enumerate(df_new['MP']):
        df_new.loc[i,'MP'] = float(MP.split(':')[0]) + float(MP.split(':')[1])/60
    temp_df[pre + 'S-Min'] = df_new.loc[:4,'MP'].sum()        
    for i, diff in enumerate(df_new['+/-']):
        df_new.loc[i,'+/-'] = diff.strip('+')        
    df_new.drop(['FG%', '3P%', 'FT%', 'MP'], axis=1, inplace=True)    
    for col in df_new.columns[:-1]:
        df_new[col] = df_new[col].astype(float)
        temp_df[pre + col] = df_new[col].sum()        
    temp_df[pre + 'FG%'] = temp_df[pre + 'FG']/temp_df[pre + 'FGA']
    temp_df[pre + '3P%'] = temp_df[pre + '3P']/temp_df[pre + '3PA']
    temp_df[pre + 'FT%'] = temp_df[pre + 'FT']/temp_df[pre + 'FTA']



# Connecting to MongoDB and initializing NoSQL DB for raw source code storage
client = MongoClient('localhost', 27017)
db = client.NBA
games_list = db.games_list
box_scores = db.box_scores
betting_list = db.betting_list
betting_odds = db.betting_odds



# Games List
# Web scraping a list of all NBA games from 2001-02 to 2019-20 seasons
months = ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may', 'june']
all_rows = []
rejects = {}
for i in range(2001, 2021):
    if i == 2020:
        months = months[:-3]
    for month in months:
        # scrape and store page source in MongoDB
        url = 'https://www.basketball-reference.com/leagues/NBA_' + str(i) + '_games-' + month + '.html'
        r = requests.get(url)
        games_list.insert_one({'year': i, 'month': month, 'html': r.content})
        

# Parsing data from MongoDB
for i in range(2001, 2021):
    if i == 2020:
        months = months[:-3]
    for month in months:        
        # parse table data into rows
        query = {"$and":
                [{"year": i},
                {"month": month}]
                }
        page = games_list.find_one(query)['html']
        soup = BeautifulSoup(page)
        if i == 2001 and month == 'october':
            headers = [h.text for h in soup.find_all('th')]
            headers = headers[1:10]
            headers[-1] = 'Link'
            all_rows.append(headers)
        rows = soup.find_all('tr')[1:]
        for row in rows:
            if len(row)==1:
                continue
            link = str(row.find_all('a')[-1]).split('"')[1]
            data = [entry.text for entry in row.find_all('td')]
            data[-1] = link
            all_rows.append(data)

# Rename headers
all_rows[0] = ['Start', 'Away', 'A-Pts-F', 'Home', 'H-Pts-F',
                    'Box', 'OT?', 'Attend.', 'Link']

# Store games list in CSV
df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
df.to_csv('data/games.csv')



# Box Scores
# Initializing custom headers for game box-score data
headers = ['MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 'FT', 'FTA', 'FT%',
           'ORB', 'DRB', 'TRB', 'AST', 'STL', 'BLK', 'TOV', 'PF', 'PTS', '+/-']

# Web scraping box scores for all NBA games from 2001-2002 to 2019-2020 seasons
for i, l in enumerate(df['Link']):
    # scrape and store page source in MongoDB
    url= 'https://www.basketball-reference.com' + l
    r = requests.get(url)
    box_scores.insert_one({'Link': l, 'html': r.content})
    
# Retrieving data from MongoDB to parse
for i, l in enumerate(df['Link']):
    page = box_scores.find_one({'Link': l})['html']
    soup = BeautifulSoup(page)
    # parse box-score page into table rows
    H1_away = soup.find_all('table')[3]
    if df.loc[i, 'OT?']=='OT':
        H1_home = soup.find_all('table')[12]
    elif df.loc[i, 'OT?']=='2OT':
        H1_home = soup.find_all('table')[13]
    elif df.loc[i, 'OT?']=='3OT':
        H1_home = soup.find_all('table')[14]
    elif df.loc[i, 'OT?']=='4OT':
        H1_home = soup.find_all('table')[15]
    else:
        H1_home = soup.find_all('table')[11]
    
    H1_away_rows = [headers]
    for row in H1_away.find_all('tr')[2:7]:
        new_row = [d.text for d in row.find_all('td')]
        H1_away_rows.append(new_row)
    for row in H1_away.find_all('tr')[8:-1]:
        new_row = [d.text for d in row.find_all('td')]
        H1_away_rows.append(new_row)
    
    H1_home_rows = [headers]
    for row in H1_home.find_all('tr')[2:7]:
        new_row = [d.text for d in row.find_all('td')]
        H1_home_rows.append(new_row)
    for row in H1_home.find_all('tr')[8:-1]:
        new_row = [d.text for d in row.find_all('td')]
        H1_home_rows.append(new_row)
        
    # Build temp dF with one row for each game
    temp_df = pd.DataFrame([l], columns=['Link'])
    build_temp_df(temp_df, H1_away_rows, home=False)
    build_temp_df(temp_df, H1_home_rows)

    # Append new game temp DF as row in Game Stats DF
    if i == 0:
        game_stats_df = temp_df.copy()
    else:
        game_stats_df = game_stats_df.append(temp_df)

# Store games stats in CSV
game_stats_df.to_csv('data/game_stats.csv')



# Betting Odds
# Creating values used to scrape betting odds data
df['year'] = df['Link'].apply(lambda x: x[11:15])
df['date'] = df['Link'].apply(lambda x: x[11:19])

# Initializing a df with each game and a link to the page to scrape the betting odds
df_games = pd.DataFrame(columns=['date','link', 'home', 'away'])

# Web scraping each days games for all NBA games from 2015-2016 to 2019-2020 seasons
for i, day in enumerate(df['date'].unique()):
    url= 'https://www.sportsbookreview.com/betting-odds/nba-basketball/money-line/2nd-half/?date=' + day
    r = requests.get(url)
    betting_list.insert_one({'Date': day, 'html': r.content})

    
# Retrieving data from MongoDB for parsing
for i, day in enumerate(df['date'].unique()):
    page = betting_list.find_one({'Date': day})['html']
    soup = BeautifulSoup(page, "html.parser")
    days = []
    link = []
    home = []
    away = []
    for j, tag in enumerate(soup.find('div', class_='_1eZfC').find_all('a', class_='_3qi53')):
        if j % 2 == 0:
            away.append(tag.text)
            link.append(str(tag).split('"')[3])
            days.append(day)
        else:
            home.append(tag.text)
    df_games = df_games.append(pd.DataFrame({'date': days,'link': link, 'home': home, 'away': away}))

# Initializing a df with each game and it's betting odds
df_games2 = pd.DataFrame(columns=['link', 'away_ML', 'home_ML', 'away_open', 'home_open'])

# Web scraping game's 2nd half betting odds for all NBA games from 2015-2016 to 2019-2020 seasons
for i, l in enumerate(df_games['link']):
    url= 'https://www.sportsbookreview.com' + l
    r = requests.get(url)
    betting_odds.insert_one({'Link': l, 'html': r.content})

# Retrieving data from MongoDB for parsing
for i, l in enumerate(df_games['link']):
    page = betting_odds.find_one({'Link': l})['html']
    soup = BeautifulSoup(page, "html.parser")

    half2 = soup.find_all('div', class_='_398eq')[2]
    full = soup.find_all('div', class_='_398eq')[0]

    away_ML = half2.find_all('span', class_='opener')[8].text
    away_open = full.find_all('span', class_='opener')[2].text
    home_ML = half2.find_all('span', class_='opener')[9].text
    home_open = full.find_all('span', class_='opener')[4].text

    away_ML, home_ML
    temp = pd.DataFrame(data={'link': [l], 
                            'away_ML':[away_ML], 
                            'home_ML':[home_ML],
                            'away_open': [away_open],
                            'home_open': [home_open]})
    df_games2 = df_games2.append(temp)

# Resetting indexes for the two newly created DFs
df_games.reset_index(inplace=True)
df_games.drop('index', axis=1, inplace=True)
df_games2.reset_index(inplace=True)
df_games2.drop('index', axis=1, inplace=True)

# Joining two betting DFs into one DF
df_game_odds = df_games.join(df_games2, rsuffix='2')
df_game_odds.drop('link2', axis=1, inplace=True)
df_game_odds.drop(index=1026, inplace=True)

# Store games betting odds in CSV
df_game_odds.to_csv('data/game_odds.csv')