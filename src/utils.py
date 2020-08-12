import os
import time
import warnings
import joblib
import numpy as np
import pandas as pd
import extract
import settings


def create_results_folder():    
    try:
        os.mkdir(settings.PATH_RESULTS_DIR)
    except FileExistsError:
        pass
    return None


def save_to_csv(data, filepath):
    if str(filepath).strip().lower()[-4:] != '.csv':
        filepath = str(filepath) + '.csv'
    data.to_csv(path_or_buf=filepath,
                sep=',',
                encoding='utf-8',
                index=False)
    return None


def get_player_data_by_subset(players):
    """ Gets DataFrame of player-data based on list of player-names given """
    data_players = extract.read_players_data()
    data_by_player_subset = pd.DataFrame()
    for player in players:
        df_temp = data_players[data_players['player'] == player]
        data_by_player_subset = pd.concat(objs=[data_by_player_subset, df_temp], ignore_index=True, sort=False)
    return data_by_player_subset


def drop_player_duplicates(data_players):
    """ Drops duplicates from 'player' column in DataFrame, and returns resulting DataFrame """
    if 'player' not in data_players.columns:
        raise ValueError("The 'player' column is missing from the DataFrame. Cannot drop duplicates!")
    data_players.drop_duplicates(subset=['player'], keep='last', inplace=True, ignore_index=True)
    return data_players


def search_player(name):
    """
    Returns DataFrame of string-matches found by player.
    Gives empty DataFrame if no string-matches exist.
    """
    name = str(name).lower().strip()
    data_players = extract.read_players_data()
    data_players = data_players.loc[:, ['player', 'squad']]
    data_players = drop_player_duplicates(data_players=data_players)
    data_players.sort_values(by='player', ascending=True, inplace=True, ignore_index=True)
    data_players['player_lowercase'] = data_players['player'].str.lower()
    data_players_found = data_players[data_players['player_lowercase'].str.contains(name)]
    data_players_found.drop(labels='player_lowercase', axis=1, inplace=True)
    if data_players_found.empty:
        if settings.PRINT_DETAILS:
            print("No matches found!")
        return pd.DataFrame()
    return data_players_found.reset_index(drop=True)


def pickle_load(filename):
    """ Loads data from pickle file, via joblib module """
    data_obj = joblib.load(filename=filename)
    return data_obj


def pickle_save(data_obj, filename):
    """ Stores data as pickle file, via joblib module """
    joblib.dump(value=data_obj, filename=filename)
    return None


def get_timetaken_fstring(num_seconds):
    """ Returns formatted-string of time elapsed, given the number of seconds (int) elapsed """
    if num_seconds < 60:
        secs = num_seconds
        fstring_timetaken = f"{secs}s"
    elif 60 < num_seconds < 3600:
        mins, secs = divmod(num_seconds, 60)
        fstring_timetaken = f"{mins}m {secs}s"
    else:
        hrs, secs_remainder = divmod(num_seconds, 3600)
        mins, secs = divmod(secs_remainder, 60)
        fstring_timetaken = f"{hrs}h {mins}m {secs}s"
    return fstring_timetaken


def run_and_timeit(func):
    """
    Takes in function-name; then runs it, times it, and prints out the time taken.
    Parameters:
        - func (object): Object of the function you want to execute.
    """
    start = time.time()
    warnings.filterwarnings(action='ignore')
    func()
    end = time.time()
    print(f"\nDone! Time taken: {get_timetaken_fstring(num_seconds=int(np.ceil(end - start)))}")
    return None