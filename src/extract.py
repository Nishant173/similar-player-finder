import pandas as pd
import utils

def read_games_data():
    return pd.read_csv(filepath_or_buffer="../data/games.csv", sep=',')


def read_players_data():
    return pd.read_csv(filepath_or_buffer="../data/players.csv", sep=',')


def read_player_feature_columns():
    return utils.pickle_load(filename="../data_pickled/player_feature_columns.pkl")


def read_user_input():
    """ Gets user input regarding which data is to be fetched """
    data_inputs = pd.read_csv(filepath_or_buffer="../inputs/user_inputs.csv", sep=',')
    data_inputs.dropna(inplace=True)
    dictionary_user_inputs = data_inputs.set_index(keys='variable').to_dict()['value']
    return dictionary_user_inputs