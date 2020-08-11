import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import extract
import settings
import utils


def subset_shot_data(data):
    columns = ['goals', 'xg', 'npxg', 'shots_total', 'shots_on_target', 'shots_free_kicks', 'shots_on_target_pct',
               'goals_per_shot', 'goals_per_shot_on_target', 'npxg_per_shot', 'xg_net', 'npxg_net']
    return data.loc[:, columns]


def subset_chance_creation_data(data):
    columns = ['sca', 'sca_passes_live', 'sca_passes_dead', 'sca_dribbles', 'sca_shots', 'sca_fouled', 'assisted_shots',
               'through_balls', 'gca', 'gca_passes_live', 'gca_passes_dead', 'gca_dribbles', 'gca_shots', 'gca_fouled',
               'gca_og_for','assists','xa']
    return data.loc[:, columns]


def subset_passing_data(data):
    columns = ['passes_completed', 'passes', 'passes_pct', 'passes_total_distance', 'passes_progressive_distance',
               'passes_completed_short', 'passes_short', 'passes_pct_short', 'passes_completed_medium', 'passes_medium',
               'passes_pct_medium', 'passes_completed_long', 'passes_long', 'passes_pct_long', 'passes_into_final_third',
               'passes_into_penalty_area', 'crosses_into_penalty_area', 'progressive_passes', 'passes_live', 'passes_dead',
               'passes_free_kicks', 'passes_pressure', 'passes_switches', 'crosses', 'corner_kicks', 'corner_kicks_in',
               'corner_kicks_out', 'corner_kicks_straight', 'passes_ground', 'passes_low', 'passes_high', 'passes_left_foot',
               'passes_right_foot', 'passes_head', 'throw_ins', 'passes_other_body', 'passes_offsides', 'passes_oob',
               'passes_intercepted', 'passes_blocked']
    return data.loc[:, columns]


def subset_defending_data(data):
    columns = ['tackles', 'tackles_won', 'tackles_def_3rd', 'tackles_mid_3rd', 'tackles_att_3rd', 'dribble_tackles',
               'dribbles_vs', 'dribble_tackles_pct', 'dribbled_past', 'pressures', 'pressure_regains', 'pressure_regain_pct',
               'pressures_def_3rd', 'pressures_mid_3rd', 'pressures_att_3rd', 'blocks', 'blocked_shots', 'blocked_shots_saves',
               'blocked_passes', 'interceptions', 'clearances', 'errors']
    return data.loc[:, columns]


def subset_possession_data(data):
    columns = ['touches', 'touches_def_pen_area', 'touches_def_3rd', 'touches_mid_3rd', 'touches_att_3rd',
               'touches_att_pen_area', 'touches_live_ball', 'dribbles_completed', 'dribbles', 'dribbles_completed_pct',
               'players_dribbled_past', 'nutmegs', 'carries', 'carry_distance', 'carry_progressive_distance',
               'pass_targets', 'passes_received_pct', 'miscontrols', 'dispossessed']
    return data.loc[:, columns]


def get_points_from_result(result_obj):
    result_obj = str(result_obj).strip().upper()
    if result_obj == 'W':
        points = 3
    elif result_obj == 'D':
        points = 1
    elif result_obj == 'L':
        points = 0
    else:
        raise ValueError(f"Result object not in ['W', 'L', 'D']. Object is: '{result_obj}'")
    return points


def get_per90_stats(data_players):
    player_feature_columns = extract.read_player_feature_columns()
    for player_feature in player_feature_columns:
        if player_feature != 'minutes':
            data_players[player_feature] = (data_players[player_feature] / data_players['minutes']) * 90
    columns_to_drop = ['goals_per90', 'cards_yellow', 'cards_red', 'assists_per90', 'goals_assists_per90',
                       'goals_pens_per90', 'goals_assists_pens_per90', 'xg_per90', 'xa_per90', 'xg_xa_per90',
                       'npxg_per90', 'npxg_xa_per90', 'minutes_90s', 'shots_total_per90', 'shots_on_target_per90',
                       'xa_net', 'sca_per90', 'gca_per90', 'passes_received', 'cards_yellow_red', 'fouls',
                       'fouled', 'offsides', 'pens_won', 'pens_conceded', 'own_goals', 'ball_recoveries',
                       'aerials_won', 'aerials_lost', 'aerials_won_pct']
    data_players.drop(labels=columns_to_drop, axis=1, inplace=True)
    return data_players


def drop_irrelevant_features(data_players):
    irrelevant_features = ['player', 'nationality', 'position', 'squad', 'age', 'birth_year',
                           'games', 'games_starts', 'minutes']
    data_players.drop(labels=irrelevant_features, axis=1, inplace=True)
    return data_players


def scale_features(data_players):
    scaler = MinMaxScaler()
    columns = data_players.columns.tolist()
    for column in columns:
        data_players[column] = scaler.fit_transform(data_players[[column]])
    return data_players


def transform_games_data(data_games):
    data_games['points_obtained'] = data_games['result'].apply(get_points_from_result)
    data_games.rename(columns={'xg_for': 'xg'}, inplace=True)
    return data_games


def transform_players_data(data_players):
    data_players = utils.drop_player_duplicates(data_players=data_players)
    data_players = get_per90_stats(data_players=data_players)
    data_players = drop_irrelevant_features(data_players=data_players)
    data_players = scale_features(data_players=data_players)
    return data_players


def rank_similar_players(data_pca_features, player, skill, num_results):
    """
    Takes in DataFrame of PCA features, along with `player`, `skill`, `num_results`; and calculates
    distances b/w `player` and similar players based on `skill` given.
    Returns DataFrame sorted by top `num_results` matches of similar players.
    """
    columns_to_drop = ['player', 'squad', 'position', 'age']
    df_stats_by_player = data_pca_features[data_pca_features['player'] == player]
    df_stats_by_player.drop(labels=columns_to_drop, axis=1, inplace=True)

    columns_of_stats = df_stats_by_player.columns.tolist()
    distance = (data_pca_features[columns_of_stats] - np.array(df_stats_by_player)).pow(2).sum(1).pow(0.5)
    data_pca_features['distance'] = distance

    distance_at_quantile = data_pca_features['distance'].quantile(q=0.95)
    series_percent_match = (100 - (data_pca_features['distance'].mul(100) / distance_at_quantile)).apply(round, args=[3])
    data_pca_features['percent_match'] = series_percent_match
    data_pca_features.sort_values(by='distance', ascending=True, inplace=True, ignore_index=True)
    data_pca_features = data_pca_features.head(num_results + 1)
    data_pca_features['skill'] = skill
    columns_to_show = ['player', 'squad', 'position', 'age', 'skill', 'percent_match']
    data_ranked_similar_players = data_pca_features.loc[:, columns_to_show]
    return data_ranked_similar_players


def get_similar_players(df_games_transformed, df_players_transformed, df_players_raw, dictionary_user_inputs):
    """
    Definition:
        Builds PCA model that ranks players based on similarities with given player, based on skill.
        Returns Pandas DataFrame of ranked similar players.
    Parameters:
        - df_games_transformed (DataFrame): Transformed games data
        - df_players_transformed (DataFrame): Transformed players data
        - df_players_raw (DataFrame): Original (raw) players data
        - dictionary_user_inputs (dict): User inputs with keys: ['player', 'team', 'skill', 'num_results']
    """
    if not isinstance(dictionary_user_inputs, dict):
        raise TypeError(f"Expected `dictionary_user_inputs` to be of type 'dict', but got type '{type(dictionary_user_inputs)}'")
    player = dictionary_user_inputs.get('player')
    team = dictionary_user_inputs.get('team')
    skill = dictionary_user_inputs.get('skill')
    num_results = int(dictionary_user_inputs.get('num_results'))
    if settings.PRINT_DETAILS:
        print(
            f"\nPlayer: {player}",
            f"\nTeam: {team}",
            f"\nSkill: {skill}",
            f"\nNumber of results: {num_results}"
        )

    skills = ['Overall', 'Possession', 'Shooting', 'Passing', 'ChanceCreation', 'DefensiveWork']
    if skill not in skills:
        raise ValueError(f"Invalid skill entered. Must be one of {skills}. It's case-sensitive!")
    
    df_games_transformed = df_games_transformed.loc[:, ~df_games_transformed.T.duplicated(keep='first')]
    df_games_transformed = df_games_transformed.loc[:, ~df_games_transformed.columns.duplicated()]

    if team == 'Overall':
        df_corr_matrix = df_games_transformed.corr()
    else:
        df_corr_matrix = df_games_transformed[df_games_transformed['for'] == team].corr()
    
    player_features = df_players_transformed.columns.tolist()
    for player_feature in player_features:
        df_players_transformed[player_feature] = (df_players_transformed[player_feature]) * (df_corr_matrix['points_obtained'][player_feature])
    
    df_players_by_skill = pd.DataFrame()
    if skill == 'Overall':
        df_players_by_skill = df_players_transformed.copy()
    elif skill == 'Possession':
        df_players_by_skill = subset_possession_data(data=df_players_transformed)
    elif skill == 'Shooting':
        df_players_by_skill = subset_shot_data(data=df_players_transformed)
    elif skill == 'Passing':
        df_players_by_skill = subset_passing_data(data=df_players_transformed)
    elif skill == 'ChanceCreation':
        df_players_by_skill = subset_chance_creation_data(data=df_players_transformed)
    elif skill == 'DefensiveWork':
        df_players_by_skill = subset_defending_data(data=df_players_transformed)

    features = df_players_by_skill.columns.tolist()
    X = df_players_by_skill.loc[:, features].values
    X = np.nan_to_num(X)
    pca = PCA(n_components=0.9)
    nd_array_principal_components = pca.fit_transform(X)
    if settings.PRINT_DETAILS:
        print(f"\nNumber of principal components used in PCA: {pca.n_components_}")
    df_pca_features = pd.DataFrame(data=nd_array_principal_components)
    df_players_raw = utils.drop_player_duplicates(data_players=df_players_raw)
    if len(df_pca_features) != len(df_players_raw):
        length1, length2 = len(df_pca_features), len(df_players_raw)
        raise ValueError(f"DataFrames to be concatenated by columns have un-equal lengths! ({length1} and {length2})")
    df_pca_features = pd.concat(objs=[df_pca_features, df_players_raw[['player']]], axis=1)
    df_pca_features = pd.concat(objs=[df_pca_features, df_players_raw[['squad']]], axis=1)
    df_pca_features = pd.concat(objs=[df_pca_features, df_players_raw[['position']]], axis=1)
    df_pca_features = pd.concat(objs=[df_pca_features, df_players_raw[['age']]], axis=1)
    df_ranked_similar_players = rank_similar_players(data_pca_features=df_pca_features,
                                                     player=player,
                                                     skill=skill,
                                                     num_results=num_results)
    if settings.PRINT_DETAILS:
        print(f"\nRanked similar players:\n{df_ranked_similar_players}")
    return df_ranked_similar_players