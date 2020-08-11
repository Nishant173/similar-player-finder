import extract
import transform
import utils

def execute_pipeline():
    """ Executes ETL pipeline """
    # Extract
    data_games = extract.read_games_data()
    data_players = extract.read_players_data()
    data_players_raw = data_players.copy()
    dictionary_user_inputs = extract.read_user_input()
    player = dictionary_user_inputs.get('player')
    skill = dictionary_user_inputs.get('skill')

    # Transform
    df_games_transformed = transform.transform_games_data(data_games=data_games)
    df_players_transformed = transform.transform_players_data(data_players=data_players)
    df_similar_players = transform.get_similar_players(df_games_transformed=df_games_transformed,
                                                       df_players_transformed=df_players_transformed,
                                                       df_players_raw=data_players_raw,
                                                       dictionary_user_inputs=dictionary_user_inputs)
    df_player_stats_subset = utils.get_player_data_by_subset(players=df_similar_players['player'].tolist())

    # Load
    utils.save_to_csv(data=df_similar_players,
                      filepath=f"../results/{player} - Similar players ({skill}).csv")
    utils.save_to_csv(data=df_player_stats_subset,
                      filepath=f"../results/{player} - Similar players' stats ({skill}).csv")
    return None