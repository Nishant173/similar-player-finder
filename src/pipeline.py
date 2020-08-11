import extract
import transform


def execute_pipeline():
    """ Executes ETL pipeline """
    # Extract
    data_games = extract.read_games_data()
    data_players = extract.read_players_data()
    data_players_raw = data_players.copy()
    dictionary_user_inputs = extract.read_user_input()
    player = dictionary_user_inputs.get('player')

    # Transform
    df_games_transformed = transform.transform_games_data(data_games=data_games)
    df_players_transformed = transform.transform_players_data(data_players=data_players)
    df_similar_players = transform.get_similar_players(df_games_transformed=df_games_transformed,
                                                       df_players_transformed=df_players_transformed,
                                                       df_players_raw=data_players_raw,
                                                       dictionary_user_inputs=dictionary_user_inputs)

    # Load
    df_similar_players.to_csv(path_or_buf=f"../results/{player} - Similar players.csv",
                              sep=',',
                              encoding='utf-8',
                              index=False)
    return None