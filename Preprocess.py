import pandas as pd
import os
from datetime import datetime


def preprocess_data(players_df, player_valuations_df, appearances_df):
    """
    Preprocesses player data before merging.
    This function calculates player age, contract remaining time,
    and adds year columns to valuations and appearances.

    Parameters:
        players_df (pd.DataFrame): Players dataset.
        player_valuations_df (pd.DataFrame): Player valuations dataset.
        appearances_df (pd.DataFrame): Appearances dataset.

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: Processed players, valuations, and appearances DataFrames.
    """
    print("Running Preprocessing Steps...")

    # Convert date columns to datetime format
    players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'], errors='coerce')
    players_df['contract_expiration_date'] = pd.to_datetime(players_df['contract_expiration_date'], errors='coerce')

    # Remove players with missing birth date
    players_df = players_df.dropna(subset=['date_of_birth']).copy()



    # Ensure the expected column is present in player_valuations_df
    if 'date' in player_valuations_df.columns:
        player_valuations_df['date'] = pd.to_datetime(player_valuations_df['date'], format="%Y-%m-%d", errors='coerce')
        player_valuations_df['year'] = player_valuations_df['date'].dt.year
    else:
        print("⚠ Warning: Column 'date' not found in player_valuations_df. Skipping year extraction.")

    # Ensure the expected column is present in appearances_df
    if 'date' in appearances_df.columns:
        appearances_df['date'] = pd.to_datetime(appearances_df['date'], format="%Y-%m-%d", errors='coerce')
        appearances_df['year'] = appearances_df['date'].dt.year
    else:
        print("⚠ Warning: Column 'date' not found in appearances_df. Skipping year extraction.")

    print("Preprocessing complete.")
    return players_df, player_valuations_df, appearances_df








def merge_players_data(players_df, appearances_df, games_df, player_valuations_df):
    """
    Merges datasets, groups players by season, and calculates performance stats including clean sheets.
    """
    print("🔄 Merging Player Data...")

    players_df = players_df.drop_duplicates(subset=['player_id'])
    appearances_df = appearances_df.drop_duplicates(subset=['player_id', 'game_id'])
    games_df = games_df.drop_duplicates(subset=['game_id'])

    players_info = players_df.set_index("player_id")

    full_players_info = appearances_df.merge(
        games_df[['game_id', 'season', 'home_club_id', 'away_club_id', 'home_club_goals', 'away_club_goals']],
        on='game_id', how='left'
    )

    full_players_info = full_players_info.merge(players_df, on='player_id', how='left')

    player_valuations_df['date'] = pd.to_datetime(player_valuations_df['date'], format="%d/%m/%Y", errors='coerce')
    player_valuations_df['year'] = player_valuations_df['date'].dt.year
    player_valuations_df = player_valuations_df.sort_values(['player_id', 'year', 'date'])
    player_valuations_df = player_valuations_df.drop_duplicates(subset=['player_id', 'year'], keep='last')

    full_players_info = full_players_info.merge(
        player_valuations_df[['player_id', 'year', 'market_value_in_eur']],
        left_on=['player_id', 'season'],
        right_on=['player_id', 'year'],
        how='left'
    )

    full_players_info.rename(columns={'market_value_in_eur': 'actual_market_value'}, inplace=True)

    sum_columns = ['yellow_cards', 'red_cards', 'goals', 'assists', 'minutes_played', 'goals_against']
    sum_columns = [col for col in sum_columns if col in full_players_info.columns]

    for col in sum_columns:
        full_players_info[col] = pd.to_numeric(full_players_info[col], errors='coerce').fillna(0)

    full_players_info['games_played'] = 1

    # ✅ Clean sheet calculation
    def calculate_clean_sheet(row):
        if row.get('position') != 'Goalkeeper':
            return -1
        team_id = row.get('player_club_id')
        if team_id == row.get('home_club_id') and row.get('away_club_goals', 1) == 0:
            return 1
        elif team_id == row.get('away_club_id') and row.get('home_club_goals', 1) == 0:
            return 1
        return 0

    full_players_info['clean_sheet'] = full_players_info.apply(calculate_clean_sheet, axis=1)
    sum_columns.append('clean_sheet')

    all_columns = set(full_players_info.columns)
    keep_first_columns = list(all_columns - set(sum_columns) - {'player_id', 'season'})

    grouped = full_players_info.groupby(['player_id', 'season'], as_index=False).agg({
        **{col: 'sum' for col in sum_columns + ['games_played']},
        **{col: 'first' for col in keep_first_columns}
    })

    for col in players_info.columns:
        if col in grouped.columns:
            grouped[col] = grouped[col].fillna(grouped['player_id'].map(players_info[col]))

    grouped['season_date'] = pd.to_datetime(grouped['season'].astype(str) + "-01-01")
    grouped['season_date'] -= pd.DateOffset(years=1)

    grouped['age'] = ((grouped['season_date'] - grouped['date_of_birth']).dt.days / 365.25).round().astype('Int64')

    grouped['term_days_remaining'] = (
        grouped['contract_expiration_date'] - grouped['season_date']
    ).dt.days.fillna(-1).astype('Int64')

    grouped = grouped.drop(columns=['season_date'])

    print("✅ Age and contract remaining time calculated correctly!")
    return grouped






def create_team_year_market_value(players_df, output_file):
    """
    Creates a new CSV file containing team_id, year, and total market value using actual market value.

    Parameters:
        players_df (pd.DataFrame): Processed Players DataFrame.
        output_file (str): Path to save the new dataset.
    """
    print("🔄 Creating Team-Year Market Value Dataset...")

    # Ensure necessary columns are numeric
    players_df['market_value_in_eur_y'] = pd.to_numeric(players_df['market_value_in_eur_y'], errors='coerce')
    players_df['player_club_id'] = pd.to_numeric(players_df['player_club_id'], errors='coerce')
    players_df['season'] = pd.to_numeric(players_df['season'], errors='coerce')

    # Drop rows with missing values in required columns
    players_df = players_df.dropna(subset=['market_value_in_eur_y', 'player_club_id', 'season'])

    # Group by year and club ID, summing player market values
    team_year_market_values = (
        players_df.groupby(['season', 'player_club_id'])['market_value_in_eur_y']
        .sum()
        .reset_index()
    )

    # Rename columns for clarity
    team_year_market_values.rename(columns={
        'season': 'year',
        'player_club_id': 'team_id',
        'market_value_in_eur_y': 'total_market_value'
    }, inplace=True)

    # Save the new dataset
    team_year_market_values.to_csv(output_file, index=False)

    print(f"✅ New Team-Year Market Value dataset saved at: {output_file}")







def remove_unnecessary_columns(input_file, output_file):
    """
    Cleans the dataset by removing unnecessary columns and filtering invalid player data.

    Parameters:
        input_file (str): Path to the original dataset.
        output_file (str): Path to save the cleaned dataset.

    Returns:
        pd.DataFrame: The cleaned dataset.
    """
    print("🔄 Cleaning Dataset...")

    # Define columns to remove
    cols_to_remove = [
        'player_current_club_id', 'year_x', 'first_name', 'last_name', 'name',
        'last_season', 'current_club_id', 'image_url', 'url',
        'market_value_in_eur_x', 'year_y','player_code','agent_name','term_days_remaining','contract_expiration_date'
    ]

    # Load dataset
    df = pd.read_csv(input_file)

    # Remove unnecessary columns if they exist
    df = df.drop(columns=[col for col in cols_to_remove if col in df.columns], errors='ignore')

    # Remove players with height < 50 cm
    if 'height_in_cm' in df.columns:
        df = df[df['height_in_cm'] >= 50]

    # Remove players with missing position (NaN or "Missing")
    if 'position' in df.columns:
        df = df[df['position'].notna()]  # Remove NaN values
        df = df[df['position'] != 'Missing']  # Remove rows where position is explicitly "Missing"

    # Save the cleaned dataset
    df.to_csv(output_file, index=False)

    print(f"✅ Cleaned dataset saved at: {output_file}")
    return df
def add_team_year_market_value_to_players(players_df, team_market_value_file):
    """
    Adds total market value per team-year to the players DataFrame.

    Parameters:
        players_df (pd.DataFrame): Cleaned player data with 'season' and 'player_club_id'.
        team_market_value_file (str): Path to the CSV containing team-year market values.

    Returns:
        pd.DataFrame: Updated players_df with 'total_team_market_value' column.
    """
    print("🔄 Merging team-year market values into player data...")

    # Load team-year market value dataset
    team_year_df = pd.read_csv(team_market_value_file)

    # Ensure proper types for merge
    team_year_df['year'] = pd.to_numeric(team_year_df['year'], errors='coerce')
    team_year_df['team_id'] = pd.to_numeric(team_year_df['team_id'], errors='coerce')
    players_df['season'] = pd.to_numeric(players_df['season'], errors='coerce')
    players_df['player_club_id'] = pd.to_numeric(players_df['player_club_id'], errors='coerce')

    # Merge on season and team ID
    merged_df = players_df.merge(
        team_year_df,
        left_on=['season', 'player_club_id'],
        right_on=['year', 'team_id'],
        how='left'
    )

    # Rename the new column for clarity
    merged_df.rename(columns={'total_market_value': 'total_team_market_value'}, inplace=True)

    # Drop extra merge columns if desired
    merged_df.drop(columns=['year', 'team_id'], inplace=True, errors='ignore')

    print("✅ Team market values added to player data.")
    return merged_df


def main():
    """
    Main function to:
    1. Preprocess data before merging.
    2. Merge datasets.
    3. Remove unnecessary columns.
    4. Create a new dataset containing only team_id, year, and market value.
    5. Save final processed data.
    """
    print("🚀 Starting Data Processing...")

    # Define file paths
    data_dir = r"C:\Users\Asus\Documents\Work\Transfermarket\archive"
    players_file = os.path.join(data_dir, "players.csv")
    player_valuations_file = os.path.join(data_dir, "player_valuations.csv")
    appearances_file = os.path.join(data_dir, "appearances.csv")
    games_file = os.path.join(data_dir, "games.csv")
    raw_output_file = os.path.join(data_dir, "final_cleaned_players_data_raw.csv")  # Before column removal
    cleaned_output_file = os.path.join(data_dir, "final_cleaned_players_data.csv")  # After column removal
    team_market_value_file = os.path.join(data_dir, "team_year_market_value.csv")

    # Load datasets
    players_df = pd.read_csv(players_file)
    player_valuations_df = pd.read_csv(player_valuations_file)
    appearances_df = pd.read_csv(appearances_file)
    games_df = pd.read_csv(games_file)

    # Preprocess data
    players_df, player_valuations_df, appearances_df = preprocess_data(players_df, player_valuations_df, appearances_df)

    # Save preprocessed player data for debugging
    players_df.to_csv(os.path.join(data_dir, "preprocessed_players.csv"), index=False)

    # Merge and clean player data
    merged_data = merge_players_data(players_df, appearances_df, games_df, player_valuations_df)

    # Save raw merged data before column removal
    merged_data.to_csv(raw_output_file, index=False)

    cleaned_data = remove_unnecessary_columns(raw_output_file, cleaned_output_file)

    # Create team-year market value dataset
    create_team_year_market_value(cleaned_data, team_market_value_file)

    # Add team-year market value back into player data
    players_with_team_value = add_team_year_market_value_to_players(cleaned_data, team_market_value_file)
    players_with_team_value.to_csv(os.path.join(data_dir, "players_with_team_value.csv"), index=False)

    print(f"✅ Final cleaned player data saved at: {cleaned_output_file}")
    print(f"✅ New team-year market value dataset saved at: {team_market_value_file}")
    print("🎉 Data Processing Complete!")


if __name__ == "__main__":
    main()