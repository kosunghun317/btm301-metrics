import pandas as pd
import pybaseball
import numpy as np

# Load the CSV file
csv_file = './class_project/data/players.csv'  # Update with your file's path
players_df = pd.read_csv(csv_file)

# Define a function to get the handedness of a player
def get_handedness(last_name, first_name, contract_year):
    try:
        # Lookup the player ID
        data = pybaseball.playerid_lookup(last=last_name, first=first_name)
        if data.empty:
            return np.nan  # Return NaN if no data is found
        
        # Extract MLBAM ID
        player_id = int(data['key_mlbam'].iloc[0])
        
        # Calculate the start and end date for stat query
        end_date = f"{contract_year - 1}-12-31"
        start_date = f"{contract_year - 1}-01-01"  # Arbitrary long past range
        
        # Get the player's batting stats
        stats = pybaseball.statcast_batter(
            start_dt=start_date,
            end_dt=end_date,
            player_id=player_id
        )
        if stats.empty:
            return np.nan  # Return NaN if no stats are found
        
        # Return the last recorded 'stand' value
        return stats['stand'].iloc[-1]
    except Exception as e:
        print(f"Error processing {first_name} {last_name}: {e}")
        return np.nan

# Apply the function to the DataFrame
for i in range(len(players_df)):
    last_name = players_df.loc[i, 'last_name']
    first_name = players_df.loc[i, 'first_name']
    contract_year = players_df.loc[i, 'contract_year']
    players_df.loc[i, 'handedness'] = get_handedness(last_name, first_name, contract_year)