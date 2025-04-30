import xarray as xr
import pandas as pd
import traceback
import time as timer

# --- Configuration ---
file_path = 'datasets/vhr-rea.nc'
u_wind_component = "U_10M"
v_wind_component = "V_10M"
output_csv_path = 'wind_data.csv'

# --- Main Script ---
start_time = timer.time()
print(f"Starting script...")

try:
    print(f"Opening dataset: {file_path}")
    with xr.open_dataset(file_path, mask_and_scale=True) as ds:
        print("--- Full Dataset Information ---")
        print(ds)

        # Select only the necessary variables and coordinates for efficiency
        # Note: 'lat' and 'lon' are coordinates associated with rlat/rlon dimensions
        # and will be included automatically by to_dataframe()
        # We explicitly select the data variables we need.
        print("Selecting necessary data variables...")
        ds_subset = ds[[u_wind_component, v_wind_component]]

        # Convert the xarray Dataset to a pandas DataFrame
        # It will create a DataFrame with a MultiIndex (time, rlat, rlon)
        # and columns for U_10M, V_10M, lat, lon.
        print("Converting xarray Dataset to pandas DataFrame...")
        df = ds_subset.to_dataframe()
        print("Conversion complete.")

        # Reset the index to turn 'time', 'rlat', 'rlon' into columns
        print("Resetting DataFrame index...")
        df = df.reset_index()

        # Select and rename the desired columns
        output_df = df[['time', 'lon', 'lat', u_wind_component, v_wind_component]].copy() # Use .copy() to avoid SettingWithCopyWarning
        output_df.rename(columns={
            'lon': 'long', # Rename 'lon' to 'long' as requested
            u_wind_component: 'u',
            v_wind_component: 'v'
        }, inplace=True)

        # Handle potential NaN values if necessary
        # For instance, drop rows where any of the essential columns are NaN
        initial_rows = len(output_df)
        output_df.dropna(subset=['time', 'long', 'lat', 'u', 'v'], inplace=True)
        if initial_rows > len(output_df):
            print(f"Dropped {initial_rows - len(output_df)} rows with NaN values.")

        # SAVE DATAFRAME TO FILE
        # # Write the DataFrame to a CSV file
        # print(f"Writing data to CSV: {output_csv_path}...")
        # # index=False prevents pandas from writing the DataFrame index as a column
        # output_df.to_csv(output_csv_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        # print(f"Successfully wrote CSV file with {len(output_df)} rows.")
        print(output_df)


except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please ensure the path is correct.")
except KeyError as e:
    print(f"Error: Missing variable, coordinate, or column in the dataset/dataframe: {e}")
    try:
        with xr.open_dataset(file_path) as temp_ds:
            print(f"Available data variables: {list(temp_ds.data_vars.keys())}")
            print(f"Available coordinates: {list(temp_ds.coords.keys())}")
    except Exception as inner_e:
        print(f"Could not read dataset to list available keys: {inner_e}")
    if 'df' in locals():
         print(f"Available DataFrame columns: {df.columns.tolist()}")

except MemoryError:
     print(f"Error: Insufficient memory to convert the dataset to a DataFrame.")
     print("Consider processing the data in chunks (e.g., time step by time step) if the dataset is too large.")
     traceback.print_exc()

except Exception as e:
    print(f"An unexpected error occurred: {e}")
    traceback.print_exc()

finally:
    end_time = timer.time()
    print(f"Script finished in {end_time - start_time:.2f} seconds.")