import xarray as xr
import traceback
import time as timer

file_path = 'datasets/vhr-rea.nc'
u_wind_component = "U_10M"
v_wind_component = "V_10M"
output_csv_path = 'wind_data.csv'

start_time = timer.time()
print(f"Starting script...")

try:
    print(f"Opening dataset: {file_path}")
    with xr.open_dataset(file_path, mask_and_scale=True) as ds:
        print("--- Full Dataset Information ---")
        print(ds)

        print("Selecting necessary data variables...")
        ds_subset = ds[[u_wind_component, v_wind_component]]

        print("Converting xarray Dataset to pandas DataFrame...")
        df = ds_subset.to_dataframe()
        print("Conversion complete.")
        
        # reset the index to turn 'time', 'rlat', 'rlon' into columns
        print("Resetting DataFrame index...")
        df = df.reset_index()
        
        output_df = df[['time', 'lon', 'lat', u_wind_component, v_wind_component]].copy()

        # handle potential NaN values
        initial_rows = len(output_df)
        output_df.dropna(subset=['time', 'lon', 'lat', 'U_10M', 'V_10M'], inplace=True)
        if initial_rows > len(output_df):
            print(f"Dropped {initial_rows - len(output_df)} rows with NaN values.")

        # # write the DataFrame to a CSV file
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