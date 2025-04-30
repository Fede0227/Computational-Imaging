import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import traceback # Keep for debugging if needed

file_path = 'datasets/vhr-rea.nc'
u_wind_component = "U_10M"
v_wind_component = "V_10M"
speed_colormap = "viridis"

try:
    # Open the dataset using a 'with' statement (good practice)
    with xr.open_dataset(file_path, mask_and_scale=True) as ds:
        print("--- Dataset Information ---")
        print(ds)

        time_index_to_plot = 0
        # --- Data Selection ---
        u_data = ds[u_wind_component].isel(time=time_index_to_plot)
        v_data = ds[v_wind_component].isel(time=time_index_to_plot)

        print("--- u data ---")
        print(u_data)
        print("--- v data ---")
        print(v_data)


        # # --- Calculate Wind Speed ---
        # # xarray handles element-wise operations and keeps metadata
        # wind_speed = np.sqrt(u_data**2 + v_data**2)

        # # Add descriptive name and units (if available) for the plot
        # wind_speed.attrs['long_name'] = 'Wind Speed at 10m' # Be more specific
        # if 'units' in u_data.attrs:
        #      wind_speed.attrs['units'] = u_data.attrs['units']
        # else:
        #      wind_speed.attrs['units'] = 'm/s' # Make a reasonable assumption if missing

        # # --- Get Time for Title ---
        # time_val = wind_speed['time'] 
        # # Format it nicely using pandas datetime properties accessed via .dt
        # time_str = time_val.dt.strftime('%Y-%m-%d %H:%M:%S').item() # .item() extracts scalar value

        # # --- Plotting (Simplified using xarray's .plot) ---
        # print(f"\nGenerating plot for wind speed at {time_str}...")

        # # Create the figure and axes explicitly for better control (optional but good practice)
        # fig, ax = plt.subplots(figsize=(10, 8))

        # # Use xarray's built-in plotting. It usually infers coordinates correctly.
        # # Tell it explicitly to use 'lon' and 'lat' which are 2D coordinates in your file
        # # It's good practice to specify coordinates if they are not the dimensions
        # wind_speed.plot.pcolormesh(
        #     ax=ax,               # Specify the axes to plot on
        #     x='lon',             # Use the 'lon' coordinate for the x-axis
        #     y='lat',             # Use the 'lat' coordinate for the y-axis
        #     cmap=speed_colormap, # Set the colormap
        #     cbar_kwargs={'label': f"{wind_speed.attrs['long_name']} ({wind_speed.attrs['units']})"} # Label for colorbar
        # )

        # ax.set_title(f"Wind Speed at {time_str}")

        # plt.tight_layout() # Adjust layout to prevent labels overlapping
        # plt.show()         # Display the plot

except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please ensure the path is correct.")
except KeyError as e:
    print(f"Error: Missing variable or coordinate in the dataset: {e}")
    # Ensure ds is defined before trying to access its keys if the error happens early
    try:
        with xr.open_dataset(file_path) as temp_ds:
            print(f"Available variables: {list(temp_ds.data_vars.keys())}")
            print(f"Available coordinates: {list(temp_ds.coords.keys())}")
    except Exception as inner_e:
        print(f"Could not read dataset to list available variables/coords: {inner_e}")
except IndexError as e:
    # This error would happen if time_index_to_plot is invalid
    print(f"Error: time_index_to_plot ({time_index_to_plot}) is out of bounds for the 'time' dimension.")
    try:
        with xr.open_dataset(file_path) as temp_ds:
            time_dim_size = temp_ds.dims.get('time', 'Not Found')
            print(f"Size of 'time' dimension: {time_dim_size}")
    except Exception as inner_e:
        print(f"Could not read dataset to check time dimension size: {inner_e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    traceback.print_exc()


