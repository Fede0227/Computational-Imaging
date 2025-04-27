import xarray as xr
import matplotlib.pyplot as plt
import numpy as np # For potential datetime decoding

# --- Configuration ---
file_path = 'datasets/era5-downscaled-over-italy_hourly_80383.nc'

# --- Choose what to plot ---
u_wind_component = 'U_10M'  # Variable name (e.g., 'U_10M' or 'V_10M')
v_wind_component = 'V_10M'

time_index_to_plot = 0     # Index of the time step (0 for the first, 1 for the second, etc.)

# --- Choose a colormap ---
# See: https://matplotlib.org/stable/tutorials/colors/colormaps.html
# 'viridis', 'plasma', 'inferno', 'magma', 'cividis' - good sequential maps
# 'coolwarm', 'bwr', 'seismic' - good diverging maps (useful if data spans +/- values like wind)
# 'jet' - common but often discouraged due to perception issues
colormap = 'viridis'

# --- Plotting ---
try:
    # Open the dataset using xarray
    # mask_and_scale=True automatically handles _FillValue, scale_factor, add_offset
    with xr.open_dataset(file_path, mask_and_scale=True) as ds:

        print("--- Dataset Information ---")
        print(ds) # Print summary to confirm structure

        # --- Data Selection ---
        # Check if the chosen variable exists
        if u_wind_component not in ds.data_vars:
            raise KeyError(f"Variable '{u_wind_component}' not found in the dataset. "
                           f"Available variables: {list(ds.data_vars.keys())}")
        
        if v_wind_component not in ds.data_vars:
            raise KeyError(f"Variable '{v_wind_component}' not found in the dataset. "
                           f"Available variables: {list(ds.data_vars.keys())}")

        # Select the variable
        var_data_u = ds[u_wind_component]
        var_data_v = ds[v_wind_component]

        # Check if the variable has a time dimension
        if 'time' in var_data_u.dims:
            # Check if the time index is valid
            if time_index_to_plot >= len(ds['time']):
                 raise IndexError(f"time_index_to_plot={time_index_to_plot} is out of bounds. "
                                  f"Time dimension size is {len(ds['time'])}.")
            # Select the specific time slice using index selection (.isel)
            data_slice_u = var_data_u.isel(time=time_index_to_plot)
            data_slice_v = var_data_v.isel(time=time_index_to_plot)

            # Get the actual time value for the title (attempt decoding)
            time_val = data_slice_u['time']
            try:
                # np.datetime_as_string needs a 0-d array or scalar
                time_str = np.datetime_as_string(time_val.values.item(), unit='s')
                plot_title = f"{var_data_u.attrs.get('long_name', u_wind_component)} at time {time_str}"
            except Exception: # Fallback if time decoding fails
                time_str = f"index {time_index_to_plot}"
                plot_title = f"{var_data_u.attrs.get('long_name', u_wind_component)} at time index {time_index_to_plot}"

        else:
            # Handle variables that might not have a time dimension (e.g., static fields)
            print(f"Warning: Variable '{u_wind_component}' does not seem to have a 'time' dimension. Plotting the variable as is.")
            data_slice_u = var_data_u
            plot_title = f"{var_data_u.attrs.get('long_name', u_wind_component)} (Static Field)"

        # --- Get Coordinates ---
        # xarray usually correctly identifies coordinates like 'lon' and 'lat'
        # These are 2D coordinate arrays in your case.
        lons_u = data_slice_u['lon']
        lats_u = data_slice_u['lat']

        lons_v = data_slice_v['lon']
        lats_v = data_slice_v['lat']

        lons = np.sqrt(lons_u**2 + lons_v**2)
        lats = np.sqrt(lats_u**2 + lats_v**2)

        # --- Create the Plot ---
        print(f"\nGenerating plot for '{u_wind_component}' at time {time_str}...")
        fig, ax = plt.subplots(figsize=(10, 8)) # Adjust figure size as needed

        # Use pcolormesh for plotting data on a 2D grid defined by lons/lats
        # It handles the 2D coordinate arrays correctly.
        # data_slice.values gives the raw numpy array data.
        mesh = ax.pcolormesh(lons, lats, data_slice_u.values,
                             cmap=colormap,
                             shading='auto') # 'auto' or 'gouraud' often look better than 'flat' for non-rectilinear grids

        # Add a color bar to show the scale
        cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02)
        cbar_label = f"{var_data_u.attrs.get('long_name', u_wind_component)}"
        if 'units' in var_data_u.attrs:
            cbar_label += f" ({var_data_u.attrs['units']})"
        cbar.set_label(cbar_label)

        # Add titles and labels (using attributes from the NetCDF file)
        ax.set_title(plot_title)
        ax.set_xlabel(f"{lons.attrs.get('long_name', 'Longitude')} ({lons.attrs.get('units', 'degrees_east')})")
        ax.set_ylabel(f"{lats.attrs.get('long_name', 'Latitude')} ({lats.attrs.get('units', 'degrees_north')})")

        # Optional: Set aspect ratio if needed, but be aware it might distort
        # geographic representation if the projection isn't handled.
        # ax.set_aspect('equal')

        plt.tight_layout() # Adjust spacing
        plt.show() # Display the plot



except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please ensure the path is correct.")
except KeyError as e:
    print(f"Error: {e}")
except IndexError as e:
     print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc() # Print detailed error information
