import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration ---
file_path = 'datasets/vhr-rea.nc'

# --- Choose wind components ---
u_wind_component = 'U_10M'
v_wind_component = 'V_10M'

time_index_to_plot = 0     # Index of the time step

# --- Plotting Settings ---
# Colormap for wind speed (sequential recommended)
speed_colormap = 'viridis' #'plasma', 'magma', 'inferno' are also good

# --- Plotting ---
try:
    # Open the dataset
    with xr.open_dataset(file_path, mask_and_scale=True) as ds:

        print("--- Dataset Information ---")
        print(ds)

        # --- Data Selection ---
        if u_wind_component not in ds.data_vars:
            raise KeyError(f"Variable '{u_wind_component}' not found. Available: {list(ds.data_vars.keys())}")
        if v_wind_component not in ds.data_vars:
            raise KeyError(f"Variable '{v_wind_component}' not found. Available: {list(ds.data_vars.keys())}")

        var_data_u = ds[u_wind_component]
        var_data_v = ds[v_wind_component]

        # --- Time Slicing ---
        if 'time' in var_data_u.dims and 'time' in var_data_v.dims:
            if time_index_to_plot >= len(ds['time']):
                 raise IndexError(f"time_index_to_plot={time_index_to_plot} out of bounds. Time size: {len(ds['time'])}.")

            data_slice_u = var_data_u.isel(time=time_index_to_plot)
            data_slice_v = var_data_v.isel(time=time_index_to_plot)

            # Get time for title
            time_val = data_slice_u['time']
            try:
                time_str = np.datetime_as_string(time_val.values.item(), unit='s') # Use .item() for 0d array
                base_title = f"Wind Speed & Direction at {time_str}"
            except Exception:
                time_str = f"index {time_index_to_plot}"
                base_title = f"Wind Speed & Direction at time index {time_index_to_plot}"

        # --- Handle variables without time ---
        # This section might need adjustment depending on the desired behavior
        # if wind components lack a time dimension. For now, we assume they have time.
        elif 'time' not in var_data_u.dims and 'time' not in var_data_v.dims:
             print("Warning: Wind components do not seem to have a 'time' dimension. Plotting as is.")
             data_slice_u = var_data_u
             data_slice_v = var_data_v
             base_title = f"Wind Speed & Direction (Static Field)"
             time_str = "N/A" # Indicate no specific time
        else:
            raise ValueError("Mismatch in time dimension between U and V components.")


        # --- Calculate Wind Speed ---
        # Assuming U and V are on the same grid and have compatible units
        wind_speed = np.sqrt(data_slice_u**2 + data_slice_v**2)
        # Attempt to copy attributes for the colorbar label
        wind_speed.attrs['long_name'] = 'Wind Speed'
        if 'units' in data_slice_u.attrs: # Assume U and V have same units
             wind_speed.attrs['units'] = data_slice_u.attrs['units']


        # --- Get Coordinates ---
        # Assume U and V components share the same coordinate grid
        if not ('lon' in data_slice_u.coords and 'lat' in data_slice_u.coords):
             raise ValueError("Could not find 'lon'/'lat' coordinates for U component.")
        lons = data_slice_u['rlon'] # Use coordinates from one variable
        lats = data_slice_u['rlat']

        # --- Create the Plot ---
        print(f"\nGenerating plot for wind at {time_str}...")
        fig, ax = plt.subplots(figsize=(10, 8)) # Adjust figure size

        # 1. Plot Wind Speed using pcolormesh
        #    Use the calculated wind_speed DataArray
        mesh = ax.pcolormesh(lons.values, lats.values, wind_speed.values,
                             cmap=speed_colormap,
                             shading='auto') # or 'gouraud'

        cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar_label = f"{wind_speed.attrs.get('long_name', 'Wind Speed')}"
        if 'units' in wind_speed.attrs:
            cbar_label += f" ({wind_speed.attrs['units']})"
        cbar.set_label(cbar_label)


        # Add titles and labels
        ax.set_title(base_title)
        ax.set_xlabel(f"{lons.attrs.get('long_name', 'Longitude')} ({lons.attrs.get('units', 'degrees_east')})")
        ax.set_ylabel(f"{lats.attrs.get('long_name', 'Latitude')} ({lats.attrs.get('units', 'degrees_north')})")

        # Optional: Set aspect ratio if desired, but be careful with map projections
        # ax.set_aspect('equal')

        plt.tight_layout()
        plt.show()

except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please ensure the path is correct.")
except KeyError as e:
    print(f"Error: Data Variable or Coordinate Key not found - {e}")
except IndexError as e:
     print(f"Error: Time Index out of bounds - {e}")
except ValueError as e:
    print(f"Error: Value mismatch or configuration error - {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    import traceback
    traceback.print_exc()
