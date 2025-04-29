import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import traceback # Import traceback for detailed error logging

# --- Configuration ---
file_path = 'datasets/regridded_era5.nc'

# --- Choose wind components ---
u_wind_component = 'u10'
v_wind_component = 'v10'

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
        print("\n--- Coordinates ---")
        print(ds.coords)
        print("\n--- Data Variables ---")
        print(list(ds.data_vars.keys()))


        # --- Data Selection ---
        if u_wind_component not in ds.data_vars:
            raise KeyError(f"Variable '{u_wind_component}' not found. Available: {list(ds.data_vars.keys())}")
        if v_wind_component not in ds.data_vars:
            raise KeyError(f"Variable '{v_wind_component}' not found. Available: {list(ds.data_vars.keys())}")

        var_data_u = ds[u_wind_component]
        var_data_v = ds[v_wind_component]

        # --- Determine Time Dimension Name (Common variations) ---
        time_dim = None
        possible_time_dims = ['time', 'valid_time', 'step', 't'] # Add other possibilities if needed
        for dim_name in possible_time_dims:
             if dim_name in var_data_u.dims and dim_name in var_data_v.dims:
                 time_dim = dim_name
                 print(f"Detected time dimension: '{time_dim}'")
                 break

        # --- Time Slicing ---
        if time_dim:
            if time_index_to_plot >= len(ds[time_dim]):
                 raise IndexError(f"time_index_to_plot={time_index_to_plot} out of bounds for dimension '{time_dim}'. Size: {len(ds[time_dim])}.")

            data_slice_u = var_data_u.isel({time_dim: time_index_to_plot})
            data_slice_v = var_data_v.isel({time_dim: time_index_to_plot})

            # Get time for title
            time_val = data_slice_u[time_dim] # Use the detected time dimension name
            try:
                # Handle different time representations (scalar vs 0d array)
                time_val_scalar = time_val.values
                if np.isscalar(time_val_scalar):
                    time_str = np.datetime_as_string(time_val_scalar, unit='s')
                else: # Try item() for 0d array
                    time_str = np.datetime_as_string(time_val_scalar.item(), unit='s')
                base_title = f"Wind Speed & Direction at {time_str}"
            except Exception as e:
                print(f"Warning: Could not format time value ({time_val.values}). Error: {e}")
                time_str = f"{time_dim} index {time_index_to_plot}"
                base_title = f"Wind Speed & Direction at {time_dim} index {time_index_to_plot}"

        # --- Handle variables without a detected time dimension ---
        elif not any(dim in var_data_u.dims for dim in possible_time_dims) and \
             not any(dim in var_data_v.dims for dim in possible_time_dims):
             print(f"Warning: Could not detect a common time dimension ({possible_time_dims}) in wind components. Plotting as static field.")
             data_slice_u = var_data_u
             data_slice_v = var_data_v
             base_title = f"Wind Speed & Direction (Static Field)"
             time_str = "N/A" # Indicate no specific time
        else:
            raise ValueError(f"Mismatch in potential time dimensions ({possible_time_dims}) between U and V components or time dimension missing in one.")

        # --- Calculate Wind Speed ---
        # Assuming U and V are on the same grid and have compatible units
        wind_speed = np.sqrt(data_slice_u**2 + data_slice_v**2)
        # Attempt to copy attributes for the colorbar label
        wind_speed.attrs['long_name'] = 'Wind Speed'
        if 'units' in data_slice_u.attrs: # Assume U and V have same units
             wind_speed.attrs['units'] = data_slice_u.attrs['units']


        # --- Determine Coordinate Names ---
        # Inspect the dataset coordinates to find the correct names
        lon_coord_name = None
        lat_coord_name = None
        # Common names - add others if needed based on your dataset inspection
        possible_lon_names = ['longitude', 'lon', 'grid_lon', 'x']
        possible_lat_names = ['latitude', 'lat', 'grid_lat', 'y']

        available_coords = list(data_slice_u.coords.keys()) # Check coords of the sliced data

        for name in possible_lon_names:
            if name in available_coords:
                lon_coord_name = name
                break # Found one, stop looking

        for name in possible_lat_names:
            if name in available_coords:
                lat_coord_name = name
                break # Found one, stop looking

        if lon_coord_name is None or lat_coord_name is None:
             raise ValueError(f"Could not automatically determine longitude/latitude coordinate names "
                              f"from possibilities {possible_lon_names}/{possible_lat_names}. "
                              f"Available coordinates in data slice: {available_coords}")

        print(f"Using coordinate names: lon='{lon_coord_name}', lat='{lat_coord_name}'")

        # --- Get Coordinates using detected names ---
        lons = data_slice_u[lon_coord_name]
        lats = data_slice_u[lat_coord_name]

        # --- Create the Plot ---
        print(f"\nGenerating plot for wind at {time_str}...")
        fig, ax = plt.subplots(figsize=(12, 10)) # Adjust figure size

        # 1. Plot Wind Speed using pcolormesh
        #    Use the calculated wind_speed DataArray
        #    Use .values to pass numpy arrays to pcolormesh
        mesh = ax.pcolormesh(lons.values, lats.values, wind_speed.values,
                             cmap=speed_colormap,
                             shading='auto') # 'auto' or 'gouraud' often better than default 'flat'

        cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
        cbar_label = f"{wind_speed.attrs.get('long_name', 'Wind Speed')}"
        if 'units' in wind_speed.attrs:
            cbar_label += f" ({wind_speed.attrs['units']})"
        cbar.set_label(cbar_label)

        # Add titles and labels using attributes from detected coordinates
        ax.set_title(base_title)
        ax.set_xlabel(f"{lons.attrs.get('long_name', lon_coord_name)} ({lons.attrs.get('units', 'unknown units')})")
        ax.set_ylabel(f"{lats.attrs.get('long_name', lat_coord_name)} ({lats.attrs.get('units', 'unknown units')})")

        # Optional: Set aspect ratio if desired, but be careful with map projections
        # ax.set_aspect('equal') # Usually not desired for lat/lon plots unless near equator

        plt.tight_layout() # Adjust layout to prevent labels overlapping
        plt.show()

except FileNotFoundError:
    print(f"Error: File not found at '{file_path}'. Please ensure the path is correct.")
except KeyError as e:
    print(f"Error: Data Variable or Coordinate Key not found - {e}")
    print("Please check the 'Dataset Information' printout above for available variable/coordinate names.")
except IndexError as e:
     print(f"Error: Time Index out of bounds - {e}")
except ValueError as e:
    print(f"Error: Value mismatch or configuration error - {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    traceback.print_exc() # Print detailed traceback for unexpected errors