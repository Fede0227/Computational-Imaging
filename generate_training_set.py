import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def plot_dataset_image_from_path(file_path, time_index_to_plot=0):

    speed_colormap = "viridis"

    if file_path == "datasets/regridded_era5.nc":
        u_wind_component = "u10"
        v_wind_component = "v10"
        time_label = "valid_time"
        lon_label, lat_label = "rlon", "rlat"
    elif file_path == "datasets/vhr-rea.nc":
        u_wind_component = "U_10M"
        v_wind_component = "V_10M"
        time_label = "time"
        lon_label, lat_label = "rlon", "rlat"
    else:
        raise ValueError(f"Unknown file path '{file_path}'")

    try:
        with xr.open_dataset(file_path, mask_and_scale=True) as ds:

            print("--- Dataset Information ---")
            print(ds)
            print("---------------------------")

            # --- Data Selection ---
            if u_wind_component not in ds.data_vars:
                raise KeyError(f"Variable '{u_wind_component}' not found. Available: {list(ds.data_vars.keys())}")
            if v_wind_component not in ds.data_vars:
                raise KeyError(f"Variable '{v_wind_component}' not found. Available: {list(ds.data_vars.keys())}")

            var_data_u = ds[u_wind_component]
            var_data_v = ds[v_wind_component]

            # --- Time Slicing ---
            if time_label in ds.dims: # Check dimension existence in Dataset
                if time_index_to_plot >= len(ds[time_label]): # ds[time_label] gives the coordinate array
                    raise IndexError(f"time_index_to_plot={time_index_to_plot} out of bounds. Time size: {len(ds[time_label])}.")

                # Use a dictionary for isel to handle different time dimension names
                data_slice_u = var_data_u.isel({time_label: time_index_to_plot})
                data_slice_v = var_data_v.isel({time_label: time_index_to_plot})
                
                # Get time for title from the sliced data variable (which now has a scalar time coord)
                # or directly from the dataset's time coordinate array
                time_val_coord = ds[time_label].isel({time_label: time_index_to_plot})

                try:
                    # .item() is crucial for 0-dim array to get scalar
                    time_str = np.datetime_as_string(time_val_coord.values.item(), unit='s')
                    base_title = f"Wind Speed & Direction at {time_str}"
                except Exception:
                    time_str = f"index {time_index_to_plot}" # Fallback
                    base_title = f"Wind Speed & Direction at time index {time_index_to_plot}"
            else:
                # If there's no time dimension, assume data is already 2D
                data_slice_u = var_data_u
                data_slice_v = var_data_v
                base_title = "Wind Speed & Direction (Static Data)"
                time_str = "N/A"
                if data_slice_u.ndim > 2 or data_slice_v.ndim > 2:
                     raise ValueError(f"Data for '{u_wind_component}' or '{v_wind_component}' is not 2D and no time dimension '{time_label}' found for slicing.")


            # --- Calculate Wind Speed ---
            wind_speed = np.sqrt(data_slice_u**2 + data_slice_v**2)
            wind_speed.attrs['long_name'] = 'Wind Speed'
            if 'units' in data_slice_u.attrs:
                wind_speed.attrs['units'] = data_slice_u.attrs['units']

            if not (lon_label in data_slice_u.coords and lat_label in data_slice_u.coords):
                 # Check dataset level coords if not found in variable's coords
                if not (lon_label in ds.coords and lat_label in ds.coords):
                    raise ValueError(f"Could not find '{lon_label}'/'{lat_label}' coordinates.")
                lons = ds[lon_label]
                lats = ds[lat_label]
            else:
                lons = data_slice_u[lon_label]
                lats = data_slice_u[lat_label]


            # --- Create the Plot ---
            print(f"\nGenerating plot from path for wind at {time_str}...")
            fig, ax = plt.subplots(figsize=(10, 8))

            mesh = ax.pcolormesh(lons.values, lats.values, wind_speed.values, cmap=speed_colormap, shading='auto')

            cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', pad=0.02, shrink=0.8)
            cbar_label = f"{wind_speed.attrs.get('long_name', 'Wind Speed')}"
            if 'units' in wind_speed.attrs:
                cbar_label += f" ({wind_speed.attrs['units']})"
            cbar.set_label(cbar_label)

            ax.set_title(base_title)
            ax.set_xlabel(f"{lons.attrs.get('long_name', 'Longitude')} ({lons.attrs.get('units', 'degrees_east')})")
            ax.set_ylabel(f"{lats.attrs.get('long_name', 'Latitude')} ({lats.attrs.get('units', 'degrees_north')})")

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


def plot_slice_wind_data(dataset_slice):
    """
    Plots wind speed and direction from an xarray.Dataset slice.
    The slice is assumed to be 2D for wind components after time selection.
    It attempts to infer variable and coordinate names.
    """
    u_comp_name, v_comp_name = None, None
    time_coord_name, lon_coord_name, lat_coord_name = None, None, None
    source_name = "Data Slice"

    # Infer variable names
    if 'u10' in dataset_slice.data_vars and 'v10' in dataset_slice.data_vars:
        u_comp_name, v_comp_name = 'u10', 'v10'
        source_name = "ERA5-like Slice"
        time_coord_name = 'valid_time' # Expected time coord for ERA5 slices
    elif 'U_10M' in dataset_slice.data_vars and 'V_10M' in dataset_slice.data_vars:
        u_comp_name, v_comp_name = 'U_10M', 'V_10M'
        source_name = "VHR-like Slice"
        time_coord_name = 'time' # Expected time coord for VHR slices
    else:
        raise ValueError(f"Cannot determine wind components in the dataset slice. Vars: {list(dataset_slice.data_vars.keys())}")

    # Infer spatial coordinates (assuming they are coordinates of the dataset_slice)
    if 'rlon' in dataset_slice.coords and 'rlat' in dataset_slice.coords:
        lon_coord_name, lat_coord_name = 'rlon', 'rlat'
    elif 'lon' in dataset_slice.coords and 'lat' in dataset_slice.coords: # common alternatives
        lon_coord_name, lat_coord_name = 'lon', 'lat'
    elif 'longitude' in dataset_slice.coords and 'latitude' in dataset_slice.coords:
        lon_coord_name, lat_coord_name = 'longitude', 'latitude'
    else:
        raise ValueError(f"Cannot determine lon/lat coordinates in the dataset slice. Coords: {list(dataset_slice.coords.keys())}")

    u_data = dataset_slice[u_comp_name]
    v_data = dataset_slice[v_comp_name]

    # Get time for title (dataset_slice should have a scalar time coordinate)
    title_time_str = "Unknown Time"
    if time_coord_name in dataset_slice.coords:
        time_val_da = dataset_slice[time_coord_name]
        if time_val_da.size == 1: # Ensure it's a scalar coordinate
            try:
                # .item() gets the scalar from a 0-dim array
                title_time_str = np.datetime_as_string(time_val_da.item(), unit='s')
            except Exception as e:
                print(f"Warning: Could not format time for title: {e}. Using raw value: {time_val_da.values}")
                title_time_str = str(time_val_da.values)
        else:
            print(f"Warning: Time coordinate '{time_coord_name}' is not scalar. Title might be incorrect.")
            title_time_str = f"{time_coord_name} (array)"
    else:
        print(f"Warning: Time coordinate '{time_coord_name}' not found for title.")

    base_title = f"Wind Speed ({source_name}) at {title_time_str}"

    wind_speed = np.sqrt(u_data**2 + v_data**2)
    wind_speed.attrs['long_name'] = 'Wind Speed'
    if 'units' in u_data.attrs:
        wind_speed.attrs['units'] = u_data.attrs['units']

    lons = dataset_slice[lon_coord_name]
    lats = dataset_slice[lat_coord_name]

    print(f"\nPlotting {source_name} - Wind Speed at {title_time_str}")
    print(f"Data shape: {wind_speed.shape}") # This will show (5,5) for X, (21,21) for Y

    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = ax.pcolormesh(lons.values, lats.values, wind_speed.values, cmap="viridis", shading='auto')
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
    cbar_label = wind_speed.attrs.get('long_name', 'Wind Speed')
    if 'units' in wind_speed.attrs:
        cbar_label += f" ({wind_speed.attrs['units']})"
    cbar.set_label(cbar_label)

    ax.set_title(base_title)
    ax.set_xlabel(f"{lons.attrs.get('long_name', 'Longitude')} ({lons.attrs.get('units', 'degrees_east')})")
    ax.set_ylabel(f"{lats.attrs.get('long_name', 'Latitude')} ({lats.attrs.get('units', 'degrees_north')})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    era5_path = "datasets/regridded_era5.nc"
    vhr_path = "datasets/vhr-rea.nc"

    era5_data = xr.open_dataset(era5_path, mask_and_scale=True)
    print("--- ERA5 Data Loaded ---")
    print(era5_data)

    vhr_data = xr.open_dataset(vhr_path, mask_and_scale=True)
    print("\n--- VHR Data Loaded ---")
    print(vhr_data)

    print("\n----------------------------")
    print("Creating training_set...")
    training_set = []
    # Loop for first two time steps (indices 0 and 1)
    for i in range(min(len(era5_data.valid_time), len(vhr_data.time), 2)):
        print(f"Processing time index {i}")
        X_slice = era5_data.isel(valid_time=i)
        y_slice = vhr_data.isel(time=i)
        print(f"  X_slice (from ERA5) valid_time: {X_slice.valid_time.values}")
        print(f"  y_slice (from VHR)  time: {y_slice.time.values}")
        training_set.append((X_slice, y_slice))
    print("----------------------------")

    # Define X and y for tuple indexing
    X_INDEX = 0
    Y_INDEX = 1
    
    dataset_pair_index = 0 # Which pair from training_set to plot (e.g., 0 for the first time step)

    if training_set:
        print(f"\nPlotting X component (ERA5-like) for training pair index {dataset_pair_index}:")
        plot_slice_wind_data(training_set[dataset_pair_index][X_INDEX])

        print(f"\nPlotting Y component (VHR-like) for training pair index {dataset_pair_index}:")
        plot_slice_wind_data(training_set[dataset_pair_index][Y_INDEX])
    else:
        print("Training set is empty, skipping plotting.")

    # # Your original calls to plot_dataset_image_from_path can also be used for comparison
    # print("\nPlotting VHR data directly from path (first time step):")
    # plot_dataset_image_from_path(vhr_path, time_index_to_plot=0)
    # print("\nPlotting ERA5 data directly from path (first time step):")
    # plot_dataset_image_from_path(era5_path, time_index_to_plot=0)