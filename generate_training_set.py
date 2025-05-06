import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

def plot_dataset_image(file_path, time_index_to_plot=0):

    speed_colormap = "viridis"

    if file_path == "datasets/regridded_era5.nc":
        u_wind_component = "u10"
        v_wind_component = "v10"
        time_label = "valid_time"
    else:
        u_wind_component = "U_10M"
        v_wind_component = "V_10M"
        time_label = "time"

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
            if time_label in var_data_u.dims and time_label in var_data_v.dims:
                if time_index_to_plot >= len(ds[time_label]):
                    raise IndexError(f"time_index_to_plot={time_index_to_plot} out of bounds. Time size: {len(ds[time_label])}.")

                if file_path == "datasets/regridded_era5.nc":
                    data_slice_u = var_data_u.isel(valid_time=time_index_to_plot)
                    data_slice_v = var_data_v.isel(valid_time=time_index_to_plot)
                else:
                    data_slice_u = var_data_u.isel(time=time_index_to_plot)
                    data_slice_v = var_data_v.isel(time=time_index_to_plot)

                # Get time for title
                time_val = data_slice_u[time_label]
                try:
                    time_str = np.datetime_as_string(time_val.values.item(), unit='s')
                    base_title = f"Wind Speed & Direction at {time_str}"
                except Exception:
                    time_str = f"index {time_index_to_plot}"
                    base_title = f"Wind Speed & Direction at time index {time_index_to_plot}"

            else:
                raise ValueError("Mismatch in time dimension between U and V components.")


            # --- Calculate Wind Speed ---
            wind_speed = np.sqrt(data_slice_u**2 + data_slice_v**2)
            wind_speed.attrs['long_name'] = 'Wind Speed'
            if 'units' in data_slice_u.attrs:
                wind_speed.attrs['units'] = data_slice_u.attrs['units']

            if not ('rlon' in data_slice_u.coords and 'rlat' in data_slice_u.coords):
                raise ValueError("Could not find 'rlon'/'rlat' coordinates for U component.")
            lons = data_slice_u['rlon']
            lats = data_slice_u['rlat']

            # --- Create the Plot ---
            print(f"\nGenerating plot for wind at {time_str}...")
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

if __name__ == "__main__":
    plot_dataset_image("datasets/vhr-rea.nc")
    plot_dataset_image("datasets/regridded_era5.nc")