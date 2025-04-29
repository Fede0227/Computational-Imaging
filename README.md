# Computational-Imaging

## How to run the project

### Setup (first time only)

```
conda create -n imenv python=3.11 -c conda-forge
```

On macos:
```
conda init zsh
conda activate imenv
```

### Required packages
```
conda activate imenv
pip install -r requirements.txt
conda install -c conda-forge xesmf
```

### Running the project

```
conda activate imenv
```

## Steps

1. Download the datasets in the folder `datasets`
2. Run the reprojection script `reproject.py`
3. Run the nn training

## Datasets

The Local datasets are smaller (only november 2023) and are thought to be worked on locally.
The full datasets will run on the training machine.
ERA5 is the dataset of the world. VHR-REA is the dataset at high resolution over italy.

- [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download)
- [VHR-REA](https://dds.cmcc.it/#/dataset/era5-downscaled-over-italy)

### ERA5 (Europe)

- [Full ERA5 (6.06 GB)](https://object-store.os-api.cci2.ecmwf.int/cci2-prod-cache-2/2025-04-27/42e4121736540999777187fc5da557c8.nc)

- [Local ERA5 (489 MB)](https://object-store.os-api.cci2.ecmwf.int/cci2-prod-cache-1/2025-04-27/32e2e20e27eba67591a1af5e33d302ed.nc)


### VHR-REA (Italy only)

- [Full VHR-REA part 1 (2.00 GB)](https://ddshub.cmcc.it/api/v2//download/80379)
- [Full VHR-REA part 2 (1.97 GB)](https://ddshub.cmcc.it/api/v2//download/80380)

- [Local VHR-REA (300 MB)](https://ddshub.cmcc.it/api/v2//download/80383)

