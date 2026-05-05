<!--
SPDX-FileCopyrightText: Contributors to the s4casting project

SPDX-License-Identifier: MPL-2.0
-->
# DATA SOURCES
Datasets used in this repository are publicly available open data and are licensed under Creative Commons Attribution, version 4.0 - see [LICENSE](LICENSES/CC-BY-4.0.txt) for details. Datasets can be downloaded at [Liander Open data] (https://www.liander.nl/over-ons/open-data#lianderpower).

This project uses multiple data sources:

- **Historical Power Measurements**: 5-minute averaged active power readings (in MW) collected across Alliander's electrical grid, measured in UTC timezone.  
- **Historical Weather Data**: Sourced from providers such as Open-Meteo, offering temperature, wind, and shortwave_radiation records, measured in UTC timezone.
- **Powerpile dataset**: Publicly available datasets containing power consumption data from various locations.                                    

## Data Structure                                                                                                                                 

The data is organized to link measurements, metadata, and locations. We use numpy.memmaps because they allow efficient access to large arrays stored on disk without loading the entire array into memory. The core components are:
                                                                                                                                                  
### 1. `.np` File (NumPy Memory-Mapped Array)                                                                                                     
- Contains a **1D NumPy array** with all historical measurement values. Each value corresponds to a 5-minute averaged active power reading.       

### 2. `.parquet` File (Span Metadata)                                                                                                            
- Stores **spans** of measurement data, each representing a time window for a specific location.                                                  
- Columns:
  - `id`: Start index in the NumPy array                                                                                                          
  - `location`: Location identifier                                                                                                               
  - `datetime_start`: Start time (Unix timestamp)
  - `num_values`: Number of values in the span
                                                                                                                                                  
### 3. `.json` File (Dataset Descriptor)                                                                                                          
- Connects the `.np` and `.parquet` files
- Includes metadata for locations, such as:                                                                                                       
  - Name 
  - Longitude and latitude

## Croissant Format

The dataset is also available in the [Croissant](https://github.com/mlcommons/croissant) format (ML Commons v1.1), a JSON-LD standard for making ML datasets discoverable and interoperable across platforms like HuggingFace, Kaggle, and OpenML. The Croissant release is self-contained — it does not require the memmap or the internal `.json` descriptor.

The release consists of two Parquet files described by `croissant.json`:

- **`measurements.parquet`** — one row per (location, span), with columns for `location_id`, `location_name`, `lat`, `lon`, `start_time`, `sample_interval_s`, `num_values`, and a `values` list column containing the power readings.
- **`weather.parquet`** — one row per (location, feature, span), with columns for `weather_location_id`, `lat`, `lon`, `feature_name`, `feature_unit`, `start_time`, `sample_interval_s`, `num_values`, and a `values` list column containing the weather readings.

To load the Croissant release into the project's internal `NumpyData` format, use the adapter in `croissant_adapter.py`:

```python
from croissant_adapter import load_measurements_from_croissant, load_weather_from_croissant

measurements = load_measurements_from_croissant(Path("data/LianderPower"))
weather = load_weather_from_croissant(Path("data/LianderPower"))
```

The adapter rebuilds the flat memmap-backed layout that the rest of the pipeline expects (`NumpyData`, `IntervalDataset`, `TimeseriesDataset`), so downstream code runs unchanged.

## Loading External Data (Optional)
                                                                                                                                                  
You can integrate data from external providers as long as it follows the same structure used by the core dataset:                                 

- Prepare a **Parquet file** containing at least `time` and `measurements` columns.                                                               
  This can be created with pandas. The measurement values should be provided in **watts (W)**; the system automatically converts them to **megawatts (MW)** during loading.
- Add the new location to the dataset's `.json` descriptor, including a `filename` field pointing to the Parquet file you created.                
  
You can follow the layout used in the benchmarking dataset for reference.
