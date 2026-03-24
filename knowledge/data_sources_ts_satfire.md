# TS-SatFire Data Sources

## Dataset Identity

- `name`: TS-SatFire supporting data sources
- `primary_paper`: TS-SatFire: A Multi-Task Satellite Image Time-Series Dataset for Wildfire Detection and Prediction
- `task_scope`: active fire detection (`af`), burned area mapping (`ba`), wildfire progression prediction (`pred`)

## TS-SatFire Overview

- The dataset covers wildfire events from January 2017 to October 2021.
- The paper reports 3552 surface reflectance images and auxiliary data totaling 71 GB.
- Training fires are from 2017-2020 in the contiguous U.S.
- Validation uses 13 wildfire events from 2017-2020.
- BA and prediction test data use 24 fire events from 2021.
- AF test data uses 17 wildfire events from 2018-2022 across multiple continents.

## Folder Structure Assumed By The Legacy Code

- One folder per fire event
- `VIIRS_Day/`: daytime VIIRS GeoTIFFs
- `VIIRS_Night/`: nighttime VIIRS GeoTIFFs
- `FirePred/`: auxiliary layers for progression prediction

This matches the code in [`legacy/satimg_dataset_processor/satimg_dataset_processor.py`](/Users/lee/Github%20Clone/TS_Agentic_AI/legacy/satimg_dataset_processor/satimg_dataset_processor.py).

GitHub repository check:

- The public TS-SatFire repository contains code that reads from `FirePred/` and packs those bands into prediction tensors.
- The public repository does not appear to contain the upstream export script that creates the `FirePred` GeoTIFF files themselves.
- The README describes dataset preparation from downloaded GeoTIFFs, but it does not document the separate auxiliary-data export pipeline.

## Channel Inventory Used By TS-SatFire

From the TS-SatFire paper, the 27-channel prediction stack is:

| Channel | Variable |
| --- | --- |
| 1 | Band I1 |
| 2 | Band I2 |
| 3 | Band I3 |
| 4 | Band I4 |
| 5 | Band I5 |
| 6 | Band M11 |
| 7 | Band I4 Night |
| 8 | Band I5 Night |
| 9 | NDVI or `NDVI_last` |
| 10 | EVI or `EVI2_last` |
| 11 | Total Precipitation |
| 12 | Wind Speed |
| 13 | Wind Direction |
| 14 | Min Temperature |
| 15 | Max Temperature |
| 16 | Energy Release Component |
| 17 | Specific Humidity |
| 18 | Slope |
| 19 | Aspect |
| 20 | Elevation |
| 21 | PDSI |
| 22 | Land Cover |
| 23 | Total Precipitation Surface |
| 24 | Forecast Wind Speed |
| 25 | Forecast Wind Direction |
| 26 | Forecast Temperature |
| 27 | Forecast Specific Humidity |

## Source Datasets

### VIIRS

- Used for the spectral wildfire imagery.
- The TS-SatFire paper states that six daytime bands are used: `I1-I5` and `M11`.
- The paper also uses nighttime captures for `I4` and `I5`.
- NASA Earthdata describes VIIRS as an operational instrument on Suomi NPP and NOAA platforms that collects visible and infrared imagery and global observations of land, atmosphere, cryosphere, and ocean.
- The TS-SatFire paper states that imagery bands provide 375 m spatial resolution at nadir and moderate bands provide 750 m.

Operational note:

- AF/BA inputs use the 8-channel stack from VIIRS day and night imagery.
- Prediction inputs use those VIIRS channels plus auxiliary `FirePred` layers.

### VNP13A1

- Source for `NDVI_last` and the vegetation-index layer exposed as `EVI2_last` in the sample `FirePred` file.
- NASA Earthdata describes VNP13A1 as a VIIRS vegetation indices product that selects the best available pixel over a 16-day acquisition period.
- Spatial resolution: 500 m
- Temporal resolution: 16 days
- Spatial extent: global

Operational note:

- Used only for progression prediction auxiliary features, not the basic AF/BA 8-channel stack.

### gridMET

- Source for historical weather features in the progression prediction setup.
- The TS-SatFire paper lists weather variables including precipitation, wind, temperature, humidity, PDSI, and Energy Release Component.
- The paper states gridMET has 4638 m spatial resolution.
- The Google Earth Engine catalog describes gridMET as daily surface meteorological data at about 4 km resolution for the contiguous United States from 1979 to near present.
- The Earth Engine catalog lists pixel size as 4638.3 meters and cadence as 1 day.
- The Earth Engine catalog lists bands including precipitation (`pr`), specific humidity (`sph`), wind direction (`th`), minimum temperature (`tmmn`), maximum temperature (`tmmx`), wind speed (`vs`), and Energy Release Component (`erc`).

Operational note:

- Use as observed weather context, distinct from forecast weather from GFS.

### GFS

- Source for forecast weather features in progression prediction.
- NOAA NCEI states GFS is a global model with base horizontal resolution of about 28 km between grid points.
- The referenced forecast products are issued 4 times per day at `00, 06, 12, 18 UTC`.
- The NOAA page shows 3-hourly forecast outputs.
- The TS-SatFire paper states the dataset averages 24 hours of forecast data based on forecasts made at the end of the current day, without using future information.

Operational note:

- Distinguish forecast variables from gridMET observation variables.

### SRTM

- Source for topographic features.
- NASA Earthdata states SRTM collected near-global land elevation data over nearly 80% of Earth’s land surfaces.
- The TS-SatFire paper uses NASA SRTM Digital Elevation data and derives `slope`, `aspect`, and `elevation`.
- The paper specifically says SRTM version 3 is used and cites 90 m spatial resolution.

Operational note:

- The legacy code expects topography already aligned into the `FirePred` GeoTIFF stack.

### MCD12Q1

- Source for land cover.
- The TS-SatFire paper says land cover is based on the MODIS Land Cover Type Yearly Global product `MCD12Q1.061`.
- The Earth Engine page for `MODIS/061/MCD12Q1` states yearly cadence and 500 m pixel size.
- The Earth Engine page states Version 6.1 provides global land cover types from Terra and Aqua MODIS reflectance data.
- The same page shows multiple classification layers, including `LC_Type1`, `LC_Type2`, `LC_Type3`, `LC_Type4`, and `LC_Type5`, plus land cover property, QC, and land/water mask layers.
- The HPC `FirePred` sample confirms the exported land-cover band is `LC_Type1`.

Operational note:

- In the prediction loader, land cover is treated as a categorical variable and one-hot encoded.

### LP DAAC

- LP DAAC is the archive and distribution center for land remote sensing data products relevant here, including MODIS and VIIRS.
- NASA Earthdata describes LP DAAC as a USGS-NASA partnership focused on land surface processes.
- LP DAAC processes, archives, and distributes products from MODIS and VIIRS missions and provides access via Earthdata Search and supporting tools.

Operational note:

- Treat LP DAAC as the canonical archive reference when documenting MODIS and VIIRS-derived products.

### GeoTIFF / GDAL Band Ordering

- The legacy code reads GeoTIFFs with `rasterio.read()`, which returns data in band-first order.
- GDAL documents two common GeoTIFF interleave modes: `PIXEL` and `BAND`.
- In `PIXEL` interleave, pixel values for all bands are stored together for each spatial block.
- In `BAND` interleave, data for the first band is written first, then the second band, and so on.
- This means TS-SatFire should rely on explicit band indices in the exported GeoTIFF stack, not on informal assumptions from visualization order.

Operational note:

- For `FirePred`, semantic band meaning should be taken from the file descriptions or export documentation, not inferred from storage layout alone.

## Confirmed FirePred Band Mapping From The HPC Dataset

The following mapping was confirmed from:

- `/home/jlc3q/data/SatFire/ts-satfire/24462335/FirePred/2020-09-26_FirePred.tif`
- inspected with `gdalinfo`

| FirePred Band | Description |
| --- | --- |
| 1 | `NDVI_last` |
| 2 | `EVI2_last` |
| 3 | `total precipitation` |
| 4 | `wind speed` |
| 5 | `wind direction` |
| 6 | `minimum temperature` |
| 7 | `maximum temperature` |
| 8 | `energy release component` |
| 9 | `specific humidity` |
| 10 | `slope` |
| 11 | `aspect` |
| 12 | `elevation` |
| 13 | `pdsi` |
| 14 | `LC_Type1` |
| 15 | `total_precipitation_surface_last` |
| 16 | `forecast wind speed` |
| 17 | `forecast wind direction` |
| 18 | `forecast temperature` |
| 19 | `forecast specific humidity` |

Operational interpretation:

- Prediction input = `VIIRS_Day` 6 bands + `VIIRS_Night` 2 bands + `FirePred` 19 bands = 27 channels
- The land-cover layer used in practice is `LC_Type1`
- The vegetation layer exported in band 2 is labeled `EVI2_last`, so downstream documentation should preserve that exact name unless later export code proves otherwise

## Label Definitions

### Active Fire

- Training labels are sourced from the NASA VIIRS active fire product.
- The paper says training labels undergo manual quality control.
- AF test labels are manually thresholded from VIIRS I4/I5 imagery.

### Burned Area

- Built from accumulated VIIRS AF detections and NIFC daily burned area perimeters.
- In some cases the final label uses only accumulated VIIRS AF hotspots to avoid poor NIFC perimeters.

### Progression Prediction

- The label is the newly burned area between the last observed day and the next day.
- This avoids rewarding trivial persistence of the last known burned area mask.

## Preprocessing Rules Already Reflected In The Legacy Code

- Missing values are replaced with zeros.
- AF uses VIIRS imagery directly after normalization.
- BA and prediction aggregate heat-sensitive bands `I4` and `I5` across the current and previous timestamps using a pixelwise maximum.
- Temporal windows are sampled from the full wildfire lifecycle.
- Temporal models use pixel time series.
- Spatial models use per-image samples.
- Spatial-temporal models use the whole image time series window.
- In prediction, angular features are transformed and land cover is one-hot encoded in [`legacy/satimg_dataset_processor/data_generator_pred_torch.py`](/Users/lee/Github%20Clone/TS_Agentic_AI/legacy/satimg_dataset_processor/data_generator_pred_torch.py).

## Gaps To Resolve Later

- The public GitHub repository appears to consume `FirePred` files but not generate them, so the missing export pipeline remains an external dependency.
- The exact export script is still missing, but the actual band order has been verified directly from an HPC sample file.
- The gridMET journal paper URL was not directly readable in the current browser session, but the Earth Engine catalog provides the citation and practical dataset metadata.

## Sources

- TS-SatFire paper: [arXiv PDF](https://arxiv.org/pdf/2412.11555)
- TS-SatFire GitHub: [repository](https://github.com/zhaoyutim/TS-SatFire)
- VIIRS: [NASA Earthdata](https://www.earthdata.nasa.gov/data/instruments/viirs)
- VNP13A1: [NASA Earthdata](https://www.earthdata.nasa.gov/data/catalog/lpcloud-vnp13a1-002)
- gridMET: [USGS catalog link provided](https://water.usgs.gov/catalog/datasets/ef98187e-8703-4ec6-afc1-4dbc72c9d6d8/)
- gridMET Earth Engine: [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/IDAHO_EPSCOR_GRIDMET)
- GFS: [NOAA NCEI](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast)
- SRTM: [NASA Earthdata](https://www.earthdata.nasa.gov/data/instruments/srtm)
- MCD12Q1 V6.1: [Google Earth Engine](https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD12Q1)
- LP DAAC: [NASA Earthdata](https://www.earthdata.nasa.gov/centers/lp-daac)
- GeoTIFF / GTiff: [GDAL documentation](https://gdal.org/en/stable/drivers/raster/gtiff.html)
