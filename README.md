# Solar Farm Model

This repository contains a Monte Carlo based radiative transfer model that calculates radiative fluxes within a photovoltaic solar farm.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Modules](#modules)
- [Notebooks](#notebooks)
- [License](#license)

## Introduction

The Solar Farm Model employs Monte Carlo methods to simulate radiative transfer processes within a photovoltaic solar farm. Traditionally, we use the effective albedo method to characterize solar farms' effect in global climate models, where we simply modify the surface albedo of the model grid points. The value assigned is the sum of the surface albedo of the solar panels and the fraction of absorbed solar energy converted to electricity. However, it ignores the fact that real-world solar panels do not fully cover the ground. They are placed strategically to maximize the output without waste of panel area. Furthermore, the longwave effect of solar panels is ignored in previous literature.

In this model, we create an idealized solar farm with rows of solar panels that are spaced apart, elevated from the ground, and tilted southward at a fixed angle. The model treats the panels and the ground as separate objects and captures their interactions. The Monte-Carlo-based radiative transfer solver can accurately calculate the radiative fluxes within the solar farm. Lookup tables can be generated for use in climate models to avoid heavy computational costs from the Monte Carlo method. This approach provides a more accurate representation of the radiative effects of photovoltaic solar farms, enhancing our understanding of their impact on radiative fluxes.

**Features**

- Solar panels separated from the ground: interactions between panels, ground, and atmosphere are captured
- Configurable solar farm parameters
  - Panel spacing
  - Panel length
  - Panel height
  - Panel tilt angle
  - Surface albedo of front and back panel surface
  - Surface albedo of ground
  - Surface emissivity of front and back panel surface
  - Surface emissivity of ground
- Shortwave radiative transfer calculation: direct and diffuse solar radiation from above
- Longwave radiative transfer calculation: four photon packet sources separately calculated
  - Downward longwave radiation from the atmosphere
  - Longwave emission from front panel surface
  - Longwave emission from back panel surface
  - Longwave emission from ground
- Lookup table generators: taking varying solar angles and ground albedo into account
- Can be further extended
  - Include spectral surface albedo and spectral surface emissivity
  - Full 3D version of solar farms
  - Variable solar panel angles
  - More types of solar panels (e.g., CSP)

Corresponding Author: Chongxing Fan ([cxfan@umich.edu](mailto:cxfan@umich.edu))

## Prerequisites

- numpy
- xarray
- matplotlib (necessary for plotting figures)
- ipywidgets (necessary for running notebooks)
- [mpi4py](https://mpi4py.readthedocs.io/en/stable/) (necessary for using MPI to generate lookup tables in `SolarFarmSpecTableMPI.py`)

## Modules

### `SolarFarmMonteCarlo.py`

Monte Carlo model that does the heavylifting work to calculate radiative fluxes within solar farms.

**High-Level Functions**

- `doDownwardRadiation2D`: Simulate downward radiation from atmosphere and calculate absorbed radiative fluxes by solar panels and ground and outgoing radiative flux to the atmosphere. This function is used to handle direct solar radiation, diffuse solar radiation, and incoming longwave radiation from the atmosphere.
- `doPanelEmission2D`: Simulate emitted longwave radiation from solar panels and calculate absorbed radiative fluxes by solar panels as well as outgoing radiative flux to the atmosphere. This function is used to handle front panel longwave emission and back panel longwave emission.
- `doGroundEmission2D`: Simulate emitted longwave radiation from ground and calculate absorbed radiative fluxes by solar panels as well as outgoing radiative flux to the atmosphere. This function is used to handle ground longwave emission.
- `plotSingleRayTracing2D`: Plot the ray tracing figure that shows the trajectory of a single photon packet.
- `plotDirectSolar2D`: Plot direct solar radiation in a solar farm and demonstrate solar panels' shading effect.

For input parameters and usage, please see the source code for documentation.

### `SolarFarmRadiation.py`

A sample radiative transfer model implementation with the Monte Carlo based solver.

### `SolarFarmSpecTablePool.py`

Generate lookup tables for radiative fluxes within solar farms depending on solar angles and ground albedo. This module uses Python's `multiprocessing` library and can run without additional dependence.

Before running the script

### `SolarFarmSpecTableMPI.py`

Generate lookup tables for radiative fluxes within solar farms depending on solar angles and ground albedo. This module uses the [mpi4py](https://mpi4py.readthedocs.io/en/stable/) library, and it requires MPI parallelization to be available on the machine.

### `SolarAngle.py`

Calculate solar zenith angle and solar azimuth angle given input latitude, longitude, solar declination angle (varying with season), and Julian day (the integer part representing the day of the year, and the decimal part representing the hour of the day).

## Notebooks

### `SolarFarmSWAnalysis.ipynb`

The notebook contains analyses of the shortwave radiative processes within a typical solar farm. It plots most figures in the manuscript except Figures 5 and 6.

To run the blocks that produce Figure 7, a lookup table should be generated, and the path to this table should be specified in the `SOLAR_FARM_SPEC` variable.

The notebook has implemented a mechanism to store necessary data for figure plotting so that you don't have to redo the simulations again and again. The cache path is specified in the `DATA_PATH` variable.

Given the nature of Monte Carlo's randomness, the figures may differ slightly across each run. The more samples you simulate (by default it is 10^6), the less variability you should observe.

### `SolarFarmLWAnalysis.ipynb`

The notebook contains analyses of the longwave radiative processes within a typical solar farm. It plots Figures 5 and 6 in the corresponding manuscript.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
