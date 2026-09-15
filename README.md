# Testing for Stable Intervals in Non-Stationary Time Series

This repository contains the code for the methods and experiments presented in the paper titled:

Title: A Functional Central Limit Theorem for Locally Stationary Time Series in Banach Spaces

Author(s): Florian Heinrichs, Luis A. Rodríguez

### Overview

This repository includes code for the different tests, simulation (data generation and experiments) and case studies.

### Requirements

To use the proposed methods, only NumPy and SciPy are required. Additional Python packages are required for the real data applications.

### Usage

The high-level functions implementing the proposed tests are defined in `experiments.py`, whereas the test is implemented in `self_normalization.py`.

### Datasets

The "Consumer-Grade EEG and Eye-Tracking Dataset" is available from [Zenodo](https://zenodo.org/records/14860668). 
The temperature data is provided by the [Deutscher Wetterdienst (DWD) Climate Data Center](https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/air_temperature/). 
The temperature curves used for the experiments are in the subfolder `data` of this repository. **Note that the DWD data is licensed under Creative Commons Attribution 4.0 International (CC BY 4.0).**

### Citation

If you use this code in your own work, please cite the following pre-print (or the peer reviewed paper, once available):

 - tba
    

### License

This project (expect of the data) is licensed under the MIT License - see the LICENSE file for details.
