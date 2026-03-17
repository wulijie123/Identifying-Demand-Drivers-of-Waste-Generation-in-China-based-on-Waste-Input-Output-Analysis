# Identifying Demand Drivers of Waste Generation in China Based on Waste Input-Output Analysis

This repository implements the calculation workflow for identifying demand drivers of waste generation in China based on waste input-output analysis, covering the quantitative calculation of waste generation, recycling, and net waste footprints for 2020. The core logic follows the sequence of "data preprocessing → WIO model construction → footprint calculation → scenario analysis".

## 📁 Project Structure

```
WIO_China_2020/
│
├── data/
│   ├── WIO_table(2020).xlsx             # 2020 China WIO table
│   └── y_scenario.xlsx                  # Final demand matrix for different scenarios
│
├── generation_footprint_cal.py          # Solid waste generation footprint calculation
├── recycle_footprint_cal.py             # Solid waste recycling footprint calculation
├── net_footprint_cal.py                 # Net solid waste footprint calculation
├── scenarios_cal.py                     # Net solid waste footprint calculation for different scenarios
│
└── result/
    ├── result_generation.xlsx           # Solid waste generation footprint results
    ├── result_recycle.xlsx              # Solid waste recycling footprint results
    ├── result_net_all.xlsx              # Net solid waste footprint results
    ├── result_net_p.xlsx                # Net solid waste footprint results by final demand
    │
    └── scenarios/                       # Results for different scenarios
        ├── RCC/
        ├── UCC/
        └── GCC/
```
