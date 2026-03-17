# Identifying Demand Drivers of Waste Generation in China Based on Waste Input-Output Analysis

This repository implements the calculation workflow for identifying demand drivers of waste generation in China based on waste input-output analysis, covering the quantitative calculation of waste generation, recycling, and net waste footprints for 2020. The core logic follows the sequence of "data preprocessing → WIO model construction → footprint calculation → scenario analysis".

## 📁 Project Structure

```
WIO_China_2020/
│
├── data/
│   ├── WIO_table(2020).xlsx
│   └── y_scenario.xlsx
│
├── generation_footprint_cal.py
├── recycle_footprint_cal.py
├── net_footprint_cal.py
├── scenarios_cal.py
│
└── result/
    ├── result_generation.xlsx
    ├── result_recycle.xlsx
    ├── result_net_all.xlsx
    ├── result_net_p.xlsx
    │
    └── scenarios/
        ├── RCC/
        ├── UCC/
        └── GCC/
```
