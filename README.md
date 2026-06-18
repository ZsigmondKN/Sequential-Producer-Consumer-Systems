# Sequential Producer-Consumer System

![Sequentially composed dynamics example](docs//images/closed-loop/get_pi_control_sequential_three_processes_balanced.png)
## Overview
A discrete event simulation tool used to simulate sequential composed producer–consumer systems. The internal dynamics of the simulation are recorded and expressed as diagrams of throughput and queue occupancy. The simulator is used to understand the effect of the composition of such systems. The experiments explore different sequence lengths, feedback types and external failures.

## Tech Stack
- Python 3  
- Optuna (parameter tuning)

## Key Features
- Event-driven simulation of multi-stage producer–consumer systems
- Support for different feedback strategies (input/output)  
- Visualisation of throughput and queue dynamics over time
- System stability diagrams and parameter tuning using Optuna 

## How to Run

1. Install dependencies using: `pip install -r requirements.txt`
2. Run the simulation using: `python sim_runtime.py`

## Experiments

### Atomic Subsystem

![Atomic second-order system balanced](docs/images/closed-loop/get_atomic_second_order_system_balanced.png)

### Failure Scenario Comparison

![Atomic second-order system failure](docs/images/disturbances/get_atomic_second_order_system_balanced_failure.png)

![PI control sequential three processes failure](docs/images/disturbances/get_pi_control_sequential_three_processes_balanced_failure.png)