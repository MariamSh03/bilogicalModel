# Tumor-Dystrophin ODE Model

## Overview
This project implements a mathematical model to study the interaction between tumor growth and dystrophin expression using a system of Ordinary Differential Equations (ODEs). The model is solved numerically using Python's `scipy.integrate.odeint` and visualized using `Streamlit` for an interactive user interface.

The research behind this model is inspired by the study published in **SpringerOpen**: [Dynamical analysis of tumor–dystrophin interaction model with impact of age of onset and staging](https://fixedpointtheoryandalgorithms.springeropen.com/articles/10.1186/s13663-025-00786-5).

## Features
- Solves a system of three coupled ODEs representing tumor size, dystrophin level, and impact factor.
- Uses the **LSODA** algorithm for adaptive numerical integration.
- Visualizes results as **time series plots** and **phase portraits**.
- Provides an interactive web interface using **Streamlit**.

## Installation
### Prerequisites
Ensure you have Python installed (version 3.8 or later). You can check your Python version with:
```sh
python --version
```

### Install Dependencies
Clone the repository and navigate to the project directory:
```sh
git clone <your-repo-url>
cd <your-repo-folder>
```
Then, install the required dependencies:
```sh
pip install -r requirements.txt
```

## How to Run
Run the Streamlit app using:
```sh
streamlit run proj1.py
```
This will open a web-based interactive interface where you can modify parameters and visualize results.

## Usage
1. **Input Parameters**: Adjust the initial conditions and parameters affecting tumor growth and dystrophin levels.
2. **Run Simulation**: Click the "Run" button to compute and visualize the ODE solutions.
3. **Interpret Graphs**: Analyze time series and phase portraits to understand the system dynamics.

## Reference
For more details on the mathematical background, refer to the original research paper: [SpringerOpen Article](https://fixedpointtheoryandalgorithms.springeropen.com/articles/10.1186/s13663-025-00786-5).

## License
This project is open-source and available for academic and research purposes. Feel free to modify and extend it!

---
Developed with ❤️ using Python & Streamlit.

