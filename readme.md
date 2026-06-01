## README

### 1. General Information
- Title of Dataset: Task-specific neural mechanisms shape human orientation perception
- Author(s): XXX
- Date of Data Collection: Psychophysics data in 2021 and 2024
- Geographic Location: XXX
- Funding Source: XXX

### 2. File Overview
This project is split into two main sections.

**Psychophysics/**
- Contains the data analysis code for the behavioural data.
- Please see the Jupyter notebooks contained within this folder:
  - `Psychophysics Analysis.ipynb`.
  - `Psychophysics Graphs.ipynb`.

**Modelling eLife/** 
- This directory contains the model code.
- For an overview of the model architecture, please see:
  - `Modelling Flow.ipynb`
- To see how best-fitting parameters are obtained, please see:
  - `ParamFits.py`
  - Guidance in `parameter_fitting_walkthrough.ipynb`
- To see how the model graphs are created in practice, please see:
  - `fitGraphs.py`

**/mytools/** 
<br>Contains custom Python code for data analysis 

**/Modelling/components/**
<br>Contains custom Python code for modelling

### 3. Data Details
- Behavioural data includes the anonymised participant number, experimental condition/s, and threshold measurement for each trial.
- Modelling data includes the model parameters of each simulation.
- Jupyter notebooks are used in each section.
- Filetypes: .py .csv .ipynb

### 4. Methods
- Psychophysical data was collected under laboratory conditions.
  - 2AFC Adaptive staircase procedure
  - PsychoPy
- Modelling data was simulated as described in the Jupyter notebook in the Modelling directory

### 5. Reuse and Access
- Licensing
    - CCO
- Citation request
    - Please cite the associated paper if you reuse this dataset.
- Limitations or known issues
    - Two participants did not complete the 8 c/deg condition in the behavioural experiment.
    - There are no other issues to the best of the authors' knowledge.

### 6. Software Requirements
- Python version 3.10.0
- For required packages, please see requirements.txt
