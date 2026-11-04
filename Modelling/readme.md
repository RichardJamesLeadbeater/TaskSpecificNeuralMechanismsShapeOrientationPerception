
### Modelling
- This directory contains the model code and data analysis for the modelling data.
- For an overview of the model architecture, please see:
  - `Modelling Flow.ipynb`
- To see how best-fitting parameters are obtained, please see:
  - Formal modelling script in `ParamFits.py`
  - Differential evolution is a global search method. It explores widely across the parameter space and is less likely to become trapped in a local minimum. Across generations, candidate parameter values are gradually refined.
- To see how the model graphs are created in practice, please see:
  - `fitGraphs.py`

**/mytools/** 
<br>Contains custom Python code supporting various scripts

**/Modelling/components/**
<br>Contains custom Python code supporting the model