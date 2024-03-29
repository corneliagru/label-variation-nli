# More Labels or Cases? Assessing Label Variation in Natural Language Inference.

Cornelia Gruber, Katharina Hechinger, Matthias Assenmacher, Göran Kauermann, and Barbara Plank. 2024. More Labels or Cases? Assessing Label Variation in Natural Language Inference. In Proceedings of the Third Workshop on Understanding Implicit and Underspecified Language, pages 22–32, Malta. Association for Computational Linguistics.

Full paper here: https://aclanthology.org/2024.unimplicit-1.2/

In this work, we analyze the uncertainty that is inherently present in the labels used for supervised machine learning in natural language inference (NLI). In cases where multiple annotations per instance are available, neither the majority vote nor the frequency of individual class votes is a trustworthy representation of the labeling uncertainty. 
We propose modeling the votes via a Bayesian mixture model to recover the data-generating process, i.e., the posterior distribution of the ``true'' latent classes, and thus gain insight into the class variations. This will enable a better understanding of the confusion happening during the annotation process. We also assess the stability of the proposed estimation procedure by systematically varying the numbers of i) instances and ii) labels. Thereby, we observe that few instances with many labels can predict the latent class borders reasonably well, while the estimation fails for many instances with only a few labels. This leads us to conclude that multiple labels are a crucial building block for properly analyzing label uncertainty.

## Repository

The file structure of this project is as follows:

```
├── README.md
├── data
│   ├── bootstrap
│   ├── final
│   └── raw
├── figs
│   ├── appendix
│   ├── full_bootstrap.png
│   └── scatter_latent.png
├── notebooks
│   ├── 0_descriptives.ipynb
│   ├── 1_bayesian_mixture_model.ipynb
│   └── 2_stability_estimation.ipynb
└── src
    ├── __init__.py
    ├── __pycache__
    ├── bootstrap_funcs.py
    ├── config.py
    ├── load_data.py
    ├── model_funcs.py
    ├── plotting_funcs.py
    └── utils
```

The folder `data` contains 
- `raw`: the raw data as given in https://github.com/easonnie/ChaosNLI, 
- `final`: the cleaned data used for the final analysis, and 
- `bootstrap`: the data generated by the bootstrapping procedure, which can be used to exactly recreate the results. 

The folder `figs` contains the figures used in the paper.

The folder `notebooks` contains the notebooks used to generate the results.
- `0_descriptives.ipynb`: code used for the initial descriptive analysis.
- `1_bayesian_mixture_model.ipynb`: code used for the estimation of the Bayesian mixture model.
- `2_stability_estimation.ipynb`: code used for the bootstrap estimation of the stability of the Bayesian mixture model.

The folder `src` contains the code files that were used throughout the project.
- `bootstrap_funcs.py`: functions used for the bootstrapping procedure.
- `config.py`: configuration file for the project.
- `load_data.py`: load the data from `data/raw`, clean it and save the results to `data/final`.
- `model_funcs.py`: functions used for the Bayesian mixture model.
- `plotting_funcs.py`: functions used for plotting the results.


