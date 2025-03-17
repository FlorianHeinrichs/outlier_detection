# Outlier Detection Research Code

This repository contains the code for the methods and experiments presented in the paper titled:

Title: Outlier Detection in Non-Stationary Time Series

Authors: Florian Heinrichs, Patrick Bastian, Holger Dette

### Overview

This repository includes code for the outlier detection method(s) proposed in the research paper. The goal of this work is to explore novel methods for identifying outliers in locally stationary time series. The alternative methods discussed include statistical and machine learning-based approaches, as well as their comparative evaluation using publicly available datasets.

### Requirements

To use the proposed methods, only NumPy and SciPy are required. Additional Python packages and R libraries are required for the alternative methods.

### Usage

The "partial" and "full" version of our proposed method are referred to as "sequential" and "parallel" versions in the code. The high-level functions to use these methods are defined in `experiments.py` (`partial_test()` and `sequential_test()`).   

### Datasets

The datasets used for evaluation in the paper are available for download here: [Australian Government - Bureau of Meteorology](http://www.bom.gov.au/climate/data/)

### Citation

If you use this code in your own work, please cite the following pre-print (or the peer reviewed paper, once available):

Heinrichs, F., Bastian, P., & Dette, H. (2025). Sequential Outlier Detection in Non-Stationary Time Series. *arXiv preprint arXiv:2502.18038*.

    @article{heinrichs2025sequential,
      title={Sequential Outlier Detection in Non-Stationary Time Series},
      author={Heinrichs, Florian and Bastian, Patrick and Dette, Holger},
      journal={arXiv preprint arXiv:2502.18038},
      year={2025}
    }

### License

This project is licensed under the MIT License - see the LICENSE file for details.
