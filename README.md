# BIMig
Bayesian Inference of Animal Migration Phenology and Confounding Effect

# Notice
This package is under developement and should not be used for now.

# Prerequisite packages
```
jax                   : 0.3.23
pytensor              : 2.11.2
matplotlib            : 3.7.1
scipy                 : 1.10.1
tensorflow_probability: 0.20.0
pandas                : 2.0.1
arviz                 : 0.15.1
numpy                 : 1.24.3
seaborn               : 0.11.2
pymc                  : 5.3.1
```

# Usage
### 1. Import
```

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import arviz as az
import pymc as pm
import pickle
import seaborn as sns
from tqdm import tqdm
import time
from warnings import filterwarnings
import contextily as cx
import pickle
import pytensor as pt
filterwarnings('ignore')
pd.set_option('display.max_row', 100)

from utils.model import * ### this import BIMig

```

### 2. Read test data
```
data = pd.read_csv('/data/test.csv')
```

### 3. Modeling
```
bimig = BIMig()
model = bimig.Make_model(data, occ='Golden-crowned Kinglet') 
### it reads the "DOY" and occ columns, 
### you can change the occ to any species column. Including wintering, breeding and passengers.

with model:
    idata = pmjax.sample_numpyro_nuts(1000,tune=1000, chains=1, cores=1)
    
    
```
### 4. Diagnostic
```
az.plot_trace(idata)
plt.tight_layout()
plt.show()

```

### 5. Ploting
```
p_, cdf_ = bimig.predict(idata, data)
bimig.plot_prediction(p_, data)
```
![fig1](/assets/fig1.png)

```
bimig.plot_midpoint(idata)
```
![fig2](/assets/fig2.png)

```
bimig.plot_mean_mid_and_sigma(idata)
```
![fig3](/assets/fig3.png)

### 6. Ouput data

All data are stored in the idata samples. Check it out.





