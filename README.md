# BIMig
Bayesian Inference of Animal Migration Phenology and Confounding Effect

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
0. Import
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

2. Read test data
```
data = pd.read_csv('test.csv')
```

3. Modeling
```
bimig = BIMig()
model = bimig.Make_model(data) ### it reads the "DOY" and "occ" columns
# idata = bimig.sample_model(model)

with model:
    idata = pmjax.sample_numpyro_nuts(1000,tune=1000, chains=1, cores=1)
    # 
    
    
```
4. Diagnostic
```
az.plot_trace(idata)
plt.tight_layout()
plt.show()


p_, cdf_ = bimig.predict(idata, data)
bimig.plot_prediction(p_, data)

```

5. 








