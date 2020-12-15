# Projeto

- Aumentar a base de dados
- Utilizar o transformada wavelet no caso do EMG, ou  densidade espectral nas 5 faixas específicas do EEG como features


```python
import pywt
import scipy.io
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
```


```python
file = scipy.io.loadmat('./EMG_Dataset/Database1/male_1.mat')
```

# Resumo



# Introdução



# Metodologia



# Resultados


```python

```

# Conclusão


```python

```

# Metodologia

1. Será utilizado o método Wavelet para calcular os coeficientes dos dados de entrada para cada canal
2. Para cada coeficiente (6) retornados será calculado o valor médio
3. Os dois vetores de 6 médias (1 para cada canal) serão concatenados e usados como entrada para o MLP


```python
levels = 3
mother_function = 'coif5'
```


```python
movements = {
    'cyl': {
        'c1': np.array(file['cyl_ch1'][0:30]),
        'c2': np.array(file['cyl_ch2'][0:30])
    },
    'tip': {
        'c1': np.array(file['tip_ch1'][0:30]),
        'c2': np.array(file['tip_ch2'][0:30])
    },
    'hook': {
        'c1': np.array(file['hook_ch1'][0:30]),
        'c2': np.array(file['hook_ch2'][0:30])
    },
    'palm': {
        'c1': np.array(file['palm_ch1'][0:30]),
        'c2': np.array(file['palm_ch2'][0:30])
    },
    'spher': {
        'c1': np.array(file['spher_ch1'][0:30]),
        'c2': np.array(file['spher_ch2'][0:30])
    },
    'lat': {
        'c1': np.array(file['lat_ch1'][0:30]),
        'c2': np.array(file['lat_ch2'][0:30])
    }
}
```


```python
coefs = {
    'cyl': [],
    'tip': [],
    'hook': [],
    'palm': [],
    'spher': [],
    'lat': []
}
```


```python
out_counter = 0
for movement, channels in movements.items():
    experiments_c1 = channels['c1']
    experiments_c2 = channels['c2']
    
    for experiment in range(30):
        c1 = experiments_c1[experiment]
        c2 = experiments_c2[experiment]

        coefficients = pywt.wavedec(c1, mother_function, level=levels) + pywt.wavedec(c2, mother_function, level=levels)
        max_coefs = []
        for coefficient in coefficients:
            max_coef = max(coefficient)

            max_coefs.append(max_coef)

        coefs[movement].append(max_coefs)
        
    coefs[movement] = pd.DataFrame(coefs[movement])
    coefs[movement]['out'] = out_counter
    out_counter += 1
```


```python
dataset = pd.DataFrame()
for key, data in coefs.items():
    dataset = dataset.append(data)
```


```python
y = dataset['out']
x = dataset.drop(['out'], axis=1)
```

# Testes


```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
mlp = MLPClassifier(
    hidden_layer_sizes=(9,9,9), 
    max_iter=10000, 
    learning_rate_init=0.005, 
    #tol=1e-7, 
    #warm_start=True,
    n_iter_no_change=100
    #verbose=True
)
mlp.fit(x_train, y_train)
mlp.score(x_test, y_test)
```




    0.9




```python
mlp.
```
