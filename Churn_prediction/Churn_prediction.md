
![png](icemd.png)

# MBDM Modulo 5: Churn Prediction

#### 138070

## Introduccion

![png](linea.png)

El problema del churn es un problema que afecta a todas las compañías, pero en especial a las Telcos, con una tasa superior al 30% debido a la alta competencia entre ellas y la gran facilidad de cambiar de una a otra
compañía.
Según un estudio de Daemon Quest, retener a un cliente cuesta entre cinco y quince veces menos que captar a uno nuevo. Es por ello por lo que las telcos, al igual que otras compañías, están poniendo mucho interés en la retención
de los clientes.

 El objetivo de este NoteBook es predecir el comportamiento de los clientes para su posterior retencion en la compañia. Para ello se analizarán todos los datos relevantes de los clientes descargados en el siguiete enlace:

 https://www.kaggle.com/blastchar/telco-customer-churn

## Descripcion del DataSet 

![png](linea.png)

Esta descripción del data set es dada por el enunciado, posteriormente analizaremos paso a paso los datos del enunciado.

Cada fila representa un cliente, cada columna contiene atributos de cliente descritos en la columna Metadata.
Los datos brutos contienen **7043 filas** (clientes) y **21 columnas**(variables/atributos).
La Columna **“Churn”** es nuestro target.
* **CustomerID** (Alfanumérica): Customer ID
* **Gender** (Alfanumérica) si el clientes es hombre se codifica con “male” y si es mujer con “female”
* **SeniorCitizen** (numérica): si el cliente es senior o no (1,0)
* **Partner** (Alfanumérica): Si el cliente tiene un partner o no (Yes,No)
* **Dependents** (Alfanumérica): si el cliente tiene dependientes o no (yes,No)
* **Ternure** (numérica): número de meses que el cliente ha estado en la compañía
* **PhoneService** (Alfanumérica): Si el cliente tiene un servicio móvil o no (Yes,No)
* **MultipleLines** (Alfanumérica): Si el cliente tiene multiples líneas (Yes, No, No pone service)
* **InternetService** (Alfanumérica): Proveedor de servicio de internet del cliente (DSL,Fiber optic, No)
* **OnlineSecurity** (Alfanumérica). Si el cliente tiene seguridad online o no (Yes,No, No internet service)
* **OnlineBackup** (Alfanumérica). Si el cliente tiene backup online o no (Yes,No, No internet service)
* **DeviceProtection** (Alfanumérica). Si el cliente tiene protección para el sispositivo o no (Yes, No, No internet service)
* **TechSupport** (Alfanumérica): Si el cliente tien soporte o no (Yes, No, No internet service)
* **StreamingTV** (Alfanumérica). Si el cliente tiene TV en streaming o no (Yes, No, No internet sercice)
* **StreamingMovies** (Alfanumérica). Si el cliente tiene películas en streaming o no (Yes, No, No internet service)
* **Contract** (Alfanumérica). Los términos del contrato del cliente. (Month-toMonth, One year, Two year)
* **PaperlessBilling** (Alfanumérica). Si el cliente tiene factura digital o no (Yes, No)
* **PaymentMethod** (Alfanumérica). El método de pago del cliente.(Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)
* **MonthlyCharges** (numérica). La cantidad cargada por el cliente mensualmente
* **TotalCharges** (numérica). La cantidad total cargada al cliente
* **Churn** (Alfanumérica). Si el cliente se ha ido de la compañía o no (Yes o No) 

## Exploracion del DataSet

![png](linea.png)

Cargamos el dataset y librerias para su posterior manipulación:


```python
import pandas as pd
import numpy as np

churn =pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
churn.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7043 entries, 0 to 7042
    Data columns (total 21 columns):
    customerID          7043 non-null object
    gender              7043 non-null object
    SeniorCitizen       7043 non-null int64
    Partner             7043 non-null object
    Dependents          7043 non-null object
    tenure              7043 non-null int64
    PhoneService        7043 non-null object
    MultipleLines       7043 non-null object
    InternetService     7043 non-null object
    OnlineSecurity      7043 non-null object
    OnlineBackup        7043 non-null object
    DeviceProtection    7043 non-null object
    TechSupport         7043 non-null object
    StreamingTV         7043 non-null object
    StreamingMovies     7043 non-null object
    Contract            7043 non-null object
    PaperlessBilling    7043 non-null object
    PaymentMethod       7043 non-null object
    MonthlyCharges      7043 non-null float64
    TotalCharges        7043 non-null object
    Churn               7043 non-null object
    dtypes: float64(1), int64(2), object(18)
    memory usage: 1.1+ MB


Como podemos observar coinciden tanto el numero de columnas como el numero de filas que tenemos en el dataset. Pero el tipo columna no esta definido como tal.
Posteriormente procedemos a pasar las columnas correspondientes de tipo `object` a tipo `float`. Pero antes de ello vamos a explorar un poco el contenido de cada columna y si estas contienen valores nulos.


```python
# Contenido de cada columna
churn.apply(set,axis = 0)
```




    customerID          {9504-DSHWM, 8158-WPEZG, 8819-IMISP, 5399-ZIMK...
    gender                                                 {Male, Female}
    SeniorCitizen                                                  {0, 1}
    Partner                                                     {No, Yes}
    Dependents                                                  {No, Yes}
    tenure              {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,...
    PhoneService                                                {No, Yes}
    MultipleLines                             {No, Yes, No phone service}
    InternetService                                {DSL, No, Fiber optic}
    OnlineSecurity                         {No internet service, No, Yes}
    OnlineBackup                           {No internet service, No, Yes}
    DeviceProtection                       {No internet service, No, Yes}
    TechSupport                            {No internet service, No, Yes}
    StreamingTV                            {No internet service, No, Yes}
    StreamingMovies                        {No internet service, No, Yes}
    Contract                         {One year, Two year, Month-to-month}
    PaperlessBilling                                            {No, Yes}
    PaymentMethod       {Mailed check, Credit card (automatic), Bank t...
    MonthlyCharges      {18.95, 19.8, 20.65, 20.15, 20.2, 20.75, 24.95...
    TotalCharges        {3480.35, 1412.4, 890.35, 1277.75, 7396.15, 10...
    Churn                                                       {No, Yes}
    dtype: object




```python
# Conteo de valores nulos
churn.isnull().sum()
```




    customerID          0
    gender              0
    SeniorCitizen       0
    Partner             0
    Dependents          0
    tenure              0
    PhoneService        0
    MultipleLines       0
    InternetService     0
    OnlineSecurity      0
    OnlineBackup        0
    DeviceProtection    0
    TechSupport         0
    StreamingTV         0
    StreamingMovies     0
    Contract            0
    PaperlessBilling    0
    PaymentMethod       0
    MonthlyCharges      0
    TotalCharges        0
    Churn               0
    dtype: int64



Porcentaje de custumer-churn para saber si es posible con el dataset generar la prediccion o debereiamos hacer un tratamiento para generar mas custumer-churn


```python
churn['Churn'].value_counts(sort=True,normalize = True)
```




    No     0.73463
    Yes    0.26537
    Name: Churn, dtype: float64



## Definición de Variables

![png](linea.png)

Tras un pequeño analisis, coincidimos con el enunciado del dataset. El siguiente paso es diferenciar nuestras varibales numericas de las categoricas, y realizar los cambios de tipologia necesarios:


```python
col_id          = ['customerID']
col_churn      = ["Churn"]
col_numericas   = ['tenure','MonthlyCharges','TotalCharges']
col_categoricas = ['gender', 'SeniorCitizen', 'Partner', 'Dependents','PhoneService', 'MultipleLines', 
                   'InternetService','OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                   'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod']
```

## Exploración de Variables Numéricas

![png](linea.png)

Tras intentar realizar la siguiente operación:
`churn[var_numericas] = churn[var_numericas].astype(float)`
Nos muestra un error, el cual nos indica que dentro de estos numeros hay una string. Si hacemos la vista atras, podemos obvservar que la unica variable que no estaba en formato `float` era `TotalCharges`.


```python
(churn.loc[:,['customerID','TotalCharges']]).sort_values(by =['TotalCharges']).head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>TotalCharges</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>936</th>
      <td>5709-LVOEQ</td>
      <td></td>
    </tr>
    <tr>
      <th>3826</th>
      <td>3213-VVOLG</td>
      <td></td>
    </tr>
    <tr>
      <th>4380</th>
      <td>2520-SGTTA</td>
      <td></td>
    </tr>
    <tr>
      <th>753</th>
      <td>3115-CZMZD</td>
      <td></td>
    </tr>
    <tr>
      <th>5218</th>
      <td>2923-ARZLG</td>
      <td></td>
    </tr>
  </tbody>
</table>
</div>



Para encontrar estos valores hemos ordenado los valores, obligandole a colocar en primer lugar los valores de caracter `string` y por ultimo los tipo `float`


```python
churn[churn.TotalCharges == ' '].loc[:,['TotalCharges']].count()
```




    TotalCharges    11
    dtype: int64



Estos valores respresentan un 0.15% del dataSet, con lo cual procedemos a eliminarlos.


```python
churn = churn[churn.TotalCharges != ' ']
churn = churn.reset_index()[churn.columns]
churn[col_numericas] = churn[col_numericas].astype(float)
```

Una vez eleminado estos outliers, empecemos un analisis gráfico con boxplots y la distribución de los datos

### Boxplot variables numericas

#### Tenure


```python
import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots()
fig.set_size_inches(5, 7)
ax = sns.boxplot(x="Churn", y="tenure", data=churn,width=0.3, notch=True)
```


![png](output_32_0.png)


Si hubieramos hecho un analisis sin la diferencia de churn-custumer no hubieramos visto estos outliers dentro de yes-churn. Como estos datos son escasos no podemos permitirnos el lujo de quitarnoslos, con lo cual vamos a suavizarlos poniendo como toque maximo, el punto maximo de nuestro percentil. Hacemos esto debido a que son muy pocos y ademas no se salen demaisado del marco.


```python
churn['tenure'] = np.where((churn.Churn == 'Yes')&(churn.tenure > 67),69,churn.tenure)
```


```python
fig, ax = plt.subplots()
fig.set_size_inches(5, 7)
ax = sns.boxplot(x="Churn", y="tenure", data=churn,width=0.3, notch=True)
```


![png](output_35_0.png)


#### MonthlyCharges


```python
fig, ax = plt.subplots()
fig.set_size_inches(5, 7)
ax = sns.boxplot(x="Churn", y="MonthlyCharges", data=churn,width=0.3, notch=True)
```


![png](output_37_0.png)


#### TotalCharges


```python
fig, ax = plt.subplots()
fig.set_size_inches(5, 7)
ax = sns.boxplot(x="Churn", y="TotalCharges", data=churn,width=0.3, notch=True)
```


![png](output_39_0.png)



```python
churn[(churn.Churn == 'Yes')&(churn.TotalCharges > 6800)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>tenure</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>InternetService</th>
      <th>OnlineSecurity</th>
      <th>...</th>
      <th>DeviceProtection</th>
      <th>TechSupport</th>
      <th>StreamingTV</th>
      <th>StreamingMovies</th>
      <th>Contract</th>
      <th>PaperlessBilling</th>
      <th>PaymentMethod</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>Churn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>104</th>
      <td>3192-NQECA</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>110.00</td>
      <td>7611.85</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>402</th>
      <td>0979-PHULV</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>99.45</td>
      <td>7007.60</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>634</th>
      <td>7207-RMRDB</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>65.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>105.50</td>
      <td>6985.65</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>809</th>
      <td>4853-RULSV</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>104.00</td>
      <td>7250.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>972</th>
      <td>2834-JRTUA</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>108.05</td>
      <td>7532.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1302</th>
      <td>0201-OAMXR</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>115.55</td>
      <td>8127.60</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1335</th>
      <td>3838-OZURD</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>66.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>105.00</td>
      <td>7133.25</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1542</th>
      <td>2886-KEFUM</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>63.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>107.50</td>
      <td>6873.75</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1835</th>
      <td>6990-YNRIO</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>65.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>108.65</td>
      <td>6937.95</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>7694-VLBWQ</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>67.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>No</td>
      <td>Electronic check</td>
      <td>104.10</td>
      <td>7040.85</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2199</th>
      <td>2659-VXMWZ</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>67.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>111.30</td>
      <td>7482.10</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2272</th>
      <td>3571-RFHAR</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>65.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>109.15</td>
      <td>6941.20</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2277</th>
      <td>1587-FKLZB</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>66.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>99.50</td>
      <td>6822.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2282</th>
      <td>5440-FLBQG</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>108.40</td>
      <td>7318.20</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2807</th>
      <td>3763-GCZHZ</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>66.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>104.05</td>
      <td>6890.00</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>2874</th>
      <td>4550-VBOFE</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>102.95</td>
      <td>7101.50</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3035</th>
      <td>7317-GGVPB</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>108.60</td>
      <td>7690.90</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3106</th>
      <td>8809-RIHDD</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>103.40</td>
      <td>7372.65</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3433</th>
      <td>0917-EZOLA</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>104.15</td>
      <td>7689.95</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3511</th>
      <td>0748-RDGGM</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>109.50</td>
      <td>7534.65</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3688</th>
      <td>1150-WFARN</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>67.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>108.75</td>
      <td>7156.20</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3883</th>
      <td>3886-CERTZ</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>109.25</td>
      <td>8109.80</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4076</th>
      <td>0324-BRPCJ</td>
      <td>Female</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>100.20</td>
      <td>6851.65</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4258</th>
      <td>2632-UCGVD</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>66.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>100.05</td>
      <td>6871.90</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4280</th>
      <td>6425-YQLLO</td>
      <td>Female</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>66.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>105.95</td>
      <td>6975.25</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4387</th>
      <td>5502-RLUYV</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>103.95</td>
      <td>7446.90</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4602</th>
      <td>2889-FPWRM</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>117.80</td>
      <td>8684.80</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4676</th>
      <td>6305-YLBMM</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>No</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>104.05</td>
      <td>7262.00</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>4784</th>
      <td>7067-KSAZT</td>
      <td>Female</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>65.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>106.25</td>
      <td>6979.80</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5042</th>
      <td>7762-URZQH</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>66.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>106.05</td>
      <td>6981.35</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5119</th>
      <td>8199-ZLLSA</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>67.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>118.35</td>
      <td>7804.15</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5249</th>
      <td>8634-CILSZ</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>104.70</td>
      <td>7220.35</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5405</th>
      <td>2722-VOJQL</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>64.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>105.65</td>
      <td>6903.10</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5572</th>
      <td>5271-YNWVR</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>113.15</td>
      <td>7856.00</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5688</th>
      <td>1984-FCOWB</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>109.50</td>
      <td>7674.55</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5693</th>
      <td>5287-QWLKY</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>105.10</td>
      <td>7548.10</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6007</th>
      <td>4250-ZBWLV</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>No</td>
      <td>Electronic check</td>
      <td>108.45</td>
      <td>7176.55</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6023</th>
      <td>9090-SGQXL</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>105.30</td>
      <td>7299.65</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6026</th>
      <td>9835-ZIITK</td>
      <td>Male</td>
      <td>1</td>
      <td>Yes</td>
      <td>No</td>
      <td>66.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>Yes</td>
      <td>Electronic check</td>
      <td>110.85</td>
      <td>7491.75</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6029</th>
      <td>1555-DJEQW</td>
      <td>Female</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>114.20</td>
      <td>7723.90</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6280</th>
      <td>9053-JZFKV</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>67.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>116.20</td>
      <td>7752.30</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6389</th>
      <td>3259-FDWOY</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>106.00</td>
      <td>7723.70</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6399</th>
      <td>5748-RNCJT</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Bank transfer (automatic)</td>
      <td>106.50</td>
      <td>7348.80</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6528</th>
      <td>1444-VVSGW</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>No</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>115.65</td>
      <td>7968.85</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6596</th>
      <td>7632-MNYOY</td>
      <td>Male</td>
      <td>1</td>
      <td>No</td>
      <td>No</td>
      <td>66.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>One year</td>
      <td>No</td>
      <td>Credit card (automatic)</td>
      <td>110.90</td>
      <td>7432.05</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6774</th>
      <td>3090-HAWSU</td>
      <td>Male</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>61.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>111.60</td>
      <td>6876.05</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>6934</th>
      <td>6797-LNAQX</td>
      <td>Male</td>
      <td>0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>69.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>No</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Two year</td>
      <td>No</td>
      <td>Bank transfer (automatic)</td>
      <td>98.30</td>
      <td>6859.50</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7023</th>
      <td>0639-TSIQW</td>
      <td>Female</td>
      <td>0</td>
      <td>No</td>
      <td>No</td>
      <td>67.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Fiber optic</td>
      <td>Yes</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Month-to-month</td>
      <td>Yes</td>
      <td>Credit card (automatic)</td>
      <td>102.95</td>
      <td>6886.25</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
<p>48 rows × 21 columns</p>
</div>



Como hemos dicho anteriormente eliminar estos datos no es lo correcto debido a que escasea informacion para los casos de custumer - churn es positivo. Lo que vamamos hacer es suavizar estos datos para que no entorpecer el aprendizaje. Para este caso vamos a colocarle la media de los valores que se salen de lo normal.


```python
churn[(churn.Churn == 'Yes')&(churn.TotalCharges > 6800)].loc[:,['TotalCharges']].mean()
```




    TotalCharges    7337.694792
    dtype: float64




```python
churn['TotalCharges'] = np.where((churn.Churn == 'Yes')&(churn.TotalCharges > 6800),7337.694792,churn.TotalCharges)
fig, ax = plt.subplots()
fig.set_size_inches(5, 7)
ax = sns.boxplot(x="Churn", y="TotalCharges", data=churn,width=0.3, notch=True)
```


![png](output_43_0.png)


Hemos descubierto que hay un pequeño salto entre los outliers, vamos a utilizar ese salto para cortar.


```python
churn['TotalCharges'] = np.where((churn.Churn == 'Yes')&(churn.TotalCharges > 6500),6500,churn.TotalCharges)
```

### Variables numéricas/Churn


```python
import seaborn as sns
sns.pairplot(churn[col_numericas + col_churn ], kind="scatter", diag_kind= 'kde',height=4, hue="Churn")
```




    <seaborn.axisgrid.PairGrid at 0x1a2183cd68>




![png](output_47_1.png)


* **Tenure** nos indicaba el tiempo de meses que ha estado el cliente en la compañia, como podemos observar las personas que estan entre los 20 primeros meses son los más propensos a irse, es algo logico porque normalmente es el tiempo de permanencia que sueles tener al coger ofertas. Pero si observamos los que no se van se mantienen constantes, pero es normal que sean muchas menos personas de las que suelen irse.
    * **Tenure - MonthlyCharges** : es muy homogenio, pero podemos observar como los puntos naranjas desaparecen cuando tienden a la derecha, debido a que son clientes que siempre mantienen un cosumo alto, por ello se se van en busca de nuvas ofertas.
    * **Tenure - TotalCharges** : Observamos una correlacion que es normal, porque a medida que pasa el tiempo los cargos se van acumulando y sumando. En esta gráfica concluimos nuestra teoria anterior, los clinetes con consumos altos son mas propensos a irse, por eso los puntos estan por encima de los azules.
    
    
* **MonthlyCharges** : Son los cargos mensuales del ciente donde se diferencian dos picos. El primero es debido a las personas con un consumo bajo y se mantienen en la compañia, y el segundo pico es debido a las personas con un alto consumo y acaban marchandose. Aqui observamos perfectamente que las dos tendencias se mantienen en una misma proporcion!
    * **MonthlyCharges - TotalCharges** : Es normal esta correlación de los datos, ya que se refieren al el cúmulo de estos. Y es algo logico que los puntos azules esten un poquito por encima que los narajas, debido que al permanecer mas tiempo en la compañia, los totales son mas altos, pero recalco lo de un poquito, debido que a que el consumo elevado de los que se van, llegan en algunos casos a igualar los montos.
    
    
* **TotalCharges** : Los gastos totales en la compañia se suelen mantener entre los 1500 euros para los dos grupos.

### Variables numéricas/Variables categóricas


```python
# Cambiamos una varible binaria a categórica para poder hacer el plot
churn['SeniorCitizen'] = churn['SeniorCitizen'].replace({1:"Yes",0:"No"})
for i in col_categoricas:
    sns.pairplot(churn[col_numericas + col_categoricas ], kind="scatter", diag_kind= 'kde',height=4, hue=i)
   
```


![png](output_50_0.png)



![png](output_50_1.png)



![png](output_50_2.png)



![png](output_50_3.png)



![png](output_50_4.png)



![png](output_50_5.png)



![png](output_50_6.png)



![png](output_50_7.png)



![png](output_50_8.png)



![png](output_50_9.png)



![png](output_50_10.png)



![png](output_50_11.png)



![png](output_50_12.png)



![png](output_50_13.png)



![png](output_50_14.png)



![png](output_50_15.png)


Una de las conclusiones a destacar tras el plot, es la modificación de las variables categoricas: `['OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies']`
Que tienen como variables `'Yes', 'No', 'No internet services'`, las cuales podemos reducirlas a `'Yes', 'No'`. Hemos decidido hacer esto debido a la cantidad y distribucíon de los datos.

## Exploración de Variables Categóricas

![png](linea.png)

Procederemos a la exploración de las variables categoricas, tomando como primera acción el cambio mencionado anteriormente y convirtiendo las varibales a numéricas:



```python
churn[col_categoricas] = churn[col_categoricas].replace({"No internet service":"No"})
churn[col_categoricas] = churn[col_categoricas].replace({"No phone service":"No"})
churn[col_categoricas] = churn[col_categoricas].replace({"Female":1,"Male":0})
churn[col_categoricas] = churn[col_categoricas].replace({"Yes":1,"No":0})
churn[col_churn] = churn[col_churn].replace({"Yes":1,"No":0})
```


```python
# Realizamos un Get dummies para las variables con mas de dos categorias
churn_dummies = pd.get_dummies(churn[col_numericas + col_categoricas + col_churn])
churn = (churn.loc[:,['customerID']]).join(churn_dummies)
col_categoricas = churn.nunique()[churn.nunique() == 2].keys()
col_categoricas
```




    Index(['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
           'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
           'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
           'Churn', 'InternetService_0', 'InternetService_DSL',
           'InternetService_Fiber optic', 'Contract_Month-to-month',
           'Contract_One year', 'Contract_Two year',
           'PaymentMethod_Bank transfer (automatic)',
           'PaymentMethod_Credit card (automatic)',
           'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'],
          dtype='object')



### Variables categóricas/Churn


```python
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
py.init_notebook_mode(connected=True)

dat_rad = churn_dummies[col_categoricas]


def plot_radar(df,aggregate,title) :
    data_frame = df[df["Churn"] == aggregate] 
    data_frame_x = data_frame[col_categoricas].sum().reset_index()
    data_frame_x.columns  = ["feature","yes"]
    data_frame_x["no"]    = data_frame.shape[0]  - data_frame_x["yes"]
    data_frame_x  = data_frame_x[data_frame_x["feature"] != "Churn"]
    
    #contamos los Yes que son 1
    trace1 = go.Scatterpolar(r = data_frame_x["yes"].values.tolist(),
                             theta = data_frame_x["feature"].tolist(),
                             fill  = "toself",name = "Yes",
                             mode = "markers+lines",
                             marker = dict(size = 5)
                            )
    #contamos los No que son 0
    trace2 = go.Scatterpolar(r = data_frame_x["no"].values.tolist(),
                             theta = data_frame_x["feature"].tolist(),
                             fill  = "toself",name = "No",
                             mode = "markers+lines",
                             marker = dict(size = 5)
                            ) 
    layout = go.Layout(dict(polar = dict(radialaxis = dict(visible = True,
                                                           side = "counterclockwise",
                                                           showline = True,
                                                           linewidth = 2,
                                                           tickwidth = 2,
                                                           gridcolor = "white",
                                                           gridwidth = 2),
                                         angularaxis = dict(tickfont = dict(size = 10),
                                                            layer = "below traces"
                                                           ),
                                         bgcolor  = "rgb(243,243,243)",
                                        ),
                            paper_bgcolor = "rgb(243,243,243)",
                            title = title,height = 700))
    
    data = [trace2,trace1]
    fig = go.Figure(data=data,layout=layout)
    py.iplot(fig)

#Ploteamos
plot_radar(dat_rad,1,"Churn")
plot_radar(dat_rad,0,"Non Churn")

```

    /anaconda3/lib/python3.7/site-packages/dask/config.py:168: YAMLLoadWarning:
    
    calling yaml.load() without Loader=... is deprecated, as the default Loader is unsafe. Please read https://msg.pyyaml.org/load for full details.
    



        <script type="text/javascript">
        window.PlotlyConfig = {MathJaxConfig: 'local'};
        if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}
        if (typeof require !== 'undefined') {
        require.undef("plotly");
        requirejs.config({
            paths: {
                'plotly': ['https://cdn.plot.ly/plotly-latest.min']
            }
        });
        require(['plotly'], function(Plotly) {
            window._Plotly = Plotly;
        });
        }
        </script>
        



<div>
        
        
            <div id="4f2c7932-bd53-40ae-8f42-6d46e46ebedc" class="plotly-graph-div" style="height:700px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("4f2c7932-bd53-40ae-8f42-6d46e46ebedc")) {
                    Plotly.newPlot(
                        '4f2c7932-bd53-40ae-8f42-6d46e46ebedc',
                        [{"fill": "toself", "marker": {"size": 5}, "mode": "markers+lines", "name": "No", "r": [930, 1393, 1200, 1543, 170, 1019, 1574, 1346, 1324, 1559, 1055, 1051, 469, 1756, 1410, 572, 214, 1703, 1821, 1611, 1637, 798, 1561], "theta": ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "InternetService_0", "InternetService_DSL", "InternetService_Fiber optic", "Contract_Month-to-month", "Contract_One year", "Contract_Two year", "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"], "type": "scatterpolar", "uid": "fec4d406-0a61-48bd-86e4-a6dbe0bef276"}, {"fill": "toself", "marker": {"size": 5}, "mode": "markers+lines", "name": "Yes", "r": [939, 476, 669, 326, 1699, 850, 295, 523, 545, 310, 814, 818, 1400, 113, 459, 1297, 1655, 166, 48, 258, 232, 1071, 308], "theta": ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "InternetService_0", "InternetService_DSL", "InternetService_Fiber optic", "Contract_Month-to-month", "Contract_One year", "Contract_Two year", "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"], "type": "scatterpolar", "uid": "33562d93-65f6-4b47-a330-7e5e9a3a3dcd"}],
                        {"height": 700, "paper_bgcolor": "rgb(243,243,243)", "polar": {"angularaxis": {"layer": "below traces", "tickfont": {"size": 10}}, "bgcolor": "rgb(243,243,243)", "radialaxis": {"gridcolor": "white", "gridwidth": 2, "linewidth": 2, "showline": true, "side": "counterclockwise", "tickwidth": 2, "visible": true}}, "title": {"text": "Churn"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('4f2c7932-bd53-40ae-8f42-6d46e46ebedc');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>



<div>
        
        
            <div id="a4e86a81-9421-41c9-80ee-4a1a748b98d3" class="plotly-graph-div" style="height:700px; width:100%;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("a4e86a81-9421-41c9-80ee-4a1a748b98d3")) {
                    Plotly.newPlot(
                        'a4e86a81-9421-41c9-80ee-4a1a748b98d3',
                        [{"fill": "toself", "marker": {"size": 5}, "mode": "markers+lines", "name": "No", "r": [2619, 4497, 2439, 3390, 510, 3046, 3443, 3261, 3290, 3433, 3274, 3250, 2395, 3756, 3206, 3364, 2943, 3857, 3526, 3879, 3874, 3869, 3867], "theta": ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "InternetService_0", "InternetService_DSL", "InternetService_Fiber optic", "Contract_Month-to-month", "Contract_One year", "Contract_Two year", "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"], "type": "scatterpolar", "uid": "7ad9b86c-e23e-4031-996f-8da37f18c336"}, {"fill": "toself", "marker": {"size": 5}, "mode": "markers+lines", "name": "Yes", "r": [2544, 666, 2724, 1773, 4653, 2117, 1720, 1902, 1873, 1730, 1889, 1913, 2768, 1407, 1957, 1799, 2220, 1306, 1637, 1284, 1289, 1294, 1296], "theta": ["gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "InternetService_0", "InternetService_DSL", "InternetService_Fiber optic", "Contract_Month-to-month", "Contract_One year", "Contract_Two year", "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"], "type": "scatterpolar", "uid": "0f1a4090-92d7-44b7-9331-e0efa3708f8e"}],
                        {"height": 700, "paper_bgcolor": "rgb(243,243,243)", "polar": {"angularaxis": {"layer": "below traces", "tickfont": {"size": 10}}, "bgcolor": "rgb(243,243,243)", "radialaxis": {"gridcolor": "white", "gridwidth": 2, "linewidth": 2, "showline": true, "side": "counterclockwise", "tickwidth": 2, "visible": true}}, "title": {"text": "Non Churn"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('a4e86a81-9421-41c9-80ee-4a1a748b98d3');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


## Feature Ingeniering

![png](linea.png)

En este apartado generaremos nuevas dimensiones de las variables continuas:


```python
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, interaction_only=False,  
                        include_bias=False)
features = pf.fit_transform(churn[col_numericas])
features
```




    array([[1.00000000e+00, 2.98500000e+01, 2.98500000e+01, ...,
            8.91022500e+02, 8.91022500e+02, 8.91022500e+02],
           [3.40000000e+01, 5.69500000e+01, 1.88950000e+03, ...,
            3.24330250e+03, 1.07607025e+05, 3.57021025e+06],
           [2.00000000e+00, 5.38500000e+01, 1.08150000e+02, ...,
            2.89982250e+03, 5.82387750e+03, 1.16964225e+04],
           ...,
           [1.10000000e+01, 2.96000000e+01, 3.46450000e+02, ...,
            8.76160000e+02, 1.02549200e+04, 1.20027602e+05],
           [4.00000000e+00, 7.44000000e+01, 3.06600000e+02, ...,
            5.53536000e+03, 2.28110400e+04, 9.40035600e+04],
           [6.60000000e+01, 1.05650000e+02, 6.84450000e+03, ...,
            1.11619225e+04, 7.23121425e+05, 4.68471802e+07]])



Una vez generada una matriz de caracteristicas, miremos con que grados a jugado en cada columna para crear el dataframe


```python
pd.DataFrame(pf.powers_, columns=['tenure_degree','MonthlyCharges_degree','TotalCharges_degree'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tenure_degree</th>
      <th>MonthlyCharges_degree</th>
      <th>TotalCharges_degree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0</td>
      <td>0</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
col_features = ['tenure','MonthlyCharges','TotalCharges','tenure_2','tenure*MonthlyCharges','tenure*TotalCharges',
                'MonthlyCharges_2','MonthlyCharges*TotalCharges','TotalCharges*2']
churn_features = pd.DataFrame(features, columns = col_features)

# Añadimos algunas medias que nos parecen interesantes 
churn_features['MonthlyCharges_mean'] = churn_features['MonthlyCharges'].mean()
churn_features['tenure_mean'] = churn_features['tenure'].mean()

# Genreamos los dos cuerpos de matrices para el posterior modelado
churn_B = (churn.loc[:,['customerID']]).join((churn_features.join(churn[col_categoricas])))
churn_A = churn
```

#### Matriz A

En esta matriz hemos dejado las varibles continuas sin manipular, adjuntando sus variables categoricas que anteriormemte hemos tratato.


```python
churn_A.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>tenure</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>gender</th>
      <th>SeniorCitizen</th>
      <th>Partner</th>
      <th>Dependents</th>
      <th>PhoneService</th>
      <th>MultipleLines</th>
      <th>...</th>
      <th>InternetService_0</th>
      <th>InternetService_DSL</th>
      <th>InternetService_Fiber optic</th>
      <th>Contract_Month-to-month</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>PaymentMethod_Bank transfer (automatic)</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>1.0</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>34.0</td>
      <td>56.95</td>
      <td>1889.50</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>2.0</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>45.0</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>2.0</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>




```python
churn_A.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7032 entries, 0 to 7031
    Data columns (total 28 columns):
    customerID                                 7032 non-null object
    tenure                                     7032 non-null float64
    MonthlyCharges                             7032 non-null float64
    TotalCharges                               7032 non-null float64
    gender                                     7032 non-null int64
    SeniorCitizen                              7032 non-null int64
    Partner                                    7032 non-null int64
    Dependents                                 7032 non-null int64
    PhoneService                               7032 non-null int64
    MultipleLines                              7032 non-null int64
    OnlineSecurity                             7032 non-null int64
    OnlineBackup                               7032 non-null int64
    DeviceProtection                           7032 non-null int64
    TechSupport                                7032 non-null int64
    StreamingTV                                7032 non-null int64
    StreamingMovies                            7032 non-null int64
    PaperlessBilling                           7032 non-null int64
    Churn                                      7032 non-null int64
    InternetService_0                          7032 non-null uint8
    InternetService_DSL                        7032 non-null uint8
    InternetService_Fiber optic                7032 non-null uint8
    Contract_Month-to-month                    7032 non-null uint8
    Contract_One year                          7032 non-null uint8
    Contract_Two year                          7032 non-null uint8
    PaymentMethod_Bank transfer (automatic)    7032 non-null uint8
    PaymentMethod_Credit card (automatic)      7032 non-null uint8
    PaymentMethod_Electronic check             7032 non-null uint8
    PaymentMethod_Mailed check                 7032 non-null uint8
    dtypes: float64(3), int64(14), object(1), uint8(10)
    memory usage: 1.0+ MB


#### Matriz B

En esta matriz hemos jugado con las variables continuas añadiendo medias y combinaciones lineas de estas. Además de las variables categóricas.


```python
churn_B.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customerID</th>
      <th>tenure</th>
      <th>MonthlyCharges</th>
      <th>TotalCharges</th>
      <th>tenure_2</th>
      <th>tenure*MonthlyCharges</th>
      <th>tenure*TotalCharges</th>
      <th>MonthlyCharges_2</th>
      <th>MonthlyCharges*TotalCharges</th>
      <th>TotalCharges*2</th>
      <th>...</th>
      <th>InternetService_0</th>
      <th>InternetService_DSL</th>
      <th>InternetService_Fiber optic</th>
      <th>Contract_Month-to-month</th>
      <th>Contract_One year</th>
      <th>Contract_Two year</th>
      <th>PaymentMethod_Bank transfer (automatic)</th>
      <th>PaymentMethod_Credit card (automatic)</th>
      <th>PaymentMethod_Electronic check</th>
      <th>PaymentMethod_Mailed check</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7590-VHVEG</td>
      <td>1.0</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>1.0</td>
      <td>29.85</td>
      <td>29.85</td>
      <td>891.0225</td>
      <td>891.0225</td>
      <td>8.910225e+02</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5575-GNVDE</td>
      <td>34.0</td>
      <td>56.95</td>
      <td>1889.50</td>
      <td>1156.0</td>
      <td>1936.30</td>
      <td>64243.00</td>
      <td>3243.3025</td>
      <td>107607.0250</td>
      <td>3.570210e+06</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3668-QPYBK</td>
      <td>2.0</td>
      <td>53.85</td>
      <td>108.15</td>
      <td>4.0</td>
      <td>107.70</td>
      <td>216.30</td>
      <td>2899.8225</td>
      <td>5823.8775</td>
      <td>1.169642e+04</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7795-CFOCW</td>
      <td>45.0</td>
      <td>42.30</td>
      <td>1840.75</td>
      <td>2025.0</td>
      <td>1903.50</td>
      <td>82833.75</td>
      <td>1789.2900</td>
      <td>77863.7250</td>
      <td>3.388361e+06</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237-HQITU</td>
      <td>2.0</td>
      <td>70.70</td>
      <td>151.65</td>
      <td>4.0</td>
      <td>141.40</td>
      <td>303.30</td>
      <td>4998.4900</td>
      <td>10721.6550</td>
      <td>2.299772e+04</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 36 columns</p>
</div>




```python
churn_B.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7032 entries, 0 to 7031
    Data columns (total 36 columns):
    customerID                                 7032 non-null object
    tenure                                     7032 non-null float64
    MonthlyCharges                             7032 non-null float64
    TotalCharges                               7032 non-null float64
    tenure_2                                   7032 non-null float64
    tenure*MonthlyCharges                      7032 non-null float64
    tenure*TotalCharges                        7032 non-null float64
    MonthlyCharges_2                           7032 non-null float64
    MonthlyCharges*TotalCharges                7032 non-null float64
    TotalCharges*2                             7032 non-null float64
    MonthlyCharges_mean                        7032 non-null float64
    tenure_mean                                7032 non-null float64
    gender                                     7032 non-null int64
    SeniorCitizen                              7032 non-null int64
    Partner                                    7032 non-null int64
    Dependents                                 7032 non-null int64
    PhoneService                               7032 non-null int64
    MultipleLines                              7032 non-null int64
    OnlineSecurity                             7032 non-null int64
    OnlineBackup                               7032 non-null int64
    DeviceProtection                           7032 non-null int64
    TechSupport                                7032 non-null int64
    StreamingTV                                7032 non-null int64
    StreamingMovies                            7032 non-null int64
    PaperlessBilling                           7032 non-null int64
    Churn                                      7032 non-null int64
    InternetService_0                          7032 non-null uint8
    InternetService_DSL                        7032 non-null uint8
    InternetService_Fiber optic                7032 non-null uint8
    Contract_Month-to-month                    7032 non-null uint8
    Contract_One year                          7032 non-null uint8
    Contract_Two year                          7032 non-null uint8
    PaymentMethod_Bank transfer (automatic)    7032 non-null uint8
    PaymentMethod_Credit card (automatic)      7032 non-null uint8
    PaymentMethod_Electronic check             7032 non-null uint8
    PaymentMethod_Mailed check                 7032 non-null uint8
    dtypes: float64(11), int64(14), object(1), uint8(10)
    memory usage: 1.5+ MB


## Nomalizacion, distribución y correlacion de las variables.

![png](linea.png)

Uno de los fallos mas habituales es dar a la maquina datos no escalados. Esto entorpeceria el aprendizaje de la maquina debido al peso de las varibales de alto valor, dejando aun lado la importancia casi de las variables categóricas. Por ello es necesario realziar un escalado a corde al dataset. Otro punto importantes es ver las correlaciones de las variables para ver si estamos dando la misma información.

### Matriz A


```python
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
norm_churn_A = std.fit_transform(churn_A[col_numericas])
norm_churn_A = pd.DataFrame(norm_churn_A,columns=col_numericas)
sns.pairplot(norm_churn_A, kind="scatter", diag_kind= 'kde',height=4)
```




    <seaborn.axisgrid.PairGrid at 0x1a2a820d30>




![png](output_77_1.png)



```python
churn_A_final = (churn_A.loc[:,['customerID']]).join((norm_churn_A.join(churn_A[col_categoricas])))
churn_A_final.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7032 entries, 0 to 7031
    Data columns (total 28 columns):
    customerID                                 7032 non-null object
    tenure                                     7032 non-null float64
    MonthlyCharges                             7032 non-null float64
    TotalCharges                               7032 non-null float64
    gender                                     7032 non-null int64
    SeniorCitizen                              7032 non-null int64
    Partner                                    7032 non-null int64
    Dependents                                 7032 non-null int64
    PhoneService                               7032 non-null int64
    MultipleLines                              7032 non-null int64
    OnlineSecurity                             7032 non-null int64
    OnlineBackup                               7032 non-null int64
    DeviceProtection                           7032 non-null int64
    TechSupport                                7032 non-null int64
    StreamingTV                                7032 non-null int64
    StreamingMovies                            7032 non-null int64
    PaperlessBilling                           7032 non-null int64
    Churn                                      7032 non-null int64
    InternetService_0                          7032 non-null uint8
    InternetService_DSL                        7032 non-null uint8
    InternetService_Fiber optic                7032 non-null uint8
    Contract_Month-to-month                    7032 non-null uint8
    Contract_One year                          7032 non-null uint8
    Contract_Two year                          7032 non-null uint8
    PaymentMethod_Bank transfer (automatic)    7032 non-null uint8
    PaymentMethod_Credit card (automatic)      7032 non-null uint8
    PaymentMethod_Electronic check             7032 non-null uint8
    PaymentMethod_Mailed check                 7032 non-null uint8
    dtypes: float64(3), int64(14), object(1), uint8(10)
    memory usage: 1.0+ MB



```python
#Preparamos el frame de correlaciones
correlation = churn_A_final.corr()
#Extraemos el nombre de las columnas
matrix_cols = correlation.columns.tolist()
#convertimos el frame en array
corr_array  = np.array(correlation)

#Visualizacion de datos con plotly
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   colorscale = [[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'],
        [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'],
        [0.6666666666666666, 'rgb(171,217,233)'],[0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'],
        [1.0, 'rgb(49,54,149)']],
                   colorbar   = dict(title = "Pearson Correlation coefficient",
                                     titleside = "right"
                                    ) ,
                  )

layout = go.Layout(dict(title = "Matriz A de correlaciones",
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                      ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9))
                       )
                  )

data = [trace]
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)
```


<div>
        
        
            <div id="1e8befcc-1fbf-4b83-9020-0aed7e7349eb" class="plotly-graph-div" style="height:720px; width:800px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("1e8befcc-1fbf-4b83-9020-0aed7e7349eb")) {
                    Plotly.newPlot(
                        '1e8befcc-1fbf-4b83-9020-0aed7e7349eb',
                        [{"colorbar": {"title": {"side": "right", "text": "Pearson Correlation coefficient"}}, "colorscale": [[0.0, "rgb(165,0,38)"], [0.1111111111111111, "rgb(215,48,39)"], [0.2222222222222222, "rgb(244,109,67)"], [0.3333333333333333, "rgb(253,174,97)"], [0.4444444444444444, "rgb(254,224,144)"], [0.5555555555555556, "rgb(224,243,248)"], [0.6666666666666666, "rgb(171,217,233)"], [0.7777777777777778, "rgb(116,173,209)"], [0.8888888888888888, "rgb(69,117,180)"], [1.0, "rgb(49,54,149)"]], "type": "heatmap", "uid": "fb7b7191-5100-456e-8e9d-3a9b12b42332", "x": ["tenure", "MonthlyCharges", "TotalCharges", "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn", "InternetService_0", "InternetService_DSL", "InternetService_Fiber optic", "Contract_Month-to-month", "Contract_One year", "Contract_Two year", "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"], "y": ["tenure", "MonthlyCharges", "TotalCharges", "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn", "InternetService_0", "InternetService_DSL", "InternetService_Fiber optic", "Contract_Month-to-month", "Contract_One year", "Contract_Two year", "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"], "z": [[1.0, 0.24673947491397813, 0.8264606208294916, -0.005242266994018822, 0.015675375704983785, 0.3818758374863098, 0.16342735045857737, 0.007936580687400586, 0.332360049569408, 0.32831731240661505, 0.3610600281117624, 0.36142893937055787, 0.32528359550205393, 0.28009772993820486, 0.28526361779277526, 0.004776972324833932, -0.35445958922540277, -0.03744230643343394, 0.013814579589352732, 0.01783065441244216, -0.6493272080616425, 0.20239239382808324, 0.5637275588452099, 0.243836903276757, 0.23282406272287987, -0.2102609569975324, -0.2321468378557078], [0.24673947491397813, 1.0, 0.6507996667280491, 0.013779327268354435, 0.21987422950593646, 0.09782497186892049, -0.11234295350128225, 0.2480330664757158, 0.4909121973267493, 0.2964469592375873, 0.4415290881871007, 0.48260691224313995, 0.33830139143424953, 0.6296678921767406, 0.6272347301103788, 0.3519304153712528, 0.1928582184700881, -0.7631910615169571, -0.16136793538251534, 0.7871948529419658, 0.0589334582292538, 0.004809615120440435, -0.07325607300641665, 0.04240972759459293, 0.030054608328366407, 0.27111737739356384, -0.37656828808083825], [0.8264606208294916, 0.6507996667280491, 1.0, 0.0012124638769354028, 0.10261720923516729, 0.3195983331517227, 0.06523236887204308, 0.11277212339029269, 0.46880629333699014, 0.41398501008905886, 0.5101708876194947, 0.5227497580254298, 0.43303338862365776, 0.5152483219994549, 0.5194915077585068, 0.15754649978292518, -0.20489465846264848, -0.37555276529366216, -0.05059155772251129, 0.3597989695050238, -0.44727599103944105, 0.16998268276634915, 0.3591774649590733, 0.1864274258095615, 0.1830206321507009, -0.06084867381034115, -0.2948982749093637], [-0.005242266994018822, 0.013779327268354435, 0.0012124638769354028, 1.0, 0.0018193906134190093, 0.0013790513218355396, -0.01034891712761414, 0.0075149799091999234, 0.008882737146285997, 0.016327823070706123, 0.013092839264555147, 0.0008067457759128009, 0.008507162405232742, 0.007124396986724594, 0.010105418366566175, 0.011901894766838502, 0.008544643224947239, -0.004744965758850031, -0.007583576103039617, 0.011189259276385852, 0.0032507651194556174, -0.007754852914267272, 0.0036031674135729897, 0.015973079031173832, -0.0016318725186135986, -0.0008437084888754001, -0.01319936726545174], [0.015675375704983785, 0.21987422950593646, 0.10261720923516729, 0.0018193906134190093, 1.0, 0.01695661453202187, -0.21055006112684216, 0.008391611911217043, 0.14299625086621018, -0.0385763901686064, 0.06666279065142021, 0.059513871482029225, -0.060576839406188035, 0.10544501753678828, 0.11984236746151568, 0.15625775052783097, 0.1505410534156757, -0.18251949495535458, -0.10827563872848943, 0.25492331502717946, 0.13775207088551514, -0.0464907545657889, -0.11620511425710835, -0.01623474200582214, -0.024359419683712323, 0.17132216591713703, -0.15298719260173027], [0.3818758374863098, 0.09782497186892049, 0.3195983331517227, 0.0013790513218355396, 0.01695661453202187, 1.0, 0.45226888584550023, 0.018397189302703662, 0.1425612874681736, 0.14334606167364233, 0.14184917072520303, 0.1535564364182745, 0.12020601780298455, 0.12448262672518244, 0.11810820943292764, -0.013956696136191696, -0.14998192562006138, -0.0002855204740384597, -0.0010430787434336079, 0.0012346095228208073, -0.2802019157901561, 0.08306706395255747, 0.24733370647615796, 0.11140561212215645, 0.08232738919649572, -0.08320661736633733, -0.09694798339506473], [0.16342735045857737, -0.11234295350128225, 0.06523236887204308, -0.01034891712761414, -0.21055006112684216, 0.45226888584550023, 1.0, -0.001077812708067608, -0.024306661314620996, 0.08078553224088346, 0.023638813060609963, 0.013899668260943368, 0.06305315799997827, -0.01649868035801052, -0.03837492560091315, -0.11013068597336993, -0.16312843938822, 0.13838288994798562, 0.05159321756471237, -0.16410089031864167, -0.2297147909213749, 0.06922205672629726, 0.2016993304297039, 0.05236890928127545, 0.06113408322897806, -0.1492739811934336, 0.05644841359590588], [0.007936580687400586, 0.2480330664757158, 0.11277212339029269, 0.0075149799091999234, 0.008391611911217043, 0.018397189302703662, -0.001077812708067608, 1.0, 0.2795295400049995, -0.09167570469500017, -0.05213341919796151, -0.07007561533228862, -0.09513849428922488, -0.021382711870578185, -0.03347749718236395, 0.016696123642784135, 0.011691398865422323, 0.1718171065699321, -0.45225528090657086, 0.29018311793843365, -0.0012425134067734023, -0.0031417807184111624, 0.0044422513152849235, 0.008271245210923577, -0.006916252198127548, 0.0027471183312986857, -0.004462839400732194], [0.332360049569408, 0.4909121973267493, 0.46880629333699014, 0.008882737146285997, 0.14299625086621018, 0.1425612874681736, -0.024306661314620996, 0.2795295400049995, 1.0, 0.0985919934252315, 0.2022283972825372, 0.201732824517757, 0.10042125595413272, 0.25780350066730157, 0.2591943175468362, 0.1637457730112591, 0.040032739872523634, -0.21079354712189471, -0.2003183214156725, 0.3664202566051166, -0.08855832218643561, -0.003594461398434266, 0.10661820819152797, 0.07542871730303179, 0.06031899094084278, 0.08358299305536028, -0.22767156949803466], [0.32831731240661505, 0.2964469592375873, 0.41398501008905886, 0.016327823070706123, -0.0385763901686064, 0.14334606167364233, 0.08078553224088346, -0.09167570469500017, 0.0985919934252315, 1.0, 0.28328454262626757, 0.274875003842449, 0.35445796164509147, 0.1755144708953688, 0.18742584957299618, -0.004051250607988492, -0.17126992353351678, -0.33279949932167546, 0.3203433737595294, -0.03050626904076453, -0.24684428487400414, 0.10065777311969464, 0.19169819815673583, 0.09436639279979232, 0.11547320256635303, -0.11229466175861408, -0.07991768713640306], [0.3610600281117624, 0.4415290881871007, 0.5101708876194947, 0.013092839264555147, 0.06666279065142021, 0.14184917072520303, 0.023638813060609963, -0.05213341919796151, 0.2022283972825372, 0.28328454262626757, 1.0, 0.30305766643807824, 0.29370469187781045, 0.2816010622259753, 0.27452301070772545, 0.12705603268686044, -0.08230696876508349, -0.3809903317320751, 0.15676460995441888, 0.16594028590307677, -0.16439302987919688, 0.08411316021806066, 0.11139068731904943, 0.0869415675760231, 0.09045518641091457, -0.00036426636786500763, -0.17407470231312427], [0.36142893937055787, 0.48260691224313995, 0.5227497580254298, 0.0008067457759128009, 0.059513871482029225, 0.1535564364182745, 0.013899668260943368, -0.07007561533228862, 0.201732824517757, 0.274875003842449, 0.30305766643807824, 1.0, 0.33285005080469243, 0.3899237975094597, 0.4023088228216018, 0.10407904724402045, -0.06619251684228997, -0.3801513548956378, 0.14514955473692903, 0.17635617323471664, -0.22598757731112262, 0.10291089629343353, 0.16524753554250074, 0.08304690185342707, 0.11125168129784246, -0.003308493511411254, -0.18732483013668594], [0.32528359550205393, 0.33830139143424953, 0.43303338862365776, 0.008507162405232742, -0.060576839406188035, 0.12020601780298455, 0.06305315799997827, -0.09513849428922488, 0.10042125595413272, 0.35445796164509147, 0.29370469187781045, 0.33285005080469243, 1.0, 0.277548599200549, 0.2801552432906342, 0.03753587307318976, -0.16471590834411207, -0.33569508671869736, 0.3121832985222757, -0.020298967520709605, -0.28549086901033593, 0.09625836225380952, 0.24092408252256528, 0.10047200087443556, 0.11702370730984143, -0.11480726996437085, -0.08463055196615278], [0.28009772993820486, 0.6296678921767406, 0.5152483219994549, 0.007124396986724594, 0.10544501753678828, 0.12448262672518244, -0.01649868035801052, -0.021382711870578185, 0.25780350066730157, 0.1755144708953688, 0.2816010622259753, 0.3899237975094597, 0.277548599200549, 1.0, 0.5333800979319763, 0.22424119793848596, 0.06325398027519404, -0.41495062156578044, 0.014973379079172382, 0.3297441152730512, -0.11254989712217289, 0.061929689963200855, 0.07212357537070835, 0.04612070051242859, 0.040010276337768505, 0.1447470086556032, -0.2477115493728633], [0.28526361779277526, 0.6272347301103788, 0.5194915077585068, 0.010105418366566175, 0.11984236746151568, 0.11810820943292764, -0.03837492560091315, -0.03347749718236395, 0.2591943175468362, 0.18742584957299618, 0.27452301070772545, 0.4023088228216018, 0.2801552432906342, 0.5333800979319763, 1.0, 0.21158250423808916, 0.06085993668146301, -0.41844975538334045, 0.02562310861129719, 0.3224574540559222, -0.11786687989290552, 0.06477997824381859, 0.07560257919382919, 0.04875484714119087, 0.048398314068082315, 0.13742008269944622, -0.2502897149395328], [0.004776972324833932, 0.3519304153712528, 0.15754649978292518, 0.011901894766838502, 0.15625775052783097, -0.013956696136191696, -0.11013068597336993, 0.016696123642784135, 0.1637457730112591, -0.004051250607988492, 0.12705603268686044, 0.10407904724402045, 0.03753587307318976, 0.22424119793848596, 0.21158250423808916, 1.0, 0.19145432108006671, -0.3205922451174622, -0.06338966821876392, 0.32647017160380964, 0.16829626845602835, -0.052278164693773076, -0.1462807050684952, -0.017468900682392235, -0.013726285095880284, 0.20842668228002995, -0.20398064814312206], [-0.35445958922540277, 0.1928582184700881, -0.20489465846264848, 0.008544643224947239, 0.1505410534156757, -0.14998192562006138, -0.16312843938822, 0.011691398865422323, 0.040032739872523634, -0.17126992353351678, -0.08230696876508349, -0.06619251684228997, -0.16471590834411207, 0.06325398027519404, 0.06085993668146301, 0.19145432108006671, 1.0, -0.22757762044656818, -0.12414142842590645, 0.30746259069818205, 0.40456455007784087, -0.17822502328994053, -0.30155233962397837, -0.1181359978280296, -0.1346868372340906, 0.30145463790858057, -0.09077284582582087], [-0.03744230643343394, -0.7631910615169571, -0.37555276529366216, -0.004744965758850031, -0.18251949495535458, -0.0002855204740384597, 0.13838288994798562, 0.1718171065699321, -0.21079354712189471, -0.33279949932167546, -0.3809903317320751, -0.3801513548956378, -0.33569508671869736, -0.41495062156578044, -0.41844975538334045, -0.3205922451174622, -0.22757762044656818, 1.0, -0.3799117751052334, -0.4657363343235562, -0.217823505545489, 0.038061459724733744, 0.21754205606911817, -0.0010943032992050863, 0.0018701145623306124, -0.2846082097225956, 0.3196937439459745], [0.013814579589352732, -0.16136793538251534, -0.05059155772251129, -0.007583576103039617, -0.10827563872848943, -0.0010430787434336079, 0.05159321756471237, -0.45225528090657086, -0.2003183214156725, 0.3203433737595294, 0.15676460995441888, 0.14514955473692903, 0.3121832985222757, 0.014973379079172382, 0.02562310861129719, -0.06338966821876392, -0.12414142842590645, -0.3799117751052334, 1.0, -0.6416356650534906, -0.06522632996531619, 0.04729967349374622, 0.030923714855574744, 0.02475954021490298, 0.05122176476699011, -0.10429333541089097, 0.04275388869901973], [0.01783065441244216, 0.7871948529419658, 0.3597989695050238, 0.011189259276385852, 0.25492331502717946, 0.0012346095228208073, -0.16410089031864167, 0.29018311793843365, 0.3664202566051166, -0.03050626904076453, 0.16594028590307677, 0.17635617323471664, -0.020298967520709605, 0.3297441152730512, 0.3224574540559222, 0.32647017160380964, 0.30746259069818205, -0.4657363343235562, -0.6416356650534906, 1.0, 0.24301351814831268, -0.07680902975096314, -0.20996452908904545, -0.022778855291876635, -0.050551991558036385, 0.3357634768102139, -0.3059839224841771], [-0.6493272080616425, 0.0589334582292538, -0.44727599103944105, 0.0032507651194556174, 0.13775207088551514, -0.2802019157901561, -0.2297147909213749, -0.0012425134067734023, -0.08855832218643561, -0.24684428487400414, -0.16439302987919688, -0.22598757731112262, -0.28549086901033593, -0.11254989712217289, -0.11786687989290552, 0.16829626845602835, 0.40456455007784087, -0.217823505545489, -0.06522632996531619, 0.24301351814831268, 1.0, -0.5700527848944215, -0.6219327447713561, -0.18015909738683936, -0.20496021669474843, 0.33087881370583405, 0.006208692442055045], [0.20239239382808324, 0.004809615120440435, 0.16998268276634915, -0.007754852914267272, -0.0464907545657889, 0.08306706395255747, 0.06922205672629726, -0.0031417807184111624, -0.003594461398434266, 0.10065777311969464, 0.08411316021806066, 0.10291089629343353, 0.09625836225380952, 0.061929689963200855, 0.06477997824381859, -0.052278164693773076, -0.17822502328994053, 0.038061459724733744, 0.04729967349374622, -0.07680902975096314, -0.5700527848944215, 1.0, -0.28884268256780254, 0.05762874929825631, 0.06758968145792751, -0.10954646682258033, 0.0001971241890499786], [0.5637275588452099, -0.07325607300641665, 0.3591774649590733, 0.0036031674135729897, -0.11620511425710835, 0.24733370647615796, 0.2016993304297039, 0.0044422513152849235, 0.10661820819152797, 0.19169819815673583, 0.11139068731904943, 0.16524753554250074, 0.24092408252256528, 0.07212357537070835, 0.07560257919382919, -0.1462807050684952, -0.30155233962397837, 0.21754205606911817, 0.030923714855574744, -0.20996452908904545, -0.6219327447713561, -0.28884268256780254, 1.0, 0.1550042180767056, 0.17440993895950943, -0.28114743393395547, -0.007422540118618717], [0.243836903276757, 0.04240972759459293, 0.1864274258095615, 0.015973079031173832, -0.01623474200582214, 0.11140561212215645, 0.05236890928127545, 0.008271245210923577, 0.07542871730303179, 0.09436639279979232, 0.0869415675760231, 0.08304690185342707, 0.10047200087443556, 0.04612070051242859, 0.04875484714119087, -0.017468900682392235, -0.1181359978280296, -0.0010943032992050863, 0.02475954021490298, -0.022778855291876635, -0.18015909738683936, 0.05762874929825631, 0.1550042180767056, 1.0, -0.27842319712065355, -0.3772703602158491, -0.28809669563791657], [0.23282406272287987, 0.030054608328366407, 0.1830206321507009, -0.0016318725186135986, -0.024359419683712323, 0.08232738919649572, 0.06113408322897806, -0.006916252198127548, 0.06031899094084278, 0.11547320256635303, 0.09045518641091457, 0.11125168129784246, 0.11702370730984143, 0.040010276337768505, 0.048398314068082315, -0.013726285095880284, -0.1346868372340906, 0.0018701145623306124, 0.05122176476699011, -0.050551991558036385, -0.20496021669474843, 0.06758968145792751, 0.17440993895950943, -0.27842319712065355, 1.0, -0.37397801626928234, -0.28558254792863036], [-0.2102609569975324, 0.27111737739356384, -0.06084867381034115, -0.0008437084888754001, 0.17132216591713703, -0.08320661736633733, -0.1492739811934336, 0.0027471183312986857, 0.08358299305536028, -0.11229466175861408, -0.00036426636786500763, -0.003308493511411254, -0.11480726996437085, 0.1447470086556032, 0.13742008269944622, 0.20842668228002995, 0.30145463790858057, -0.2846082097225956, -0.10429333541089097, 0.3357634768102139, 0.33087881370583405, -0.10954646682258033, -0.28114743393395547, -0.3772703602158491, -0.37397801626928234, 1.0, -0.3869714587097063], [-0.2321468378557078, -0.37656828808083825, -0.2948982749093637, -0.01319936726545174, -0.15298719260173027, -0.09694798339506473, 0.05644841359590588, -0.004462839400732194, -0.22767156949803466, -0.07991768713640306, -0.17407470231312427, -0.18732483013668594, -0.08463055196615278, -0.2477115493728633, -0.2502897149395328, -0.20398064814312206, -0.09077284582582087, 0.3196937439459745, 0.04275388869901973, -0.3059839224841771, 0.006208692442055045, 0.0001971241890499786, -0.007422540118618717, -0.28809669563791657, -0.28558254792863036, -0.3869714587097063, 1.0]]}],
                        {"autosize": false, "height": 720, "margin": {"b": 210, "l": 210, "r": 0, "t": 25}, "title": {"text": "Matriz A de correlaciones"}, "width": 800, "xaxis": {"tickfont": {"size": 9}}, "yaxis": {"tickfont": {"size": 9}}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('1e8befcc-1fbf-4b83-9020-0aed7e7349eb');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


### Matriz B


```python
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
norm_churn_B = std.fit_transform(churn_B[col_features])
norm_churn_B = pd.DataFrame(norm_churn_B,columns=col_features)
sns.pairplot(norm_churn_B, kind="scatter", diag_kind= 'kde',height=4)
```




    <seaborn.axisgrid.PairGrid at 0x1a26933dd8>




![png](output_81_1.png)



```python
churn_B_final = (churn_B.loc[:,['customerID']]).join((norm_churn_B.join(churn_B[col_categoricas])))
churn_B_final.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 7032 entries, 0 to 7031
    Data columns (total 34 columns):
    customerID                                 7032 non-null object
    tenure                                     7032 non-null float64
    MonthlyCharges                             7032 non-null float64
    TotalCharges                               7032 non-null float64
    tenure_2                                   7032 non-null float64
    tenure*MonthlyCharges                      7032 non-null float64
    tenure*TotalCharges                        7032 non-null float64
    MonthlyCharges_2                           7032 non-null float64
    MonthlyCharges*TotalCharges                7032 non-null float64
    TotalCharges*2                             7032 non-null float64
    gender                                     7032 non-null int64
    SeniorCitizen                              7032 non-null int64
    Partner                                    7032 non-null int64
    Dependents                                 7032 non-null int64
    PhoneService                               7032 non-null int64
    MultipleLines                              7032 non-null int64
    OnlineSecurity                             7032 non-null int64
    OnlineBackup                               7032 non-null int64
    DeviceProtection                           7032 non-null int64
    TechSupport                                7032 non-null int64
    StreamingTV                                7032 non-null int64
    StreamingMovies                            7032 non-null int64
    PaperlessBilling                           7032 non-null int64
    Churn                                      7032 non-null int64
    InternetService_0                          7032 non-null uint8
    InternetService_DSL                        7032 non-null uint8
    InternetService_Fiber optic                7032 non-null uint8
    Contract_Month-to-month                    7032 non-null uint8
    Contract_One year                          7032 non-null uint8
    Contract_Two year                          7032 non-null uint8
    PaymentMethod_Bank transfer (automatic)    7032 non-null uint8
    PaymentMethod_Credit card (automatic)      7032 non-null uint8
    PaymentMethod_Electronic check             7032 non-null uint8
    PaymentMethod_Mailed check                 7032 non-null uint8
    dtypes: float64(9), int64(14), object(1), uint8(10)
    memory usage: 1.4+ MB



```python
#Preparamos el frame de correlaciones
correlation = churn_B_final.corr()
#Extraemos el nombre de las columnas
matrix_cols = correlation.columns.tolist()
#convertimos el frame en array
corr_array  = np.array(correlation)

#Visualizacion de datos con plotly
trace = go.Heatmap(z = corr_array,
                   x = matrix_cols,
                   y = matrix_cols,
                   colorscale = [[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'],
        [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'],
        [0.6666666666666666, 'rgb(171,217,233)'],[0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'],
        [1.0, 'rgb(49,54,149)']],
                   colorbar   = dict(title = "Pearson Correlation coefficient",
                                     titleside = "right"
                                    ) ,
                  )

layout = go.Layout(dict(title = "Matriz B de correlaciones",
                        autosize = False,
                        height  = 720,
                        width   = 800,
                        margin  = dict(r = 0 ,l = 210,
                                       t = 25,b = 210,
                                      ),
                        yaxis   = dict(tickfont = dict(size = 9)),
                        xaxis   = dict(tickfont = dict(size = 9))
                       )
                  )

data = [trace]
fig = go.Figure(data=data,layout=layout)
py.iplot(fig)
```


<div>
        
        
            <div id="863349f7-a5b8-4455-a818-e979891e9393" class="plotly-graph-div" style="height:720px; width:800px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("863349f7-a5b8-4455-a818-e979891e9393")) {
                    Plotly.newPlot(
                        '863349f7-a5b8-4455-a818-e979891e9393',
                        [{"colorbar": {"title": {"side": "right", "text": "Pearson Correlation coefficient"}}, "colorscale": [[0.0, "rgb(165,0,38)"], [0.1111111111111111, "rgb(215,48,39)"], [0.2222222222222222, "rgb(244,109,67)"], [0.3333333333333333, "rgb(253,174,97)"], [0.4444444444444444, "rgb(254,224,144)"], [0.5555555555555556, "rgb(224,243,248)"], [0.6666666666666666, "rgb(171,217,233)"], [0.7777777777777778, "rgb(116,173,209)"], [0.8888888888888888, "rgb(69,117,180)"], [1.0, "rgb(49,54,149)"]], "type": "heatmap", "uid": "3f0ad5d5-b718-4a0c-8570-6ab11310a736", "x": ["tenure", "MonthlyCharges", "TotalCharges", "tenure_2", "tenure*MonthlyCharges", "tenure*TotalCharges", "MonthlyCharges_2", "MonthlyCharges*TotalCharges", "TotalCharges*2", "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn", "InternetService_0", "InternetService_DSL", "InternetService_Fiber optic", "Contract_Month-to-month", "Contract_One year", "Contract_Two year", "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"], "y": ["tenure", "MonthlyCharges", "TotalCharges", "tenure_2", "tenure*MonthlyCharges", "tenure*TotalCharges", "MonthlyCharges_2", "MonthlyCharges*TotalCharges", "TotalCharges*2", "gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn", "InternetService_0", "InternetService_DSL", "InternetService_Fiber optic", "Contract_Month-to-month", "Contract_One year", "Contract_Two year", "PaymentMethod_Bank transfer (automatic)", "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"], "z": [[1.0, 0.24673947491397813, 0.8264606208294916, 0.968103997411089, 0.8261839267575941, 0.8329551368697643, 0.28972430013783385, 0.7050251415052948, 0.7237742233918172, -0.005242266994018822, 0.015675375704983785, 0.3818758374863098, 0.16342735045857737, 0.007936580687400586, 0.332360049569408, 0.32831731240661505, 0.3610600281117624, 0.36142893937055787, 0.32528359550205393, 0.28009772993820486, 0.28526361779277526, 0.004776972324833932, -0.35445958922540277, -0.03744230643343394, 0.013814579589352732, 0.01783065441244216, -0.6493272080616425, 0.20239239382808324, 0.5637275588452099, 0.243836903276757, 0.23282406272287987, -0.2102609569975324, -0.2321468378557078], [0.24673947491397813, 1.0, 0.6507996667280491, 0.23723840019351838, 0.6514929993745142, 0.5443413558891584, 0.9798239052347385, 0.7168541393442216, 0.6090887877182735, 0.013779327268354435, 0.21987422950593646, 0.09782497186892049, -0.11234295350128225, 0.2480330664757158, 0.4909121973267493, 0.2964469592375873, 0.4415290881871007, 0.48260691224313995, 0.33830139143424953, 0.6296678921767406, 0.6272347301103788, 0.3519304153712528, 0.1928582184700881, -0.7631910615169571, -0.16136793538251534, 0.7871948529419658, 0.0589334582292538, 0.004809615120440435, -0.07325607300641665, 0.04240972759459293, 0.030054608328366407, 0.27111737739356384, -0.37656828808083825], [0.8264606208294916, 0.6507996667280491, 1.0, 0.8161233349410527, 0.999104994100068, 0.971584852684071, 0.6920909997095896, 0.9741817888612448, 0.9557275951296712, 0.0012124638769354028, 0.10261720923516729, 0.3195983331517227, 0.06523236887204308, 0.11277212339029269, 0.46880629333699014, 0.41398501008905886, 0.5101708876194947, 0.5227497580254298, 0.43303338862365776, 0.5152483219994549, 0.5194915077585068, 0.15754649978292518, -0.20489465846264848, -0.37555276529366216, -0.05059155772251129, 0.3597989695050238, -0.44727599103944105, 0.16998268276634915, 0.3591774649590733, 0.1864274258095615, 0.1830206321507009, -0.06084867381034115, -0.2948982749093637], [0.968103997411089, 0.23723840019351838, 0.8161233349410527, 1.0, 0.8163045495895697, 0.8705123323495255, 0.2809212564777575, 0.7064582260398368, 0.7629314825231891, -0.007225535580163918, 0.009720589243021487, 0.36616896986301933, 0.15129636449079023, 0.010197030103282403, 0.31935228247047154, 0.32459504662829747, 0.3544860172246109, 0.3503528174620447, 0.3203299680705391, 0.26683894346363934, 0.27286257583091933, 0.002852341560838162, -0.31947528632149524, -0.03545799351781017, 0.01725520025815402, 0.012893831702433668, -0.6278464018186047, 0.1372764170782222, 0.60075882882873, 0.23468725303127846, 0.22425323338306308, -0.20634651716823613, -0.21912236727938975], [0.8261839267575941, 0.6514929993745142, 0.999104994100068, 0.8163045495895697, 1.0, 0.9714294026335367, 0.6933145998038428, 0.9742597865154126, 0.9550526274621957, -4.366515882430437e-05, 0.10269799460500303, 0.3190116294010489, 0.06487716875604223, 0.11318014767185407, 0.46930985808829095, 0.4128627029069421, 0.5106465385186568, 0.5226473772205817, 0.43285605546812955, 0.5159607322991684, 0.5197335423375478, 0.15810039642387086, -0.20007986234276773, -0.3753941377432183, -0.05208357887454562, 0.3610947817631769, -0.44666222995876576, 0.17016947871189894, 0.35828424721783236, 0.18675094471437545, 0.18262682965208013, -0.06081920577782405, -0.2948640659549553], [0.8329551368697643, 0.5443413558891584, 0.971584852684071, 0.8705123323495255, 0.9714294026335367, 1.0, 0.591838917773528, 0.943279979569241, 0.976311997709901, -0.0008199963417565758, 0.07055233846915253, 0.32118127850900935, 0.08281073935280951, 0.08591854247366074, 0.4195852181231613, 0.4016291091167322, 0.47910916229838996, 0.48216028498920166, 0.413945401985833, 0.44926504998410366, 0.4542185537587962, 0.11581136158976209, -0.22559803494271718, -0.28956001860542946, -0.027799850688649645, 0.2666918761619192, -0.4868484276580345, 0.13652022778792106, 0.43718204258550464, 0.18842917746800963, 0.18921268728791288, -0.10371998256574284, -0.2546775806450459], [0.28972430013783385, 0.9798239052347385, 0.6920909997095896, 0.2809212564777575, 0.6933145998038428, 0.591838917773528, 1.0, 0.7763876236753195, 0.6714834747365426, 0.014002465211667035, 0.21371292299847253, 0.11552732090595876, -0.09845595673952634, 0.2783266213795512, 0.5144472313588228, 0.2700194598315499, 0.4351479647243336, 0.4850987799545087, 0.3228315406203701, 0.6520313786420349, 0.6483046646145301, 0.33277660467425685, 0.16672079838541454, -0.6477234585948888, -0.2808685679404608, 0.8057718738814815, 0.008739218075596395, 0.01962028951180728, -0.028883352763437408, 0.04870319041643038, 0.03591716649619229, 0.2432221173257656, -0.35711822495685847], [0.7050251415052948, 0.7168541393442216, 0.9741817888612448, 0.7064582260398368, 0.9742597865154126, 0.943279979569241, 0.7763876236753195, 1.0, 0.974743831630852, 0.004478281493445422, 0.12027626099798366, 0.2733702609588143, 0.030578494699395286, 0.16260941087738232, 0.4829396273009106, 0.3794385350834015, 0.49897134096164814, 0.5184300288857341, 0.4111182966263771, 0.5508166737595369, 0.552910167190139, 0.18798595561893655, -0.13931315129471694, -0.39410651188691237, -0.14295002769600637, 0.4635381343811918, -0.35841643517901006, 0.14217019901023947, 0.28214205191366354, 0.15052556502275072, 0.14938522178524932, -0.011890581259552627, -0.2816170803030248], [0.7237742233918172, 0.6090887877182735, 0.9557275951296712, 0.7629314825231891, 0.9550526274621957, 0.976311997709901, 0.6714834747365426, 0.974743831630852, 1.0, 0.003597971479578409, 0.08666521741975955, 0.2810876411065623, 0.0529895332214611, 0.12884766158761515, 0.43279242115563316, 0.3774582319405636, 0.473416070598031, 0.4821784735932278, 0.39979827987252936, 0.4823242047785849, 0.48554676060865554, 0.14501805816315513, -0.17741633011462576, -0.316389424453676, -0.10265626987513667, 0.36054969798542325, -0.4078925727712864, 0.121531765167998, 0.3594643779592965, 0.1557575701167035, 0.16083217614164116, -0.060309801589733814, -0.2434911660934916], [-0.005242266994018822, 0.013779327268354435, 0.0012124638769354028, -0.007225535580163918, -4.366515882430437e-05, -0.0008199963417565758, 0.014002465211667035, 0.004478281493445422, 0.003597971479578409, 1.0, 0.0018193906134190093, 0.0013790513218355396, -0.01034891712761414, 0.0075149799091999234, 0.008882737146285997, 0.016327823070706123, 0.013092839264555147, 0.0008067457759128009, 0.008507162405232742, 0.007124396986724594, 0.010105418366566175, 0.011901894766838502, 0.008544643224947239, -0.004744965758850031, -0.007583576103039617, 0.011189259276385852, 0.0032507651194556174, -0.007754852914267272, 0.0036031674135729897, 0.015973079031173832, -0.0016318725186135986, -0.0008437084888754001, -0.01319936726545174], [0.015675375704983785, 0.21987422950593646, 0.10261720923516729, 0.009720589243021487, 0.10269799460500303, 0.07055233846915253, 0.21371292299847253, 0.12027626099798366, 0.08666521741975955, 0.0018193906134190093, 1.0, 0.01695661453202187, -0.21055006112684216, 0.008391611911217043, 0.14299625086621018, -0.0385763901686064, 0.06666279065142021, 0.059513871482029225, -0.060576839406188035, 0.10544501753678828, 0.11984236746151568, 0.15625775052783097, 0.1505410534156757, -0.18251949495535458, -0.10827563872848943, 0.25492331502717946, 0.13775207088551514, -0.0464907545657889, -0.11620511425710835, -0.01623474200582214, -0.024359419683712323, 0.17132216591713703, -0.15298719260173027], [0.3818758374863098, 0.09782497186892049, 0.3195983331517227, 0.36616896986301933, 0.3190116294010489, 0.32118127850900935, 0.11552732090595876, 0.2733702609588143, 0.2810876411065623, 0.0013790513218355396, 0.01695661453202187, 1.0, 0.45226888584550023, 0.018397189302703662, 0.1425612874681736, 0.14334606167364233, 0.14184917072520303, 0.1535564364182745, 0.12020601780298455, 0.12448262672518244, 0.11810820943292764, -0.013956696136191696, -0.14998192562006138, -0.0002855204740384597, -0.0010430787434336079, 0.0012346095228208073, -0.2802019157901561, 0.08306706395255747, 0.24733370647615796, 0.11140561212215645, 0.08232738919649572, -0.08320661736633733, -0.09694798339506473], [0.16342735045857737, -0.11234295350128225, 0.06523236887204308, 0.15129636449079023, 0.06487716875604223, 0.08281073935280951, -0.09845595673952634, 0.030578494699395286, 0.0529895332214611, -0.01034891712761414, -0.21055006112684216, 0.45226888584550023, 1.0, -0.001077812708067608, -0.024306661314620996, 0.08078553224088346, 0.023638813060609963, 0.013899668260943368, 0.06305315799997827, -0.01649868035801052, -0.03837492560091315, -0.11013068597336993, -0.16312843938822, 0.13838288994798562, 0.05159321756471237, -0.16410089031864167, -0.2297147909213749, 0.06922205672629726, 0.2016993304297039, 0.05236890928127545, 0.06113408322897806, -0.1492739811934336, 0.05644841359590588], [0.007936580687400586, 0.2480330664757158, 0.11277212339029269, 0.010197030103282403, 0.11318014767185407, 0.08591854247366074, 0.2783266213795512, 0.16260941087738232, 0.12884766158761515, 0.0075149799091999234, 0.008391611911217043, 0.018397189302703662, -0.001077812708067608, 1.0, 0.2795295400049995, -0.09167570469500017, -0.05213341919796151, -0.07007561533228862, -0.09513849428922488, -0.021382711870578185, -0.03347749718236395, 0.016696123642784135, 0.011691398865422323, 0.1718171065699321, -0.45225528090657086, 0.29018311793843365, -0.0012425134067734023, -0.0031417807184111624, 0.0044422513152849235, 0.008271245210923577, -0.006916252198127548, 0.0027471183312986857, -0.004462839400732194], [0.332360049569408, 0.4909121973267493, 0.46880629333699014, 0.31935228247047154, 0.46930985808829095, 0.4195852181231613, 0.5144472313588228, 0.4829396273009106, 0.43279242115563316, 0.008882737146285997, 0.14299625086621018, 0.1425612874681736, -0.024306661314620996, 0.2795295400049995, 1.0, 0.0985919934252315, 0.2022283972825372, 0.201732824517757, 0.10042125595413272, 0.25780350066730157, 0.2591943175468362, 0.1637457730112591, 0.040032739872523634, -0.21079354712189471, -0.2003183214156725, 0.3664202566051166, -0.08855832218643561, -0.003594461398434266, 0.10661820819152797, 0.07542871730303179, 0.06031899094084278, 0.08358299305536028, -0.22767156949803466], [0.32831731240661505, 0.2964469592375873, 0.41398501008905886, 0.32459504662829747, 0.4128627029069421, 0.4016291091167322, 0.2700194598315499, 0.3794385350834015, 0.3774582319405636, 0.016327823070706123, -0.0385763901686064, 0.14334606167364233, 0.08078553224088346, -0.09167570469500017, 0.0985919934252315, 1.0, 0.28328454262626757, 0.274875003842449, 0.35445796164509147, 0.1755144708953688, 0.18742584957299618, -0.004051250607988492, -0.17126992353351678, -0.33279949932167546, 0.3203433737595294, -0.03050626904076453, -0.24684428487400414, 0.10065777311969464, 0.19169819815673583, 0.09436639279979232, 0.11547320256635303, -0.11229466175861408, -0.07991768713640306], [0.3610600281117624, 0.4415290881871007, 0.5101708876194947, 0.3544860172246109, 0.5106465385186568, 0.47910916229838996, 0.4351479647243336, 0.49897134096164814, 0.473416070598031, 0.013092839264555147, 0.06666279065142021, 0.14184917072520303, 0.023638813060609963, -0.05213341919796151, 0.2022283972825372, 0.28328454262626757, 1.0, 0.30305766643807824, 0.29370469187781045, 0.2816010622259753, 0.27452301070772545, 0.12705603268686044, -0.08230696876508349, -0.3809903317320751, 0.15676460995441888, 0.16594028590307677, -0.16439302987919688, 0.08411316021806066, 0.11139068731904943, 0.0869415675760231, 0.09045518641091457, -0.00036426636786500763, -0.17407470231312427], [0.36142893937055787, 0.48260691224313995, 0.5227497580254298, 0.3503528174620447, 0.5226473772205817, 0.48216028498920166, 0.4850987799545087, 0.5184300288857341, 0.4821784735932278, 0.0008067457759128009, 0.059513871482029225, 0.1535564364182745, 0.013899668260943368, -0.07007561533228862, 0.201732824517757, 0.274875003842449, 0.30305766643807824, 1.0, 0.33285005080469243, 0.3899237975094597, 0.4023088228216018, 0.10407904724402045, -0.06619251684228997, -0.3801513548956378, 0.14514955473692903, 0.17635617323471664, -0.22598757731112262, 0.10291089629343353, 0.16524753554250074, 0.08304690185342707, 0.11125168129784246, -0.003308493511411254, -0.18732483013668594], [0.32528359550205393, 0.33830139143424953, 0.43303338862365776, 0.3203299680705391, 0.43285605546812955, 0.413945401985833, 0.3228315406203701, 0.4111182966263771, 0.39979827987252936, 0.008507162405232742, -0.060576839406188035, 0.12020601780298455, 0.06305315799997827, -0.09513849428922488, 0.10042125595413272, 0.35445796164509147, 0.29370469187781045, 0.33285005080469243, 1.0, 0.277548599200549, 0.2801552432906342, 0.03753587307318976, -0.16471590834411207, -0.33569508671869736, 0.3121832985222757, -0.020298967520709605, -0.28549086901033593, 0.09625836225380952, 0.24092408252256528, 0.10047200087443556, 0.11702370730984143, -0.11480726996437085, -0.08463055196615278], [0.28009772993820486, 0.6296678921767406, 0.5152483219994549, 0.26683894346363934, 0.5159607322991684, 0.44926504998410366, 0.6520313786420349, 0.5508166737595369, 0.4823242047785849, 0.007124396986724594, 0.10544501753678828, 0.12448262672518244, -0.01649868035801052, -0.021382711870578185, 0.25780350066730157, 0.1755144708953688, 0.2816010622259753, 0.3899237975094597, 0.277548599200549, 1.0, 0.5333800979319763, 0.22424119793848596, 0.06325398027519404, -0.41495062156578044, 0.014973379079172382, 0.3297441152730512, -0.11254989712217289, 0.061929689963200855, 0.07212357537070835, 0.04612070051242859, 0.040010276337768505, 0.1447470086556032, -0.2477115493728633], [0.28526361779277526, 0.6272347301103788, 0.5194915077585068, 0.27286257583091933, 0.5197335423375478, 0.4542185537587962, 0.6483046646145301, 0.552910167190139, 0.48554676060865554, 0.010105418366566175, 0.11984236746151568, 0.11810820943292764, -0.03837492560091315, -0.03347749718236395, 0.2591943175468362, 0.18742584957299618, 0.27452301070772545, 0.4023088228216018, 0.2801552432906342, 0.5333800979319763, 1.0, 0.21158250423808916, 0.06085993668146301, -0.41844975538334045, 0.02562310861129719, 0.3224574540559222, -0.11786687989290552, 0.06477997824381859, 0.07560257919382919, 0.04875484714119087, 0.048398314068082315, 0.13742008269944622, -0.2502897149395328], [0.004776972324833932, 0.3519304153712528, 0.15754649978292518, 0.002852341560838162, 0.15810039642387086, 0.11581136158976209, 0.33277660467425685, 0.18798595561893655, 0.14501805816315513, 0.011901894766838502, 0.15625775052783097, -0.013956696136191696, -0.11013068597336993, 0.016696123642784135, 0.1637457730112591, -0.004051250607988492, 0.12705603268686044, 0.10407904724402045, 0.03753587307318976, 0.22424119793848596, 0.21158250423808916, 1.0, 0.19145432108006671, -0.3205922451174622, -0.06338966821876392, 0.32647017160380964, 0.16829626845602835, -0.052278164693773076, -0.1462807050684952, -0.017468900682392235, -0.013726285095880284, 0.20842668228002995, -0.20398064814312206], [-0.35445958922540277, 0.1928582184700881, -0.20489465846264848, -0.31947528632149524, -0.20007986234276773, -0.22559803494271718, 0.16672079838541454, -0.13931315129471694, -0.17741633011462576, 0.008544643224947239, 0.1505410534156757, -0.14998192562006138, -0.16312843938822, 0.011691398865422323, 0.040032739872523634, -0.17126992353351678, -0.08230696876508349, -0.06619251684228997, -0.16471590834411207, 0.06325398027519404, 0.06085993668146301, 0.19145432108006671, 1.0, -0.22757762044656818, -0.12414142842590645, 0.30746259069818205, 0.40456455007784087, -0.17822502328994053, -0.30155233962397837, -0.1181359978280296, -0.1346868372340906, 0.30145463790858057, -0.09077284582582087], [-0.03744230643343394, -0.7631910615169571, -0.37555276529366216, -0.03545799351781017, -0.3753941377432183, -0.28956001860542946, -0.6477234585948888, -0.39410651188691237, -0.316389424453676, -0.004744965758850031, -0.18251949495535458, -0.0002855204740384597, 0.13838288994798562, 0.1718171065699321, -0.21079354712189471, -0.33279949932167546, -0.3809903317320751, -0.3801513548956378, -0.33569508671869736, -0.41495062156578044, -0.41844975538334045, -0.3205922451174622, -0.22757762044656818, 1.0, -0.3799117751052334, -0.4657363343235562, -0.217823505545489, 0.038061459724733744, 0.21754205606911817, -0.0010943032992050863, 0.0018701145623306124, -0.2846082097225956, 0.3196937439459745], [0.013814579589352732, -0.16136793538251534, -0.05059155772251129, 0.01725520025815402, -0.05208357887454562, -0.027799850688649645, -0.2808685679404608, -0.14295002769600637, -0.10265626987513667, -0.007583576103039617, -0.10827563872848943, -0.0010430787434336079, 0.05159321756471237, -0.45225528090657086, -0.2003183214156725, 0.3203433737595294, 0.15676460995441888, 0.14514955473692903, 0.3121832985222757, 0.014973379079172382, 0.02562310861129719, -0.06338966821876392, -0.12414142842590645, -0.3799117751052334, 1.0, -0.6416356650534906, -0.06522632996531619, 0.04729967349374622, 0.030923714855574744, 0.02475954021490298, 0.05122176476699011, -0.10429333541089097, 0.04275388869901973], [0.01783065441244216, 0.7871948529419658, 0.3597989695050238, 0.012893831702433668, 0.3610947817631769, 0.2666918761619192, 0.8057718738814815, 0.4635381343811918, 0.36054969798542325, 0.011189259276385852, 0.25492331502717946, 0.0012346095228208073, -0.16410089031864167, 0.29018311793843365, 0.3664202566051166, -0.03050626904076453, 0.16594028590307677, 0.17635617323471664, -0.020298967520709605, 0.3297441152730512, 0.3224574540559222, 0.32647017160380964, 0.30746259069818205, -0.4657363343235562, -0.6416356650534906, 1.0, 0.24301351814831268, -0.07680902975096314, -0.20996452908904545, -0.022778855291876635, -0.050551991558036385, 0.3357634768102139, -0.3059839224841771], [-0.6493272080616425, 0.0589334582292538, -0.44727599103944105, -0.6278464018186047, -0.44666222995876576, -0.4868484276580345, 0.008739218075596395, -0.35841643517901006, -0.4078925727712864, 0.0032507651194556174, 0.13775207088551514, -0.2802019157901561, -0.2297147909213749, -0.0012425134067734023, -0.08855832218643561, -0.24684428487400414, -0.16439302987919688, -0.22598757731112262, -0.28549086901033593, -0.11254989712217289, -0.11786687989290552, 0.16829626845602835, 0.40456455007784087, -0.217823505545489, -0.06522632996531619, 0.24301351814831268, 1.0, -0.5700527848944215, -0.6219327447713561, -0.18015909738683936, -0.20496021669474843, 0.33087881370583405, 0.006208692442055045], [0.20239239382808324, 0.004809615120440435, 0.16998268276634915, 0.1372764170782222, 0.17016947871189894, 0.13652022778792106, 0.01962028951180728, 0.14217019901023947, 0.121531765167998, -0.007754852914267272, -0.0464907545657889, 0.08306706395255747, 0.06922205672629726, -0.0031417807184111624, -0.003594461398434266, 0.10065777311969464, 0.08411316021806066, 0.10291089629343353, 0.09625836225380952, 0.061929689963200855, 0.06477997824381859, -0.052278164693773076, -0.17822502328994053, 0.038061459724733744, 0.04729967349374622, -0.07680902975096314, -0.5700527848944215, 1.0, -0.28884268256780254, 0.05762874929825631, 0.06758968145792751, -0.10954646682258033, 0.0001971241890499786], [0.5637275588452099, -0.07325607300641665, 0.3591774649590733, 0.60075882882873, 0.35828424721783236, 0.43718204258550464, -0.028883352763437408, 0.28214205191366354, 0.3594643779592965, 0.0036031674135729897, -0.11620511425710835, 0.24733370647615796, 0.2016993304297039, 0.0044422513152849235, 0.10661820819152797, 0.19169819815673583, 0.11139068731904943, 0.16524753554250074, 0.24092408252256528, 0.07212357537070835, 0.07560257919382919, -0.1462807050684952, -0.30155233962397837, 0.21754205606911817, 0.030923714855574744, -0.20996452908904545, -0.6219327447713561, -0.28884268256780254, 1.0, 0.1550042180767056, 0.17440993895950943, -0.28114743393395547, -0.007422540118618717], [0.243836903276757, 0.04240972759459293, 0.1864274258095615, 0.23468725303127846, 0.18675094471437545, 0.18842917746800963, 0.04870319041643038, 0.15052556502275072, 0.1557575701167035, 0.015973079031173832, -0.01623474200582214, 0.11140561212215645, 0.05236890928127545, 0.008271245210923577, 0.07542871730303179, 0.09436639279979232, 0.0869415675760231, 0.08304690185342707, 0.10047200087443556, 0.04612070051242859, 0.04875484714119087, -0.017468900682392235, -0.1181359978280296, -0.0010943032992050863, 0.02475954021490298, -0.022778855291876635, -0.18015909738683936, 0.05762874929825631, 0.1550042180767056, 1.0, -0.27842319712065355, -0.3772703602158491, -0.28809669563791657], [0.23282406272287987, 0.030054608328366407, 0.1830206321507009, 0.22425323338306308, 0.18262682965208013, 0.18921268728791288, 0.03591716649619229, 0.14938522178524932, 0.16083217614164116, -0.0016318725186135986, -0.024359419683712323, 0.08232738919649572, 0.06113408322897806, -0.006916252198127548, 0.06031899094084278, 0.11547320256635303, 0.09045518641091457, 0.11125168129784246, 0.11702370730984143, 0.040010276337768505, 0.048398314068082315, -0.013726285095880284, -0.1346868372340906, 0.0018701145623306124, 0.05122176476699011, -0.050551991558036385, -0.20496021669474843, 0.06758968145792751, 0.17440993895950943, -0.27842319712065355, 1.0, -0.37397801626928234, -0.28558254792863036], [-0.2102609569975324, 0.27111737739356384, -0.06084867381034115, -0.20634651716823613, -0.06081920577782405, -0.10371998256574284, 0.2432221173257656, -0.011890581259552627, -0.060309801589733814, -0.0008437084888754001, 0.17132216591713703, -0.08320661736633733, -0.1492739811934336, 0.0027471183312986857, 0.08358299305536028, -0.11229466175861408, -0.00036426636786500763, -0.003308493511411254, -0.11480726996437085, 0.1447470086556032, 0.13742008269944622, 0.20842668228002995, 0.30145463790858057, -0.2846082097225956, -0.10429333541089097, 0.3357634768102139, 0.33087881370583405, -0.10954646682258033, -0.28114743393395547, -0.3772703602158491, -0.37397801626928234, 1.0, -0.3869714587097063], [-0.2321468378557078, -0.37656828808083825, -0.2948982749093637, -0.21912236727938975, -0.2948640659549553, -0.2546775806450459, -0.35711822495685847, -0.2816170803030248, -0.2434911660934916, -0.01319936726545174, -0.15298719260173027, -0.09694798339506473, 0.05644841359590588, -0.004462839400732194, -0.22767156949803466, -0.07991768713640306, -0.17407470231312427, -0.18732483013668594, -0.08463055196615278, -0.2477115493728633, -0.2502897149395328, -0.20398064814312206, -0.09077284582582087, 0.3196937439459745, 0.04275388869901973, -0.3059839224841771, 0.006208692442055045, 0.0001971241890499786, -0.007422540118618717, -0.28809669563791657, -0.28558254792863036, -0.3869714587097063, 1.0]]}],
                        {"autosize": false, "height": 720, "margin": {"b": 210, "l": 210, "r": 0, "t": 25}, "title": {"text": "Matriz B de correlaciones"}, "width": 800, "xaxis": {"tickfont": {"size": 9}}, "yaxis": {"tickfont": {"size": 9}}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('863349f7-a5b8-4455-a818-e979891e9393');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


Podemos observar como en la matriz de correlaciones aparece nuestro feature engineering, marcandonos con un color mas azul estas variables correlacionadas.

## Train and Test

![png](linea.png)

Antes de empezar a lanzar los modelos debemos separar nuestros datos de entrenamiento y testing.


```python
from sklearn.model_selection import train_test_split
train_A,test_A = train_test_split(churn_A_final, test_size = .25 ,random_state = 111)
train_B,test_B = train_test_split(churn_B_final, test_size = .25 ,random_state = 111)
```

A continuacion separamos las variables a predecir y col_id que no interviene en nuestro entrenamiento


```python
# Matriz A
cols_A    = [i for i in churn_A_final.columns if i not in col_id + col_churn]
train_XA = train_A[cols_A]
train_YA = train_A[col_churn]
test_XA  = test_A[cols_A]
test_YA  = test_A[col_churn]

# Matriz B
cols_B    = [i for i in churn_B_final.columns if i not in col_id + col_churn]
train_XB = train_B[cols_B]
train_YB = train_B[col_churn]
test_XB  = test_B[cols_B]
test_YB  = test_B[col_churn]
```

Preparamos una funcion que nos enseñe las carecteristicas importantes de cada entrenamiento, para posteriromente comparar y sacar conclusiones


```python
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import roc_auc_score,roc_curve,scorer
import statsmodels.api as sm

def churn_prediction(algoritmo, training_x, testing_x, training_y, testing_y):
    algoritmo.fit(training_x,training_y)
    prediccion   = algoritmo.predict(testing_x)
    probabilidad = algoritmo.predict_proba(testing_x)
    conf_matrix = confusion_matrix(testing_y,prediccion)
    model_roc_auc = roc_auc_score(testing_y,prediccion)
    fpr,tpr,thresholds = roc_curve(testing_y,probabilidad[:,1])
    
    print (algoritmo)
    print ("Accuracy   Score : ",accuracy_score(testing_y,prediccion))
    print ("Area bajo la curva : ",model_roc_auc,"\n")
    
    #Curva ROC
    grafico1 = go.Scatter(x = fpr,y = tpr,
                        name = "ROC : " + str(model_roc_auc),
                        line = dict(color = ('rgb(22, 96, 167)'),width = 2),
                       )
    grafico1B = go.Scatter(x = [0,1],y=[0,1],
                        line = dict(color = ('rgb(205, 12, 24)'),width = 2,
                        dash = 'dot'))
    
    #plot confusion matrix
    grafico2 = go.Heatmap(z = conf_matrix ,x = ["Not churn","Churn"],
                        y = ["Not churn","Churn"],
                        showscale  = False,colorscale = [[0.0, 'rgb(165,0,38)'], [0.1111111111111111, 'rgb(215,48,39)'], [0.2222222222222222, 'rgb(244,109,67)'],
        [0.3333333333333333, 'rgb(253,174,97)'], [0.4444444444444444, 'rgb(254,224,144)'], [0.5555555555555556, 'rgb(224,243,248)'],
        [0.6666666666666666, 'rgb(171,217,233)'],[0.7777777777777778, 'rgb(116,173,209)'], [0.8888888888888888, 'rgb(69,117,180)'],
        [1.0, 'rgb(49,54,149)']],name = "matrix",
                        xaxis = "x2",yaxis = "y2"
                       )
    
    layout = go.Layout(dict(title="Caracteristicas del modelo" ,
                            autosize = False,height = 600,width = 900,
                            showlegend = False,
                            plot_bgcolor  = "rgb(243,243,243)",
                            paper_bgcolor = "rgb(243,243,243)",
                            xaxis = dict(title = "Ratio falso positivo",
                                         gridcolor = 'rgb(255, 255, 255)',
                                         domain=[0, 0.6],
                                         ticklen=5,gridwidth=2),
                            yaxis = dict(title = "Ratio verdadero positivo",
                                         gridcolor = 'rgb(255, 255, 255)',
                                         zerolinewidth=1,
                                         ticklen=5,gridwidth=2),
                            margin = dict(b=200),
                            xaxis2=dict(domain=[0.7, 1],tickangle = 90,
                                        gridcolor = 'rgb(255, 255, 255)'),
                            yaxis2=dict(anchor='x2',gridcolor = 'rgb(255, 255, 255)')
                           )
                  )
    data = [grafico1,grafico1B,grafico2]
    fig = go.Figure(data=data,layout=layout)
    
    py.iplot(fig)

   
```

## Modelos

![png](linea.png)

Para este apartado se a decidido utilizar los modelos mas utilizados hoy en dia, los cuales son:
* LogisticRegression
* RandomForestClassifier
* GaussianNB
* XGBClassifier

### LogisticRegression


```python
from sklearn.linear_model import LogisticRegression

LogisticRegression  = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
          verbose=0, warm_start=False)

churn_prediction(LogisticRegression, train_XA, test_XA, train_YA, test_YA)
churn_prediction(LogisticRegression, train_XB, test_XB, train_YB, test_YB)

```

    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)
    Accuracy   Score :  0.8020477815699659
    Area bajo la curva :  0.72190658597824 
    


    /anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py:761: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    



<div>
        
        
            <div id="f11a3922-63e0-4aa2-8a2e-172433d1fb9b" class="plotly-graph-div" style="height:600px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("f11a3922-63e0-4aa2-8a2e-172433d1fb9b")) {
                    Plotly.newPlot(
                        'f11a3922-63e0-4aa2-8a2e-172433d1fb9b',
                        [{"line": {"color": "rgb(22, 96, 167)", "width": 2}, "name": "ROC : 0.72190658597824", "type": "scatter", "uid": "55682e08-4c70-413b-b197-04bb9aa83b42", "x": [0.0, 0.0, 0.0, 0.0007886435331230284, 0.0007886435331230284, 0.0015772870662460567, 0.0015772870662460567, 0.002365930599369085, 0.002365930599369085, 0.0031545741324921135, 0.0031545741324921135, 0.003943217665615142, 0.003943217665615142, 0.00473186119873817, 0.00473186119873817, 0.005520504731861199, 0.005520504731861199, 0.006309148264984227, 0.006309148264984227, 0.007097791798107256, 0.007097791798107256, 0.008675078864353312, 0.008675078864353312, 0.00946372239747634, 0.00946372239747634, 0.01025236593059937, 0.01025236593059937, 0.011041009463722398, 0.011041009463722398, 0.012618296529968454, 0.012618296529968454, 0.014195583596214511, 0.014195583596214511, 0.015772870662460567, 0.015772870662460567, 0.016561514195583597, 0.016561514195583597, 0.017350157728706624, 0.017350157728706624, 0.021293375394321766, 0.021293375394321766, 0.022082018927444796, 0.022082018927444796, 0.022870662460567823, 0.022870662460567823, 0.02365930599369085, 0.02365930599369085, 0.02444794952681388, 0.02444794952681388, 0.025236593059936908, 0.025236593059936908, 0.026025236593059938, 0.026025236593059938, 0.026813880126182965, 0.026813880126182965, 0.027602523659305992, 0.027602523659305992, 0.02917981072555205, 0.02917981072555205, 0.02996845425867508, 0.02996845425867508, 0.030757097791798107, 0.030757097791798107, 0.031545741324921134, 0.031545741324921134, 0.032334384858044164, 0.032334384858044164, 0.033123028391167195, 0.033123028391167195, 0.03391167192429022, 0.03391167192429022, 0.03470031545741325, 0.03470031545741325, 0.03548895899053628, 0.03548895899053628, 0.03627760252365931, 0.03627760252365931, 0.038643533123028394, 0.038643533123028394, 0.03943217665615142, 0.03943217665615142, 0.04022082018927445, 0.04022082018927445, 0.04100946372239748, 0.04100946372239748, 0.04258675078864353, 0.04258675078864353, 0.04337539432176656, 0.04337539432176656, 0.04416403785488959, 0.04416403785488959, 0.044952681388012616, 0.044952681388012616, 0.04652996845425868, 0.04652996845425868, 0.0473186119873817, 0.0473186119873817, 0.04810725552050473, 0.04810725552050473, 0.04889589905362776, 0.04889589905362776, 0.04968454258675079, 0.04968454258675079, 0.051261829652996846, 0.051261829652996846, 0.05441640378548896, 0.05441640378548896, 0.055205047318611984, 0.055205047318611984, 0.055993690851735015, 0.055993690851735015, 0.0583596214511041, 0.0583596214511041, 0.05993690851735016, 0.05993690851735016, 0.06072555205047318, 0.06072555205047318, 0.061514195583596214, 0.061514195583596214, 0.062302839116719244, 0.062302839116719244, 0.06309148264984227, 0.06309148264984227, 0.06466876971608833, 0.06466876971608833, 0.06624605678233439, 0.06624605678233439, 0.06861198738170347, 0.06861198738170347, 0.0694006309148265, 0.0694006309148265, 0.07018927444794952, 0.07018927444794952, 0.07097791798107256, 0.07097791798107256, 0.07176656151419558, 0.07176656151419558, 0.07255520504731862, 0.07255520504731862, 0.07334384858044164, 0.07334384858044164, 0.07413249211356467, 0.07413249211356467, 0.0749211356466877, 0.0749211356466877, 0.07570977917981073, 0.07570977917981073, 0.07649842271293375, 0.07649842271293375, 0.07728706624605679, 0.07728706624605679, 0.07807570977917981, 0.07807570977917981, 0.07965299684542587, 0.07965299684542587, 0.08280757097791798, 0.08280757097791798, 0.08517350157728706, 0.08517350157728706, 0.08753943217665615, 0.08753943217665615, 0.08832807570977919, 0.08832807570977919, 0.08990536277602523, 0.08990536277602523, 0.0914826498422713, 0.0914826498422713, 0.09227129337539432, 0.09227129337539432, 0.09384858044164038, 0.09384858044164038, 0.0946372239747634, 0.0946372239747634, 0.09542586750788644, 0.09542586750788644, 0.09700315457413249, 0.09700315457413249, 0.09858044164037855, 0.09858044164037855, 0.09936908517350158, 0.09936908517350158, 0.10173501577287067, 0.10173501577287067, 0.10331230283911672, 0.10331230283911672, 0.10410094637223975, 0.10410094637223975, 0.10488958990536278, 0.10488958990536278, 0.10646687697160884, 0.10646687697160884, 0.10725552050473186, 0.10725552050473186, 0.111198738170347, 0.111198738170347, 0.11277602523659307, 0.11277602523659307, 0.11435331230283911, 0.11435331230283911, 0.11829652996845426, 0.11829652996845426, 0.12302839116719243, 0.12302839116719243, 0.12460567823343849, 0.12460567823343849, 0.1253943217665615, 0.1253943217665615, 0.12618296529968454, 0.12618296529968454, 0.12854889589905363, 0.12854889589905363, 0.12933753943217666, 0.12933753943217666, 0.1332807570977918, 0.1332807570977918, 0.13485804416403785, 0.13485804416403785, 0.13643533123028392, 0.13643533123028392, 0.138801261829653, 0.138801261829653, 0.13958990536277602, 0.13958990536277602, 0.14195583596214512, 0.14195583596214512, 0.14511041009463724, 0.14511041009463724, 0.1474763406940063, 0.1474763406940063, 0.15063091482649843, 0.15063091482649843, 0.15141955835962145, 0.15141955835962145, 0.15220820189274448, 0.15220820189274448, 0.15378548895899052, 0.15378548895899052, 0.1553627760252366, 0.1553627760252366, 0.15615141955835962, 0.15615141955835962, 0.15772870662460567, 0.15772870662460567, 0.15851735015772872, 0.15851735015772872, 0.16167192429022081, 0.16167192429022081, 0.1640378548895899, 0.1640378548895899, 0.16482649842271294, 0.16482649842271294, 0.16640378548895898, 0.16640378548895898, 0.16876971608832808, 0.16876971608832808, 0.1695583596214511, 0.1695583596214511, 0.17429022082018927, 0.17429022082018927, 0.1750788643533123, 0.1750788643533123, 0.17665615141955837, 0.17665615141955837, 0.18217665615141956, 0.18217665615141956, 0.1829652996845426, 0.1829652996845426, 0.18454258675078863, 0.18454258675078863, 0.18769716088328076, 0.18769716088328076, 0.18848580441640378, 0.18848580441640378, 0.19242902208201892, 0.19242902208201892, 0.19321766561514195, 0.19321766561514195, 0.1971608832807571, 0.1971608832807571, 0.20110410094637224, 0.20110410094637224, 0.20268138801261829, 0.20268138801261829, 0.20662460567823343, 0.20662460567823343, 0.20741324921135645, 0.20741324921135645, 0.2082018927444795, 0.2082018927444795, 0.21214511041009465, 0.21214511041009465, 0.21293375394321767, 0.21293375394321767, 0.21608832807570977, 0.21608832807570977, 0.221608832807571, 0.221608832807571, 0.222397476340694, 0.222397476340694, 0.22318611987381703, 0.22318611987381703, 0.22555205047318613, 0.22555205047318613, 0.2358044164037855, 0.2358044164037855, 0.23659305993690852, 0.23659305993690852, 0.23895899053627762, 0.23895899053627762, 0.24132492113564669, 0.24132492113564669, 0.2444794952681388, 0.2444794952681388, 0.24526813880126183, 0.24526813880126183, 0.24605678233438485, 0.24605678233438485, 0.25, 0.25, 0.25236593059936907, 0.25236593059936907, 0.2547318611987382, 0.2547318611987382, 0.2555205047318612, 0.2555205047318612, 0.25630914826498424, 0.25630914826498424, 0.25709779179810727, 0.25709779179810727, 0.2610410094637224, 0.2610410094637224, 0.26419558359621453, 0.26419558359621453, 0.27208201892744477, 0.27208201892744477, 0.27365930599369087, 0.27365930599369087, 0.2752365930599369, 0.2752365930599369, 0.277602523659306, 0.277602523659306, 0.27996845425867506, 0.27996845425867506, 0.28154574132492116, 0.28154574132492116, 0.2831230283911672, 0.2831230283911672, 0.28470031545741326, 0.28470031545741326, 0.29258675078864355, 0.29258675078864355, 0.29337539432176657, 0.29337539432176657, 0.29889589905362773, 0.29889589905362773, 0.2996845425867508, 0.2996845425867508, 0.30757097791798105, 0.30757097791798105, 0.31230283911671924, 0.31230283911671924, 0.3186119873817035, 0.3186119873817035, 0.3194006309148265, 0.3194006309148265, 0.32097791798107256, 0.32097791798107256, 0.3217665615141956, 0.3217665615141956, 0.32334384858044163, 0.32334384858044163, 0.3249211356466877, 0.3249211356466877, 0.32965299684542587, 0.32965299684542587, 0.33280757097791797, 0.33280757097791797, 0.333596214511041, 0.333596214511041, 0.334384858044164, 0.334384858044164, 0.3414826498422713, 0.3414826498422713, 0.3422712933753943, 0.3422712933753943, 0.3438485804416404, 0.3438485804416404, 0.34542586750788645, 0.34542586750788645, 0.35173501577287064, 0.35173501577287064, 0.36041009463722395, 0.36041009463722395, 0.36198738170347006, 0.36198738170347006, 0.3627760252365931, 0.3643533123028391, 0.36514195583596215, 0.36514195583596215, 0.3659305993690852, 0.3659305993690852, 0.3698738170347003, 0.3698738170347003, 0.37697160883280756, 0.37697160883280756, 0.3785488958990536, 0.3785488958990536, 0.3840694006309148, 0.3840694006309148, 0.3856466876971609, 0.3856466876971609, 0.3951104100946372, 0.3951104100946372, 0.39826498422712936, 0.39826498422712936, 0.4022082018927445, 0.4022082018927445, 0.4037854889589905, 0.4037854889589905, 0.40930599369085174, 0.40930599369085174, 0.41009463722397477, 0.41009463722397477, 0.4148264984227129, 0.4148264984227129, 0.4282334384858044, 0.4282334384858044, 0.42902208201892744, 0.42902208201892744, 0.4313880126182965, 0.4313880126182965, 0.4361198738170347, 0.4361198738170347, 0.444006309148265, 0.444006309148265, 0.4550473186119874, 0.4550473186119874, 0.4794952681388013, 0.4794952681388013, 0.4802839116719243, 0.4802839116719243, 0.4842271293375394, 0.4842271293375394, 0.48501577287066244, 0.48501577287066244, 0.49290220820189273, 0.49290220820189273, 0.5063091482649842, 0.5063091482649842, 0.5110410094637224, 0.5110410094637224, 0.527602523659306, 0.527602523659306, 0.5370662460567823, 0.5370662460567823, 0.5449526813880127, 0.5449526813880127, 0.5520504731861199, 0.5520504731861199, 0.5615141955835962, 0.5615141955835962, 0.5772870662460567, 0.5772870662460567, 0.5851735015772871, 0.5851735015772871, 0.5962145110410094, 0.5962145110410094, 0.6096214511041009, 0.6096214511041009, 0.6301261829652997, 0.6301261829652997, 0.6324921135646687, 0.6324921135646687, 0.6451104100946372, 0.6451104100946372, 0.6585173501577287, 0.6585173501577287, 0.667981072555205, 0.667981072555205, 0.692429022082019, 0.692429022082019, 0.7476340694006309, 0.7476340694006309, 0.777602523659306, 0.777602523659306, 0.7799684542586751, 0.7799684542586751, 0.7815457413249212, 0.7815457413249212, 0.8233438485804416, 0.8233438485804416, 0.8470031545741324, 0.8470031545741324, 0.9140378548895899, 0.9140378548895899, 0.9321766561514195, 0.9321766561514195, 1.0], "y": [0.0, 0.0020408163265306124, 0.006122448979591836, 0.006122448979591836, 0.00816326530612245, 0.00816326530612245, 0.03469387755102041, 0.03469387755102041, 0.04081632653061224, 0.04081632653061224, 0.044897959183673466, 0.044897959183673466, 0.053061224489795916, 0.053061224489795916, 0.05918367346938776, 0.05918367346938776, 0.09591836734693877, 0.09591836734693877, 0.10408163265306122, 0.10408163265306122, 0.11428571428571428, 0.11428571428571428, 0.12040816326530612, 0.12040816326530612, 0.12244897959183673, 0.12244897959183673, 0.12653061224489795, 0.12653061224489795, 0.13673469387755102, 0.13673469387755102, 0.14489795918367346, 0.14489795918367346, 0.1469387755102041, 0.1469387755102041, 0.15714285714285714, 0.15714285714285714, 0.1836734693877551, 0.1836734693877551, 0.19183673469387755, 0.19183673469387755, 0.19387755102040816, 0.19387755102040816, 0.19591836734693877, 0.19591836734693877, 0.19795918367346937, 0.19795918367346937, 0.20408163265306123, 0.20408163265306123, 0.21020408163265306, 0.21020408163265306, 0.21224489795918366, 0.21224489795918366, 0.21836734693877552, 0.21836734693877552, 0.22653061224489796, 0.22653061224489796, 0.22857142857142856, 0.22857142857142856, 0.23061224489795917, 0.23061224489795917, 0.23265306122448978, 0.23265306122448978, 0.23877551020408164, 0.23877551020408164, 0.24081632653061225, 0.24081632653061225, 0.24285714285714285, 0.24285714285714285, 0.24489795918367346, 0.24489795918367346, 0.27755102040816326, 0.27755102040816326, 0.2795918367346939, 0.2795918367346939, 0.2836734693877551, 0.2836734693877551, 0.2938775510204082, 0.2938775510204082, 0.3020408163265306, 0.3020408163265306, 0.3040816326530612, 0.3040816326530612, 0.3142857142857143, 0.3142857142857143, 0.3163265306122449, 0.3163265306122449, 0.3326530612244898, 0.3326530612244898, 0.3346938775510204, 0.3346938775510204, 0.33877551020408164, 0.33877551020408164, 0.3448979591836735, 0.3448979591836735, 0.3469387755102041, 0.3469387755102041, 0.3489795918367347, 0.3489795918367347, 0.35306122448979593, 0.35306122448979593, 0.35714285714285715, 0.35714285714285715, 0.36122448979591837, 0.36122448979591837, 0.3673469387755102, 0.3673469387755102, 0.3693877551020408, 0.3693877551020408, 0.37755102040816324, 0.37755102040816324, 0.3795918367346939, 0.3795918367346939, 0.3836734693877551, 0.3836734693877551, 0.38571428571428573, 0.38571428571428573, 0.39591836734693875, 0.39591836734693875, 0.3979591836734694, 0.3979591836734694, 0.4, 0.4, 0.4163265306122449, 0.4163265306122449, 0.41836734693877553, 0.41836734693877553, 0.42244897959183675, 0.42244897959183675, 0.42857142857142855, 0.42857142857142855, 0.4306122448979592, 0.4306122448979592, 0.4387755102040816, 0.4387755102040816, 0.44081632653061226, 0.44081632653061226, 0.44285714285714284, 0.44285714285714284, 0.45510204081632655, 0.45510204081632655, 0.45714285714285713, 0.45714285714285713, 0.46122448979591835, 0.46122448979591835, 0.463265306122449, 0.463265306122449, 0.4673469387755102, 0.4673469387755102, 0.47551020408163264, 0.47551020408163264, 0.47959183673469385, 0.47959183673469385, 0.49183673469387756, 0.49183673469387756, 0.49387755102040815, 0.49387755102040815, 0.5020408163265306, 0.5020408163265306, 0.5040816326530613, 0.5040816326530613, 0.5061224489795918, 0.5061224489795918, 0.5102040816326531, 0.5102040816326531, 0.5122448979591837, 0.5122448979591837, 0.5183673469387755, 0.5183673469387755, 0.5204081632653061, 0.5204081632653061, 0.5244897959183673, 0.5244897959183673, 0.5265306122448979, 0.5265306122448979, 0.5285714285714286, 0.5285714285714286, 0.5408163265306123, 0.5408163265306123, 0.5428571428571428, 0.5428571428571428, 0.5469387755102041, 0.5469387755102041, 0.5510204081632653, 0.5510204081632653, 0.5551020408163265, 0.5551020408163265, 0.5591836734693878, 0.5591836734693878, 0.563265306122449, 0.563265306122449, 0.5673469387755102, 0.5673469387755102, 0.5775510204081633, 0.5775510204081633, 0.5795918367346938, 0.5795918367346938, 0.5836734693877551, 0.5836734693877551, 0.5877551020408164, 0.5877551020408164, 0.5918367346938775, 0.5918367346938775, 0.5959183673469388, 0.5959183673469388, 0.6, 0.6, 0.6040816326530613, 0.6040816326530613, 0.6204081632653061, 0.6204081632653061, 0.6285714285714286, 0.6285714285714286, 0.6306122448979592, 0.6306122448979592, 0.6326530612244898, 0.6326530612244898, 0.636734693877551, 0.636734693877551, 0.6428571428571429, 0.6428571428571429, 0.6469387755102041, 0.6469387755102041, 0.6489795918367347, 0.6489795918367347, 0.6510204081632653, 0.6510204081632653, 0.6530612244897959, 0.6530612244897959, 0.6612244897959184, 0.6612244897959184, 0.6632653061224489, 0.6632653061224489, 0.6653061224489796, 0.6653061224489796, 0.673469387755102, 0.673469387755102, 0.6755102040816326, 0.6755102040816326, 0.6775510204081633, 0.6775510204081633, 0.6795918367346939, 0.6795918367346939, 0.6816326530612244, 0.6816326530612244, 0.6836734693877551, 0.6836734693877551, 0.6857142857142857, 0.6857142857142857, 0.6877551020408164, 0.6877551020408164, 0.689795918367347, 0.689795918367347, 0.6918367346938775, 0.6918367346938775, 0.6938775510204082, 0.6938775510204082, 0.6959183673469388, 0.6959183673469388, 0.7, 0.7, 0.7061224489795919, 0.7061224489795919, 0.7081632653061225, 0.7081632653061225, 0.710204081632653, 0.710204081632653, 0.7122448979591837, 0.7122448979591837, 0.7142857142857143, 0.7142857142857143, 0.7183673469387755, 0.7183673469387755, 0.7204081632653061, 0.7204081632653061, 0.7244897959183674, 0.7244897959183674, 0.7285714285714285, 0.7285714285714285, 0.7326530612244898, 0.7326530612244898, 0.7346938775510204, 0.7346938775510204, 0.736734693877551, 0.736734693877551, 0.7387755102040816, 0.7387755102040816, 0.7408163265306122, 0.7408163265306122, 0.7448979591836735, 0.7448979591836735, 0.746938775510204, 0.746938775510204, 0.7489795918367347, 0.7489795918367347, 0.7571428571428571, 0.7571428571428571, 0.7612244897959184, 0.7612244897959184, 0.763265306122449, 0.763265306122449, 0.7653061224489796, 0.7653061224489796, 0.7693877551020408, 0.7693877551020408, 0.7775510204081633, 0.7775510204081633, 0.7795918367346939, 0.7795918367346939, 0.7816326530612245, 0.7816326530612245, 0.7836734693877551, 0.7836734693877551, 0.7877551020408163, 0.7877551020408163, 0.789795918367347, 0.789795918367347, 0.7918367346938775, 0.7918367346938775, 0.7959183673469388, 0.7959183673469388, 0.7979591836734694, 0.7979591836734694, 0.8, 0.8, 0.8020408163265306, 0.8020408163265306, 0.8040816326530612, 0.8040816326530612, 0.8061224489795918, 0.8061224489795918, 0.8081632653061225, 0.8081632653061225, 0.810204081632653, 0.810204081632653, 0.8122448979591836, 0.8122448979591836, 0.8142857142857143, 0.8142857142857143, 0.8163265306122449, 0.8163265306122449, 0.8183673469387756, 0.8183673469387756, 0.8204081632653061, 0.8204081632653061, 0.8224489795918367, 0.8224489795918367, 0.8244897959183674, 0.8244897959183674, 0.826530612244898, 0.826530612244898, 0.8285714285714286, 0.8285714285714286, 0.8306122448979592, 0.8306122448979592, 0.8326530612244898, 0.8326530612244898, 0.8346938775510204, 0.8346938775510204, 0.8367346938775511, 0.8367346938775511, 0.8408163265306122, 0.8408163265306122, 0.8428571428571429, 0.8428571428571429, 0.8448979591836735, 0.8448979591836735, 0.8469387755102041, 0.8469387755102041, 0.8489795918367347, 0.8489795918367347, 0.8510204081632653, 0.8510204081632653, 0.8530612244897959, 0.8530612244897959, 0.8551020408163266, 0.8551020408163266, 0.8571428571428571, 0.8571428571428571, 0.8591836734693877, 0.8591836734693877, 0.8612244897959184, 0.8612244897959184, 0.863265306122449, 0.863265306122449, 0.8653061224489796, 0.8653061224489796, 0.8673469387755102, 0.8673469387755102, 0.8693877551020408, 0.8693877551020408, 0.8714285714285714, 0.8714285714285714, 0.8734693877551021, 0.8734693877551021, 0.8755102040816326, 0.8775510204081632, 0.8775510204081632, 0.8775510204081632, 0.8795918367346939, 0.8795918367346939, 0.8816326530612245, 0.8816326530612245, 0.8836734693877552, 0.8836734693877552, 0.8857142857142857, 0.8857142857142857, 0.889795918367347, 0.889795918367347, 0.8918367346938776, 0.8918367346938776, 0.8938775510204081, 0.8938775510204081, 0.8959183673469387, 0.8959183673469387, 0.8979591836734694, 0.8979591836734694, 0.9061224489795918, 0.9061224489795918, 0.9102040816326531, 0.9102040816326531, 0.9142857142857143, 0.9142857142857143, 0.9163265306122449, 0.9163265306122449, 0.9183673469387755, 0.9183673469387755, 0.9224489795918367, 0.9224489795918367, 0.926530612244898, 0.926530612244898, 0.9306122448979591, 0.9306122448979591, 0.9326530612244898, 0.9326530612244898, 0.936734693877551, 0.936734693877551, 0.9387755102040817, 0.9387755102040817, 0.9408163265306122, 0.9408163265306122, 0.9428571428571428, 0.9428571428571428, 0.9448979591836735, 0.9448979591836735, 0.9469387755102041, 0.9469387755102041, 0.9489795918367347, 0.9489795918367347, 0.9510204081632653, 0.9510204081632653, 0.9530612244897959, 0.9530612244897959, 0.9551020408163265, 0.9551020408163265, 0.9571428571428572, 0.9571428571428572, 0.9591836734693877, 0.9591836734693877, 0.9612244897959183, 0.9612244897959183, 0.963265306122449, 0.963265306122449, 0.9653061224489796, 0.9653061224489796, 0.9673469387755103, 0.9673469387755103, 0.9693877551020408, 0.9693877551020408, 0.9714285714285714, 0.9714285714285714, 0.9734693877551021, 0.9734693877551021, 0.9755102040816327, 0.9755102040816327, 0.9775510204081632, 0.9775510204081632, 0.9795918367346939, 0.9795918367346939, 0.9816326530612245, 0.9816326530612245, 0.9836734693877551, 0.9836734693877551, 0.9857142857142858, 0.9857142857142858, 0.9877551020408163, 0.9877551020408163, 0.9897959183673469, 0.9897959183673469, 0.9918367346938776, 0.9918367346938776, 0.9938775510204082, 0.9938775510204082, 0.9959183673469387, 0.9959183673469387, 0.9979591836734694, 0.9979591836734694, 1.0, 1.0]}, {"line": {"color": "rgb(205, 12, 24)", "dash": "dot", "width": 2}, "type": "scatter", "uid": "78b45b6f-fda2-4df9-a066-dcb46b20d6b3", "x": [0, 1], "y": [0, 1]}, {"colorscale": [[0.0, "rgb(165,0,38)"], [0.1111111111111111, "rgb(215,48,39)"], [0.2222222222222222, "rgb(244,109,67)"], [0.3333333333333333, "rgb(253,174,97)"], [0.4444444444444444, "rgb(254,224,144)"], [0.5555555555555556, "rgb(224,243,248)"], [0.6666666666666666, "rgb(171,217,233)"], [0.7777777777777778, "rgb(116,173,209)"], [0.8888888888888888, "rgb(69,117,180)"], [1.0, "rgb(49,54,149)"]], "name": "matrix", "showscale": false, "type": "heatmap", "uid": "bc2c54e7-7270-4d42-a4fa-afa2789a7c3f", "x": ["Not churn", "Churn"], "xaxis": "x2", "y": ["Not churn", "Churn"], "yaxis": "y2", "z": [[1145, 123], [225, 265]]}],
                        {"autosize": false, "height": 600, "margin": {"b": 200}, "paper_bgcolor": "rgb(243,243,243)", "plot_bgcolor": "rgb(243,243,243)", "showlegend": false, "title": {"text": "Caracteristicas del modelo"}, "width": 900, "xaxis": {"domain": [0, 0.6], "gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio falso positivo"}}, "xaxis2": {"domain": [0.7, 1], "gridcolor": "rgb(255, 255, 255)", "tickangle": 90}, "yaxis": {"gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio verdadero positivo"}, "zerolinewidth": 1}, "yaxis2": {"anchor": "x2", "gridcolor": "rgb(255, 255, 255)"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('f11a3922-63e0-4aa2-8a2e-172433d1fb9b');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


    /anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py:761: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    


    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
              intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
              penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
              verbose=0, warm_start=False)
    Accuracy   Score :  0.8100113765642776
    Area bajo la curva :  0.7311836090903239 
    



<div>
        
        
            <div id="88e5f339-1e5e-4e63-912f-d43d538e734e" class="plotly-graph-div" style="height:600px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("88e5f339-1e5e-4e63-912f-d43d538e734e")) {
                    Plotly.newPlot(
                        '88e5f339-1e5e-4e63-912f-d43d538e734e',
                        [{"line": {"color": "rgb(22, 96, 167)", "width": 2}, "name": "ROC : 0.7311836090903239", "type": "scatter", "uid": "79c7cf2f-a5f0-44ca-b2e1-4f94ecae851f", "x": [0.0, 0.0, 0.0, 0.0007886435331230284, 0.0007886435331230284, 0.0015772870662460567, 0.0015772870662460567, 0.002365930599369085, 0.002365930599369085, 0.0031545741324921135, 0.0031545741324921135, 0.003943217665615142, 0.003943217665615142, 0.00473186119873817, 0.00473186119873817, 0.005520504731861199, 0.005520504731861199, 0.006309148264984227, 0.006309148264984227, 0.007097791798107256, 0.007097791798107256, 0.008675078864353312, 0.008675078864353312, 0.00946372239747634, 0.00946372239747634, 0.01025236593059937, 0.01025236593059937, 0.011041009463722398, 0.011041009463722398, 0.012618296529968454, 0.012618296529968454, 0.014195583596214511, 0.014195583596214511, 0.01498422712933754, 0.01498422712933754, 0.016561514195583597, 0.016561514195583597, 0.018138801261829655, 0.018138801261829655, 0.01892744479495268, 0.01892744479495268, 0.01971608832807571, 0.01971608832807571, 0.021293375394321766, 0.021293375394321766, 0.022082018927444796, 0.022082018927444796, 0.022870662460567823, 0.022870662460567823, 0.02365930599369085, 0.02365930599369085, 0.02444794952681388, 0.02444794952681388, 0.025236593059936908, 0.025236593059936908, 0.026025236593059938, 0.026025236593059938, 0.026813880126182965, 0.026813880126182965, 0.028391167192429023, 0.028391167192429023, 0.02917981072555205, 0.02917981072555205, 0.02996845425867508, 0.02996845425867508, 0.032334384858044164, 0.032334384858044164, 0.033123028391167195, 0.033123028391167195, 0.03391167192429022, 0.03391167192429022, 0.03470031545741325, 0.03470031545741325, 0.03548895899053628, 0.03548895899053628, 0.03627760252365931, 0.03627760252365931, 0.03706624605678233, 0.03706624605678233, 0.03785488958990536, 0.03785488958990536, 0.038643533123028394, 0.038643533123028394, 0.04100946372239748, 0.04100946372239748, 0.0417981072555205, 0.0417981072555205, 0.044952681388012616, 0.044952681388012616, 0.04574132492113565, 0.04574132492113565, 0.04652996845425868, 0.04652996845425868, 0.0473186119873817, 0.0473186119873817, 0.04810725552050473, 0.04810725552050473, 0.050473186119873815, 0.050473186119873815, 0.051261829652996846, 0.051261829652996846, 0.0528391167192429, 0.0528391167192429, 0.05362776025236593, 0.05362776025236593, 0.05441640378548896, 0.05441640378548896, 0.055205047318611984, 0.055205047318611984, 0.055993690851735015, 0.055993690851735015, 0.056782334384858045, 0.056782334384858045, 0.057570977917981075, 0.057570977917981075, 0.05993690851735016, 0.05993690851735016, 0.06072555205047318, 0.06072555205047318, 0.061514195583596214, 0.061514195583596214, 0.06309148264984227, 0.06309148264984227, 0.06624605678233439, 0.06624605678233439, 0.06703470031545741, 0.06703470031545741, 0.06782334384858044, 0.06782334384858044, 0.06861198738170347, 0.06861198738170347, 0.07097791798107256, 0.07097791798107256, 0.07176656151419558, 0.07176656151419558, 0.07334384858044164, 0.07334384858044164, 0.07413249211356467, 0.07413249211356467, 0.0749211356466877, 0.0749211356466877, 0.07570977917981073, 0.07570977917981073, 0.07649842271293375, 0.07649842271293375, 0.07728706624605679, 0.07728706624605679, 0.07807570977917981, 0.07807570977917981, 0.07886435331230283, 0.07886435331230283, 0.07965299684542587, 0.07965299684542587, 0.0804416403785489, 0.0804416403785489, 0.08201892744479496, 0.08201892744479496, 0.08517350157728706, 0.08517350157728706, 0.0859621451104101, 0.0859621451104101, 0.08675078864353312, 0.08675078864353312, 0.08753943217665615, 0.08753943217665615, 0.08832807570977919, 0.08832807570977919, 0.08911671924290221, 0.08911671924290221, 0.08990536277602523, 0.08990536277602523, 0.09227129337539432, 0.09227129337539432, 0.0946372239747634, 0.0946372239747634, 0.09700315457413249, 0.09700315457413249, 0.09779179810725552, 0.09779179810725552, 0.09936908517350158, 0.09936908517350158, 0.10015772870662461, 0.10015772870662461, 0.10094637223974763, 0.10094637223974763, 0.10173501577287067, 0.10173501577287067, 0.10252365930599369, 0.10252365930599369, 0.10331230283911672, 0.10331230283911672, 0.10488958990536278, 0.10488958990536278, 0.1056782334384858, 0.1056782334384858, 0.10646687697160884, 0.10646687697160884, 0.10725552050473186, 0.10725552050473186, 0.11198738170347003, 0.11198738170347003, 0.11514195583596215, 0.11514195583596215, 0.11593059936908517, 0.11593059936908517, 0.11750788643533124, 0.11750788643533124, 0.12302839116719243, 0.12302839116719243, 0.12381703470031545, 0.12381703470031545, 0.12460567823343849, 0.12460567823343849, 0.12854889589905363, 0.12854889589905363, 0.13012618296529968, 0.13012618296529968, 0.13170347003154576, 0.13170347003154576, 0.13249211356466878, 0.13249211356466878, 0.1332807570977918, 0.1332807570977918, 0.13406940063091483, 0.13406940063091483, 0.13485804416403785, 0.13485804416403785, 0.13564668769716087, 0.13564668769716087, 0.13801261829652997, 0.13801261829652997, 0.14037854889589904, 0.14037854889589904, 0.1411671924290221, 0.1411671924290221, 0.1443217665615142, 0.1443217665615142, 0.14589905362776026, 0.14589905362776026, 0.14668769716088328, 0.14668769716088328, 0.1474763406940063, 0.1474763406940063, 0.14826498422712933, 0.14826498422712933, 0.14905362776025236, 0.14905362776025236, 0.15063091482649843, 0.15063091482649843, 0.15378548895899052, 0.15378548895899052, 0.15851735015772872, 0.15851735015772872, 0.15930599369085174, 0.15930599369085174, 0.16009463722397477, 0.16009463722397477, 0.16167192429022081, 0.16167192429022081, 0.16246056782334384, 0.16246056782334384, 0.1640378548895899, 0.1640378548895899, 0.167192429022082, 0.167192429022082, 0.1695583596214511, 0.1695583596214511, 0.17034700315457413, 0.17034700315457413, 0.17429022082018927, 0.17429022082018927, 0.1750788643533123, 0.1750788643533123, 0.17586750788643532, 0.17586750788643532, 0.17823343848580442, 0.17823343848580442, 0.1805993690851735, 0.1805993690851735, 0.1829652996845426, 0.1829652996845426, 0.18454258675078863, 0.18454258675078863, 0.18690851735015773, 0.18690851735015773, 0.18848580441640378, 0.18848580441640378, 0.19006309148264985, 0.19006309148264985, 0.1916403785488959, 0.1916403785488959, 0.19479495268138802, 0.19479495268138802, 0.19558359621451105, 0.19558359621451105, 0.1971608832807571, 0.1971608832807571, 0.19794952681388012, 0.19794952681388012, 0.20031545741324921, 0.20031545741324921, 0.20110410094637224, 0.20110410094637224, 0.20504731861198738, 0.20504731861198738, 0.20741324921135645, 0.20741324921135645, 0.20899053627760253, 0.20899053627760253, 0.20977917981072555, 0.20977917981072555, 0.2113564668769716, 0.2113564668769716, 0.21293375394321767, 0.21293375394321767, 0.2137223974763407, 0.2137223974763407, 0.2200315457413249, 0.2200315457413249, 0.221608832807571, 0.221608832807571, 0.22476340694006308, 0.22476340694006308, 0.2302839116719243, 0.2302839116719243, 0.23107255520504733, 0.23107255520504733, 0.23186119873817035, 0.23186119873817035, 0.23264984227129337, 0.23264984227129337, 0.2358044164037855, 0.2358044164037855, 0.23974763406940064, 0.23974763406940064, 0.24526813880126183, 0.24526813880126183, 0.25236593059936907, 0.25236593059936907, 0.2578864353312303, 0.2578864353312303, 0.25946372239747634, 0.25946372239747634, 0.26261829652996843, 0.26261829652996843, 0.2689274447949527, 0.2689274447949527, 0.2697160883280757, 0.2697160883280757, 0.2752365930599369, 0.2752365930599369, 0.27602523659305994, 0.27602523659305994, 0.278391167192429, 0.278391167192429, 0.2854889589905363, 0.2854889589905363, 0.2894321766561514, 0.2894321766561514, 0.2973186119873817, 0.2973186119873817, 0.2996845425867508, 0.2996845425867508, 0.30126182965299686, 0.30126182965299686, 0.30362776025236593, 0.30441640378548895, 0.305993690851735, 0.30757097791798105, 0.30757097791798105, 0.3138801261829653, 0.3138801261829653, 0.31782334384858046, 0.31782334384858046, 0.3194006309148265, 0.3194006309148265, 0.32018927444794953, 0.32018927444794953, 0.32097791798107256, 0.32097791798107256, 0.32334384858044163, 0.32334384858044163, 0.3257097791798107, 0.3257097791798107, 0.3264984227129338, 0.3264984227129338, 0.33280757097791797, 0.33280757097791797, 0.333596214511041, 0.333596214511041, 0.3391167192429022, 0.3391167192429022, 0.3470031545741325, 0.3470031545741325, 0.3477917981072555, 0.3477917981072555, 0.3556782334384858, 0.3556782334384858, 0.3714511041009464, 0.3714511041009464, 0.38801261829652994, 0.38801261829652994, 0.388801261829653, 0.388801261829653, 0.39037854889589907, 0.39037854889589907, 0.3919558359621451, 0.3919558359621451, 0.3951104100946372, 0.3951104100946372, 0.39668769716088326, 0.39668769716088326, 0.39747634069400634, 0.39747634069400634, 0.3990536277602524, 0.3990536277602524, 0.4022082018927445, 0.4022082018927445, 0.41009463722397477, 0.41009463722397477, 0.4108832807570978, 0.4108832807570978, 0.41246056782334384, 0.41246056782334384, 0.41324921135646686, 0.41324921135646686, 0.4148264984227129, 0.4148264984227129, 0.4250788643533123, 0.4250788643533123, 0.42586750788643535, 0.42586750788643535, 0.4361198738170347, 0.4361198738170347, 0.4400630914826498, 0.4400630914826498, 0.44952681388012616, 0.44952681388012616, 0.4503154574132492, 0.4503154574132492, 0.4613564668769716, 0.4613564668769716, 0.46529968454258674, 0.46529968454258674, 0.47634069400630913, 0.47634069400630913, 0.47712933753943215, 0.47712933753943215, 0.48974763406940064, 0.48974763406940064, 0.5055205047318612, 0.5055205047318612, 0.5063091482649842, 0.5063091482649842, 0.5110410094637224, 0.5110410094637224, 0.5141955835962145, 0.5141955835962145, 0.5205047318611987, 0.5205047318611987, 0.5370662460567823, 0.5370662460567823, 0.5402208201892744, 0.5402208201892744, 0.5607255520504731, 0.5607255520504731, 0.5694006309148265, 0.5694006309148265, 0.582018927444795, 0.582018927444795, 0.582807570977918, 0.582807570977918, 0.6522082018927445, 0.6522082018927445, 0.6545741324921136, 0.6545741324921136, 0.666403785488959, 0.666403785488959, 0.6861198738170347, 0.6861198738170347, 0.6971608832807571, 0.6971608832807571, 0.7184542586750788, 0.7184542586750788, 0.7192429022082019, 0.7192429022082019, 0.7326498422712934, 0.7326498422712934, 0.8028391167192429, 0.8028391167192429, 0.8162460567823344, 0.8162460567823344, 0.8430599369085173, 0.8430599369085173, 0.8919558359621451, 0.8919558359621451, 1.0], "y": [0.0, 0.0020408163265306124, 0.012244897959183673, 0.012244897959183673, 0.014285714285714285, 0.014285714285714285, 0.02857142857142857, 0.02857142857142857, 0.05102040816326531, 0.05102040816326531, 0.06326530612244897, 0.06326530612244897, 0.07346938775510205, 0.07346938775510205, 0.07755102040816327, 0.07755102040816327, 0.0836734693877551, 0.0836734693877551, 0.10204081632653061, 0.10204081632653061, 0.10612244897959183, 0.10612244897959183, 0.1346938775510204, 0.1346938775510204, 0.13877551020408163, 0.13877551020408163, 0.1489795918367347, 0.1489795918367347, 0.15306122448979592, 0.15306122448979592, 0.16122448979591836, 0.16122448979591836, 0.17142857142857143, 0.17142857142857143, 0.17346938775510204, 0.17346938775510204, 0.17959183673469387, 0.17959183673469387, 0.1816326530612245, 0.1816326530612245, 0.18775510204081633, 0.18775510204081633, 0.19387755102040816, 0.19387755102040816, 0.19795918367346937, 0.19795918367346937, 0.20408163265306123, 0.20408163265306123, 0.21224489795918366, 0.21224489795918366, 0.21428571428571427, 0.21428571428571427, 0.2163265306122449, 0.2163265306122449, 0.21836734693877552, 0.21836734693877552, 0.23877551020408164, 0.23877551020408164, 0.24081632653061225, 0.24081632653061225, 0.2571428571428571, 0.2571428571428571, 0.2612244897959184, 0.2612244897959184, 0.2673469387755102, 0.2673469387755102, 0.2714285714285714, 0.2714285714285714, 0.2755102040816326, 0.2755102040816326, 0.27755102040816326, 0.27755102040816326, 0.2795918367346939, 0.2795918367346939, 0.2897959183673469, 0.2897959183673469, 0.29183673469387755, 0.29183673469387755, 0.2938775510204082, 0.2938775510204082, 0.2979591836734694, 0.2979591836734694, 0.30612244897959184, 0.30612244897959184, 0.3163265306122449, 0.3163265306122449, 0.3224489795918367, 0.3224489795918367, 0.32653061224489793, 0.32653061224489793, 0.336734693877551, 0.336734693877551, 0.3408163265306122, 0.3408163265306122, 0.3489795918367347, 0.3489795918367347, 0.3510204081632653, 0.3510204081632653, 0.3551020408163265, 0.3551020408163265, 0.363265306122449, 0.363265306122449, 0.3653061224489796, 0.3653061224489796, 0.37551020408163266, 0.37551020408163266, 0.3816326530612245, 0.3816326530612245, 0.38571428571428573, 0.38571428571428573, 0.39591836734693875, 0.39591836734693875, 0.4, 0.4, 0.40816326530612246, 0.40816326530612246, 0.4142857142857143, 0.4142857142857143, 0.42244897959183675, 0.42244897959183675, 0.42448979591836733, 0.42448979591836733, 0.42857142857142855, 0.42857142857142855, 0.4326530612244898, 0.4326530612244898, 0.4448979591836735, 0.4448979591836735, 0.44693877551020406, 0.44693877551020406, 0.4489795918367347, 0.4489795918367347, 0.4530612244897959, 0.4530612244897959, 0.45714285714285713, 0.45714285714285713, 0.45918367346938777, 0.45918367346938777, 0.47551020408163264, 0.47551020408163264, 0.47959183673469385, 0.47959183673469385, 0.48775510204081635, 0.48775510204081635, 0.49795918367346936, 0.49795918367346936, 0.5102040816326531, 0.5102040816326531, 0.5122448979591837, 0.5122448979591837, 0.5163265306122449, 0.5163265306122449, 0.5183673469387755, 0.5183673469387755, 0.5224489795918368, 0.5224489795918368, 0.5285714285714286, 0.5285714285714286, 0.5326530612244897, 0.5326530612244897, 0.5346938775510204, 0.5346938775510204, 0.536734693877551, 0.536734693877551, 0.5408163265306123, 0.5408163265306123, 0.5428571428571428, 0.5428571428571428, 0.5469387755102041, 0.5469387755102041, 0.5530612244897959, 0.5530612244897959, 0.5551020408163265, 0.5551020408163265, 0.5612244897959183, 0.5612244897959183, 0.5673469387755102, 0.5673469387755102, 0.5693877551020409, 0.5693877551020409, 0.5714285714285714, 0.5714285714285714, 0.573469387755102, 0.573469387755102, 0.5755102040816327, 0.5755102040816327, 0.5775510204081633, 0.5775510204081633, 0.5795918367346938, 0.5795918367346938, 0.5836734693877551, 0.5836734693877551, 0.5877551020408164, 0.5877551020408164, 0.5918367346938775, 0.5918367346938775, 0.5979591836734693, 0.5979591836734693, 0.6020408163265306, 0.6020408163265306, 0.610204081632653, 0.610204081632653, 0.6122448979591837, 0.6122448979591837, 0.6142857142857143, 0.6142857142857143, 0.6163265306122448, 0.6163265306122448, 0.6183673469387755, 0.6183673469387755, 0.6204081632653061, 0.6204081632653061, 0.6244897959183674, 0.6244897959183674, 0.6265306122448979, 0.6265306122448979, 0.6285714285714286, 0.6285714285714286, 0.6306122448979592, 0.6306122448979592, 0.6346938775510204, 0.6346938775510204, 0.636734693877551, 0.636734693877551, 0.6387755102040816, 0.6387755102040816, 0.6408163265306123, 0.6408163265306123, 0.6469387755102041, 0.6469387755102041, 0.6489795918367347, 0.6489795918367347, 0.6510204081632653, 0.6510204081632653, 0.6530612244897959, 0.6530612244897959, 0.6591836734693878, 0.6591836734693878, 0.6632653061224489, 0.6632653061224489, 0.6653061224489796, 0.6653061224489796, 0.6693877551020408, 0.6693877551020408, 0.6775510204081633, 0.6775510204081633, 0.6795918367346939, 0.6795918367346939, 0.6816326530612244, 0.6816326530612244, 0.6836734693877551, 0.6836734693877551, 0.6877551020408164, 0.6877551020408164, 0.6918367346938775, 0.6918367346938775, 0.7020408163265306, 0.7020408163265306, 0.7040816326530612, 0.7040816326530612, 0.7081632653061225, 0.7081632653061225, 0.7122448979591837, 0.7122448979591837, 0.7142857142857143, 0.7142857142857143, 0.7163265306122449, 0.7163265306122449, 0.7244897959183674, 0.7244897959183674, 0.726530612244898, 0.726530612244898, 0.7285714285714285, 0.7285714285714285, 0.7326530612244898, 0.7326530612244898, 0.736734693877551, 0.736734693877551, 0.7387755102040816, 0.7387755102040816, 0.7408163265306122, 0.7408163265306122, 0.7428571428571429, 0.7428571428571429, 0.7448979591836735, 0.7448979591836735, 0.746938775510204, 0.746938775510204, 0.7489795918367347, 0.7489795918367347, 0.7510204081632653, 0.7510204081632653, 0.753061224489796, 0.753061224489796, 0.7551020408163265, 0.7551020408163265, 0.7571428571428571, 0.7571428571428571, 0.7591836734693878, 0.7591836734693878, 0.7653061224489796, 0.7653061224489796, 0.7673469387755102, 0.7673469387755102, 0.7693877551020408, 0.7693877551020408, 0.7714285714285715, 0.7714285714285715, 0.773469387755102, 0.773469387755102, 0.7795918367346939, 0.7795918367346939, 0.7836734693877551, 0.7836734693877551, 0.7857142857142857, 0.7857142857142857, 0.7877551020408163, 0.7877551020408163, 0.789795918367347, 0.789795918367347, 0.7918367346938775, 0.7918367346938775, 0.7938775510204081, 0.7938775510204081, 0.7959183673469388, 0.7959183673469388, 0.8, 0.8, 0.8020408163265306, 0.8020408163265306, 0.8040816326530612, 0.8040816326530612, 0.8061224489795918, 0.8061224489795918, 0.8081632653061225, 0.8081632653061225, 0.810204081632653, 0.810204081632653, 0.8122448979591836, 0.8122448979591836, 0.8163265306122449, 0.8163265306122449, 0.8183673469387756, 0.8183673469387756, 0.8224489795918367, 0.8224489795918367, 0.8244897959183674, 0.8244897959183674, 0.826530612244898, 0.826530612244898, 0.8285714285714286, 0.8285714285714286, 0.8306122448979592, 0.8306122448979592, 0.8326530612244898, 0.8326530612244898, 0.8346938775510204, 0.8346938775510204, 0.8387755102040816, 0.8387755102040816, 0.8408163265306122, 0.8408163265306122, 0.8428571428571429, 0.8428571428571429, 0.8448979591836735, 0.8448979591836735, 0.8469387755102041, 0.8469387755102041, 0.8469387755102041, 0.8489795918367347, 0.8489795918367347, 0.8530612244897959, 0.8530612244897959, 0.8551020408163266, 0.8551020408163266, 0.863265306122449, 0.863265306122449, 0.8653061224489796, 0.8653061224489796, 0.8673469387755102, 0.8673469387755102, 0.8693877551020408, 0.8693877551020408, 0.8714285714285714, 0.8714285714285714, 0.8755102040816326, 0.8755102040816326, 0.8775510204081632, 0.8775510204081632, 0.8795918367346939, 0.8795918367346939, 0.8816326530612245, 0.8816326530612245, 0.8836734693877552, 0.8836734693877552, 0.8857142857142857, 0.8857142857142857, 0.889795918367347, 0.889795918367347, 0.8918367346938776, 0.8918367346938776, 0.8938775510204081, 0.8938775510204081, 0.8979591836734694, 0.8979591836734694, 0.9, 0.9, 0.9020408163265307, 0.9020408163265307, 0.9040816326530612, 0.9040816326530612, 0.9061224489795918, 0.9061224489795918, 0.9081632653061225, 0.9081632653061225, 0.9102040816326531, 0.9102040816326531, 0.9122448979591836, 0.9122448979591836, 0.9142857142857143, 0.9142857142857143, 0.9163265306122449, 0.9163265306122449, 0.9183673469387755, 0.9183673469387755, 0.9204081632653062, 0.9204081632653062, 0.9224489795918367, 0.9224489795918367, 0.9244897959183673, 0.9244897959183673, 0.926530612244898, 0.926530612244898, 0.9285714285714286, 0.9285714285714286, 0.9306122448979591, 0.9306122448979591, 0.9326530612244898, 0.9326530612244898, 0.9346938775510204, 0.9346938775510204, 0.936734693877551, 0.936734693877551, 0.9387755102040817, 0.9387755102040817, 0.9408163265306122, 0.9408163265306122, 0.9448979591836735, 0.9448979591836735, 0.9469387755102041, 0.9469387755102041, 0.9489795918367347, 0.9489795918367347, 0.9510204081632653, 0.9510204081632653, 0.9530612244897959, 0.9530612244897959, 0.9591836734693877, 0.9591836734693877, 0.9612244897959183, 0.9612244897959183, 0.963265306122449, 0.963265306122449, 0.9653061224489796, 0.9653061224489796, 0.9673469387755103, 0.9673469387755103, 0.9693877551020408, 0.9693877551020408, 0.9714285714285714, 0.9714285714285714, 0.9734693877551021, 0.9734693877551021, 0.9755102040816327, 0.9755102040816327, 0.9795918367346939, 0.9795918367346939, 0.9816326530612245, 0.9816326530612245, 0.9836734693877551, 0.9836734693877551, 0.9857142857142858, 0.9857142857142858, 0.9877551020408163, 0.9877551020408163, 0.9897959183673469, 0.9897959183673469, 0.9918367346938776, 0.9918367346938776, 0.9938775510204082, 0.9938775510204082, 0.9959183673469387, 0.9959183673469387, 0.9979591836734694, 0.9979591836734694, 1.0, 1.0]}, {"line": {"color": "rgb(205, 12, 24)", "dash": "dot", "width": 2}, "type": "scatter", "uid": "0af3195e-a5da-4f98-be03-6bebee2d9cd5", "x": [0, 1], "y": [0, 1]}, {"colorscale": [[0.0, "rgb(165,0,38)"], [0.1111111111111111, "rgb(215,48,39)"], [0.2222222222222222, "rgb(244,109,67)"], [0.3333333333333333, "rgb(253,174,97)"], [0.4444444444444444, "rgb(254,224,144)"], [0.5555555555555556, "rgb(224,243,248)"], [0.6666666666666666, "rgb(171,217,233)"], [0.7777777777777778, "rgb(116,173,209)"], [0.8888888888888888, "rgb(69,117,180)"], [1.0, "rgb(49,54,149)"]], "name": "matrix", "showscale": false, "type": "heatmap", "uid": "52a197fc-f7e9-40a2-9713-9cfbc321bd00", "x": ["Not churn", "Churn"], "xaxis": "x2", "y": ["Not churn", "Churn"], "yaxis": "y2", "z": [[1153, 115], [219, 271]]}],
                        {"autosize": false, "height": 600, "margin": {"b": 200}, "paper_bgcolor": "rgb(243,243,243)", "plot_bgcolor": "rgb(243,243,243)", "showlegend": false, "title": {"text": "Caracteristicas del modelo"}, "width": 900, "xaxis": {"domain": [0, 0.6], "gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio falso positivo"}}, "xaxis2": {"domain": [0.7, 1], "gridcolor": "rgb(255, 255, 255)", "tickangle": 90}, "yaxis": {"gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio verdadero positivo"}, "zerolinewidth": 1}, "yaxis2": {"anchor": "x2", "gridcolor": "rgb(255, 255, 255)"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('88e5f339-1e5e-4e63-912f-d43d538e734e');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


### RandomForest


```python
from sklearn.ensemble import RandomForestClassifier

RandomForestClassifier =RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                                               max_depth=3, max_features='auto', max_leaf_nodes=None,
                                               min_impurity_decrease=0.0, min_impurity_split=None,
                                               min_samples_leaf=1, min_samples_split=2,
                                               min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                                               oob_score=False, random_state=None, verbose=0,
                                               warm_start=False)
churn_prediction(RandomForestClassifier, train_XA, test_XA, train_YA, test_YA)
churn_prediction(RandomForestClassifier, train_XB, test_XB, train_YB, test_YB)
```

    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    


    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=3, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
    Accuracy   Score :  0.7747440273037542
    Area bajo la curva :  0.6303531191656473 
    



<div>
        
        
            <div id="be241a15-fd68-4bec-83ec-7e46c2beb6f0" class="plotly-graph-div" style="height:600px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("be241a15-fd68-4bec-83ec-7e46c2beb6f0")) {
                    Plotly.newPlot(
                        'be241a15-fd68-4bec-83ec-7e46c2beb6f0',
                        [{"line": {"color": "rgb(22, 96, 167)", "width": 2}, "name": "ROC : 0.6303531191656473", "type": "scatter", "uid": "b02f22fc-4b6d-40a6-937e-54a66fe7d8cd", "x": [0.0, 0.0, 0.0, 0.0007886435331230284, 0.0007886435331230284, 0.0007886435331230284, 0.0007886435331230284, 0.002365930599369085, 0.002365930599369085, 0.002365930599369085, 0.002365930599369085, 0.0031545741324921135, 0.003943217665615142, 0.003943217665615142, 0.00473186119873817, 0.00473186119873817, 0.00473186119873817, 0.00473186119873817, 0.005520504731861199, 0.005520504731861199, 0.005520504731861199, 0.006309148264984227, 0.007097791798107256, 0.007097791798107256, 0.007886435331230283, 0.007886435331230283, 0.008675078864353312, 0.008675078864353312, 0.00946372239747634, 0.00946372239747634, 0.011829652996845425, 0.011829652996845425, 0.012618296529968454, 0.012618296529968454, 0.013406940063091483, 0.013406940063091483, 0.01498422712933754, 0.01498422712933754, 0.015772870662460567, 0.015772870662460567, 0.016561514195583597, 0.016561514195583597, 0.017350157728706624, 0.017350157728706624, 0.018138801261829655, 0.018138801261829655, 0.01971608832807571, 0.01971608832807571, 0.02050473186119874, 0.02050473186119874, 0.021293375394321766, 0.021293375394321766, 0.022082018927444796, 0.022082018927444796, 0.02444794952681388, 0.02444794952681388, 0.025236593059936908, 0.025236593059936908, 0.026025236593059938, 0.026813880126182965, 0.026813880126182965, 0.028391167192429023, 0.028391167192429023, 0.02917981072555205, 0.02917981072555205, 0.02996845425867508, 0.02996845425867508, 0.031545741324921134, 0.031545741324921134, 0.032334384858044164, 0.032334384858044164, 0.033123028391167195, 0.033123028391167195, 0.03470031545741325, 0.03470031545741325, 0.03627760252365931, 0.03627760252365931, 0.03785488958990536, 0.03785488958990536, 0.038643533123028394, 0.038643533123028394, 0.03943217665615142, 0.03943217665615142, 0.04022082018927445, 0.04022082018927445, 0.0417981072555205, 0.0417981072555205, 0.04337539432176656, 0.04337539432176656, 0.04416403785488959, 0.04416403785488959, 0.044952681388012616, 0.044952681388012616, 0.04574132492113565, 0.04574132492113565, 0.04652996845425868, 0.04652996845425868, 0.0473186119873817, 0.0473186119873817, 0.04889589905362776, 0.04889589905362776, 0.04968454258675079, 0.050473186119873815, 0.050473186119873815, 0.051261829652996846, 0.051261829652996846, 0.052050473186119876, 0.052050473186119876, 0.05362776025236593, 0.05362776025236593, 0.055993690851735015, 0.055993690851735015, 0.056782334384858045, 0.056782334384858045, 0.0583596214511041, 0.0583596214511041, 0.05914826498422713, 0.05914826498422713, 0.061514195583596214, 0.061514195583596214, 0.0638801261829653, 0.0638801261829653, 0.06466876971608833, 0.06466876971608833, 0.06545741324921135, 0.06545741324921135, 0.06624605678233439, 0.06624605678233439, 0.06703470031545741, 0.06703470031545741, 0.06782334384858044, 0.06782334384858044, 0.0694006309148265, 0.0694006309148265, 0.07018927444794952, 0.07018927444794952, 0.07097791798107256, 0.07097791798107256, 0.07176656151419558, 0.07176656151419558, 0.07255520504731862, 0.07255520504731862, 0.07413249211356467, 0.07413249211356467, 0.07570977917981073, 0.07570977917981073, 0.07649842271293375, 0.07649842271293375, 0.07807570977917981, 0.07807570977917981, 0.07886435331230283, 0.07886435331230283, 0.07965299684542587, 0.07965299684542587, 0.08123028391167192, 0.08123028391167192, 0.08123028391167192, 0.08123028391167192, 0.08123028391167192, 0.08201892744479496, 0.08280757097791798, 0.08438485804416404, 0.08438485804416404, 0.0859621451104101, 0.0859621451104101, 0.08675078864353312, 0.08675078864353312, 0.08753943217665615, 0.08753943217665615, 0.08832807570977919, 0.08832807570977919, 0.08832807570977919, 0.08832807570977919, 0.09069400630914827, 0.09069400630914827, 0.0914826498422713, 0.0914826498422713, 0.09305993690851735, 0.09305993690851735, 0.09542586750788644, 0.09621451104100946, 0.09779179810725552, 0.09779179810725552, 0.09936908517350158, 0.09936908517350158, 0.10094637223974763, 0.10094637223974763, 0.10173501577287067, 0.10173501577287067, 0.10252365930599369, 0.10252365930599369, 0.10804416403785488, 0.10804416403785488, 0.10804416403785488, 0.10962145110410094, 0.10962145110410094, 0.11041009463722397, 0.11041009463722397, 0.111198738170347, 0.111198738170347, 0.11277602523659307, 0.11277602523659307, 0.11356466876971609, 0.11356466876971609, 0.11435331230283911, 0.11435331230283911, 0.11435331230283911, 0.11750788643533124, 0.11750788643533124, 0.12302839116719243, 0.12302839116719243, 0.12381703470031545, 0.12381703470031545, 0.1253943217665615, 0.1253943217665615, 0.12618296529968454, 0.12618296529968454, 0.1277602523659306, 0.1277602523659306, 0.12854889589905363, 0.12933753943217666, 0.12933753943217666, 0.13012618296529968, 0.13012618296529968, 0.1332807570977918, 0.1332807570977918, 0.13958990536277602, 0.13958990536277602, 0.1411671924290221, 0.1411671924290221, 0.1443217665615142, 0.1443217665615142, 0.14905362776025236, 0.14905362776025236, 0.1498422712933754, 0.1498422712933754, 0.15457413249211358, 0.15457413249211358, 0.15694006309148265, 0.15694006309148265, 0.15772870662460567, 0.15772870662460567, 0.1632492113564669, 0.1632492113564669, 0.167192429022082, 0.167192429022082, 0.16876971608832808, 0.16876971608832808, 0.1695583596214511, 0.17034700315457413, 0.1750788643533123, 0.1750788643533123, 0.1774447949526814, 0.1774447949526814, 0.1805993690851735, 0.1805993690851735, 0.18217665615141956, 0.18217665615141956, 0.1837539432176656, 0.1837539432176656, 0.1861198738170347, 0.1861198738170347, 0.18690851735015773, 0.18690851735015773, 0.18848580441640378, 0.18848580441640378, 0.19085173501577288, 0.19085173501577288, 0.20110410094637224, 0.20110410094637224, 0.20347003154574134, 0.20347003154574134, 0.2082018927444795, 0.2082018927444795, 0.20977917981072555, 0.20977917981072555, 0.21056782334384858, 0.21056782334384858, 0.21214511041009465, 0.21214511041009465, 0.21687697160883282, 0.21687697160883282, 0.22476340694006308, 0.22476340694006308, 0.22555205047318613, 0.22555205047318613, 0.22634069400630916, 0.22634069400630916, 0.2279179810725552, 0.2279179810725552, 0.22949526813880125, 0.22949526813880125, 0.23107255520504733, 0.23107255520504733, 0.23264984227129337, 0.23264984227129337, 0.23501577287066247, 0.23501577287066247, 0.2358044164037855, 0.2358044164037855, 0.23738170347003154, 0.23738170347003154, 0.23974763406940064, 0.23974763406940064, 0.24053627760252366, 0.2421135646687697, 0.24369085173501578, 0.24369085173501578, 0.24526813880126183, 0.24526813880126183, 0.250788643533123, 0.250788643533123, 0.2547318611987382, 0.2547318611987382, 0.25630914826498424, 0.25630914826498424, 0.2618296529968454, 0.2618296529968454, 0.26261829652996843, 0.26261829652996843, 0.26419558359621453, 0.26419558359621453, 0.2689274447949527, 0.2689274447949527, 0.27287066246056785, 0.27287066246056785, 0.2752365930599369, 0.2752365930599369, 0.27602523659305994, 0.27681388012618297, 0.27681388012618297, 0.27917981072555204, 0.27917981072555204, 0.27996845425867506, 0.27996845425867506, 0.2823343848580442, 0.2823343848580442, 0.2831230283911672, 0.28391167192429023, 0.2854889589905363, 0.2854889589905363, 0.29337539432176657, 0.29337539432176657, 0.2941640378548896, 0.2941640378548896, 0.2973186119873817, 0.2973186119873817, 0.2981072555205047, 0.2981072555205047, 0.29889589905362773, 0.30126182965299686, 0.3020504731861199, 0.3020504731861199, 0.30441640378548895, 0.305993690851735, 0.30757097791798105, 0.30757097791798105, 0.3115141955835962, 0.31624605678233436, 0.31624605678233436, 0.3186119873817035, 0.32018927444794953, 0.32018927444794953, 0.3272870662460568, 0.32886435331230285, 0.3304416403785489, 0.3312302839116719, 0.33201892744479494, 0.3351735015772871, 0.3383280757097792, 0.3383280757097792, 0.3430599369085173, 0.3438485804416404, 0.3438485804416404, 0.35646687697160884, 0.3580441640378549, 0.3580441640378549, 0.361198738170347, 0.361198738170347, 0.3746056782334385, 0.3746056782334385, 0.38091482649842273, 0.38091482649842273, 0.38485804416403785, 0.38485804416403785, 0.39037854889589907, 0.39037854889589907, 0.3911671924290221, 0.3911671924290221, 0.3919558359621451, 0.3919558359621451, 0.3943217665615142, 0.3943217665615142, 0.4061514195583596, 0.4061514195583596, 0.4069400630914827, 0.4069400630914827, 0.40930599369085174, 0.4108832807570978, 0.416403785488959, 0.416403785488959, 0.41719242902208203, 0.41719242902208203, 0.41798107255520506, 0.41798107255520506, 0.4187697160883281, 0.4187697160883281, 0.4195583596214511, 0.4195583596214511, 0.42113564668769715, 0.42113564668769715, 0.4219242902208202, 0.4219242902208202, 0.4227129337539432, 0.4227129337539432, 0.4242902208201893, 0.4242902208201893, 0.42586750788643535, 0.42586750788643535, 0.4416403785488959, 0.4416403785488959, 0.444794952681388, 0.444794952681388, 0.4558359621451104, 0.4558359621451104, 0.4613564668769716, 0.46214511041009465, 0.46529968454258674, 0.46529968454258674, 0.4668769716088328, 0.4668769716088328, 0.47003154574132494, 0.47003154574132494, 0.48501577287066244, 0.48501577287066244, 0.4865930599369085, 0.4865930599369085, 0.4960567823343849, 0.4960567823343849, 0.5078864353312302, 0.5078864353312302, 0.5086750788643533, 0.5086750788643533, 0.5141955835962145, 0.5141955835962145, 0.5149842271293376, 0.5149842271293376, 0.5173501577287066, 0.5173501577287066, 0.5236593059936908, 0.5236593059936908, 0.526813880126183, 0.526813880126183, 0.5291798107255521, 0.5291798107255521, 0.5347003154574133, 0.5347003154574133, 0.5496845425867508, 0.5496845425867508, 0.5741324921135647, 0.5741324921135647, 0.5962145110410094, 0.5962145110410094, 0.6056782334384858, 0.6056782334384858, 0.6230283911671924, 0.6230283911671924, 0.6293375394321766, 0.6293375394321766, 0.6593059936908517, 0.6608832807570978, 0.6648264984227129, 0.6648264984227129, 0.6813880126182965, 0.6829652996845426, 0.693217665615142, 0.693217665615142, 0.7129337539432177, 0.7145110410094637, 0.7168769716088328, 0.7168769716088328, 0.7208201892744479, 0.722397476340694, 0.7247634069400631, 0.7271293375394322, 0.7468454258675079, 0.749211356466877, 0.7523659305993691, 0.7539432176656151, 0.7783911671924291, 0.7783911671924291, 0.8115141955835962, 0.8138801261829653, 0.8233438485804416, 0.8233438485804416, 0.8257097791798107, 0.8272870662460567, 0.832018927444795, 0.833596214511041, 0.8675078864353313, 0.8690851735015773, 0.8722397476340694, 0.8722397476340694, 0.8919558359621451, 0.8935331230283912, 0.917981072555205, 0.919558359621451, 0.9235015772870663, 0.9250788643533123, 0.9290220820189274, 0.9321766561514195, 0.942429022082019, 0.944006309148265, 0.9503154574132492, 0.9518927444794952, 0.9534700315457413, 0.9550473186119873, 0.9558359621451105, 0.9582018927444795, 0.9637223974763407, 0.9652996845425867, 0.9660883280757098, 0.9676656151419558, 0.9708201892744479, 0.973186119873817, 0.9739747634069401, 0.9787066246056783, 0.9794952681388013, 0.9810725552050473, 0.9960567823343849, 0.9976340694006309, 1.0], "y": [0.0, 0.0020408163265306124, 0.014285714285714285, 0.014285714285714285, 0.022448979591836733, 0.02857142857142857, 0.030612244897959183, 0.036734693877551024, 0.03877551020408163, 0.04285714285714286, 0.044897959183673466, 0.044897959183673466, 0.053061224489795916, 0.05714285714285714, 0.05714285714285714, 0.08571428571428572, 0.08979591836734693, 0.09591836734693877, 0.09591836734693877, 0.1, 0.10204081632653061, 0.10612244897959183, 0.10612244897959183, 0.12653061224489795, 0.12653061224489795, 0.1306122448979592, 0.1306122448979592, 0.13673469387755102, 0.13673469387755102, 0.1469387755102041, 0.1469387755102041, 0.1489795918367347, 0.1489795918367347, 0.15714285714285714, 0.15714285714285714, 0.16326530612244897, 0.16326530612244897, 0.1673469387755102, 0.1673469387755102, 0.16938775510204082, 0.16938775510204082, 0.17346938775510204, 0.17346938775510204, 0.17959183673469387, 0.17959183673469387, 0.2, 0.2, 0.20408163265306123, 0.20408163265306123, 0.21020408163265306, 0.21020408163265306, 0.21224489795918366, 0.21224489795918366, 0.22040816326530613, 0.22040816326530613, 0.22448979591836735, 0.22448979591836735, 0.23061224489795917, 0.23265306122448978, 0.23265306122448978, 0.23673469387755103, 0.23673469387755103, 0.24081632653061225, 0.24081632653061225, 0.24693877551020407, 0.24693877551020407, 0.25918367346938775, 0.25918367346938775, 0.2612244897959184, 0.2612244897959184, 0.26326530612244897, 0.26326530612244897, 0.2673469387755102, 0.2673469387755102, 0.2714285714285714, 0.2714285714285714, 0.2795918367346939, 0.2795918367346939, 0.2836734693877551, 0.2836734693877551, 0.2897959183673469, 0.2897959183673469, 0.2979591836734694, 0.2979591836734694, 0.3, 0.3, 0.3040816326530612, 0.3040816326530612, 0.30612244897959184, 0.30612244897959184, 0.31020408163265306, 0.31020408163265306, 0.3183673469387755, 0.3183673469387755, 0.3306122448979592, 0.3306122448979592, 0.3346938775510204, 0.3346938775510204, 0.33877551020408164, 0.33877551020408164, 0.3448979591836735, 0.3448979591836735, 0.3469387755102041, 0.35306122448979593, 0.35306122448979593, 0.35918367346938773, 0.35918367346938773, 0.363265306122449, 0.363265306122449, 0.3653061224489796, 0.3653061224489796, 0.3693877551020408, 0.3693877551020408, 0.37551020408163266, 0.37551020408163266, 0.3836734693877551, 0.3836734693877551, 0.3877551020408163, 0.3877551020408163, 0.39183673469387753, 0.39183673469387753, 0.40816326530612246, 0.40816326530612246, 0.4122448979591837, 0.4122448979591837, 0.4142857142857143, 0.4142857142857143, 0.41836734693877553, 0.41836734693877553, 0.4204081632653061, 0.4204081632653061, 0.42448979591836733, 0.42448979591836733, 0.42653061224489797, 0.42653061224489797, 0.4346938775510204, 0.4346938775510204, 0.4448979591836735, 0.4448979591836735, 0.44693877551020406, 0.44693877551020406, 0.4489795918367347, 0.4489795918367347, 0.4530612244897959, 0.4530612244897959, 0.45510204081632655, 0.45714285714285713, 0.45918367346938777, 0.45918367346938777, 0.46122448979591835, 0.46122448979591835, 0.47346938775510206, 0.47346938775510206, 0.47551020408163264, 0.47551020408163264, 0.47959183673469385, 0.48367346938775513, 0.4857142857142857, 0.4897959183673469, 0.4897959183673469, 0.49387755102040815, 0.49387755102040815, 0.49795918367346936, 0.49795918367346936, 0.5, 0.5, 0.5020408163265306, 0.5020408163265306, 0.5040816326530613, 0.5040816326530613, 0.5061224489795918, 0.5102040816326531, 0.5122448979591837, 0.5122448979591837, 0.5163265306122449, 0.5163265306122449, 0.5224489795918368, 0.5224489795918368, 0.5285714285714286, 0.5285714285714286, 0.5306122448979592, 0.5306122448979592, 0.536734693877551, 0.536734693877551, 0.5387755102040817, 0.5387755102040817, 0.5448979591836735, 0.5448979591836735, 0.5530612244897959, 0.5530612244897959, 0.5551020408163265, 0.5551020408163265, 0.5591836734693878, 0.563265306122449, 0.563265306122449, 0.5714285714285714, 0.5714285714285714, 0.5775510204081633, 0.5775510204081633, 0.5877551020408164, 0.5877551020408164, 0.5897959183673469, 0.5897959183673469, 0.5938775510204082, 0.5938775510204082, 0.6, 0.6040816326530613, 0.6040816326530613, 0.6061224489795919, 0.6061224489795919, 0.6081632653061224, 0.6081632653061224, 0.610204081632653, 0.610204081632653, 0.6122448979591837, 0.6122448979591837, 0.6306122448979592, 0.6306122448979592, 0.6326530612244898, 0.6326530612244898, 0.6346938775510204, 0.6387755102040816, 0.6387755102040816, 0.6428571428571429, 0.6428571428571429, 0.6469387755102041, 0.6469387755102041, 0.6489795918367347, 0.6489795918367347, 0.6530612244897959, 0.6530612244897959, 0.6551020408163265, 0.6551020408163265, 0.6591836734693878, 0.6591836734693878, 0.6612244897959184, 0.6612244897959184, 0.6632653061224489, 0.6632653061224489, 0.6653061224489796, 0.6653061224489796, 0.6673469387755102, 0.6673469387755102, 0.673469387755102, 0.673469387755102, 0.6755102040816326, 0.6755102040816326, 0.6795918367346939, 0.6795918367346939, 0.6836734693877551, 0.6836734693877551, 0.6877551020408164, 0.6877551020408164, 0.6959183673469388, 0.6959183673469388, 0.6979591836734694, 0.6979591836734694, 0.7020408163265306, 0.7020408163265306, 0.7040816326530612, 0.7040816326530612, 0.7061224489795919, 0.7061224489795919, 0.7081632653061225, 0.7081632653061225, 0.7142857142857143, 0.7142857142857143, 0.7163265306122449, 0.7163265306122449, 0.7224489795918367, 0.7224489795918367, 0.726530612244898, 0.726530612244898, 0.7285714285714285, 0.7285714285714285, 0.7306122448979592, 0.7306122448979592, 0.7346938775510204, 0.7346938775510204, 0.736734693877551, 0.736734693877551, 0.7387755102040816, 0.7387755102040816, 0.7408163265306122, 0.7408163265306122, 0.7428571428571429, 0.7428571428571429, 0.7489795918367347, 0.7489795918367347, 0.753061224489796, 0.753061224489796, 0.7571428571428571, 0.7571428571428571, 0.7673469387755102, 0.7673469387755102, 0.7693877551020408, 0.7693877551020408, 0.7714285714285715, 0.7714285714285715, 0.773469387755102, 0.773469387755102, 0.7755102040816326, 0.7755102040816326, 0.7795918367346939, 0.7795918367346939, 0.7795918367346939, 0.7795918367346939, 0.7816326530612245, 0.7816326530612245, 0.7836734693877551, 0.7836734693877551, 0.7857142857142857, 0.7857142857142857, 0.7877551020408163, 0.7877551020408163, 0.789795918367347, 0.789795918367347, 0.7918367346938775, 0.7918367346938775, 0.7938775510204081, 0.7938775510204081, 0.7959183673469388, 0.7959183673469388, 0.7979591836734694, 0.7979591836734694, 0.8, 0.8, 0.8020408163265306, 0.8020408163265306, 0.8040816326530612, 0.8061224489795918, 0.8081632653061225, 0.810204081632653, 0.810204081632653, 0.8122448979591836, 0.8122448979591836, 0.8163265306122449, 0.8163265306122449, 0.8183673469387756, 0.8183673469387756, 0.8204081632653061, 0.8204081632653061, 0.8224489795918367, 0.8224489795918367, 0.8244897959183674, 0.8244897959183674, 0.826530612244898, 0.826530612244898, 0.8285714285714286, 0.8285714285714286, 0.8285714285714286, 0.8285714285714286, 0.8306122448979592, 0.8306122448979592, 0.8306122448979592, 0.8326530612244898, 0.8346938775510204, 0.8367346938775511, 0.8367346938775511, 0.8408163265306122, 0.8448979591836735, 0.8448979591836735, 0.8469387755102041, 0.8469387755102041, 0.8469387755102041, 0.8469387755102041, 0.8489795918367347, 0.8489795918367347, 0.8489795918367347, 0.8489795918367347, 0.8530612244897959, 0.8530612244897959, 0.8551020408163266, 0.8591836734693877, 0.8591836734693877, 0.8591836734693877, 0.8612244897959184, 0.8612244897959184, 0.863265306122449, 0.863265306122449, 0.8673469387755102, 0.8673469387755102, 0.8693877551020408, 0.8693877551020408, 0.8714285714285714, 0.8714285714285714, 0.8734693877551021, 0.8734693877551021, 0.8775510204081632, 0.8775510204081632, 0.8836734693877552, 0.8836734693877552, 0.8857142857142857, 0.8857142857142857, 0.8877551020408163, 0.8877551020408163, 0.8918367346938776, 0.8918367346938776, 0.8918367346938776, 0.8918367346938776, 0.8959183673469387, 0.8959183673469387, 0.9020408163265307, 0.9020408163265307, 0.9061224489795918, 0.9061224489795918, 0.9081632653061225, 0.9081632653061225, 0.9102040816326531, 0.9102040816326531, 0.9142857142857143, 0.9142857142857143, 0.9163265306122449, 0.9163265306122449, 0.9183673469387755, 0.9183673469387755, 0.9224489795918367, 0.9224489795918367, 0.9244897959183673, 0.9244897959183673, 0.926530612244898, 0.926530612244898, 0.9285714285714286, 0.9285714285714286, 0.9306122448979591, 0.9306122448979591, 0.9326530612244898, 0.9326530612244898, 0.9346938775510204, 0.9346938775510204, 0.936734693877551, 0.936734693877551, 0.9408163265306122, 0.9408163265306122, 0.9428571428571428, 0.9428571428571428, 0.9469387755102041, 0.9469387755102041, 0.9510204081632653, 0.9510204081632653, 0.9530612244897959, 0.9530612244897959, 0.9551020408163265, 0.9551020408163265, 0.9571428571428572, 0.9571428571428572, 0.9591836734693877, 0.9591836734693877, 0.9612244897959183, 0.9612244897959183, 0.9653061224489796, 0.9653061224489796, 0.9673469387755103, 0.9673469387755103, 0.9693877551020408, 0.9693877551020408, 0.9714285714285714, 0.9714285714285714, 0.9734693877551021, 0.9734693877551021, 0.9775510204081632, 0.9775510204081632, 0.9795918367346939, 0.9795918367346939, 0.9816326530612245, 0.9816326530612245, 0.9836734693877551, 0.9836734693877551, 0.9877551020408163, 0.9877551020408163, 0.9877551020408163, 0.9877551020408163, 0.9897959183673469, 0.9897959183673469, 0.9897959183673469, 0.9897959183673469, 0.9918367346938776, 0.9918367346938776, 0.9918367346938776, 0.9918367346938776, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9959183673469387, 0.9959183673469387, 0.9959183673469387, 0.9959183673469387, 0.9979591836734694, 0.9979591836734694, 0.9979591836734694, 0.9979591836734694, 0.9979591836734694, 0.9979591836734694, 0.9979591836734694, 0.9979591836734694, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, {"line": {"color": "rgb(205, 12, 24)", "dash": "dot", "width": 2}, "type": "scatter", "uid": "741722a4-7718-477f-a4d5-72ab6d53306e", "x": [0, 1], "y": [0, 1]}, {"colorscale": [[0.0, "rgb(165,0,38)"], [0.1111111111111111, "rgb(215,48,39)"], [0.2222222222222222, "rgb(244,109,67)"], [0.3333333333333333, "rgb(253,174,97)"], [0.4444444444444444, "rgb(254,224,144)"], [0.5555555555555556, "rgb(224,243,248)"], [0.6666666666666666, "rgb(171,217,233)"], [0.7777777777777778, "rgb(116,173,209)"], [0.8888888888888888, "rgb(69,117,180)"], [1.0, "rgb(49,54,149)"]], "name": "matrix", "showscale": false, "type": "heatmap", "uid": "ae588f37-62ee-43e5-8a94-43daf6267641", "x": ["Not churn", "Churn"], "xaxis": "x2", "y": ["Not churn", "Churn"], "yaxis": "y2", "z": [[1213, 55], [341, 149]]}],
                        {"autosize": false, "height": 600, "margin": {"b": 200}, "paper_bgcolor": "rgb(243,243,243)", "plot_bgcolor": "rgb(243,243,243)", "showlegend": false, "title": {"text": "Caracteristicas del modelo"}, "width": 900, "xaxis": {"domain": [0, 0.6], "gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio falso positivo"}}, "xaxis2": {"domain": [0.7, 1], "gridcolor": "rgb(255, 255, 255)", "tickangle": 90}, "yaxis": {"gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio verdadero positivo"}, "zerolinewidth": 1}, "yaxis2": {"anchor": "x2", "gridcolor": "rgb(255, 255, 255)"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('be241a15-fd68-4bec-83ec-7e46c2beb6f0');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


    /anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:6: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples,), for example using ravel().
    


    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                max_depth=3, max_features='auto', max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
                oob_score=False, random_state=None, verbose=0,
                warm_start=False)
    Accuracy   Score :  0.7724687144482366
    Area bajo la curva :  0.6300280048928087 
    



<div>
        
        
            <div id="2d2516fe-e419-4726-b169-936a8020eece" class="plotly-graph-div" style="height:600px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("2d2516fe-e419-4726-b169-936a8020eece")) {
                    Plotly.newPlot(
                        '2d2516fe-e419-4726-b169-936a8020eece',
                        [{"line": {"color": "rgb(22, 96, 167)", "width": 2}, "name": "ROC : 0.6300280048928087", "type": "scatter", "uid": "b943ebca-28e8-45a0-b856-f343627a495c", "x": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0007886435331230284, 0.0007886435331230284, 0.0015772870662460567, 0.0031545741324921135, 0.0031545741324921135, 0.003943217665615142, 0.003943217665615142, 0.003943217665615142, 0.003943217665615142, 0.003943217665615142, 0.003943217665615142, 0.003943217665615142, 0.003943217665615142, 0.003943217665615142, 0.00473186119873817, 0.00473186119873817, 0.006309148264984227, 0.006309148264984227, 0.007886435331230283, 0.007886435331230283, 0.008675078864353312, 0.008675078864353312, 0.00946372239747634, 0.00946372239747634, 0.01025236593059937, 0.011041009463722398, 0.011041009463722398, 0.011829652996845425, 0.011829652996845425, 0.012618296529968454, 0.012618296529968454, 0.013406940063091483, 0.013406940063091483, 0.015772870662460567, 0.015772870662460567, 0.016561514195583597, 0.016561514195583597, 0.017350157728706624, 0.017350157728706624, 0.018138801261829655, 0.018138801261829655, 0.01892744479495268, 0.01892744479495268, 0.01971608832807571, 0.01971608832807571, 0.02050473186119874, 0.02050473186119874, 0.02444794952681388, 0.02444794952681388, 0.026025236593059938, 0.026025236593059938, 0.02917981072555205, 0.02917981072555205, 0.02996845425867508, 0.02996845425867508, 0.031545741324921134, 0.031545741324921134, 0.032334384858044164, 0.032334384858044164, 0.032334384858044164, 0.032334384858044164, 0.033123028391167195, 0.033123028391167195, 0.03785488958990536, 0.03785488958990536, 0.03943217665615142, 0.03943217665615142, 0.04022082018927445, 0.04022082018927445, 0.04100946372239748, 0.0417981072555205, 0.0417981072555205, 0.04258675078864353, 0.04258675078864353, 0.04337539432176656, 0.04337539432176656, 0.04416403785488959, 0.04416403785488959, 0.044952681388012616, 0.044952681388012616, 0.04652996845425868, 0.04652996845425868, 0.0473186119873817, 0.0473186119873817, 0.04810725552050473, 0.04810725552050473, 0.04889589905362776, 0.04968454258675079, 0.051261829652996846, 0.051261829652996846, 0.0528391167192429, 0.0528391167192429, 0.0528391167192429, 0.0528391167192429, 0.05362776025236593, 0.05362776025236593, 0.05441640378548896, 0.05441640378548896, 0.055993690851735015, 0.055993690851735015, 0.056782334384858045, 0.056782334384858045, 0.057570977917981075, 0.057570977917981075, 0.05914826498422713, 0.05914826498422713, 0.05993690851735016, 0.05993690851735016, 0.06072555205047318, 0.06072555205047318, 0.061514195583596214, 0.061514195583596214, 0.062302839116719244, 0.062302839116719244, 0.06309148264984227, 0.06309148264984227, 0.06466876971608833, 0.06466876971608833, 0.06545741324921135, 0.06545741324921135, 0.06624605678233439, 0.06624605678233439, 0.06782334384858044, 0.06782334384858044, 0.06861198738170347, 0.06861198738170347, 0.07018927444794952, 0.07018927444794952, 0.07097791798107256, 0.07097791798107256, 0.07176656151419558, 0.07413249211356467, 0.07413249211356467, 0.0749211356466877, 0.0749211356466877, 0.07570977917981073, 0.07570977917981073, 0.07649842271293375, 0.07649842271293375, 0.07728706624605679, 0.07728706624605679, 0.07886435331230283, 0.07886435331230283, 0.0804416403785489, 0.08123028391167192, 0.08123028391167192, 0.08201892744479496, 0.08201892744479496, 0.08280757097791798, 0.08280757097791798, 0.083596214511041, 0.083596214511041, 0.08438485804416404, 0.08438485804416404, 0.08517350157728706, 0.08517350157728706, 0.0859621451104101, 0.0859621451104101, 0.08675078864353312, 0.08675078864353312, 0.08675078864353312, 0.08753943217665615, 0.08753943217665615, 0.08753943217665615, 0.08753943217665615, 0.08753943217665615, 0.08911671924290221, 0.08911671924290221, 0.09069400630914827, 0.09069400630914827, 0.09069400630914827, 0.0914826498422713, 0.09305993690851735, 0.09305993690851735, 0.09384858044164038, 0.09384858044164038, 0.0946372239747634, 0.0946372239747634, 0.09779179810725552, 0.09779179810725552, 0.09858044164037855, 0.09858044164037855, 0.09936908517350158, 0.10094637223974763, 0.10173501577287067, 0.10173501577287067, 0.10331230283911672, 0.10331230283911672, 0.10488958990536278, 0.10488958990536278, 0.1056782334384858, 0.1056782334384858, 0.10725552050473186, 0.10962145110410094, 0.11198738170347003, 0.11198738170347003, 0.11277602523659307, 0.11277602523659307, 0.11356466876971609, 0.11356466876971609, 0.11435331230283911, 0.11593059936908517, 0.11829652996845426, 0.11829652996845426, 0.11908517350157728, 0.11987381703470032, 0.12066246056782334, 0.12066246056782334, 0.1222397476340694, 0.1222397476340694, 0.12697160883280756, 0.12697160883280756, 0.12854889589905363, 0.12854889589905363, 0.13012618296529968, 0.1309148264984227, 0.1309148264984227, 0.13170347003154576, 0.13170347003154576, 0.1332807570977918, 0.1332807570977918, 0.13485804416403785, 0.13485804416403785, 0.13722397476340695, 0.13722397476340695, 0.138801261829653, 0.138801261829653, 0.14274447949526814, 0.14274447949526814, 0.15063091482649843, 0.15063091482649843, 0.15378548895899052, 0.1553627760252366, 0.15694006309148265, 0.15851735015772872, 0.15930599369085174, 0.1608832807570978, 0.1608832807570978, 0.16561514195583596, 0.16561514195583596, 0.16640378548895898, 0.16640378548895898, 0.167192429022082, 0.167192429022082, 0.16876971608832808, 0.16876971608832808, 0.17271293375394323, 0.17271293375394323, 0.17350157728706625, 0.17350157728706625, 0.17586750788643532, 0.17586750788643532, 0.17665615141955837, 0.17665615141955837, 0.17981072555205047, 0.18690851735015773, 0.18769716088328076, 0.18769716088328076, 0.1892744479495268, 0.1892744479495268, 0.19085173501577288, 0.19085173501577288, 0.19321766561514195, 0.19400630914826497, 0.19558359621451105, 0.19637223974763407, 0.19637223974763407, 0.1995268138801262, 0.1995268138801262, 0.20031545741324921, 0.20425867507886436, 0.20425867507886436, 0.2058359621451104, 0.2058359621451104, 0.2058359621451104, 0.21056782334384858, 0.21056782334384858, 0.2113564668769716, 0.2113564668769716, 0.21451104100946372, 0.21608832807570977, 0.21766561514195584, 0.21766561514195584, 0.2192429022082019, 0.2192429022082019, 0.2200315457413249, 0.2200315457413249, 0.22082018927444794, 0.22082018927444794, 0.222397476340694, 0.222397476340694, 0.22397476340694006, 0.22397476340694006, 0.23264984227129337, 0.23264984227129337, 0.2334384858044164, 0.2358044164037855, 0.23895899053627762, 0.23895899053627762, 0.24684542586750788, 0.24842271293375395, 0.25, 0.25, 0.25236593059936907, 0.25236593059936907, 0.25630914826498424, 0.25630914826498424, 0.25709779179810727, 0.25709779179810727, 0.26813880126182965, 0.26813880126182965, 0.27129337539432175, 0.27129337539432175, 0.27602523659305994, 0.27602523659305994, 0.278391167192429, 0.27996845425867506, 0.28391167192429023, 0.28391167192429023, 0.28470031545741326, 0.28470031545741326, 0.2862776025236593, 0.2886435331230284, 0.2917981072555205, 0.2917981072555205, 0.30126182965299686, 0.30126182965299686, 0.30441640378548895, 0.30441640378548895, 0.3107255520504732, 0.3107255520504732, 0.31703470031545744, 0.31703470031545744, 0.32334384858044163, 0.3249211356466877, 0.3257097791798107, 0.3257097791798107, 0.3272870662460568, 0.3272870662460568, 0.32886435331230285, 0.32886435331230285, 0.3304416403785489, 0.3304416403785489, 0.33201892744479494, 0.33201892744479494, 0.333596214511041, 0.333596214511041, 0.3351735015772871, 0.3351735015772871, 0.3359621451104101, 0.3359621451104101, 0.33675078864353314, 0.33675078864353314, 0.3383280757097792, 0.3383280757097792, 0.3430599369085173, 0.3438485804416404, 0.34463722397476343, 0.34463722397476343, 0.34858044164037855, 0.34858044164037855, 0.35173501577287064, 0.35173501577287064, 0.3548895899053628, 0.3548895899053628, 0.35646687697160884, 0.35725552050473186, 0.35962145110410093, 0.35962145110410093, 0.36041009463722395, 0.36041009463722395, 0.3643533123028391, 0.3643533123028391, 0.3698738170347003, 0.3698738170347003, 0.37381703470031546, 0.37381703470031546, 0.37697160883280756, 0.37697160883280756, 0.37933753943217663, 0.37933753943217663, 0.3801261829652997, 0.3801261829652997, 0.38485804416403785, 0.38485804416403785, 0.3856466876971609, 0.3856466876971609, 0.3864353312302839, 0.3864353312302839, 0.3872239747634069, 0.3872239747634069, 0.39037854889589907, 0.39037854889589907, 0.39747634069400634, 0.39826498422712936, 0.3998422712933754, 0.3998422712933754, 0.40063091482649843, 0.40063091482649843, 0.4022082018927445, 0.4037854889589905, 0.4061514195583596, 0.4061514195583596, 0.415615141955836, 0.415615141955836, 0.41798107255520506, 0.41798107255520506, 0.4195583596214511, 0.4195583596214511, 0.4219242902208202, 0.4219242902208202, 0.42586750788643535, 0.42586750788643535, 0.4282334384858044, 0.4282334384858044, 0.43217665615141954, 0.43217665615141954, 0.43690851735015773, 0.43690851735015773, 0.4400630914826498, 0.4400630914826498, 0.4479495268138801, 0.4479495268138801, 0.45425867507886436, 0.45425867507886436, 0.4550473186119874, 0.4550473186119874, 0.45741324921135645, 0.45741324921135645, 0.4582018927444795, 0.4582018927444795, 0.4629337539432177, 0.4629337539432177, 0.4637223974763407, 0.4637223974763407, 0.4668769716088328, 0.4668769716088328, 0.47318611987381703, 0.47318611987381703, 0.4747634069400631, 0.4747634069400631, 0.47791798107255523, 0.47791798107255523, 0.48264984227129337, 0.48264984227129337, 0.48580441640378547, 0.48580441640378547, 0.49369085173501576, 0.49369085173501576, 0.4944794952681388, 0.4944794952681388, 0.4960567823343849, 0.49763406940063093, 0.500788643533123, 0.500788643533123, 0.5070977917981072, 0.5070977917981072, 0.5102523659305994, 0.5102523659305994, 0.5236593059936908, 0.5252365930599369, 0.5252365930599369, 0.526813880126183, 0.526813880126183, 0.5354889589905363, 0.5354889589905363, 0.5528391167192429, 0.5528391167192429, 0.555993690851735, 0.555993690851735, 0.5615141955835962, 0.5638801261829653, 0.5646687697160884, 0.5646687697160884, 0.5678233438485805, 0.5678233438485805, 0.5686119873817035, 0.5686119873817035, 0.612776025236593, 0.612776025236593, 0.6143533123028391, 0.6143533123028391, 0.6529968454258676, 0.6529968454258676, 0.6553627760252366, 0.6892744479495269, 0.6892744479495269, 0.7247634069400631, 0.7247634069400631, 0.7294952681388013, 0.7310725552050473, 0.7318611987381703, 0.7350157728706624, 0.7531545741324921, 0.7547318611987381, 0.777602523659306, 0.7807570977917981, 0.7831230283911672, 0.7831230283911672, 0.7886435331230284, 0.7886435331230284, 0.7965299684542587, 0.7965299684542587, 0.8146687697160884, 0.8162460567823344, 0.8170347003154574, 0.8186119873817035, 0.8241324921135647, 0.8257097791798107, 0.8470031545741324, 0.8485804416403786, 0.8501577287066246, 0.8517350157728707, 0.8541009463722398, 0.8556782334384858, 0.8580441640378549, 0.8643533123028391, 0.8690851735015773, 0.8706624605678234, 0.8730283911671924, 0.8746056782334385, 0.8769716088328076, 0.8785488958990536, 0.8793375394321766, 0.8809148264984227, 0.8903785488958991, 0.8919558359621451, 0.9085173501577287, 0.9100946372239748, 0.917981072555205, 0.9203470031545742, 0.9250788643533123, 0.9282334384858044, 0.9313880126182965, 0.9337539432176656, 0.9345425867507886, 0.9361198738170347, 0.9392744479495269, 0.944006309148265, 0.9471608832807571, 0.9495268138801262, 0.9534700315457413, 0.9566246056782335, 0.9684542586750788, 0.9700315457413249, 0.9794952681388013, 0.9810725552050473, 0.9834384858044164, 0.9858044164037855, 0.9881703470031545, 0.9897476340694006, 0.9921135646687698, 0.9944794952681388, 0.9976340694006309, 1.0], "y": [0.0, 0.0020408163265306124, 0.00816326530612245, 0.014285714285714285, 0.018367346938775512, 0.022448979591836733, 0.026530612244897958, 0.030612244897959183, 0.036734693877551024, 0.036734693877551024, 0.04285714285714286, 0.04897959183673469, 0.053061224489795916, 0.05714285714285714, 0.06326530612244897, 0.0673469387755102, 0.06938775510204082, 0.07346938775510205, 0.07551020408163266, 0.07959183673469387, 0.08775510204081632, 0.08775510204081632, 0.09387755102040816, 0.09387755102040816, 0.11020408163265306, 0.11020408163265306, 0.11836734693877551, 0.11836734693877551, 0.1346938775510204, 0.1346938775510204, 0.13877551020408163, 0.13877551020408163, 0.14285714285714285, 0.15306122448979592, 0.15306122448979592, 0.15918367346938775, 0.15918367346938775, 0.16122448979591836, 0.16122448979591836, 0.16326530612244897, 0.16326530612244897, 0.1673469387755102, 0.1673469387755102, 0.17142857142857143, 0.17142857142857143, 0.17346938775510204, 0.17346938775510204, 0.1836734693877551, 0.1836734693877551, 0.19387755102040816, 0.19387755102040816, 0.19591836734693877, 0.19591836734693877, 0.19795918367346937, 0.19795918367346937, 0.20204081632653062, 0.20204081632653062, 0.20408163265306123, 0.20408163265306123, 0.20612244897959184, 0.20612244897959184, 0.21020408163265306, 0.21020408163265306, 0.22857142857142856, 0.22857142857142856, 0.24081632653061225, 0.24489795918367346, 0.2510204081632653, 0.2510204081632653, 0.2530612244897959, 0.2530612244897959, 0.25510204081632654, 0.25510204081632654, 0.2571428571428571, 0.2571428571428571, 0.2612244897959184, 0.2612244897959184, 0.26326530612244897, 0.2653061224489796, 0.2653061224489796, 0.2693877551020408, 0.2693877551020408, 0.27346938775510204, 0.27346938775510204, 0.2795918367346939, 0.2795918367346939, 0.28775510204081634, 0.28775510204081634, 0.2938775510204082, 0.2938775510204082, 0.3, 0.3, 0.3122448979591837, 0.3122448979591837, 0.3142857142857143, 0.3142857142857143, 0.336734693877551, 0.336734693877551, 0.3489795918367347, 0.35306122448979593, 0.35714285714285715, 0.35714285714285715, 0.3673469387755102, 0.3673469387755102, 0.37755102040816324, 0.37755102040816324, 0.3877551020408163, 0.3877551020408163, 0.39183673469387753, 0.39183673469387753, 0.3979591836734694, 0.3979591836734694, 0.4, 0.4020408163265306, 0.40408163265306124, 0.40408163265306124, 0.40816326530612246, 0.41020408163265304, 0.4204081632653061, 0.4204081632653061, 0.42244897959183675, 0.42244897959183675, 0.42653061224489797, 0.42653061224489797, 0.42857142857142855, 0.42857142857142855, 0.4306122448979592, 0.4306122448979592, 0.4326530612244898, 0.4326530612244898, 0.43673469387755104, 0.43673469387755104, 0.4387755102040816, 0.4387755102040816, 0.44081632653061226, 0.44081632653061226, 0.44285714285714284, 0.4448979591836735, 0.4448979591836735, 0.44693877551020406, 0.44693877551020406, 0.4530612244897959, 0.4530612244897959, 0.45714285714285713, 0.45918367346938777, 0.46938775510204084, 0.46938775510204084, 0.47346938775510206, 0.47346938775510206, 0.47551020408163264, 0.47551020408163264, 0.47551020408163264, 0.47959183673469385, 0.4857142857142857, 0.48775510204081635, 0.48775510204081635, 0.4897959183673469, 0.4897959183673469, 0.4959183673469388, 0.4959183673469388, 0.5, 0.5, 0.5040816326530613, 0.5040816326530613, 0.5102040816326531, 0.5102040816326531, 0.5142857142857142, 0.5244897959183673, 0.5244897959183673, 0.5326530612244897, 0.5387755102040817, 0.5428571428571428, 0.5469387755102041, 0.5469387755102041, 0.5489795918367347, 0.5489795918367347, 0.5510204081632653, 0.5571428571428572, 0.5591836734693878, 0.5612244897959183, 0.5673469387755102, 0.5673469387755102, 0.5714285714285714, 0.5714285714285714, 0.573469387755102, 0.573469387755102, 0.5775510204081633, 0.5775510204081633, 0.5795918367346938, 0.5836734693877551, 0.5836734693877551, 0.5836734693877551, 0.5877551020408164, 0.5877551020408164, 0.5938775510204082, 0.5938775510204082, 0.5959183673469388, 0.6061224489795919, 0.610204081632653, 0.610204081632653, 0.610204081632653, 0.610204081632653, 0.6142857142857143, 0.6142857142857143, 0.6224489795918368, 0.6224489795918368, 0.6285714285714286, 0.6285714285714286, 0.6326530612244898, 0.6326530612244898, 0.6346938775510204, 0.636734693877551, 0.636734693877551, 0.6387755102040816, 0.6408163265306123, 0.6408163265306123, 0.6469387755102041, 0.6469387755102041, 0.6530612244897959, 0.6530612244897959, 0.6591836734693878, 0.6591836734693878, 0.6612244897959184, 0.6632653061224489, 0.6632653061224489, 0.6653061224489796, 0.6653061224489796, 0.6693877551020408, 0.6693877551020408, 0.6714285714285714, 0.6714285714285714, 0.673469387755102, 0.673469387755102, 0.6755102040816326, 0.6755102040816326, 0.6775510204081633, 0.6775510204081633, 0.6795918367346939, 0.689795918367347, 0.689795918367347, 0.689795918367347, 0.689795918367347, 0.6918367346938775, 0.6918367346938775, 0.6938775510204082, 0.6938775510204082, 0.6959183673469388, 0.6959183673469388, 0.7, 0.7, 0.7020408163265306, 0.7020408163265306, 0.7040816326530612, 0.7040816326530612, 0.7081632653061225, 0.7081632653061225, 0.710204081632653, 0.710204081632653, 0.7122448979591837, 0.7122448979591837, 0.7163265306122449, 0.7163265306122449, 0.7244897959183674, 0.7244897959183674, 0.726530612244898, 0.726530612244898, 0.7285714285714285, 0.7285714285714285, 0.7306122448979592, 0.7306122448979592, 0.7306122448979592, 0.7326530612244898, 0.7326530612244898, 0.736734693877551, 0.736734693877551, 0.7387755102040816, 0.7408163265306122, 0.7408163265306122, 0.7428571428571429, 0.7428571428571429, 0.7448979591836735, 0.7489795918367347, 0.7489795918367347, 0.753061224489796, 0.753061224489796, 0.7551020408163265, 0.7551020408163265, 0.7551020408163265, 0.7551020408163265, 0.7571428571428571, 0.7571428571428571, 0.7591836734693878, 0.7591836734693878, 0.763265306122449, 0.763265306122449, 0.7673469387755102, 0.7673469387755102, 0.7693877551020408, 0.7693877551020408, 0.773469387755102, 0.773469387755102, 0.7775510204081633, 0.7775510204081633, 0.7775510204081633, 0.7775510204081633, 0.7795918367346939, 0.7795918367346939, 0.7795918367346939, 0.7795918367346939, 0.7857142857142857, 0.7857142857142857, 0.7877551020408163, 0.7877551020408163, 0.789795918367347, 0.789795918367347, 0.7938775510204081, 0.7938775510204081, 0.7959183673469388, 0.7959183673469388, 0.7979591836734694, 0.7979591836734694, 0.8, 0.8, 0.8, 0.8, 0.8020408163265306, 0.8020408163265306, 0.8040816326530612, 0.8040816326530612, 0.8061224489795918, 0.8061224489795918, 0.8081632653061225, 0.8081632653061225, 0.810204081632653, 0.810204081632653, 0.8122448979591836, 0.8122448979591836, 0.8142857142857143, 0.8142857142857143, 0.8163265306122449, 0.8163265306122449, 0.8163265306122449, 0.8163265306122449, 0.8183673469387756, 0.8183673469387756, 0.8224489795918367, 0.8224489795918367, 0.8244897959183674, 0.8244897959183674, 0.8285714285714286, 0.8285714285714286, 0.8306122448979592, 0.8306122448979592, 0.8326530612244898, 0.8326530612244898, 0.8346938775510204, 0.8346938775510204, 0.8367346938775511, 0.8367346938775511, 0.8387755102040816, 0.8387755102040816, 0.8448979591836735, 0.8448979591836735, 0.8469387755102041, 0.8469387755102041, 0.8489795918367347, 0.8489795918367347, 0.8510204081632653, 0.8510204081632653, 0.8530612244897959, 0.8530612244897959, 0.8551020408163266, 0.8551020408163266, 0.8571428571428571, 0.8571428571428571, 0.8591836734693877, 0.8591836734693877, 0.8612244897959184, 0.8612244897959184, 0.863265306122449, 0.863265306122449, 0.8653061224489796, 0.8653061224489796, 0.8673469387755102, 0.8673469387755102, 0.8693877551020408, 0.8693877551020408, 0.8714285714285714, 0.8714285714285714, 0.8734693877551021, 0.8734693877551021, 0.8755102040816326, 0.8755102040816326, 0.8816326530612245, 0.8816326530612245, 0.8836734693877552, 0.8836734693877552, 0.8877551020408163, 0.8877551020408163, 0.889795918367347, 0.889795918367347, 0.8918367346938776, 0.8918367346938776, 0.8938775510204081, 0.8938775510204081, 0.8959183673469387, 0.8959183673469387, 0.8959183673469387, 0.8959183673469387, 0.9, 0.9, 0.9061224489795918, 0.9061224489795918, 0.9102040816326531, 0.9102040816326531, 0.9122448979591836, 0.9122448979591836, 0.9142857142857143, 0.9142857142857143, 0.9163265306122449, 0.9163265306122449, 0.9183673469387755, 0.9183673469387755, 0.9204081632653062, 0.9204081632653062, 0.9224489795918367, 0.9224489795918367, 0.9244897959183673, 0.9244897959183673, 0.926530612244898, 0.926530612244898, 0.9285714285714286, 0.9285714285714286, 0.9306122448979591, 0.9306122448979591, 0.9326530612244898, 0.9326530612244898, 0.9346938775510204, 0.9346938775510204, 0.936734693877551, 0.936734693877551, 0.9387755102040817, 0.9387755102040817, 0.9408163265306122, 0.9408163265306122, 0.9428571428571428, 0.9428571428571428, 0.9448979591836735, 0.9448979591836735, 0.9489795918367347, 0.9489795918367347, 0.9510204081632653, 0.9510204081632653, 0.9530612244897959, 0.9530612244897959, 0.9551020408163265, 0.9551020408163265, 0.9571428571428572, 0.9571428571428572, 0.9571428571428572, 0.9571428571428572, 0.9591836734693877, 0.9591836734693877, 0.9612244897959183, 0.9612244897959183, 0.963265306122449, 0.963265306122449, 0.963265306122449, 0.9653061224489796, 0.9653061224489796, 0.9693877551020408, 0.9693877551020408, 0.9714285714285714, 0.9714285714285714, 0.9734693877551021, 0.9734693877551021, 0.9755102040816327, 0.9755102040816327, 0.9755102040816327, 0.9755102040816327, 0.9775510204081632, 0.9775510204081632, 0.9795918367346939, 0.9795918367346939, 0.9816326530612245, 0.9816326530612245, 0.9857142857142858, 0.9857142857142858, 0.9877551020408163, 0.9877551020408163, 0.9897959183673469, 0.9897959183673469, 0.9897959183673469, 0.9918367346938776, 0.9918367346938776, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9959183673469387, 0.9959183673469387, 0.9979591836734694, 0.9979591836734694, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, {"line": {"color": "rgb(205, 12, 24)", "dash": "dot", "width": 2}, "type": "scatter", "uid": "d82fd237-d81a-4a12-86db-1e9c78b44b81", "x": [0, 1], "y": [0, 1]}, {"colorscale": [[0.0, "rgb(165,0,38)"], [0.1111111111111111, "rgb(215,48,39)"], [0.2222222222222222, "rgb(244,109,67)"], [0.3333333333333333, "rgb(253,174,97)"], [0.4444444444444444, "rgb(254,224,144)"], [0.5555555555555556, "rgb(224,243,248)"], [0.6666666666666666, "rgb(171,217,233)"], [0.7777777777777778, "rgb(116,173,209)"], [0.8888888888888888, "rgb(69,117,180)"], [1.0, "rgb(49,54,149)"]], "name": "matrix", "showscale": false, "type": "heatmap", "uid": "a76c5dfa-60ff-4140-b761-c528bc6405a8", "x": ["Not churn", "Churn"], "xaxis": "x2", "y": ["Not churn", "Churn"], "yaxis": "y2", "z": [[1207, 61], [339, 151]]}],
                        {"autosize": false, "height": 600, "margin": {"b": 200}, "paper_bgcolor": "rgb(243,243,243)", "plot_bgcolor": "rgb(243,243,243)", "showlegend": false, "title": {"text": "Caracteristicas del modelo"}, "width": 900, "xaxis": {"domain": [0, 0.6], "gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio falso positivo"}}, "xaxis2": {"domain": [0.7, 1], "gridcolor": "rgb(255, 255, 255)", "tickangle": 90}, "yaxis": {"gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio verdadero positivo"}, "zerolinewidth": 1}, "yaxis2": {"anchor": "x2", "gridcolor": "rgb(255, 255, 255)"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('2d2516fe-e419-4726-b169-936a8020eece');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


### Gaussian NB


```python
from sklearn.naive_bayes import GaussianNB
GaussianNB = GaussianNB(priors=None)
churn_prediction(GaussianNB, train_XA, test_XA, train_YA, test_YA)
churn_prediction(GaussianNB, train_XB, test_XB, train_YB, test_YB)
```

    GaussianNB(priors=None, var_smoothing=1e-09)
    Accuracy   Score :  0.7582480091012515
    Area bajo la curva :  0.7654220047640508 
    


    /anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py:761: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    



<div>
        
        
            <div id="f153b755-2967-4c80-bd1d-155ad9e5ecc4" class="plotly-graph-div" style="height:600px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("f153b755-2967-4c80-bd1d-155ad9e5ecc4")) {
                    Plotly.newPlot(
                        'f153b755-2967-4c80-bd1d-155ad9e5ecc4',
                        [{"line": {"color": "rgb(22, 96, 167)", "width": 2}, "name": "ROC : 0.7654220047640508", "type": "scatter", "uid": "d59db550-ca0b-402e-b5a4-42ea0e2e8a82", "x": [0.0, 0.0, 0.0, 0.0007886435331230284, 0.0007886435331230284, 0.0015772870662460567, 0.0015772870662460567, 0.002365930599369085, 0.002365930599369085, 0.003943217665615142, 0.003943217665615142, 0.00473186119873817, 0.00473186119873817, 0.006309148264984227, 0.006309148264984227, 0.007886435331230283, 0.007886435331230283, 0.008675078864353312, 0.008675078864353312, 0.00946372239747634, 0.00946372239747634, 0.01025236593059937, 0.01025236593059937, 0.011041009463722398, 0.011041009463722398, 0.011829652996845425, 0.011829652996845425, 0.012618296529968454, 0.012618296529968454, 0.013406940063091483, 0.013406940063091483, 0.014195583596214511, 0.014195583596214511, 0.01498422712933754, 0.01498422712933754, 0.015772870662460567, 0.015772870662460567, 0.016561514195583597, 0.016561514195583597, 0.017350157728706624, 0.017350157728706624, 0.018138801261829655, 0.018138801261829655, 0.01971608832807571, 0.01971608832807571, 0.02050473186119874, 0.02050473186119874, 0.021293375394321766, 0.021293375394321766, 0.022082018927444796, 0.022082018927444796, 0.022870662460567823, 0.022870662460567823, 0.02365930599369085, 0.02365930599369085, 0.02444794952681388, 0.02444794952681388, 0.025236593059936908, 0.025236593059936908, 0.026025236593059938, 0.026025236593059938, 0.028391167192429023, 0.028391167192429023, 0.02917981072555205, 0.02917981072555205, 0.02996845425867508, 0.02996845425867508, 0.030757097791798107, 0.030757097791798107, 0.031545741324921134, 0.031545741324921134, 0.032334384858044164, 0.032334384858044164, 0.03470031545741325, 0.03470031545741325, 0.03627760252365931, 0.03627760252365931, 0.03785488958990536, 0.03785488958990536, 0.03943217665615142, 0.03943217665615142, 0.04022082018927445, 0.04022082018927445, 0.04100946372239748, 0.04100946372239748, 0.04416403785488959, 0.04416403785488959, 0.04574132492113565, 0.04574132492113565, 0.04810725552050473, 0.04810725552050473, 0.04889589905362776, 0.04889589905362776, 0.051261829652996846, 0.051261829652996846, 0.0528391167192429, 0.0528391167192429, 0.05362776025236593, 0.05362776025236593, 0.055205047318611984, 0.055205047318611984, 0.055993690851735015, 0.055993690851735015, 0.056782334384858045, 0.056782334384858045, 0.05993690851735016, 0.05993690851735016, 0.062302839116719244, 0.062302839116719244, 0.06466876971608833, 0.06466876971608833, 0.06545741324921135, 0.06545741324921135, 0.06624605678233439, 0.06624605678233439, 0.06703470031545741, 0.06703470031545741, 0.06782334384858044, 0.06782334384858044, 0.07018927444794952, 0.07018927444794952, 0.07413249211356467, 0.07413249211356467, 0.0749211356466877, 0.0749211356466877, 0.07649842271293375, 0.07649842271293375, 0.07728706624605679, 0.07728706624605679, 0.07807570977917981, 0.07807570977917981, 0.07886435331230283, 0.07886435331230283, 0.07965299684542587, 0.07965299684542587, 0.08123028391167192, 0.08123028391167192, 0.08201892744479496, 0.08201892744479496, 0.083596214511041, 0.083596214511041, 0.0859621451104101, 0.0859621451104101, 0.08675078864353312, 0.08675078864353312, 0.08753943217665615, 0.08753943217665615, 0.0914826498422713, 0.0914826498422713, 0.09227129337539432, 0.09227129337539432, 0.09384858044164038, 0.09384858044164038, 0.09858044164037855, 0.09858044164037855, 0.10173501577287067, 0.10173501577287067, 0.10252365930599369, 0.10252365930599369, 0.10488958990536278, 0.10488958990536278, 0.10646687697160884, 0.10646687697160884, 0.10725552050473186, 0.10725552050473186, 0.10804416403785488, 0.10804416403785488, 0.10962145110410094, 0.10962145110410094, 0.11041009463722397, 0.11041009463722397, 0.111198738170347, 0.111198738170347, 0.11198738170347003, 0.11198738170347003, 0.11277602523659307, 0.11277602523659307, 0.11435331230283911, 0.11435331230283911, 0.11514195583596215, 0.11514195583596215, 0.11593059936908517, 0.11593059936908517, 0.1167192429022082, 0.1167192429022082, 0.11829652996845426, 0.11829652996845426, 0.1222397476340694, 0.1222397476340694, 0.12381703470031545, 0.12381703470031545, 0.1253943217665615, 0.1253943217665615, 0.12697160883280756, 0.12697160883280756, 0.1277602523659306, 0.1277602523659306, 0.1309148264984227, 0.1309148264984227, 0.13249211356466878, 0.13249211356466878, 0.1332807570977918, 0.1332807570977918, 0.13485804416403785, 0.13485804416403785, 0.13564668769716087, 0.13564668769716087, 0.13643533123028392, 0.13643533123028392, 0.13722397476340695, 0.13722397476340695, 0.13801261829652997, 0.13801261829652997, 0.14037854889589904, 0.14037854889589904, 0.1411671924290221, 0.1411671924290221, 0.14195583596214512, 0.14195583596214512, 0.14353312302839116, 0.14353312302839116, 0.1443217665615142, 0.1443217665615142, 0.14511041009463724, 0.14511041009463724, 0.1474763406940063, 0.1474763406940063, 0.15063091482649843, 0.15063091482649843, 0.15220820189274448, 0.15220820189274448, 0.1529968454258675, 0.1529968454258675, 0.15694006309148265, 0.15694006309148265, 0.15772870662460567, 0.15772870662460567, 0.15851735015772872, 0.15851735015772872, 0.16009463722397477, 0.16009463722397477, 0.16167192429022081, 0.16167192429022081, 0.16482649842271294, 0.16482649842271294, 0.16561514195583596, 0.16561514195583596, 0.167192429022082, 0.167192429022082, 0.16876971608832808, 0.16876971608832808, 0.1695583596214511, 0.1695583596214511, 0.17034700315457413, 0.17034700315457413, 0.1719242902208202, 0.1719242902208202, 0.17350157728706625, 0.17350157728706625, 0.17665615141955837, 0.17665615141955837, 0.17823343848580442, 0.17823343848580442, 0.17981072555205047, 0.17981072555205047, 0.18138801261829654, 0.18138801261829654, 0.18690851735015773, 0.18690851735015773, 0.18769716088328076, 0.18769716088328076, 0.19794952681388012, 0.19794952681388012, 0.19873817034700317, 0.19873817034700317, 0.20268138801261829, 0.20268138801261829, 0.20504731861198738, 0.20504731861198738, 0.20741324921135645, 0.20741324921135645, 0.2082018927444795, 0.2082018927444795, 0.20977917981072555, 0.20977917981072555, 0.2113564668769716, 0.2113564668769716, 0.21293375394321767, 0.21293375394321767, 0.21529968454258674, 0.21529968454258674, 0.21608832807570977, 0.21608832807570977, 0.21845425867507887, 0.21845425867507887, 0.221608832807571, 0.221608832807571, 0.222397476340694, 0.222397476340694, 0.22476340694006308, 0.22476340694006308, 0.2279179810725552, 0.2279179810725552, 0.23107255520504733, 0.23107255520504733, 0.2334384858044164, 0.2334384858044164, 0.23974763406940064, 0.23974763406940064, 0.24290220820189273, 0.24290220820189273, 0.24526813880126183, 0.24526813880126183, 0.24605678233438485, 0.24605678233438485, 0.2476340694006309, 0.2476340694006309, 0.25236593059936907, 0.25236593059936907, 0.2578864353312303, 0.2578864353312303, 0.2586750788643533, 0.2586750788643533, 0.2610410094637224, 0.2610410094637224, 0.26419558359621453, 0.26419558359621453, 0.2665615141955836, 0.2665615141955836, 0.26813880126182965, 0.26813880126182965, 0.27208201892744477, 0.27208201892744477, 0.27996845425867506, 0.27996845425867506, 0.28154574132492116, 0.28154574132492116, 0.28470031545741326, 0.28470031545741326, 0.2854889589905363, 0.2854889589905363, 0.2870662460567823, 0.2870662460567823, 0.2973186119873817, 0.2973186119873817, 0.30757097791798105, 0.30757097791798105, 0.31309148264984227, 0.31309148264984227, 0.31703470031545744, 0.31703470031545744, 0.32334384858044163, 0.32334384858044163, 0.3249211356466877, 0.3249211356466877, 0.3257097791798107, 0.3257097791798107, 0.3312302839116719, 0.3312302839116719, 0.33201892744479494, 0.33201892744479494, 0.334384858044164, 0.334384858044164, 0.3351735015772871, 0.3351735015772871, 0.3383280757097792, 0.3383280757097792, 0.33990536277602523, 0.33990536277602523, 0.3414826498422713, 0.3414826498422713, 0.3470031545741325, 0.3470031545741325, 0.3477917981072555, 0.3477917981072555, 0.3501577287066246, 0.3501577287066246, 0.35173501577287064, 0.35173501577287064, 0.3556782334384858, 0.3556782334384858, 0.361198738170347, 0.361198738170347, 0.3627760252365931, 0.3627760252365931, 0.3698738170347003, 0.3698738170347003, 0.37066246056782337, 0.37066246056782337, 0.3722397476340694, 0.3722397476340694, 0.37618296529968454, 0.37618296529968454, 0.3801261829652997, 0.3801261829652997, 0.3856466876971609, 0.3856466876971609, 0.3864353312302839, 0.3864353312302839, 0.388801261829653, 0.38958990536277605, 0.3911671924290221, 0.39747634069400634, 0.39747634069400634, 0.40063091482649843, 0.40063091482649843, 0.4069400630914827, 0.4069400630914827, 0.42113564668769715, 0.42113564668769715, 0.4219242902208202, 0.4219242902208202, 0.4242902208201893, 0.4242902208201893, 0.4274447949526814, 0.4274447949526814, 0.44085173501577285, 0.44085173501577285, 0.444006309148265, 0.444006309148265, 0.4471608832807571, 0.4471608832807571, 0.44873817034700314, 0.44873817034700314, 0.4503154574132492, 0.4503154574132492, 0.45110410094637227, 0.45110410094637227, 0.46214511041009465, 0.46214511041009465, 0.4629337539432177, 0.4629337539432177, 0.4645110410094637, 0.4645110410094637, 0.46529968454258674, 0.46529968454258674, 0.46845425867507884, 0.46845425867507884, 0.47082018927444796, 0.47082018927444796, 0.47318611987381703, 0.47318611987381703, 0.4802839116719243, 0.4802839116719243, 0.4810725552050473, 0.4810725552050473, 0.4842271293375394, 0.4842271293375394, 0.48580441640378547, 0.48580441640378547, 0.49053627760252366, 0.49053627760252366, 0.4921135646687697, 0.4921135646687697, 0.5, 0.5, 0.501577287066246, 0.501577287066246, 0.5023659305993691, 0.5023659305993691, 0.5047318611987381, 0.5047318611987381, 0.5078864353312302, 0.5078864353312302, 0.5630914826498423, 0.5630914826498423, 0.5749211356466877, 0.5749211356466877, 0.5788643533123028, 0.5788643533123028, 0.5843848580441641, 0.5843848580441641, 0.6175078864353313, 0.6175078864353313, 0.6269716088328076, 0.6269716088328076, 0.6317034700315457, 0.6317034700315457, 0.6348580441640379, 0.6348580441640379, 0.6616719242902208, 0.6616719242902208, 0.6805993690851735, 0.6805993690851735, 0.695583596214511, 0.695583596214511, 0.7074132492113565, 0.7074132492113565, 0.7105678233438486, 0.7105678233438486, 0.7358044164037855, 0.7358044164037855, 0.7397476340694006, 0.7397476340694006, 0.8769716088328076, 0.8769716088328076, 0.88801261829653, 0.88801261829653, 0.8935331230283912, 0.8935331230283912, 1.0], "y": [0.0, 0.0020408163265306124, 0.004081632653061225, 0.004081632653061225, 0.01020408163265306, 0.01020408163265306, 0.036734693877551024, 0.036734693877551024, 0.04693877551020408, 0.04693877551020408, 0.053061224489795916, 0.053061224489795916, 0.05714285714285714, 0.05714285714285714, 0.06326530612244897, 0.06326530612244897, 0.07551020408163266, 0.07551020408163266, 0.08571428571428572, 0.08571428571428572, 0.08979591836734693, 0.08979591836734693, 0.09795918367346938, 0.09795918367346938, 0.10816326530612246, 0.10816326530612246, 0.1306122448979592, 0.1306122448979592, 0.1469387755102041, 0.1469387755102041, 0.15510204081632653, 0.15510204081632653, 0.16122448979591836, 0.16122448979591836, 0.1673469387755102, 0.1673469387755102, 0.17346938775510204, 0.17346938775510204, 0.17551020408163265, 0.17551020408163265, 0.18571428571428572, 0.18571428571428572, 0.19387755102040816, 0.19387755102040816, 0.19591836734693877, 0.19591836734693877, 0.19795918367346937, 0.19795918367346937, 0.21224489795918366, 0.21224489795918366, 0.21428571428571427, 0.21428571428571427, 0.2163265306122449, 0.2163265306122449, 0.22448979591836735, 0.22448979591836735, 0.22857142857142856, 0.22857142857142856, 0.23061224489795917, 0.23061224489795917, 0.23673469387755103, 0.23673469387755103, 0.23877551020408164, 0.23877551020408164, 0.24285714285714285, 0.24285714285714285, 0.24489795918367346, 0.24489795918367346, 0.25510204081632654, 0.25510204081632654, 0.26326530612244897, 0.26326530612244897, 0.2714285714285714, 0.2714285714285714, 0.27755102040816326, 0.27755102040816326, 0.2816326530612245, 0.2816326530612245, 0.29591836734693877, 0.29591836734693877, 0.2979591836734694, 0.2979591836734694, 0.3, 0.3, 0.3081632653061224, 0.3081632653061224, 0.3122448979591837, 0.3122448979591837, 0.3183673469387755, 0.3183673469387755, 0.3224489795918367, 0.3224489795918367, 0.32857142857142857, 0.32857142857142857, 0.3326530612244898, 0.3326530612244898, 0.336734693877551, 0.336734693877551, 0.33877551020408164, 0.33877551020408164, 0.34285714285714286, 0.34285714285714286, 0.3469387755102041, 0.3469387755102041, 0.3489795918367347, 0.3489795918367347, 0.3510204081632653, 0.3510204081632653, 0.35714285714285715, 0.35714285714285715, 0.3673469387755102, 0.3673469387755102, 0.37551020408163266, 0.37551020408163266, 0.3816326530612245, 0.3816326530612245, 0.3836734693877551, 0.3836734693877551, 0.38979591836734695, 0.38979591836734695, 0.4122448979591837, 0.4122448979591837, 0.41836734693877553, 0.41836734693877553, 0.4387755102040816, 0.4387755102040816, 0.44081632653061226, 0.44081632653061226, 0.44285714285714284, 0.44285714285714284, 0.4448979591836735, 0.4448979591836735, 0.4489795918367347, 0.4489795918367347, 0.4530612244897959, 0.4530612244897959, 0.45510204081632655, 0.45510204081632655, 0.45918367346938777, 0.45918367346938777, 0.463265306122449, 0.463265306122449, 0.4673469387755102, 0.4673469387755102, 0.47551020408163264, 0.47551020408163264, 0.4897959183673469, 0.4897959183673469, 0.49387755102040815, 0.49387755102040815, 0.49795918367346936, 0.49795918367346936, 0.5, 0.5, 0.5020408163265306, 0.5020408163265306, 0.5040816326530613, 0.5040816326530613, 0.5122448979591837, 0.5122448979591837, 0.5142857142857142, 0.5142857142857142, 0.5163265306122449, 0.5163265306122449, 0.5183673469387755, 0.5183673469387755, 0.5204081632653061, 0.5204081632653061, 0.5244897959183673, 0.5244897959183673, 0.5265306122448979, 0.5265306122448979, 0.5285714285714286, 0.5285714285714286, 0.5306122448979592, 0.5306122448979592, 0.5346938775510204, 0.5346938775510204, 0.5408163265306123, 0.5408163265306123, 0.5469387755102041, 0.5469387755102041, 0.5489795918367347, 0.5489795918367347, 0.5510204081632653, 0.5510204081632653, 0.5551020408163265, 0.5551020408163265, 0.5612244897959183, 0.5612244897959183, 0.563265306122449, 0.563265306122449, 0.5653061224489796, 0.5653061224489796, 0.5673469387755102, 0.5673469387755102, 0.573469387755102, 0.573469387755102, 0.5755102040816327, 0.5755102040816327, 0.5775510204081633, 0.5775510204081633, 0.5795918367346938, 0.5795918367346938, 0.5816326530612245, 0.5816326530612245, 0.5877551020408164, 0.5877551020408164, 0.5897959183673469, 0.5897959183673469, 0.5959183673469388, 0.5959183673469388, 0.5979591836734693, 0.5979591836734693, 0.6, 0.6, 0.610204081632653, 0.610204081632653, 0.6122448979591837, 0.6122448979591837, 0.6183673469387755, 0.6183673469387755, 0.6224489795918368, 0.6224489795918368, 0.6326530612244898, 0.6326530612244898, 0.6346938775510204, 0.6346938775510204, 0.636734693877551, 0.636734693877551, 0.6387755102040816, 0.6387755102040816, 0.6428571428571429, 0.6428571428571429, 0.6448979591836734, 0.6448979591836734, 0.6469387755102041, 0.6469387755102041, 0.6489795918367347, 0.6489795918367347, 0.6530612244897959, 0.6530612244897959, 0.6551020408163265, 0.6551020408163265, 0.6571428571428571, 0.6571428571428571, 0.6612244897959184, 0.6612244897959184, 0.6653061224489796, 0.6653061224489796, 0.6673469387755102, 0.6673469387755102, 0.6693877551020408, 0.6693877551020408, 0.6714285714285714, 0.6714285714285714, 0.673469387755102, 0.673469387755102, 0.6775510204081633, 0.6775510204081633, 0.6857142857142857, 0.6857142857142857, 0.6877551020408164, 0.6877551020408164, 0.689795918367347, 0.689795918367347, 0.6918367346938775, 0.6918367346938775, 0.6938775510204082, 0.6938775510204082, 0.6979591836734694, 0.6979591836734694, 0.7020408163265306, 0.7020408163265306, 0.710204081632653, 0.710204081632653, 0.7142857142857143, 0.7142857142857143, 0.7163265306122449, 0.7163265306122449, 0.7183673469387755, 0.7183673469387755, 0.7204081632653061, 0.7204081632653061, 0.7244897959183674, 0.7244897959183674, 0.726530612244898, 0.726530612244898, 0.7306122448979592, 0.7306122448979592, 0.7326530612244898, 0.7326530612244898, 0.7346938775510204, 0.7346938775510204, 0.736734693877551, 0.736734693877551, 0.7387755102040816, 0.7387755102040816, 0.7448979591836735, 0.7448979591836735, 0.7489795918367347, 0.7489795918367347, 0.7591836734693878, 0.7591836734693878, 0.7612244897959184, 0.7612244897959184, 0.763265306122449, 0.763265306122449, 0.7653061224489796, 0.7653061224489796, 0.7673469387755102, 0.7673469387755102, 0.7693877551020408, 0.7693877551020408, 0.7795918367346939, 0.7795918367346939, 0.7816326530612245, 0.7816326530612245, 0.7836734693877551, 0.7836734693877551, 0.7857142857142857, 0.7857142857142857, 0.7877551020408163, 0.7877551020408163, 0.7918367346938775, 0.7918367346938775, 0.7938775510204081, 0.7938775510204081, 0.7959183673469388, 0.7959183673469388, 0.7979591836734694, 0.7979591836734694, 0.8, 0.8, 0.8040816326530612, 0.8040816326530612, 0.8061224489795918, 0.8061224489795918, 0.8081632653061225, 0.8081632653061225, 0.810204081632653, 0.810204081632653, 0.8122448979591836, 0.8122448979591836, 0.8142857142857143, 0.8142857142857143, 0.8163265306122449, 0.8163265306122449, 0.8183673469387756, 0.8183673469387756, 0.8204081632653061, 0.8204081632653061, 0.8244897959183674, 0.8244897959183674, 0.826530612244898, 0.826530612244898, 0.8326530612244898, 0.8326530612244898, 0.8346938775510204, 0.8346938775510204, 0.8367346938775511, 0.8367346938775511, 0.8387755102040816, 0.8387755102040816, 0.8428571428571429, 0.8428571428571429, 0.8448979591836735, 0.8448979591836735, 0.8469387755102041, 0.8469387755102041, 0.8489795918367347, 0.8489795918367347, 0.8510204081632653, 0.8510204081632653, 0.8530612244897959, 0.8530612244897959, 0.8551020408163266, 0.8551020408163266, 0.8571428571428571, 0.8571428571428571, 0.8591836734693877, 0.8591836734693877, 0.8612244897959184, 0.8612244897959184, 0.8653061224489796, 0.8653061224489796, 0.8673469387755102, 0.8673469387755102, 0.8693877551020408, 0.8693877551020408, 0.8714285714285714, 0.8714285714285714, 0.8734693877551021, 0.8734693877551021, 0.8775510204081632, 0.8775510204081632, 0.8816326530612245, 0.8816326530612245, 0.8836734693877552, 0.8836734693877552, 0.8857142857142857, 0.8857142857142857, 0.8857142857142857, 0.8877551020408163, 0.8877551020408163, 0.889795918367347, 0.889795918367347, 0.8918367346938776, 0.8918367346938776, 0.8938775510204081, 0.8938775510204081, 0.8959183673469387, 0.8959183673469387, 0.8979591836734694, 0.8979591836734694, 0.9, 0.9, 0.9020408163265307, 0.9020408163265307, 0.9040816326530612, 0.9040816326530612, 0.9061224489795918, 0.9061224489795918, 0.9081632653061225, 0.9081632653061225, 0.9102040816326531, 0.9102040816326531, 0.9122448979591836, 0.9122448979591836, 0.9183673469387755, 0.9183673469387755, 0.9204081632653062, 0.9204081632653062, 0.9224489795918367, 0.9224489795918367, 0.9244897959183673, 0.9244897959183673, 0.9285714285714286, 0.9285714285714286, 0.9306122448979591, 0.9306122448979591, 0.9326530612244898, 0.9326530612244898, 0.9346938775510204, 0.9346938775510204, 0.9387755102040817, 0.9387755102040817, 0.9408163265306122, 0.9408163265306122, 0.9448979591836735, 0.9448979591836735, 0.9469387755102041, 0.9469387755102041, 0.9489795918367347, 0.9489795918367347, 0.9510204081632653, 0.9510204081632653, 0.9530612244897959, 0.9530612244897959, 0.9551020408163265, 0.9551020408163265, 0.9571428571428572, 0.9571428571428572, 0.9591836734693877, 0.9591836734693877, 0.9612244897959183, 0.9612244897959183, 0.963265306122449, 0.963265306122449, 0.9653061224489796, 0.9653061224489796, 0.9673469387755103, 0.9673469387755103, 0.9693877551020408, 0.9693877551020408, 0.9714285714285714, 0.9714285714285714, 0.9734693877551021, 0.9734693877551021, 0.9755102040816327, 0.9755102040816327, 0.9775510204081632, 0.9775510204081632, 0.9795918367346939, 0.9795918367346939, 0.9816326530612245, 0.9816326530612245, 0.9836734693877551, 0.9836734693877551, 0.9897959183673469, 0.9897959183673469, 0.9918367346938776, 0.9918367346938776, 0.9938775510204082, 0.9938775510204082, 0.9959183673469387, 0.9959183673469387, 0.9979591836734694, 0.9979591836734694, 1.0, 1.0]}, {"line": {"color": "rgb(205, 12, 24)", "dash": "dot", "width": 2}, "type": "scatter", "uid": "fdab58f1-782c-4539-b142-08e133afba95", "x": [0, 1], "y": [0, 1]}, {"colorscale": [[0.0, "rgb(165,0,38)"], [0.1111111111111111, "rgb(215,48,39)"], [0.2222222222222222, "rgb(244,109,67)"], [0.3333333333333333, "rgb(253,174,97)"], [0.4444444444444444, "rgb(254,224,144)"], [0.5555555555555556, "rgb(224,243,248)"], [0.6666666666666666, "rgb(171,217,233)"], [0.7777777777777778, "rgb(116,173,209)"], [0.8888888888888888, "rgb(69,117,180)"], [1.0, "rgb(49,54,149)"]], "name": "matrix", "showscale": false, "type": "heatmap", "uid": "731ab49b-e272-44ab-bda8-a1c8f8779c58", "x": ["Not churn", "Churn"], "xaxis": "x2", "y": ["Not churn", "Churn"], "yaxis": "y2", "z": [[950, 318], [107, 383]]}],
                        {"autosize": false, "height": 600, "margin": {"b": 200}, "paper_bgcolor": "rgb(243,243,243)", "plot_bgcolor": "rgb(243,243,243)", "showlegend": false, "title": {"text": "Caracteristicas del modelo"}, "width": 900, "xaxis": {"domain": [0, 0.6], "gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio falso positivo"}}, "xaxis2": {"domain": [0.7, 1], "gridcolor": "rgb(255, 255, 255)", "tickangle": 90}, "yaxis": {"gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio verdadero positivo"}, "zerolinewidth": 1}, "yaxis2": {"anchor": "x2", "gridcolor": "rgb(255, 255, 255)"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('f153b755-2967-4c80-bd1d-155ad9e5ecc4');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


    /anaconda3/lib/python3.7/site-packages/sklearn/utils/validation.py:761: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    


    GaussianNB(priors=None, var_smoothing=1e-09)
    Accuracy   Score :  0.7531285551763367
    Area bajo la curva :  0.7656296272452199 
    



<div>
        
        
            <div id="846e9701-7c95-4ac1-96ac-1a3bf9087151" class="plotly-graph-div" style="height:600px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("846e9701-7c95-4ac1-96ac-1a3bf9087151")) {
                    Plotly.newPlot(
                        '846e9701-7c95-4ac1-96ac-1a3bf9087151',
                        [{"line": {"color": "rgb(22, 96, 167)", "width": 2}, "name": "ROC : 0.7656296272452199", "type": "scatter", "uid": "fa756fd4-2542-4991-a5bd-69bf8933ca0c", "x": [0.0, 0.0, 0.0, 0.0007886435331230284, 0.0007886435331230284, 0.0015772870662460567, 0.0015772870662460567, 0.002365930599369085, 0.002365930599369085, 0.00473186119873817, 0.00473186119873817, 0.005520504731861199, 0.005520504731861199, 0.007097791798107256, 0.007097791798107256, 0.007886435331230283, 0.007886435331230283, 0.008675078864353312, 0.008675078864353312, 0.00946372239747634, 0.00946372239747634, 0.01025236593059937, 0.01025236593059937, 0.011041009463722398, 0.011041009463722398, 0.011829652996845425, 0.011829652996845425, 0.012618296529968454, 0.012618296529968454, 0.013406940063091483, 0.013406940063091483, 0.014195583596214511, 0.014195583596214511, 0.01498422712933754, 0.01498422712933754, 0.015772870662460567, 0.015772870662460567, 0.016561514195583597, 0.016561514195583597, 0.017350157728706624, 0.017350157728706624, 0.018138801261829655, 0.018138801261829655, 0.01971608832807571, 0.01971608832807571, 0.02050473186119874, 0.02050473186119874, 0.021293375394321766, 0.021293375394321766, 0.022082018927444796, 0.022082018927444796, 0.022870662460567823, 0.022870662460567823, 0.02444794952681388, 0.02444794952681388, 0.025236593059936908, 0.025236593059936908, 0.026025236593059938, 0.026025236593059938, 0.026813880126182965, 0.026813880126182965, 0.028391167192429023, 0.028391167192429023, 0.030757097791798107, 0.030757097791798107, 0.031545741324921134, 0.031545741324921134, 0.03548895899053628, 0.03548895899053628, 0.03706624605678233, 0.03706624605678233, 0.04022082018927445, 0.04022082018927445, 0.04100946372239748, 0.04100946372239748, 0.04652996845425868, 0.04652996845425868, 0.04889589905362776, 0.04889589905362776, 0.04968454258675079, 0.04968454258675079, 0.050473186119873815, 0.050473186119873815, 0.052050473186119876, 0.052050473186119876, 0.05362776025236593, 0.05362776025236593, 0.05441640378548896, 0.05441640378548896, 0.055205047318611984, 0.055205047318611984, 0.055993690851735015, 0.055993690851735015, 0.056782334384858045, 0.056782334384858045, 0.057570977917981075, 0.057570977917981075, 0.0583596214511041, 0.0583596214511041, 0.05914826498422713, 0.05914826498422713, 0.05993690851735016, 0.05993690851735016, 0.06072555205047318, 0.06072555205047318, 0.061514195583596214, 0.061514195583596214, 0.062302839116719244, 0.062302839116719244, 0.0638801261829653, 0.0638801261829653, 0.06545741324921135, 0.06545741324921135, 0.06624605678233439, 0.06624605678233439, 0.06861198738170347, 0.06861198738170347, 0.0694006309148265, 0.0694006309148265, 0.07097791798107256, 0.07097791798107256, 0.07176656151419558, 0.07176656151419558, 0.07255520504731862, 0.07255520504731862, 0.0749211356466877, 0.0749211356466877, 0.07728706624605679, 0.07728706624605679, 0.07886435331230283, 0.07886435331230283, 0.07965299684542587, 0.07965299684542587, 0.0804416403785489, 0.0804416403785489, 0.08280757097791798, 0.08280757097791798, 0.08517350157728706, 0.08517350157728706, 0.0859621451104101, 0.0859621451104101, 0.08753943217665615, 0.08753943217665615, 0.08832807570977919, 0.08832807570977919, 0.08911671924290221, 0.08911671924290221, 0.08990536277602523, 0.08990536277602523, 0.0914826498422713, 0.0914826498422713, 0.09305993690851735, 0.09305993690851735, 0.09542586750788644, 0.09542586750788644, 0.09621451104100946, 0.09621451104100946, 0.09779179810725552, 0.09779179810725552, 0.09858044164037855, 0.09858044164037855, 0.10094637223974763, 0.10094637223974763, 0.10173501577287067, 0.10173501577287067, 0.10410094637223975, 0.10410094637223975, 0.10488958990536278, 0.10488958990536278, 0.10646687697160884, 0.10646687697160884, 0.10883280757097792, 0.10883280757097792, 0.11041009463722397, 0.11041009463722397, 0.111198738170347, 0.111198738170347, 0.11356466876971609, 0.11356466876971609, 0.11908517350157728, 0.11908517350157728, 0.11987381703470032, 0.11987381703470032, 0.1222397476340694, 0.1222397476340694, 0.12381703470031545, 0.12381703470031545, 0.12618296529968454, 0.12618296529968454, 0.12697160883280756, 0.12697160883280756, 0.1277602523659306, 0.1277602523659306, 0.12854889589905363, 0.12854889589905363, 0.13170347003154576, 0.13170347003154576, 0.13249211356466878, 0.13249211356466878, 0.13564668769716087, 0.13564668769716087, 0.13722397476340695, 0.13722397476340695, 0.13801261829652997, 0.13801261829652997, 0.13958990536277602, 0.13958990536277602, 0.14037854889589904, 0.14037854889589904, 0.14353312302839116, 0.14353312302839116, 0.1443217665615142, 0.1443217665615142, 0.14511041009463724, 0.14511041009463724, 0.14589905362776026, 0.14589905362776026, 0.14668769716088328, 0.14668769716088328, 0.14905362776025236, 0.14905362776025236, 0.15141955835962145, 0.15141955835962145, 0.15220820189274448, 0.15220820189274448, 0.1529968454258675, 0.1529968454258675, 0.15378548895899052, 0.15378548895899052, 0.15615141955835962, 0.15615141955835962, 0.15851735015772872, 0.15851735015772872, 0.16167192429022081, 0.16167192429022081, 0.1632492113564669, 0.1632492113564669, 0.1640378548895899, 0.1640378548895899, 0.16640378548895898, 0.16640378548895898, 0.16798107255520506, 0.16798107255520506, 0.17034700315457413, 0.17034700315457413, 0.1719242902208202, 0.1719242902208202, 0.17271293375394323, 0.17271293375394323, 0.1774447949526814, 0.1774447949526814, 0.18217665615141956, 0.18217665615141956, 0.18533123028391169, 0.18533123028391169, 0.18690851735015773, 0.18690851735015773, 0.18769716088328076, 0.18769716088328076, 0.19006309148264985, 0.19006309148264985, 0.19085173501577288, 0.19085173501577288, 0.19558359621451105, 0.19558359621451105, 0.19637223974763407, 0.19637223974763407, 0.1995268138801262, 0.1995268138801262, 0.20347003154574134, 0.20347003154574134, 0.20741324921135645, 0.20741324921135645, 0.20899053627760253, 0.20899053627760253, 0.21056782334384858, 0.21056782334384858, 0.21214511041009465, 0.21214511041009465, 0.21845425867507887, 0.21845425867507887, 0.2192429022082019, 0.2192429022082019, 0.22082018927444794, 0.22082018927444794, 0.221608832807571, 0.221608832807571, 0.222397476340694, 0.222397476340694, 0.22318611987381703, 0.22318611987381703, 0.22870662460567823, 0.22870662460567823, 0.22949526813880125, 0.22949526813880125, 0.23107255520504733, 0.23107255520504733, 0.2358044164037855, 0.2358044164037855, 0.23817034700315456, 0.23817034700315456, 0.23974763406940064, 0.23974763406940064, 0.2421135646687697, 0.2421135646687697, 0.2444794952681388, 0.2444794952681388, 0.24526813880126183, 0.24526813880126183, 0.2555205047318612, 0.2555205047318612, 0.2578864353312303, 0.2578864353312303, 0.2586750788643533, 0.2586750788643533, 0.2618296529968454, 0.2618296529968454, 0.2657728706624606, 0.2657728706624606, 0.26735015772870663, 0.26735015772870663, 0.2689274447949527, 0.2689274447949527, 0.2752365930599369, 0.2752365930599369, 0.277602523659306, 0.277602523659306, 0.28391167192429023, 0.28391167192429023, 0.2862776025236593, 0.2862776025236593, 0.2917981072555205, 0.2917981072555205, 0.29258675078864355, 0.29258675078864355, 0.2949526813880126, 0.2949526813880126, 0.29574132492113564, 0.29574132492113564, 0.3020504731861199, 0.3020504731861199, 0.30362776025236593, 0.30362776025236593, 0.306782334384858, 0.306782334384858, 0.31230283911671924, 0.31230283911671924, 0.3138801261829653, 0.3138801261829653, 0.31782334384858046, 0.31782334384858046, 0.3186119873817035, 0.3186119873817035, 0.3264984227129338, 0.3264984227129338, 0.333596214511041, 0.333596214511041, 0.33990536277602523, 0.33990536277602523, 0.34069400630914826, 0.34069400630914826, 0.3438485804416404, 0.34463722397476343, 0.3462145110410095, 0.35252365930599366, 0.35252365930599366, 0.35410094637223977, 0.35410094637223977, 0.35962145110410093, 0.35962145110410093, 0.3635646687697161, 0.3635646687697161, 0.3659305993690852, 0.3659305993690852, 0.37302839116719244, 0.37302839116719244, 0.37381703470031546, 0.37381703470031546, 0.38091482649842273, 0.38091482649842273, 0.3832807570977918, 0.3832807570977918, 0.3840694006309148, 0.3840694006309148, 0.39826498422712936, 0.39826498422712936, 0.4022082018927445, 0.4022082018927445, 0.4061514195583596, 0.4061514195583596, 0.4077287066246057, 0.4077287066246057, 0.40930599369085174, 0.40930599369085174, 0.41009463722397477, 0.41009463722397477, 0.41246056782334384, 0.41246056782334384, 0.41324921135646686, 0.41324921135646686, 0.4242902208201893, 0.4242902208201893, 0.4250788643533123, 0.4250788643533123, 0.42586750788643535, 0.42586750788643535, 0.444006309148265, 0.444006309148265, 0.4526813880126183, 0.4526813880126183, 0.4605678233438486, 0.4605678233438486, 0.46845425867507884, 0.46845425867507884, 0.4834384858044164, 0.4834384858044164, 0.4881703470031546, 0.4881703470031546, 0.49763406940063093, 0.49763406940063093, 0.501577287066246, 0.501577287066246, 0.5031545741324921, 0.5031545741324921, 0.5070977917981072, 0.5070977917981072, 0.5094637223974764, 0.5094637223974764, 0.5118296529968455, 0.5118296529968455, 0.5197160883280757, 0.5197160883280757, 0.5378548895899053, 0.5378548895899053, 0.5410094637223974, 0.5410094637223974, 0.5481072555205048, 0.5481072555205048, 0.5623028391167192, 0.5623028391167192, 0.5733438485804416, 0.5733438485804416, 0.582018927444795, 0.582018927444795, 0.5859621451104101, 0.5859621451104101, 0.5970031545741324, 0.5970031545741324, 0.6041009463722398, 0.6041009463722398, 0.61198738170347, 0.61198738170347, 0.613564668769716, 0.613564668769716, 0.6182965299684543, 0.6182965299684543, 0.6246056782334385, 0.6246056782334385, 0.6301261829652997, 0.6301261829652997, 0.6348580441640379, 0.6348580441640379, 0.6403785488958991, 0.6403785488958991, 0.6427444794952681, 0.6427444794952681, 0.6451104100946372, 0.6451104100946372, 0.6656151419558359, 0.6656151419558359, 0.6853312302839116, 0.6853312302839116, 0.6884858044164038, 0.6884858044164038, 0.693217665615142, 0.693217665615142, 0.6979495268138801, 0.6979495268138801, 0.7657728706624606, 0.7657728706624606, 0.7949526813880127, 0.7949526813880127, 0.8036277602523659, 0.8036277602523659, 0.8130914826498423, 0.8130914826498423, 0.8927444794952681, 0.8927444794952681, 0.8951104100946372, 0.8951104100946372, 0.9132492113564669, 0.9132492113564669, 0.9329652996845426, 0.9329652996845426, 0.9369085173501577, 0.9369085173501577, 0.9566246056782335, 0.9566246056782335, 1.0], "y": [0.0, 0.0020408163265306124, 0.004081632653061225, 0.004081632653061225, 0.012244897959183673, 0.012244897959183673, 0.026530612244897958, 0.026530612244897958, 0.04693877551020408, 0.04693877551020408, 0.061224489795918366, 0.061224489795918366, 0.06326530612244897, 0.06326530612244897, 0.0653061224489796, 0.0653061224489796, 0.0673469387755102, 0.0673469387755102, 0.07755102040816327, 0.07755102040816327, 0.0836734693877551, 0.0836734693877551, 0.09591836734693877, 0.09591836734693877, 0.09795918367346938, 0.09795918367346938, 0.11836734693877551, 0.11836734693877551, 0.1346938775510204, 0.1346938775510204, 0.16326530612244897, 0.16326530612244897, 0.1673469387755102, 0.1673469387755102, 0.17346938775510204, 0.17346938775510204, 0.18571428571428572, 0.18571428571428572, 0.19183673469387755, 0.19183673469387755, 0.19387755102040816, 0.19387755102040816, 0.20204081632653062, 0.20204081632653062, 0.20612244897959184, 0.20612244897959184, 0.20816326530612245, 0.20816326530612245, 0.21428571428571427, 0.21428571428571427, 0.2163265306122449, 0.2163265306122449, 0.21836734693877552, 0.21836734693877552, 0.22244897959183674, 0.22244897959183674, 0.23265306122448978, 0.23265306122448978, 0.23877551020408164, 0.23877551020408164, 0.24489795918367346, 0.24489795918367346, 0.2571428571428571, 0.2571428571428571, 0.2612244897959184, 0.2612244897959184, 0.2693877551020408, 0.2693877551020408, 0.2755102040816326, 0.2755102040816326, 0.2979591836734694, 0.2979591836734694, 0.30612244897959184, 0.30612244897959184, 0.31020408163265306, 0.31020408163265306, 0.3163265306122449, 0.3163265306122449, 0.3183673469387755, 0.3183673469387755, 0.32857142857142857, 0.32857142857142857, 0.3306122448979592, 0.3306122448979592, 0.336734693877551, 0.336734693877551, 0.34285714285714286, 0.34285714285714286, 0.3469387755102041, 0.3469387755102041, 0.3489795918367347, 0.3489795918367347, 0.35306122448979593, 0.35306122448979593, 0.363265306122449, 0.363265306122449, 0.3653061224489796, 0.3653061224489796, 0.37142857142857144, 0.37142857142857144, 0.373469387755102, 0.373469387755102, 0.3795918367346939, 0.3795918367346939, 0.3836734693877551, 0.3836734693877551, 0.38571428571428573, 0.38571428571428573, 0.3877551020408163, 0.3877551020408163, 0.38979591836734695, 0.38979591836734695, 0.39387755102040817, 0.39387755102040817, 0.39591836734693875, 0.39591836734693875, 0.41020408163265304, 0.41020408163265304, 0.41836734693877553, 0.41836734693877553, 0.42244897959183675, 0.42244897959183675, 0.42653061224489797, 0.42653061224489797, 0.4326530612244898, 0.4326530612244898, 0.4346938775510204, 0.4346938775510204, 0.43673469387755104, 0.43673469387755104, 0.4489795918367347, 0.4489795918367347, 0.46122448979591835, 0.46122448979591835, 0.463265306122449, 0.463265306122449, 0.47346938775510206, 0.47346938775510206, 0.47959183673469385, 0.47959183673469385, 0.4816326530612245, 0.4816326530612245, 0.48367346938775513, 0.48367346938775513, 0.4857142857142857, 0.4857142857142857, 0.48775510204081635, 0.48775510204081635, 0.4897959183673469, 0.4897959183673469, 0.49183673469387756, 0.49183673469387756, 0.4959183673469388, 0.4959183673469388, 0.49795918367346936, 0.49795918367346936, 0.5, 0.5, 0.5040816326530613, 0.5040816326530613, 0.5122448979591837, 0.5122448979591837, 0.5142857142857142, 0.5142857142857142, 0.5306122448979592, 0.5306122448979592, 0.5326530612244897, 0.5326530612244897, 0.5346938775510204, 0.5346938775510204, 0.5387755102040817, 0.5387755102040817, 0.5408163265306123, 0.5408163265306123, 0.5448979591836735, 0.5448979591836735, 0.5489795918367347, 0.5489795918367347, 0.5510204081632653, 0.5510204081632653, 0.5591836734693878, 0.5591836734693878, 0.5653061224489796, 0.5653061224489796, 0.5673469387755102, 0.5673469387755102, 0.5693877551020409, 0.5693877551020409, 0.573469387755102, 0.573469387755102, 0.5857142857142857, 0.5857142857142857, 0.5918367346938775, 0.5918367346938775, 0.5959183673469388, 0.5959183673469388, 0.5979591836734693, 0.5979591836734693, 0.6040816326530613, 0.6040816326530613, 0.6061224489795919, 0.6061224489795919, 0.6081632653061224, 0.6081632653061224, 0.610204081632653, 0.610204081632653, 0.6142857142857143, 0.6142857142857143, 0.6183673469387755, 0.6183673469387755, 0.6265306122448979, 0.6265306122448979, 0.6285714285714286, 0.6285714285714286, 0.6306122448979592, 0.6306122448979592, 0.6448979591836734, 0.6448979591836734, 0.6469387755102041, 0.6469387755102041, 0.6489795918367347, 0.6489795918367347, 0.6530612244897959, 0.6530612244897959, 0.6551020408163265, 0.6551020408163265, 0.6632653061224489, 0.6632653061224489, 0.6653061224489796, 0.6653061224489796, 0.6673469387755102, 0.6673469387755102, 0.6693877551020408, 0.6693877551020408, 0.673469387755102, 0.673469387755102, 0.6755102040816326, 0.6755102040816326, 0.6775510204081633, 0.6775510204081633, 0.6795918367346939, 0.6795918367346939, 0.6836734693877551, 0.6836734693877551, 0.6857142857142857, 0.6857142857142857, 0.6877551020408164, 0.6877551020408164, 0.689795918367347, 0.689795918367347, 0.6918367346938775, 0.6918367346938775, 0.6959183673469388, 0.6959183673469388, 0.7, 0.7, 0.7061224489795919, 0.7061224489795919, 0.7163265306122449, 0.7163265306122449, 0.7183673469387755, 0.7183673469387755, 0.7204081632653061, 0.7204081632653061, 0.7224489795918367, 0.7224489795918367, 0.7244897959183674, 0.7244897959183674, 0.726530612244898, 0.726530612244898, 0.7326530612244898, 0.7326530612244898, 0.7346938775510204, 0.7346938775510204, 0.736734693877551, 0.736734693877551, 0.7387755102040816, 0.7387755102040816, 0.7428571428571429, 0.7428571428571429, 0.7448979591836735, 0.7448979591836735, 0.7489795918367347, 0.7489795918367347, 0.7510204081632653, 0.7510204081632653, 0.753061224489796, 0.753061224489796, 0.7571428571428571, 0.7571428571428571, 0.7591836734693878, 0.7591836734693878, 0.7612244897959184, 0.7612244897959184, 0.763265306122449, 0.763265306122449, 0.7653061224489796, 0.7653061224489796, 0.7673469387755102, 0.7673469387755102, 0.773469387755102, 0.773469387755102, 0.7755102040816326, 0.7755102040816326, 0.7795918367346939, 0.7795918367346939, 0.7816326530612245, 0.7816326530612245, 0.7836734693877551, 0.7836734693877551, 0.7857142857142857, 0.7857142857142857, 0.789795918367347, 0.789795918367347, 0.7918367346938775, 0.7918367346938775, 0.7938775510204081, 0.7938775510204081, 0.7959183673469388, 0.7959183673469388, 0.7979591836734694, 0.7979591836734694, 0.8, 0.8, 0.8020408163265306, 0.8020408163265306, 0.8040816326530612, 0.8040816326530612, 0.8061224489795918, 0.8061224489795918, 0.810204081632653, 0.810204081632653, 0.8122448979591836, 0.8122448979591836, 0.8142857142857143, 0.8142857142857143, 0.8163265306122449, 0.8163265306122449, 0.8183673469387756, 0.8183673469387756, 0.8204081632653061, 0.8204081632653061, 0.8224489795918367, 0.8224489795918367, 0.8244897959183674, 0.8244897959183674, 0.8285714285714286, 0.8285714285714286, 0.8306122448979592, 0.8306122448979592, 0.8367346938775511, 0.8367346938775511, 0.8408163265306122, 0.8408163265306122, 0.8428571428571429, 0.8428571428571429, 0.8448979591836735, 0.8448979591836735, 0.8469387755102041, 0.8469387755102041, 0.8510204081632653, 0.8510204081632653, 0.8530612244897959, 0.8530612244897959, 0.8530612244897959, 0.8551020408163266, 0.8551020408163266, 0.8571428571428571, 0.8571428571428571, 0.8591836734693877, 0.8591836734693877, 0.8612244897959184, 0.8612244897959184, 0.863265306122449, 0.863265306122449, 0.8653061224489796, 0.8653061224489796, 0.8673469387755102, 0.8673469387755102, 0.8693877551020408, 0.8693877551020408, 0.8714285714285714, 0.8714285714285714, 0.8734693877551021, 0.8734693877551021, 0.8755102040816326, 0.8755102040816326, 0.8775510204081632, 0.8775510204081632, 0.8816326530612245, 0.8816326530612245, 0.8836734693877552, 0.8836734693877552, 0.8857142857142857, 0.8857142857142857, 0.8877551020408163, 0.8877551020408163, 0.889795918367347, 0.889795918367347, 0.8918367346938776, 0.8918367346938776, 0.8938775510204081, 0.8938775510204081, 0.8959183673469387, 0.8959183673469387, 0.8979591836734694, 0.8979591836734694, 0.9, 0.9, 0.9020408163265307, 0.9020408163265307, 0.9040816326530612, 0.9040816326530612, 0.9061224489795918, 0.9061224489795918, 0.9081632653061225, 0.9081632653061225, 0.9102040816326531, 0.9102040816326531, 0.9122448979591836, 0.9122448979591836, 0.9142857142857143, 0.9142857142857143, 0.9163265306122449, 0.9163265306122449, 0.9183673469387755, 0.9183673469387755, 0.9204081632653062, 0.9204081632653062, 0.9224489795918367, 0.9224489795918367, 0.9244897959183673, 0.9244897959183673, 0.926530612244898, 0.926530612244898, 0.9285714285714286, 0.9285714285714286, 0.9326530612244898, 0.9326530612244898, 0.9346938775510204, 0.9346938775510204, 0.936734693877551, 0.936734693877551, 0.9387755102040817, 0.9387755102040817, 0.9408163265306122, 0.9408163265306122, 0.9428571428571428, 0.9428571428571428, 0.9448979591836735, 0.9448979591836735, 0.9489795918367347, 0.9489795918367347, 0.9510204081632653, 0.9510204081632653, 0.9530612244897959, 0.9530612244897959, 0.9551020408163265, 0.9551020408163265, 0.9571428571428572, 0.9571428571428572, 0.9591836734693877, 0.9591836734693877, 0.9612244897959183, 0.9612244897959183, 0.963265306122449, 0.963265306122449, 0.9653061224489796, 0.9653061224489796, 0.9673469387755103, 0.9673469387755103, 0.9693877551020408, 0.9693877551020408, 0.9734693877551021, 0.9734693877551021, 0.9755102040816327, 0.9755102040816327, 0.9775510204081632, 0.9775510204081632, 0.9795918367346939, 0.9795918367346939, 0.9816326530612245, 0.9816326530612245, 0.9836734693877551, 0.9836734693877551, 0.9857142857142858, 0.9857142857142858, 0.9877551020408163, 0.9877551020408163, 0.9918367346938776, 0.9918367346938776, 0.9938775510204082, 0.9938775510204082, 0.9959183673469387, 0.9959183673469387, 0.9979591836734694, 0.9979591836734694, 1.0, 1.0]}, {"line": {"color": "rgb(205, 12, 24)", "dash": "dot", "width": 2}, "type": "scatter", "uid": "a2c752ce-07f8-4d52-98c5-30ab8cd22f96", "x": [0, 1], "y": [0, 1]}, {"colorscale": [[0.0, "rgb(165,0,38)"], [0.1111111111111111, "rgb(215,48,39)"], [0.2222222222222222, "rgb(244,109,67)"], [0.3333333333333333, "rgb(253,174,97)"], [0.4444444444444444, "rgb(254,224,144)"], [0.5555555555555556, "rgb(224,243,248)"], [0.6666666666666666, "rgb(171,217,233)"], [0.7777777777777778, "rgb(116,173,209)"], [0.8888888888888888, "rgb(69,117,180)"], [1.0, "rgb(49,54,149)"]], "name": "matrix", "showscale": false, "type": "heatmap", "uid": "a617e5cf-df9c-4e80-9e3c-81a8be89eb12", "x": ["Not churn", "Churn"], "xaxis": "x2", "y": ["Not churn", "Churn"], "yaxis": "y2", "z": [[935, 333], [101, 389]]}],
                        {"autosize": false, "height": 600, "margin": {"b": 200}, "paper_bgcolor": "rgb(243,243,243)", "plot_bgcolor": "rgb(243,243,243)", "showlegend": false, "title": {"text": "Caracteristicas del modelo"}, "width": 900, "xaxis": {"domain": [0, 0.6], "gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio falso positivo"}}, "xaxis2": {"domain": [0.7, 1], "gridcolor": "rgb(255, 255, 255)", "tickangle": 90}, "yaxis": {"gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio verdadero positivo"}, "zerolinewidth": 1}, "yaxis2": {"anchor": "x2", "gridcolor": "rgb(255, 255, 255)"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('846e9701-7c95-4ac1-96ac-1a3bf9087151');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


### XGBoost Classifier


```python
from xgboost import XGBClassifier

XGBClassifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bytree=1, gamma=0, learning_rate=0.9, max_delta_step=0,
                    max_depth = 7, min_child_weight=1, missing=None, n_estimators=100,
                    n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                    silent=True, subsample=1)

churn_prediction(XGBClassifier, train_XA, test_XA, train_YA, test_YA)
churn_prediction(XGBClassifier, train_XB, test_XB, train_YB, test_YB)
```

    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:219: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    
    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:252: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    


    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.9,
           max_delta_step=0, max_depth=7, min_child_weight=1, missing=None,
           n_estimators=100, n_jobs=1, nthread=None,
           objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
           subsample=1, verbosity=1)
    Accuracy   Score :  0.7713310580204779
    Area bajo la curva :  0.6887175690465461 
    



<div>
        
        
            <div id="d05b476a-64dd-46fc-a32a-fa5ad6103cc9" class="plotly-graph-div" style="height:600px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("d05b476a-64dd-46fc-a32a-fa5ad6103cc9")) {
                    Plotly.newPlot(
                        'd05b476a-64dd-46fc-a32a-fa5ad6103cc9',
                        [{"line": {"color": "rgb(22, 96, 167)", "width": 2}, "name": "ROC : 0.6887175690465461", "type": "scatter", "uid": "e2126bab-931a-4cb6-82ec-15f7b9eb122a", "x": [0.0, 0.0, 0.0, 0.0007886435331230284, 0.0007886435331230284, 0.0015772870662460567, 0.0015772870662460567, 0.002365930599369085, 0.002365930599369085, 0.0031545741324921135, 0.0031545741324921135, 0.003943217665615142, 0.003943217665615142, 0.00473186119873817, 0.00473186119873817, 0.005520504731861199, 0.005520504731861199, 0.006309148264984227, 0.006309148264984227, 0.007097791798107256, 0.007097791798107256, 0.007886435331230283, 0.007886435331230283, 0.008675078864353312, 0.008675078864353312, 0.00946372239747634, 0.00946372239747634, 0.01025236593059937, 0.01025236593059937, 0.011041009463722398, 0.011041009463722398, 0.011829652996845425, 0.011829652996845425, 0.01498422712933754, 0.01498422712933754, 0.015772870662460567, 0.015772870662460567, 0.016561514195583597, 0.016561514195583597, 0.017350157728706624, 0.017350157728706624, 0.01892744479495268, 0.01892744479495268, 0.01971608832807571, 0.02050473186119874, 0.02050473186119874, 0.021293375394321766, 0.021293375394321766, 0.022870662460567823, 0.022870662460567823, 0.02365930599369085, 0.02365930599369085, 0.02444794952681388, 0.02444794952681388, 0.026025236593059938, 0.026025236593059938, 0.026813880126182965, 0.026813880126182965, 0.02917981072555205, 0.02917981072555205, 0.02996845425867508, 0.02996845425867508, 0.030757097791798107, 0.030757097791798107, 0.031545741324921134, 0.031545741324921134, 0.032334384858044164, 0.032334384858044164, 0.033123028391167195, 0.033123028391167195, 0.03470031545741325, 0.03470031545741325, 0.03548895899053628, 0.03548895899053628, 0.03706624605678233, 0.03706624605678233, 0.038643533123028394, 0.038643533123028394, 0.04022082018927445, 0.04022082018927445, 0.04337539432176656, 0.04337539432176656, 0.04416403785488959, 0.04416403785488959, 0.04652996845425868, 0.04652996845425868, 0.04889589905362776, 0.04889589905362776, 0.04968454258675079, 0.04968454258675079, 0.0528391167192429, 0.0528391167192429, 0.05362776025236593, 0.05362776025236593, 0.056782334384858045, 0.056782334384858045, 0.05914826498422713, 0.05914826498422713, 0.05993690851735016, 0.05993690851735016, 0.06072555205047318, 0.06072555205047318, 0.061514195583596214, 0.061514195583596214, 0.0638801261829653, 0.0638801261829653, 0.06466876971608833, 0.06466876971608833, 0.06545741324921135, 0.06545741324921135, 0.06624605678233439, 0.06624605678233439, 0.06703470031545741, 0.06703470031545741, 0.06782334384858044, 0.06782334384858044, 0.06861198738170347, 0.06861198738170347, 0.0694006309148265, 0.0694006309148265, 0.07097791798107256, 0.07097791798107256, 0.07255520504731862, 0.07255520504731862, 0.07334384858044164, 0.07334384858044164, 0.0749211356466877, 0.0749211356466877, 0.07570977917981073, 0.07570977917981073, 0.07807570977917981, 0.07807570977917981, 0.07886435331230283, 0.07886435331230283, 0.0804416403785489, 0.0804416403785489, 0.08123028391167192, 0.08123028391167192, 0.08201892744479496, 0.08201892744479496, 0.08438485804416404, 0.08438485804416404, 0.08517350157728706, 0.08517350157728706, 0.08753943217665615, 0.08753943217665615, 0.08832807570977919, 0.08832807570977919, 0.08911671924290221, 0.08911671924290221, 0.08990536277602523, 0.08990536277602523, 0.09069400630914827, 0.09069400630914827, 0.0914826498422713, 0.0914826498422713, 0.09227129337539432, 0.09227129337539432, 0.0946372239747634, 0.0946372239747634, 0.09621451104100946, 0.09621451104100946, 0.09700315457413249, 0.09700315457413249, 0.09779179810725552, 0.09779179810725552, 0.09936908517350158, 0.09936908517350158, 0.10094637223974763, 0.10094637223974763, 0.10173501577287067, 0.10173501577287067, 0.10252365930599369, 0.10252365930599369, 0.10331230283911672, 0.10331230283911672, 0.10725552050473186, 0.10725552050473186, 0.10883280757097792, 0.10883280757097792, 0.11041009463722397, 0.11041009463722397, 0.111198738170347, 0.111198738170347, 0.11198738170347003, 0.11198738170347003, 0.11277602523659307, 0.11277602523659307, 0.11356466876971609, 0.11356466876971609, 0.11593059936908517, 0.11593059936908517, 0.1167192429022082, 0.1167192429022082, 0.11829652996845426, 0.11829652996845426, 0.12066246056782334, 0.12066246056782334, 0.12145110410094637, 0.12145110410094637, 0.12460567823343849, 0.12460567823343849, 0.12618296529968454, 0.12618296529968454, 0.12854889589905363, 0.12854889589905363, 0.12933753943217666, 0.12933753943217666, 0.13012618296529968, 0.13012618296529968, 0.1309148264984227, 0.1309148264984227, 0.13170347003154576, 0.13170347003154576, 0.13485804416403785, 0.13485804416403785, 0.13722397476340695, 0.13722397476340695, 0.14037854889589904, 0.14037854889589904, 0.1411671924290221, 0.1411671924290221, 0.14195583596214512, 0.14195583596214512, 0.14589905362776026, 0.14589905362776026, 0.14668769716088328, 0.14668769716088328, 0.14826498422712933, 0.14826498422712933, 0.1498422712933754, 0.1498422712933754, 0.15141955835962145, 0.15141955835962145, 0.15220820189274448, 0.15220820189274448, 0.15378548895899052, 0.15378548895899052, 0.1553627760252366, 0.1553627760252366, 0.15615141955835962, 0.15615141955835962, 0.15694006309148265, 0.15694006309148265, 0.15772870662460567, 0.15772870662460567, 0.1632492113564669, 0.1632492113564669, 0.16482649842271294, 0.16482649842271294, 0.16640378548895898, 0.16640378548895898, 0.16798107255520506, 0.16798107255520506, 0.17113564668769715, 0.17113564668769715, 0.17271293375394323, 0.17902208201892744, 0.17902208201892744, 0.17981072555205047, 0.17981072555205047, 0.1837539432176656, 0.1837539432176656, 0.18533123028391169, 0.18533123028391169, 0.18769716088328076, 0.18769716088328076, 0.1892744479495268, 0.1892744479495268, 0.19006309148264985, 0.19006309148264985, 0.1916403785488959, 0.1916403785488959, 0.19400630914826497, 0.19400630914826497, 0.19479495268138802, 0.19479495268138802, 0.19637223974763407, 0.1971608832807571, 0.1995268138801262, 0.1995268138801262, 0.20031545741324921, 0.20031545741324921, 0.20110410094637224, 0.20110410094637224, 0.20189274447949526, 0.20189274447949526, 0.20268138801261829, 0.20268138801261829, 0.20347003154574134, 0.20347003154574134, 0.2058359621451104, 0.2058359621451104, 0.20741324921135645, 0.20741324921135645, 0.2082018927444795, 0.2082018927444795, 0.20899053627760253, 0.20899053627760253, 0.20977917981072555, 0.20977917981072555, 0.21214511041009465, 0.21214511041009465, 0.21529968454258674, 0.21529968454258674, 0.21608832807570977, 0.21608832807570977, 0.21687697160883282, 0.21687697160883282, 0.21766561514195584, 0.21766561514195584, 0.22397476340694006, 0.22397476340694006, 0.22555205047318613, 0.22555205047318613, 0.22634069400630916, 0.22634069400630916, 0.22712933753943218, 0.22712933753943218, 0.23186119873817035, 0.23186119873817035, 0.23264984227129337, 0.23264984227129337, 0.2334384858044164, 0.2334384858044164, 0.2358044164037855, 0.2358044164037855, 0.23738170347003154, 0.23738170347003154, 0.23895899053627762, 0.23895899053627762, 0.24053627760252366, 0.24053627760252366, 0.24132492113564669, 0.24132492113564669, 0.24290220820189273, 0.24290220820189273, 0.24526813880126183, 0.24526813880126183, 0.2476340694006309, 0.2476340694006309, 0.25157728706624605, 0.25157728706624605, 0.25236593059936907, 0.25236593059936907, 0.2531545741324921, 0.2531545741324921, 0.2555205047318612, 0.2555205047318612, 0.25630914826498424, 0.25630914826498424, 0.2578864353312303, 0.2578864353312303, 0.2586750788643533, 0.2586750788643533, 0.26025236593059936, 0.26025236593059936, 0.2618296529968454, 0.2618296529968454, 0.2634069400630915, 0.2634069400630915, 0.26419558359621453, 0.26419558359621453, 0.2657728706624606, 0.2657728706624606, 0.2665615141955836, 0.2665615141955836, 0.26813880126182965, 0.26813880126182965, 0.2744479495268139, 0.2744479495268139, 0.278391167192429, 0.278391167192429, 0.27917981072555204, 0.27917981072555204, 0.28154574132492116, 0.28154574132492116, 0.2823343848580442, 0.2823343848580442, 0.28391167192429023, 0.28391167192429023, 0.28785488958990535, 0.28785488958990535, 0.2886435331230284, 0.2886435331230284, 0.29337539432176657, 0.29337539432176657, 0.2973186119873817, 0.2973186119873817, 0.30047318611987384, 0.30047318611987384, 0.30126182965299686, 0.30126182965299686, 0.30362776025236593, 0.30362776025236593, 0.305205047318612, 0.305205047318612, 0.305993690851735, 0.305993690851735, 0.306782334384858, 0.306782334384858, 0.30914826498422715, 0.30914826498422715, 0.3107255520504732, 0.3107255520504732, 0.31624605678233436, 0.31624605678233436, 0.3186119873817035, 0.3186119873817035, 0.32018927444794953, 0.32018927444794953, 0.32413249211356465, 0.32413249211356465, 0.3249211356466877, 0.3249211356466877, 0.3257097791798107, 0.3257097791798107, 0.3264984227129338, 0.3264984227129338, 0.33280757097791797, 0.33280757097791797, 0.334384858044164, 0.334384858044164, 0.3351735015772871, 0.3351735015772871, 0.33753943217665616, 0.33753943217665616, 0.3391167192429022, 0.3391167192429022, 0.3414826498422713, 0.3414826498422713, 0.3430599369085173, 0.3430599369085173, 0.34542586750788645, 0.34542586750788645, 0.35331230283911674, 0.35331230283911674, 0.3548895899053628, 0.3548895899053628, 0.35725552050473186, 0.35725552050473186, 0.35962145110410093, 0.35962145110410093, 0.3667192429022082, 0.3667192429022082, 0.36829652996845424, 0.36829652996845424, 0.3714511041009464, 0.3714511041009464, 0.37381703470031546, 0.37381703470031546, 0.3840694006309148, 0.3840694006309148, 0.38485804416403785, 0.38485804416403785, 0.388801261829653, 0.388801261829653, 0.39274447949526814, 0.39274447949526814, 0.39747634069400634, 0.39747634069400634, 0.4037854889589905, 0.4037854889589905, 0.40457413249211355, 0.40457413249211355, 0.4069400630914827, 0.4069400630914827, 0.4085173501577287, 0.4085173501577287, 0.41719242902208203, 0.41719242902208203, 0.42665615141955837, 0.42665615141955837, 0.4274447949526814, 0.4274447949526814, 0.42902208201892744, 0.42902208201892744, 0.43217665615141954, 0.43217665615141954, 0.43296529968454256, 0.43296529968454256, 0.43375394321766564, 0.43375394321766564, 0.4361198738170347, 0.4361198738170347, 0.4503154574132492, 0.4503154574132492, 0.45347003154574134, 0.45347003154574134, 0.4558359621451104, 0.4558359621451104, 0.4637223974763407, 0.4637223974763407, 0.46608832807570977, 0.46608832807570977, 0.4794952681388013, 0.4794952681388013, 0.48264984227129337, 0.48264984227129337, 0.4834384858044164, 0.4834384858044164, 0.4842271293375394, 0.4842271293375394, 0.48501577287066244, 0.48501577287066244, 0.500788643533123, 0.500788643533123, 0.5047318611987381, 0.5047318611987381, 0.5070977917981072, 0.5070977917981072, 0.5134069400630915, 0.5134069400630915, 0.5212933753943217, 0.5212933753943217, 0.5291798107255521, 0.5291798107255521, 0.5339116719242902, 0.5339116719242902, 0.5457413249211357, 0.5457413249211357, 0.5473186119873817, 0.5473186119873817, 0.5496845425867508, 0.5496845425867508, 0.5623028391167192, 0.5623028391167192, 0.5654574132492114, 0.5670347003154574, 0.582018927444795, 0.582018927444795, 0.5843848580441641, 0.5843848580441641, 0.5914826498422713, 0.5914826498422713, 0.5977917981072555, 0.5977917981072555, 0.6056782334384858, 0.6056782334384858, 0.637223974763407, 0.637223974763407, 0.6435331230283912, 0.6435331230283912, 0.6719242902208202, 0.6719242902208202, 0.6829652996845426, 0.6829652996845426, 0.6869085173501577, 0.6869085173501577, 0.7089905362776026, 0.7105678233438486, 0.7302839116719243, 0.7302839116719243, 0.7342271293375394, 0.7342271293375394, 0.7539432176656151, 0.7539432176656151, 0.7720820189274448, 0.7720820189274448, 0.7783911671924291, 0.7799684542586751, 0.8130914826498423, 0.8130914826498423, 0.831230283911672, 0.832807570977918, 0.8572555205047319, 0.8572555205047319, 0.8848580441640379, 0.8848580441640379, 0.9361198738170347, 0.9361198738170347, 0.9400630914826499, 0.9416403785488959, 0.9755520504731862, 0.9771293375394322, 1.0], "y": [0.0, 0.0020408163265306124, 0.018367346938775512, 0.018367346938775512, 0.022448979591836733, 0.022448979591836733, 0.024489795918367346, 0.024489795918367346, 0.030612244897959183, 0.030612244897959183, 0.03877551020408163, 0.03877551020408163, 0.04081632653061224, 0.04081632653061224, 0.04285714285714286, 0.04285714285714286, 0.053061224489795916, 0.053061224489795916, 0.0653061224489796, 0.0653061224489796, 0.0673469387755102, 0.0673469387755102, 0.09795918367346938, 0.09795918367346938, 0.1, 0.1, 0.12040816326530612, 0.12040816326530612, 0.12244897959183673, 0.12244897959183673, 0.12448979591836734, 0.12448979591836734, 0.12653061224489795, 0.12653061224489795, 0.12857142857142856, 0.12857142857142856, 0.1306122448979592, 0.1306122448979592, 0.1326530612244898, 0.1326530612244898, 0.13673469387755102, 0.13673469387755102, 0.14081632653061224, 0.14081632653061224, 0.14285714285714285, 0.1469387755102041, 0.1469387755102041, 0.15510204081632653, 0.15510204081632653, 0.15918367346938775, 0.15918367346938775, 0.16326530612244897, 0.16326530612244897, 0.1673469387755102, 0.1673469387755102, 0.17142857142857143, 0.17142857142857143, 0.18775510204081633, 0.18775510204081633, 0.18979591836734694, 0.18979591836734694, 0.19591836734693877, 0.19591836734693877, 0.19795918367346937, 0.19795918367346937, 0.20408163265306123, 0.20408163265306123, 0.20612244897959184, 0.20612244897959184, 0.21020408163265306, 0.21020408163265306, 0.21224489795918366, 0.21224489795918366, 0.22244897959183674, 0.22244897959183674, 0.22653061224489796, 0.22653061224489796, 0.22857142857142856, 0.22857142857142856, 0.23265306122448978, 0.23265306122448978, 0.23469387755102042, 0.23469387755102042, 0.23877551020408164, 0.23877551020408164, 0.2510204081632653, 0.2510204081632653, 0.2530612244897959, 0.2530612244897959, 0.2571428571428571, 0.2571428571428571, 0.2673469387755102, 0.2673469387755102, 0.2755102040816326, 0.2755102040816326, 0.2816326530612245, 0.2816326530612245, 0.2857142857142857, 0.2857142857142857, 0.28775510204081634, 0.28775510204081634, 0.2897959183673469, 0.2897959183673469, 0.29591836734693877, 0.29591836734693877, 0.3040816326530612, 0.3040816326530612, 0.31020408163265306, 0.31020408163265306, 0.3163265306122449, 0.3163265306122449, 0.3224489795918367, 0.3224489795918367, 0.32653061224489793, 0.32653061224489793, 0.32857142857142857, 0.32857142857142857, 0.336734693877551, 0.336734693877551, 0.3408163265306122, 0.3408163265306122, 0.3448979591836735, 0.3448979591836735, 0.3469387755102041, 0.3469387755102041, 0.3489795918367347, 0.3489795918367347, 0.35918367346938773, 0.35918367346938773, 0.36122448979591837, 0.36122448979591837, 0.363265306122449, 0.363265306122449, 0.3653061224489796, 0.3653061224489796, 0.3673469387755102, 0.3673469387755102, 0.37551020408163266, 0.37551020408163266, 0.3795918367346939, 0.3795918367346939, 0.3816326530612245, 0.3816326530612245, 0.3836734693877551, 0.3836734693877551, 0.38571428571428573, 0.38571428571428573, 0.38979591836734695, 0.38979591836734695, 0.39387755102040817, 0.39387755102040817, 0.4, 0.4, 0.40408163265306124, 0.40408163265306124, 0.4061224489795918, 0.4061224489795918, 0.41020408163265304, 0.41020408163265304, 0.4142857142857143, 0.4142857142857143, 0.4163265306122449, 0.4163265306122449, 0.41836734693877553, 0.41836734693877553, 0.4204081632653061, 0.4204081632653061, 0.42244897959183675, 0.42244897959183675, 0.4306122448979592, 0.4306122448979592, 0.4326530612244898, 0.4326530612244898, 0.4346938775510204, 0.4346938775510204, 0.43673469387755104, 0.43673469387755104, 0.4387755102040816, 0.4387755102040816, 0.44081632653061226, 0.44081632653061226, 0.44285714285714284, 0.44285714285714284, 0.44693877551020406, 0.44693877551020406, 0.4489795918367347, 0.4489795918367347, 0.45102040816326533, 0.45102040816326533, 0.45714285714285713, 0.45714285714285713, 0.463265306122449, 0.463265306122449, 0.48367346938775513, 0.48367346938775513, 0.49183673469387756, 0.49183673469387756, 0.49387755102040815, 0.49387755102040815, 0.49795918367346936, 0.49795918367346936, 0.5020408163265306, 0.5020408163265306, 0.5040816326530613, 0.5040816326530613, 0.5081632653061224, 0.5081632653061224, 0.5102040816326531, 0.5102040816326531, 0.5122448979591837, 0.5122448979591837, 0.5244897959183673, 0.5244897959183673, 0.5265306122448979, 0.5265306122448979, 0.5306122448979592, 0.5306122448979592, 0.5326530612244897, 0.5326530612244897, 0.536734693877551, 0.536734693877551, 0.5408163265306123, 0.5408163265306123, 0.5448979591836735, 0.5448979591836735, 0.5489795918367347, 0.5489795918367347, 0.5510204081632653, 0.5510204081632653, 0.5551020408163265, 0.5551020408163265, 0.5571428571428572, 0.5571428571428572, 0.563265306122449, 0.563265306122449, 0.5673469387755102, 0.5673469387755102, 0.5693877551020409, 0.5693877551020409, 0.5714285714285714, 0.5714285714285714, 0.5775510204081633, 0.5775510204081633, 0.5857142857142857, 0.5857142857142857, 0.5918367346938775, 0.5918367346938775, 0.5938775510204082, 0.5938775510204082, 0.5959183673469388, 0.5959183673469388, 0.6020408163265306, 0.6020408163265306, 0.6040816326530613, 0.6040816326530613, 0.6081632653061224, 0.6081632653061224, 0.6081632653061224, 0.610204081632653, 0.610204081632653, 0.6122448979591837, 0.6122448979591837, 0.6163265306122448, 0.6163265306122448, 0.6183673469387755, 0.6183673469387755, 0.6204081632653061, 0.6204081632653061, 0.6244897959183674, 0.6244897959183674, 0.6285714285714286, 0.6285714285714286, 0.6306122448979592, 0.6306122448979592, 0.6326530612244898, 0.6326530612244898, 0.6387755102040816, 0.6387755102040816, 0.6408163265306123, 0.6408163265306123, 0.6428571428571429, 0.6428571428571429, 0.6448979591836734, 0.6448979591836734, 0.6469387755102041, 0.6469387755102041, 0.6489795918367347, 0.6489795918367347, 0.6510204081632653, 0.6510204081632653, 0.6530612244897959, 0.6530612244897959, 0.6551020408163265, 0.6551020408163265, 0.6571428571428571, 0.6571428571428571, 0.6591836734693878, 0.6591836734693878, 0.6612244897959184, 0.6612244897959184, 0.6632653061224489, 0.6632653061224489, 0.6653061224489796, 0.6653061224489796, 0.6693877551020408, 0.6693877551020408, 0.6755102040816326, 0.6755102040816326, 0.6795918367346939, 0.6795918367346939, 0.6816326530612244, 0.6816326530612244, 0.6836734693877551, 0.6836734693877551, 0.6857142857142857, 0.6857142857142857, 0.6938775510204082, 0.6938775510204082, 0.6979591836734694, 0.6979591836734694, 0.7, 0.7, 0.7020408163265306, 0.7020408163265306, 0.7040816326530612, 0.7040816326530612, 0.7061224489795919, 0.7061224489795919, 0.710204081632653, 0.710204081632653, 0.7122448979591837, 0.7122448979591837, 0.7142857142857143, 0.7142857142857143, 0.7163265306122449, 0.7163265306122449, 0.7183673469387755, 0.7183673469387755, 0.7224489795918367, 0.7224489795918367, 0.7244897959183674, 0.7244897959183674, 0.726530612244898, 0.726530612244898, 0.7285714285714285, 0.7285714285714285, 0.7306122448979592, 0.7306122448979592, 0.7326530612244898, 0.7326530612244898, 0.7346938775510204, 0.7346938775510204, 0.736734693877551, 0.736734693877551, 0.7428571428571429, 0.7428571428571429, 0.746938775510204, 0.746938775510204, 0.7489795918367347, 0.7489795918367347, 0.7510204081632653, 0.7510204081632653, 0.7571428571428571, 0.7571428571428571, 0.763265306122449, 0.763265306122449, 0.7673469387755102, 0.7673469387755102, 0.7714285714285715, 0.7714285714285715, 0.773469387755102, 0.773469387755102, 0.7755102040816326, 0.7755102040816326, 0.7775510204081633, 0.7775510204081633, 0.7795918367346939, 0.7795918367346939, 0.7816326530612245, 0.7816326530612245, 0.7836734693877551, 0.7836734693877551, 0.7857142857142857, 0.7857142857142857, 0.7877551020408163, 0.7877551020408163, 0.789795918367347, 0.789795918367347, 0.7959183673469388, 0.7959183673469388, 0.7979591836734694, 0.7979591836734694, 0.8, 0.8, 0.8040816326530612, 0.8040816326530612, 0.8061224489795918, 0.8061224489795918, 0.8081632653061225, 0.8081632653061225, 0.810204081632653, 0.810204081632653, 0.8122448979591836, 0.8122448979591836, 0.8142857142857143, 0.8142857142857143, 0.8163265306122449, 0.8163265306122449, 0.8183673469387756, 0.8183673469387756, 0.8204081632653061, 0.8204081632653061, 0.8224489795918367, 0.8224489795918367, 0.8244897959183674, 0.8244897959183674, 0.826530612244898, 0.826530612244898, 0.8285714285714286, 0.8285714285714286, 0.8306122448979592, 0.8306122448979592, 0.8346938775510204, 0.8346938775510204, 0.8367346938775511, 0.8367346938775511, 0.8387755102040816, 0.8387755102040816, 0.8408163265306122, 0.8408163265306122, 0.8428571428571429, 0.8428571428571429, 0.8469387755102041, 0.8469387755102041, 0.8489795918367347, 0.8489795918367347, 0.8510204081632653, 0.8510204081632653, 0.8551020408163266, 0.8551020408163266, 0.8571428571428571, 0.8571428571428571, 0.8591836734693877, 0.8591836734693877, 0.8612244897959184, 0.8612244897959184, 0.863265306122449, 0.863265306122449, 0.8653061224489796, 0.8653061224489796, 0.8673469387755102, 0.8673469387755102, 0.8693877551020408, 0.8693877551020408, 0.8714285714285714, 0.8714285714285714, 0.8734693877551021, 0.8734693877551021, 0.8755102040816326, 0.8755102040816326, 0.8775510204081632, 0.8775510204081632, 0.8816326530612245, 0.8816326530612245, 0.8836734693877552, 0.8836734693877552, 0.8877551020408163, 0.8877551020408163, 0.8918367346938776, 0.8918367346938776, 0.8938775510204081, 0.8938775510204081, 0.8959183673469387, 0.8959183673469387, 0.8979591836734694, 0.8979591836734694, 0.9, 0.9, 0.9061224489795918, 0.9061224489795918, 0.9081632653061225, 0.9081632653061225, 0.9102040816326531, 0.9102040816326531, 0.9122448979591836, 0.9122448979591836, 0.9142857142857143, 0.9142857142857143, 0.9163265306122449, 0.9163265306122449, 0.9183673469387755, 0.9183673469387755, 0.9204081632653062, 0.9204081632653062, 0.9224489795918367, 0.9224489795918367, 0.9244897959183673, 0.9244897959183673, 0.926530612244898, 0.926530612244898, 0.9285714285714286, 0.9285714285714286, 0.9306122448979591, 0.9306122448979591, 0.9326530612244898, 0.9326530612244898, 0.9346938775510204, 0.9346938775510204, 0.936734693877551, 0.936734693877551, 0.9387755102040817, 0.9387755102040817, 0.9408163265306122, 0.9408163265306122, 0.9428571428571428, 0.9428571428571428, 0.9448979591836735, 0.9448979591836735, 0.9469387755102041, 0.9469387755102041, 0.9489795918367347, 0.9489795918367347, 0.9530612244897959, 0.9530612244897959, 0.9551020408163265, 0.9551020408163265, 0.9571428571428572, 0.9571428571428572, 0.9571428571428572, 0.9571428571428572, 0.9591836734693877, 0.9591836734693877, 0.9612244897959183, 0.9612244897959183, 0.9653061224489796, 0.9653061224489796, 0.9673469387755103, 0.9673469387755103, 0.9693877551020408, 0.9693877551020408, 0.9714285714285714, 0.9714285714285714, 0.9734693877551021, 0.9734693877551021, 0.9755102040816327, 0.9755102040816327, 0.9775510204081632, 0.9775510204081632, 0.9795918367346939, 0.9795918367346939, 0.9795918367346939, 0.9795918367346939, 0.9836734693877551, 0.9836734693877551, 0.9857142857142858, 0.9857142857142858, 0.9877551020408163, 0.9877551020408163, 0.9918367346938776, 0.9918367346938776, 0.9918367346938776, 0.9918367346938776, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9959183673469387, 0.9959183673469387, 0.9979591836734694, 0.9979591836734694, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]}, {"line": {"color": "rgb(205, 12, 24)", "dash": "dot", "width": 2}, "type": "scatter", "uid": "000b8deb-ba03-479a-aaa3-bdcc02c88e91", "x": [0, 1], "y": [0, 1]}, {"colorscale": [[0.0, "rgb(165,0,38)"], [0.1111111111111111, "rgb(215,48,39)"], [0.2222222222222222, "rgb(244,109,67)"], [0.3333333333333333, "rgb(253,174,97)"], [0.4444444444444444, "rgb(254,224,144)"], [0.5555555555555556, "rgb(224,243,248)"], [0.6666666666666666, "rgb(171,217,233)"], [0.7777777777777778, "rgb(116,173,209)"], [0.8888888888888888, "rgb(69,117,180)"], [1.0, "rgb(49,54,149)"]], "name": "matrix", "showscale": false, "type": "heatmap", "uid": "ac763c4f-1791-4171-ba33-ec0167c7f680", "x": ["Not churn", "Churn"], "xaxis": "x2", "y": ["Not churn", "Churn"], "yaxis": "y2", "z": [[1110, 158], [244, 246]]}],
                        {"autosize": false, "height": 600, "margin": {"b": 200}, "paper_bgcolor": "rgb(243,243,243)", "plot_bgcolor": "rgb(243,243,243)", "showlegend": false, "title": {"text": "Caracteristicas del modelo"}, "width": 900, "xaxis": {"domain": [0, 0.6], "gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio falso positivo"}}, "xaxis2": {"domain": [0.7, 1], "gridcolor": "rgb(255, 255, 255)", "tickangle": 90}, "yaxis": {"gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio verdadero positivo"}, "zerolinewidth": 1}, "yaxis2": {"anchor": "x2", "gridcolor": "rgb(255, 255, 255)"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('d05b476a-64dd-46fc-a32a-fa5ad6103cc9');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:219: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    
    /anaconda3/lib/python3.7/site-packages/sklearn/preprocessing/label.py:252: DataConversionWarning:
    
    A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
    


    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.9,
           max_delta_step=0, max_depth=7, min_child_weight=1, missing=None,
           n_estimators=100, n_jobs=1, nthread=None,
           objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
           subsample=1, verbosity=1)
    Accuracy   Score :  0.7741751990898749
    Area bajo la curva :  0.6925674370694651 
    



<div>
        
        
            <div id="c8d4b53d-2a75-49b8-9d25-b5d5bcedacd9" class="plotly-graph-div" style="height:600px; width:900px;"></div>
            <script type="text/javascript">
                require(["plotly"], function(Plotly) {
                    window.PLOTLYENV=window.PLOTLYENV || {};
                    window.PLOTLYENV.BASE_URL='https://plot.ly';
                    
                if (document.getElementById("c8d4b53d-2a75-49b8-9d25-b5d5bcedacd9")) {
                    Plotly.newPlot(
                        'c8d4b53d-2a75-49b8-9d25-b5d5bcedacd9',
                        [{"line": {"color": "rgb(22, 96, 167)", "width": 2}, "name": "ROC : 0.6925674370694651", "type": "scatter", "uid": "1d7c7b1a-592e-4dc6-b9d1-15b0e25aedd3", "x": [0.0, 0.0, 0.0, 0.0007886435331230284, 0.0007886435331230284, 0.0007886435331230284, 0.0015772870662460567, 0.0015772870662460567, 0.002365930599369085, 0.002365930599369085, 0.0031545741324921135, 0.0031545741324921135, 0.003943217665615142, 0.003943217665615142, 0.00473186119873817, 0.00473186119873817, 0.005520504731861199, 0.005520504731861199, 0.006309148264984227, 0.006309148264984227, 0.007097791798107256, 0.007097791798107256, 0.00946372239747634, 0.00946372239747634, 0.01025236593059937, 0.01025236593059937, 0.011041009463722398, 0.011041009463722398, 0.011829652996845425, 0.011829652996845425, 0.012618296529968454, 0.012618296529968454, 0.013406940063091483, 0.013406940063091483, 0.014195583596214511, 0.014195583596214511, 0.01498422712933754, 0.01498422712933754, 0.015772870662460567, 0.015772870662460567, 0.01892744479495268, 0.01892744479495268, 0.021293375394321766, 0.021293375394321766, 0.022082018927444796, 0.022082018927444796, 0.022870662460567823, 0.022870662460567823, 0.02365930599369085, 0.02365930599369085, 0.026025236593059938, 0.026025236593059938, 0.026813880126182965, 0.026813880126182965, 0.027602523659305992, 0.027602523659305992, 0.02996845425867508, 0.02996845425867508, 0.031545741324921134, 0.031545741324921134, 0.032334384858044164, 0.032334384858044164, 0.033123028391167195, 0.033123028391167195, 0.03470031545741325, 0.03470031545741325, 0.03548895899053628, 0.03548895899053628, 0.03706624605678233, 0.03706624605678233, 0.03785488958990536, 0.03785488958990536, 0.038643533123028394, 0.03943217665615142, 0.03943217665615142, 0.04100946372239748, 0.04100946372239748, 0.0417981072555205, 0.0417981072555205, 0.04337539432176656, 0.04337539432176656, 0.04416403785488959, 0.04416403785488959, 0.044952681388012616, 0.044952681388012616, 0.04574132492113565, 0.04574132492113565, 0.04652996845425868, 0.04652996845425868, 0.04810725552050473, 0.04810725552050473, 0.04889589905362776, 0.04889589905362776, 0.04968454258675079, 0.04968454258675079, 0.050473186119873815, 0.050473186119873815, 0.051261829652996846, 0.051261829652996846, 0.05362776025236593, 0.05362776025236593, 0.05441640378548896, 0.05441640378548896, 0.056782334384858045, 0.056782334384858045, 0.0583596214511041, 0.0583596214511041, 0.06072555205047318, 0.06072555205047318, 0.062302839116719244, 0.062302839116719244, 0.0638801261829653, 0.0638801261829653, 0.06466876971608833, 0.06466876971608833, 0.06624605678233439, 0.06624605678233439, 0.06703470031545741, 0.06703470031545741, 0.06782334384858044, 0.06782334384858044, 0.06861198738170347, 0.06861198738170347, 0.0694006309148265, 0.0694006309148265, 0.07018927444794952, 0.07018927444794952, 0.07255520504731862, 0.07255520504731862, 0.07334384858044164, 0.07334384858044164, 0.07413249211356467, 0.07413249211356467, 0.0749211356466877, 0.0749211356466877, 0.07570977917981073, 0.07570977917981073, 0.07649842271293375, 0.07649842271293375, 0.07807570977917981, 0.07807570977917981, 0.07965299684542587, 0.07965299684542587, 0.08201892744479496, 0.08201892744479496, 0.08280757097791798, 0.08280757097791798, 0.08517350157728706, 0.08517350157728706, 0.0859621451104101, 0.0859621451104101, 0.08675078864353312, 0.08675078864353312, 0.08753943217665615, 0.08753943217665615, 0.08911671924290221, 0.08911671924290221, 0.09069400630914827, 0.09069400630914827, 0.0914826498422713, 0.0914826498422713, 0.09227129337539432, 0.09227129337539432, 0.09305993690851735, 0.09305993690851735, 0.0946372239747634, 0.0946372239747634, 0.09621451104100946, 0.09621451104100946, 0.09700315457413249, 0.09700315457413249, 0.09858044164037855, 0.09858044164037855, 0.09936908517350158, 0.09936908517350158, 0.10094637223974763, 0.10094637223974763, 0.10173501577287067, 0.10173501577287067, 0.10331230283911672, 0.10331230283911672, 0.10410094637223975, 0.10410094637223975, 0.10883280757097792, 0.10883280757097792, 0.11277602523659307, 0.11277602523659307, 0.11356466876971609, 0.11356466876971609, 0.11435331230283911, 0.11435331230283911, 0.11514195583596215, 0.11514195583596215, 0.1167192429022082, 0.1167192429022082, 0.11750788643533124, 0.11750788643533124, 0.11829652996845426, 0.11829652996845426, 0.11987381703470032, 0.11987381703470032, 0.12066246056782334, 0.12066246056782334, 0.12145110410094637, 0.12145110410094637, 0.1222397476340694, 0.1222397476340694, 0.12302839116719243, 0.12302839116719243, 0.12381703470031545, 0.12381703470031545, 0.12460567823343849, 0.12460567823343849, 0.12618296529968454, 0.12618296529968454, 0.1277602523659306, 0.1277602523659306, 0.1309148264984227, 0.13249211356466878, 0.13249211356466878, 0.1332807570977918, 0.1332807570977918, 0.13406940063091483, 0.13406940063091483, 0.13643533123028392, 0.13643533123028392, 0.13801261829652997, 0.13801261829652997, 0.1411671924290221, 0.1411671924290221, 0.14195583596214512, 0.14195583596214512, 0.14353312302839116, 0.14353312302839116, 0.14511041009463724, 0.14511041009463724, 0.14668769716088328, 0.14668769716088328, 0.15063091482649843, 0.15063091482649843, 0.15141955835962145, 0.15141955835962145, 0.15378548895899052, 0.15378548895899052, 0.15694006309148265, 0.15694006309148265, 0.15772870662460567, 0.15772870662460567, 0.15930599369085174, 0.15930599369085174, 0.1608832807570978, 0.1608832807570978, 0.16167192429022081, 0.16167192429022081, 0.16246056782334384, 0.16246056782334384, 0.1640378548895899, 0.1640378548895899, 0.16561514195583596, 0.16561514195583596, 0.16640378548895898, 0.16640378548895898, 0.167192429022082, 0.167192429022082, 0.16876971608832808, 0.16876971608832808, 0.17034700315457413, 0.17034700315457413, 0.1719242902208202, 0.1719242902208202, 0.1750788643533123, 0.1750788643533123, 0.17586750788643532, 0.17586750788643532, 0.17665615141955837, 0.17665615141955837, 0.17823343848580442, 0.17823343848580442, 0.17981072555205047, 0.17981072555205047, 0.1805993690851735, 0.1805993690851735, 0.18217665615141956, 0.18217665615141956, 0.1829652996845426, 0.1829652996845426, 0.1861198738170347, 0.1861198738170347, 0.18769716088328076, 0.18848580441640378, 0.19085173501577288, 0.19085173501577288, 0.19321766561514195, 0.19321766561514195, 0.19479495268138802, 0.19479495268138802, 0.19558359621451105, 0.19558359621451105, 0.19794952681388012, 0.19794952681388012, 0.1995268138801262, 0.1995268138801262, 0.20031545741324921, 0.20031545741324921, 0.20110410094637224, 0.20110410094637224, 0.20347003154574134, 0.20347003154574134, 0.20425867507886436, 0.2058359621451104, 0.2058359621451104, 0.20662460567823343, 0.20662460567823343, 0.20741324921135645, 0.20741324921135645, 0.2082018927444795, 0.2082018927444795, 0.20899053627760253, 0.20899053627760253, 0.21056782334384858, 0.21056782334384858, 0.2113564668769716, 0.2113564668769716, 0.21214511041009465, 0.21214511041009465, 0.2137223974763407, 0.2137223974763407, 0.21687697160883282, 0.21687697160883282, 0.2192429022082019, 0.2192429022082019, 0.2200315457413249, 0.2200315457413249, 0.221608832807571, 0.221608832807571, 0.22476340694006308, 0.22476340694006308, 0.23817034700315456, 0.23817034700315456, 0.2421135646687697, 0.2421135646687697, 0.2444794952681388, 0.2444794952681388, 0.2476340694006309, 0.2476340694006309, 0.2547318611987382, 0.2547318611987382, 0.25709779179810727, 0.25709779179810727, 0.2578864353312303, 0.2578864353312303, 0.2586750788643533, 0.2586750788643533, 0.2618296529968454, 0.2618296529968454, 0.26498422712933756, 0.26498422712933756, 0.27129337539432175, 0.27129337539432175, 0.27287066246056785, 0.27287066246056785, 0.27996845425867506, 0.27996845425867506, 0.2823343848580442, 0.2823343848580442, 0.2870662460567823, 0.2870662460567823, 0.2910094637223975, 0.2910094637223975, 0.29337539432176657, 0.29337539432176657, 0.30047318611987384, 0.30047318611987384, 0.305993690851735, 0.305993690851735, 0.306782334384858, 0.306782334384858, 0.30757097791798105, 0.30757097791798105, 0.31230283911671924, 0.31230283911671924, 0.3146687697160883, 0.3146687697160883, 0.3186119873817035, 0.3186119873817035, 0.32097791798107256, 0.32097791798107256, 0.3272870662460568, 0.3272870662460568, 0.3359621451104101, 0.3359621451104101, 0.33990536277602523, 0.33990536277602523, 0.3422712933753943, 0.3422712933753943, 0.34463722397476343, 0.34463722397476343, 0.34542586750788645, 0.34542586750788645, 0.34936908517350157, 0.34936908517350157, 0.35173501577287064, 0.35173501577287064, 0.35331230283911674, 0.35331230283911674, 0.35410094637223977, 0.35410094637223977, 0.3548895899053628, 0.3548895899053628, 0.3588328075709779, 0.3588328075709779, 0.36041009463722395, 0.36041009463722395, 0.361198738170347, 0.361198738170347, 0.3627760252365931, 0.3627760252365931, 0.3635646687697161, 0.3635646687697161, 0.36514195583596215, 0.36514195583596215, 0.3675078864353312, 0.3675078864353312, 0.36829652996845424, 0.36829652996845424, 0.3698738170347003, 0.3698738170347003, 0.3746056782334385, 0.3746056782334385, 0.3777602523659306, 0.3777602523659306, 0.3801261829652997, 0.3801261829652997, 0.3872239747634069, 0.3872239747634069, 0.3911671924290221, 0.3911671924290221, 0.39668769716088326, 0.39668769716088326, 0.3998422712933754, 0.3998422712933754, 0.40141955835962145, 0.40141955835962145, 0.4037854889589905, 0.4037854889589905, 0.4108832807570978, 0.4108832807570978, 0.41798107255520506, 0.41798107255520506, 0.4313880126182965, 0.4313880126182965, 0.4384858044164038, 0.4384858044164038, 0.44242902208201895, 0.44242902208201895, 0.444006309148265, 0.444006309148265, 0.44873817034700314, 0.44873817034700314, 0.44952681388012616, 0.44952681388012616, 0.4503154574132492, 0.4503154574132492, 0.45425867507886436, 0.45425867507886436, 0.45662460567823343, 0.45662460567823343, 0.4597791798107255, 0.4597791798107255, 0.4692429022082019, 0.4692429022082019, 0.48264984227129337, 0.48264984227129337, 0.48501577287066244, 0.48501577287066244, 0.5078864353312302, 0.5078864353312302, 0.5094637223974764, 0.5094637223974764, 0.5126182965299685, 0.5126182965299685, 0.5141955835962145, 0.5141955835962145, 0.5244479495268138, 0.5244479495268138, 0.5291798107255521, 0.5291798107255521, 0.5378548895899053, 0.5378548895899053, 0.5402208201892744, 0.5402208201892744, 0.5473186119873817, 0.5473186119873817, 0.5512618296529969, 0.5512618296529969, 0.5536277602523659, 0.5536277602523659, 0.5591482649842271, 0.5591482649842271, 0.5623028391167192, 0.5623028391167192, 0.5662460567823344, 0.5662460567823344, 0.5709779179810726, 0.5709779179810726, 0.5788643533123028, 0.5788643533123028, 0.5796529968454258, 0.5796529968454258, 0.582018927444795, 0.582018927444795, 0.5851735015772871, 0.5851735015772871, 0.5875394321766562, 0.5875394321766562, 0.5899053627760252, 0.5899053627760252, 0.6190851735015773, 0.6190851735015773, 0.6246056782334385, 0.6246056782334385, 0.6285488958990536, 0.6285488958990536, 0.6411671924290221, 0.6411671924290221, 0.6569400630914827, 0.6569400630914827, 0.6600946372239748, 0.6600946372239748, 0.6656151419558359, 0.6656151419558359, 0.667192429022082, 0.668769716088328, 0.6837539432176656, 0.6837539432176656, 0.6876971608832808, 0.6876971608832808, 0.6884858044164038, 0.6884858044164038, 0.6908517350157729, 0.6908517350157729, 0.6987381703470031, 0.6987381703470031, 0.7105678233438486, 0.7121451104100947, 0.7192429022082019, 0.7192429022082019, 0.7247634069400631, 0.7247634069400631, 0.7271293375394322, 0.7271293375394322, 0.7310725552050473, 0.7310725552050473, 0.7358044164037855, 0.7358044164037855, 0.7626182965299685, 0.7626182965299685, 0.7673501577287066, 0.7673501577287066, 0.7847003154574133, 0.7847003154574133, 0.805205047318612, 0.805205047318612, 0.8107255520504731, 0.8123028391167192, 0.8343848580441641, 0.8359621451104101, 0.8414826498422713, 0.8414826498422713, 0.9140378548895899, 0.9140378548895899, 0.9321766561514195, 0.9321766561514195, 1.0], "y": [0.0, 0.0020408163265306124, 0.02857142857142857, 0.02857142857142857, 0.0326530612244898, 0.04693877551020408, 0.04693877551020408, 0.04897959183673469, 0.04897959183673469, 0.05510204081632653, 0.05510204081632653, 0.06938775510204082, 0.06938775510204082, 0.07142857142857142, 0.07142857142857142, 0.07551020408163266, 0.07551020408163266, 0.07959183673469387, 0.07959183673469387, 0.0836734693877551, 0.0836734693877551, 0.08979591836734693, 0.08979591836734693, 0.09183673469387756, 0.09183673469387756, 0.09591836734693877, 0.09591836734693877, 0.10204081632653061, 0.10204081632653061, 0.10612244897959183, 0.10612244897959183, 0.11020408163265306, 0.11020408163265306, 0.11224489795918367, 0.11224489795918367, 0.11836734693877551, 0.11836734693877551, 0.12653061224489795, 0.12653061224489795, 0.14489795918367346, 0.14489795918367346, 0.1469387755102041, 0.1469387755102041, 0.1489795918367347, 0.1489795918367347, 0.1510204081632653, 0.1510204081632653, 0.15918367346938775, 0.15918367346938775, 0.16122448979591836, 0.16122448979591836, 0.1673469387755102, 0.1673469387755102, 0.16938775510204082, 0.16938775510204082, 0.17142857142857143, 0.17142857142857143, 0.17346938775510204, 0.17346938775510204, 0.17755102040816326, 0.17755102040816326, 0.18571428571428572, 0.18571428571428572, 0.18775510204081633, 0.18775510204081633, 0.18979591836734694, 0.18979591836734694, 0.19183673469387755, 0.19183673469387755, 0.19387755102040816, 0.19387755102040816, 0.19591836734693877, 0.19795918367346937, 0.19795918367346937, 0.2, 0.2, 0.20816326530612245, 0.20816326530612245, 0.2163265306122449, 0.2163265306122449, 0.22653061224489796, 0.22653061224489796, 0.22857142857142856, 0.22857142857142856, 0.23061224489795917, 0.23061224489795917, 0.23877551020408164, 0.23877551020408164, 0.24081632653061225, 0.24081632653061225, 0.24489795918367346, 0.24489795918367346, 0.2530612244897959, 0.2530612244897959, 0.25510204081632654, 0.25510204081632654, 0.2612244897959184, 0.2612244897959184, 0.2653061224489796, 0.2653061224489796, 0.2673469387755102, 0.2673469387755102, 0.2693877551020408, 0.2693877551020408, 0.2714285714285714, 0.2714285714285714, 0.27346938775510204, 0.27346938775510204, 0.27755102040816326, 0.27755102040816326, 0.2816326530612245, 0.2816326530612245, 0.2836734693877551, 0.2836734693877551, 0.29591836734693877, 0.29591836734693877, 0.2979591836734694, 0.2979591836734694, 0.3081632653061224, 0.3081632653061224, 0.31020408163265306, 0.31020408163265306, 0.3224489795918367, 0.3224489795918367, 0.336734693877551, 0.336734693877551, 0.3448979591836735, 0.3448979591836735, 0.3489795918367347, 0.3489795918367347, 0.3510204081632653, 0.3510204081632653, 0.35306122448979593, 0.35306122448979593, 0.35714285714285715, 0.35714285714285715, 0.35918367346938773, 0.35918367346938773, 0.36122448979591837, 0.36122448979591837, 0.3653061224489796, 0.3653061224489796, 0.37142857142857144, 0.37142857142857144, 0.37551020408163266, 0.37551020408163266, 0.3816326530612245, 0.3816326530612245, 0.3836734693877551, 0.3836734693877551, 0.38571428571428573, 0.38571428571428573, 0.3877551020408163, 0.3877551020408163, 0.38979591836734695, 0.38979591836734695, 0.39387755102040817, 0.39387755102040817, 0.39591836734693875, 0.39591836734693875, 0.4, 0.4, 0.4020408163265306, 0.4020408163265306, 0.4122448979591837, 0.4122448979591837, 0.4163265306122449, 0.4163265306122449, 0.42244897959183675, 0.42244897959183675, 0.42448979591836733, 0.42448979591836733, 0.4306122448979592, 0.4306122448979592, 0.4326530612244898, 0.4326530612244898, 0.43673469387755104, 0.43673469387755104, 0.44081632653061226, 0.44081632653061226, 0.44693877551020406, 0.44693877551020406, 0.45102040816326533, 0.45102040816326533, 0.4530612244897959, 0.4530612244897959, 0.45510204081632655, 0.45510204081632655, 0.46938775510204084, 0.46938775510204084, 0.4714285714285714, 0.4714285714285714, 0.47959183673469385, 0.47959183673469385, 0.4816326530612245, 0.4816326530612245, 0.48775510204081635, 0.48775510204081635, 0.4897959183673469, 0.4897959183673469, 0.49183673469387756, 0.49183673469387756, 0.5020408163265306, 0.5020408163265306, 0.5061224489795918, 0.5061224489795918, 0.5081632653061224, 0.5081632653061224, 0.5163265306122449, 0.5163265306122449, 0.5183673469387755, 0.5183673469387755, 0.5265306122448979, 0.5265306122448979, 0.5326530612244897, 0.5326530612244897, 0.5387755102040817, 0.5387755102040817, 0.5387755102040817, 0.5448979591836735, 0.5448979591836735, 0.5510204081632653, 0.5510204081632653, 0.5530612244897959, 0.5530612244897959, 0.5551020408163265, 0.5551020408163265, 0.5571428571428572, 0.5571428571428572, 0.5612244897959183, 0.5612244897959183, 0.5673469387755102, 0.5673469387755102, 0.5693877551020409, 0.5693877551020409, 0.5714285714285714, 0.5714285714285714, 0.573469387755102, 0.573469387755102, 0.5755102040816327, 0.5755102040816327, 0.5775510204081633, 0.5775510204081633, 0.5816326530612245, 0.5816326530612245, 0.5836734693877551, 0.5836734693877551, 0.5857142857142857, 0.5857142857142857, 0.5877551020408164, 0.5877551020408164, 0.5897959183673469, 0.5897959183673469, 0.5959183673469388, 0.5959183673469388, 0.5979591836734693, 0.5979591836734693, 0.6, 0.6, 0.6020408163265306, 0.6020408163265306, 0.6040816326530613, 0.6040816326530613, 0.6081632653061224, 0.6081632653061224, 0.610204081632653, 0.610204081632653, 0.6163265306122448, 0.6163265306122448, 0.6183673469387755, 0.6183673469387755, 0.6306122448979592, 0.6306122448979592, 0.6346938775510204, 0.6346938775510204, 0.6387755102040816, 0.6387755102040816, 0.6428571428571429, 0.6428571428571429, 0.6469387755102041, 0.6469387755102041, 0.6510204081632653, 0.6510204081632653, 0.6530612244897959, 0.6530612244897959, 0.6551020408163265, 0.6551020408163265, 0.6571428571428571, 0.6571428571428571, 0.6591836734693878, 0.6591836734693878, 0.6653061224489796, 0.6653061224489796, 0.6673469387755102, 0.6673469387755102, 0.6714285714285714, 0.6714285714285714, 0.6755102040816326, 0.6755102040816326, 0.6775510204081633, 0.6775510204081633, 0.6795918367346939, 0.6795918367346939, 0.6836734693877551, 0.6836734693877551, 0.6857142857142857, 0.6857142857142857, 0.6877551020408164, 0.6877551020408164, 0.6877551020408164, 0.689795918367347, 0.689795918367347, 0.6938775510204082, 0.6938775510204082, 0.6979591836734694, 0.6979591836734694, 0.7, 0.7, 0.7040816326530612, 0.7040816326530612, 0.7061224489795919, 0.7061224489795919, 0.7081632653061225, 0.7081632653061225, 0.7122448979591837, 0.7122448979591837, 0.7142857142857143, 0.7142857142857143, 0.7163265306122449, 0.7163265306122449, 0.7183673469387755, 0.7183673469387755, 0.7204081632653061, 0.7204081632653061, 0.7224489795918367, 0.7224489795918367, 0.726530612244898, 0.726530612244898, 0.7285714285714285, 0.7285714285714285, 0.7326530612244898, 0.7326530612244898, 0.7346938775510204, 0.7346938775510204, 0.736734693877551, 0.736734693877551, 0.7387755102040816, 0.7387755102040816, 0.7448979591836735, 0.7448979591836735, 0.746938775510204, 0.746938775510204, 0.7489795918367347, 0.7489795918367347, 0.7510204081632653, 0.7510204081632653, 0.753061224489796, 0.753061224489796, 0.7551020408163265, 0.7551020408163265, 0.7571428571428571, 0.7571428571428571, 0.7591836734693878, 0.7591836734693878, 0.7612244897959184, 0.7612244897959184, 0.7653061224489796, 0.7653061224489796, 0.7693877551020408, 0.7693877551020408, 0.773469387755102, 0.773469387755102, 0.7755102040816326, 0.7755102040816326, 0.7795918367346939, 0.7795918367346939, 0.7816326530612245, 0.7816326530612245, 0.7836734693877551, 0.7836734693877551, 0.7857142857142857, 0.7857142857142857, 0.789795918367347, 0.789795918367347, 0.7918367346938775, 0.7918367346938775, 0.7938775510204081, 0.7938775510204081, 0.7959183673469388, 0.7959183673469388, 0.7979591836734694, 0.7979591836734694, 0.8, 0.8, 0.8020408163265306, 0.8020408163265306, 0.8040816326530612, 0.8040816326530612, 0.810204081632653, 0.810204081632653, 0.8122448979591836, 0.8122448979591836, 0.8142857142857143, 0.8142857142857143, 0.8204081632653061, 0.8204081632653061, 0.8224489795918367, 0.8224489795918367, 0.8244897959183674, 0.8244897959183674, 0.826530612244898, 0.826530612244898, 0.8285714285714286, 0.8285714285714286, 0.8306122448979592, 0.8306122448979592, 0.8326530612244898, 0.8326530612244898, 0.8367346938775511, 0.8367346938775511, 0.8387755102040816, 0.8387755102040816, 0.8408163265306122, 0.8408163265306122, 0.8428571428571429, 0.8428571428571429, 0.8489795918367347, 0.8489795918367347, 0.8510204081632653, 0.8510204081632653, 0.8530612244897959, 0.8530612244897959, 0.8551020408163266, 0.8551020408163266, 0.8571428571428571, 0.8571428571428571, 0.8612244897959184, 0.8612244897959184, 0.8653061224489796, 0.8653061224489796, 0.8673469387755102, 0.8673469387755102, 0.8693877551020408, 0.8693877551020408, 0.8714285714285714, 0.8714285714285714, 0.8734693877551021, 0.8734693877551021, 0.8755102040816326, 0.8755102040816326, 0.8775510204081632, 0.8775510204081632, 0.8816326530612245, 0.8816326530612245, 0.8836734693877552, 0.8836734693877552, 0.8857142857142857, 0.8857142857142857, 0.8877551020408163, 0.8877551020408163, 0.889795918367347, 0.889795918367347, 0.8938775510204081, 0.8938775510204081, 0.8959183673469387, 0.8959183673469387, 0.8979591836734694, 0.8979591836734694, 0.9, 0.9, 0.9020408163265307, 0.9020408163265307, 0.9040816326530612, 0.9040816326530612, 0.9061224489795918, 0.9061224489795918, 0.9081632653061225, 0.9081632653061225, 0.9102040816326531, 0.9102040816326531, 0.9122448979591836, 0.9122448979591836, 0.9142857142857143, 0.9142857142857143, 0.9163265306122449, 0.9163265306122449, 0.9183673469387755, 0.9183673469387755, 0.9204081632653062, 0.9204081632653062, 0.9224489795918367, 0.9224489795918367, 0.9244897959183673, 0.9244897959183673, 0.926530612244898, 0.926530612244898, 0.9285714285714286, 0.9285714285714286, 0.9306122448979591, 0.9306122448979591, 0.9326530612244898, 0.9326530612244898, 0.9346938775510204, 0.9346938775510204, 0.936734693877551, 0.936734693877551, 0.9387755102040817, 0.9387755102040817, 0.9408163265306122, 0.9408163265306122, 0.9428571428571428, 0.9428571428571428, 0.9448979591836735, 0.9448979591836735, 0.9469387755102041, 0.9469387755102041, 0.9510204081632653, 0.9510204081632653, 0.9530612244897959, 0.9530612244897959, 0.9551020408163265, 0.9551020408163265, 0.9571428571428572, 0.9571428571428572, 0.9591836734693877, 0.9591836734693877, 0.9612244897959183, 0.9612244897959183, 0.963265306122449, 0.963265306122449, 0.9653061224489796, 0.9653061224489796, 0.9653061224489796, 0.9653061224489796, 0.9673469387755103, 0.9673469387755103, 0.9693877551020408, 0.9693877551020408, 0.9714285714285714, 0.9714285714285714, 0.9734693877551021, 0.9734693877551021, 0.9755102040816327, 0.9755102040816327, 0.9755102040816327, 0.9755102040816327, 0.9775510204081632, 0.9775510204081632, 0.9795918367346939, 0.9795918367346939, 0.9816326530612245, 0.9816326530612245, 0.9836734693877551, 0.9836734693877551, 0.9857142857142858, 0.9857142857142858, 0.9877551020408163, 0.9877551020408163, 0.9897959183673469, 0.9897959183673469, 0.9918367346938776, 0.9918367346938776, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9938775510204082, 0.9959183673469387, 0.9959183673469387, 0.9979591836734694, 0.9979591836734694, 1.0, 1.0]}, {"line": {"color": "rgb(205, 12, 24)", "dash": "dot", "width": 2}, "type": "scatter", "uid": "c5d66ed3-c0ce-4de4-8e2a-c2ce7b347105", "x": [0, 1], "y": [0, 1]}, {"colorscale": [[0.0, "rgb(165,0,38)"], [0.1111111111111111, "rgb(215,48,39)"], [0.2222222222222222, "rgb(244,109,67)"], [0.3333333333333333, "rgb(253,174,97)"], [0.4444444444444444, "rgb(254,224,144)"], [0.5555555555555556, "rgb(224,243,248)"], [0.6666666666666666, "rgb(171,217,233)"], [0.7777777777777778, "rgb(116,173,209)"], [0.8888888888888888, "rgb(69,117,180)"], [1.0, "rgb(49,54,149)"]], "name": "matrix", "showscale": false, "type": "heatmap", "uid": "b17603d0-d685-439c-bed9-f9d77a2eb691", "x": ["Not churn", "Churn"], "xaxis": "x2", "y": ["Not churn", "Churn"], "yaxis": "y2", "z": [[1112, 156], [241, 249]]}],
                        {"autosize": false, "height": 600, "margin": {"b": 200}, "paper_bgcolor": "rgb(243,243,243)", "plot_bgcolor": "rgb(243,243,243)", "showlegend": false, "title": {"text": "Caracteristicas del modelo"}, "width": 900, "xaxis": {"domain": [0, 0.6], "gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio falso positivo"}}, "xaxis2": {"domain": [0.7, 1], "gridcolor": "rgb(255, 255, 255)", "tickangle": 90}, "yaxis": {"gridcolor": "rgb(255, 255, 255)", "gridwidth": 2, "ticklen": 5, "title": {"text": "Ratio verdadero positivo"}, "zerolinewidth": 1}, "yaxis2": {"anchor": "x2", "gridcolor": "rgb(255, 255, 255)"}},
                        {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly", "responsive": true}
                    ).then(function(){
                            
var gd = document.getElementById('c8d4b53d-2a75-49b8-9d25-b5d5bcedacd9');
var x = new MutationObserver(function (mutations, observer) {{
        var display = window.getComputedStyle(gd).display;
        if (!display || display === 'none') {{
            console.log([gd, 'removed!']);
            Plotly.purge(gd);
            observer.disconnect();
        }}
}});

// Listen for the removal of the full notebook cells
var notebookContainer = gd.closest('#notebook-container');
if (notebookContainer) {{
    x.observe(notebookContainer, {childList: true});
}}

// Listen for the clearing of the current output cell
var outputEl = gd.closest('.output');
if (outputEl) {{
    x.observe(outputEl, {childList: true});
}}

                        })
                };
                });
            </script>
        </div>


## Conclusion

![png](linea.png)

Tras visualizar los graficos de los modelos, llegamos a la conclusion que el mejor algoritmo para este caso es la **LogisticRegression**, para la matriz B, dandonos los siguientes resultados:

Accuracy   Score :  0.8100113765642776
Area bajo la curva :  0.7311836090903239

Cabe descatar que el feature engeniering ha mejorado los resultados respecto a la matriz A, muy poco, pero la mejora ha sido buena. Recalco, que este resultado puede mejorarse si se estudia la posibilidad de mayores combinaciones lineales entre las variables continuas. Además, los algoritmos que hemos probado, han sido con parametros sencillos, podría caber la posibilidad que los algoritmos como el XGBoost o el RamdomForest superen este score jugando con sus hyperparametros.
