
# coding: utf-8

# In[3]:


from google.cloud import storage
from io import BytesIO
import pandas as pd
from io import StringIO
from pyspark.sql.types import *
from pyspark.sql import SparkSession
import pyspark.sql.functions
from pyspark.sql.functions import *
import datetime

#Obtenemos el año en el que se realiza el procesamiento
now = datetime.datetime.now()
now_year= now.year

#Función que permitirá obtener los dias del año del data set
def day_set (df,columna_hora):
    meses = df.Mes.unique()
    days=[]
    for i in meses:
        day = 1
        df1 = df[df['Mes']== i]
        for i in range(df1.index[1],df1.index[len(df1)-1]+1):
            if (df1[columna_hora][i-1] == "23" and df1[columna_hora][i] == "0"):
                day = day + 1
            days.append(day)
        days.append(1)

    df['day']=days
    return df

#Funcion de subida de archivos a google cloud storage
def upload_file(df,year,step):
    bucket_name = 'tfm-samur-bucket'
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob('Process_data/step_'+str(step)+'/samur_activaciones_proces_' + str(year) + '.csv')
    
    data = df.toPandas()
    data.to_csv('samur_activaciones_process_' + str(year) + '.csv',encoding ='latin1')

    blob.upload_from_filename('samur_activaciones_process_' + str(year) + '.csv')
    

#Función que procesará los datos en pyspark 
def function_spark(numero_activacion):
    
    # Descargamos los datos del bucket a variables locales
    storage_client = storage.Client()
    bucket= storage_client.get_bucket('tfm-samur-bucket')
    blob = bucket.get_blob('Raw_data/samur_activaciones_'+str(numero_activacion)+'.csv')
    content = blob.download_as_string()

    #Preparamos el dato para poder convertirlo en un RDD de spark
    data = pd.read_csv(BytesIO(content), index_col=0)
    data = data.values.tolist()

    #Generamos el esquema de unestro RDD
    
    schema = StructType([
        StructField("Año", IntegerType(), True),
        StructField("Mes", StringType(), True),
        StructField("Solicitud", StringType(), True),
        StructField("Intervencion", StringType(), True),
        StructField("codigo", StringType(), True),
        StructField("Distrito", StringType(), True),
        StructField("Hospital", StringType(), True)
        ])

    #Comenzamos nuestra aplicación de Spark
    
    spark = SparkSession.builder.appName("samur").config("spark.some.config.option", "some-value").getOrCreate()

    #Generamos nuestro RDD
    df = spark.createDataFrame(data,schema)

    #Extraemos los datos la hora de solicitud no sea Nan
    df= df.filter(df.Solicitud!='NaN')

    #Expliteamos el campos solicitud para obtener la hora
    split_col = pyspark.sql.functions.split(df['Solicitud'], ':')
    df = df.withColumn('Hora', split_col.getItem(0))

    #Lo convertimos en un pdDataFrame para poder hacer una iteración y obtener array de meses
    data = df.toPandas()
    meses = data.Mes.unique()

    #Llamada a la funcion que me permitira setear los dias de cada mes
    data = day_set(data,'Hora')

    #Volvemos a preparar el dataframe para convertirlo en un RDD
    data= data.values.tolist()
    schema = StructType([
        StructField("Año", IntegerType(), True),
        StructField("Mes", StringType(), True),
        StructField("Solicitud", StringType(), True),
        StructField("Intervencion", StringType(), True),
        StructField("Codigo", StringType(), True),
        StructField("Distrito", StringType(), True),
        StructField("Hospital", StringType(), True),
        StructField("Hora_Solicitud", StringType(), True),
        StructField("Dia", StringType(), True)
        ])

    #Generamos el RDD
    df = spark.createDataFrame(data,schema)

    #Covertimos columna de str a int
    df = df.withColumn("Dia", df["Dia"].cast(IntegerType()))

    #Creamos columna 'Month' en formato numerico
    df = df.withColumn("Mes_num",lit(0))
    numero = 1
    for i in meses:
        df = df.withColumn('Mes_num', when(col('Mes') == i,numero).otherwise(col('Mes_num')))
        numero = numero + 1 

    #Creamos la columna 'Finalizado' que servirá para el conteo total de llamdas al dia
    df = df.withColumn("Finalizado",lit(1))
    df = df.withColumn('Finalizado', when(col('Intervencion') != 'NaN',1).otherwise(0))
    
    #Eliminamos las columnas que en este caso no vamos a necesitar para el objetivo a conseguir,
    #No obstante podrían ser de gran utilidad para otro tipo de procesos.
    df = df.drop('Hora_Solicitud','Mes')
    
    #Generamos una nueva columna fecha, que servirá para poder realizar pruebas con algoritmos de SSTT
    df= df.withColumn('Fecha_activacion',concat(col('Año'), lit('-'), col('Mes_num'),lit('-'),col('Dia')))
    
    #Colocamos el dataframe de una forma mas ordenada
    df = df.select(col('Fecha_activacion'),col('Año'),col('Mes_num'),col('Dia'),col('Solicitud'),
                  col('Intervencion'),col('Distrito'),col('Hospital'),col('Finalizado'))
    
    #Subimos el archivo creado a google cloud storage de nuestro primer paso.
    upload_file(df,numero_activacion,1)
    
    #Agrupamos por fecha y distrito para ver el numero total de llamadas por zona cada dia.
    df = df.select(col('Fecha_activacion'),col('Distrito'),col('Finalizado'))
    df = df.groupBy("Fecha_activacion",'Distrito').sum().orderBy("Fecha_activacion")
    
    #Subimos el archivo creado a google cloud storage de nuestro primer paso.
    upload_file(df,numero_activacion,2)
    
for i in range(2017,now_year+1):
    function_spark(i)

