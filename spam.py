#! /usr/bin/python3

import numpy as np
from joblib import dump, load
import os.path
from pyspark.sql.session import SparkSession
from pyspark import SparkContext
import pyspark.sql.types as tp
from pyspark.streaming import StreamingContext

my_schema = tp.StructType([
  tp.StructField(name= 'id',          dataType= tp.IntegerType(),  nullable= True),
  tp.StructField(name= 'text',       dataType= tp.StringType(),  nullable= True)
])


def split_batches(df):
    rd=spark.read.json(df)
    dataset=rd.head()
    splt1='spam'
    splt2='ham'
    if dataset:
        spam_list=list()
        data_frame=list()
        for i in dataset._fields_:
            message=str(dataset[i]['feature1'])
            subject=str(dataset[i]['feature0'])
            text = str(message + ' ' + subject)
            if (str(dataset[i]['feature2'])==splt1):
                spam_list.append(1)
            else:
                spam_list.append(0)
            ap={'id':i,'text':text}
            data_frame.append(ap)
        data_frame=session.createDataFrame(data_frame,schema=my_schema)
    #proc_data_frame=process_data_frame(data_frame,spam_list) #anirudh
    # classifiers, Jaywanth
    
sc = SparkContext.getOrCreate()
sc.setLogLevel("OFF")
stream = StreamingContext(sc,1)
spark = SparkSession(sc)
record = stream.socketTextStream("localhost",6100)
record.foreachRDD(lambda rdd:split_batches(rdd))
stream.start()
stream.awaitTermination()
stream.stop(stopSparkContext=True)