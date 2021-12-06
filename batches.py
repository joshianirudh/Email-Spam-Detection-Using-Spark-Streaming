#! /usr/bin/python3

from pyspark.sql import SparkSession
from pyspark.streaming import StreamingContext
from pyspark import SparkContext
from pyspark.sql.types import *

def split_batches(df):
    dataset=(spark.read.json(df)).head()
    li=dataset._fields_
    splt1='spam'
    splt2='ham'
    spam_list=list()
    data_frame=list()
    for i in li:
        message=str(dataset[i]['feature1'])
        subject=str(dataset[i]['feature0'])
        if (str(dataset([i]['feature2'])==splt1):
            spam_list.append(1)
        else:
            spam_list.append(0)
        ap={'id':i,'subject':subject,'message':message}
        data_frame.append(ap)
    data_frame=spark.createDataFrame(data_frame,schema=mySchema)
    proc_data_frame=process_data_frame(data_frame,spam_list) #anirudh
    # classifiers, Jaywanth


