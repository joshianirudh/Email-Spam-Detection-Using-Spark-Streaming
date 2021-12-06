#! /usr/bin/python3

import numpy as np
import json
from joblib import dump, load
import os.path
from pyspark.sql.session import SparkSession
from pyspark import SparkContext
import pyspark.sql.types as tp
from pyspark.streaming import StreamingContext
import pyspark
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer,IDF
from pyspark.ml.classification import *
from nltk.corpus import stopwords
from sklearn.metrics import *
from sklearn.model_selection import train_test_split

nb = NaiveBayes(smoothing=1.0,modelType="multinomial")
gbt = GBTClassifier(maxIter=5,seed=42)
dt = DecisionTreeClassifier(maxDepth=5)
rf = RandomForestClassifier(numTrees=10,maxDepth=5,seed=42)
logr = LogisticRegression(regParam=0.1,family='multinomial')
svc = LinearSVC(maxIter=5,regParam=0.1)
mlp = MultilayerPerceptronClassifier(maxIter=100,layers=[2,4,2],blockSize=1,seed=42)

classifiers = {
    'nb' : nb,
    'gbt': gbt,
    'dt' : dt,
    'rf' : rf,
    'logr': logr,
    'svc' : svc,
    'mlp' : mlp
}

def train_classifiers(clf,train_df,test_df):
    clf.fit(train_df)
    y_pred = clf.predict(test_df)
    accuracy = accuracy_score(y_pred,test_df['Spam/Ham'])
    precision = precision_score(y_pred,test_df['Spam/Ham'])
    recall = recall_score(y_pred,test_df['Spam/Ham'])
    return accuracy,precision,recall



def process_data_frame(data):
    document = DocumentAssembler().setInputCol('text').setOutputCol('data')
    tokenizer = Tokenizer().setInputCols(['data']).setOutputCol('token')
    normalizer = Normalizer().setInputCols(['token']).setOutputCol('normalized').setLowercase(True)
    lemmatizer = LemmatizerModel.pretrained().setInputCols(['normalized']).setOutputCol('lemmatized')
    stop_words = stopwords.words('english')
    stopwords_cleaner = StopWordsCleaner().setInputCols(['lemmatized']).setOutputCol('stopped').setStopWords(stop_words)
    finisher = Finisher().setInputCols('stopped').setOutputCols('cleaned')
    pipeline = Pipeline().setStages([
                                    document,
                                    tokenizer,
                                    normalizer,
                                    lemmatizer,
                                    stopwords_cleaner,
                                    finisher
                                   ])
    p_data = pipeline.fit(data).transform(data)
    return p_data

my_schema = tp.StructType([
  tp.StructField(name= 'id',          dataType= tp.IntegerType(),  nullable= True),
  tp.StructField(name= 'text',       dataType= tp.StringType(),  nullable= True)
])


def split_batches(df):
    rd=json.loads(df)
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
        data_frame= spark.createDataFrame(data_frame,schema=my_schema)
    p_data = process_data_frame(data_frame)
    tf = CountVectorizer(inputCol='cleaned',outputCol='tf')
    tf_model = tf.fit(p_data).transform(p_data)
    idf = IDF(inputCol='tf',outputCol='idf')
    idf_model = idf.fit(tf).transform(tf)
    splits = idf_model.randomSplit([0.6,0.4])
    train_df = splits[0]
    test_df = splits[1] 
    for name,clf in classifiers.items():
        acc,precision,recall = train_classifiers(name,train_df,test_df)
        print(name,":")
        print("Accuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("\n\n")
    

    
sc = SparkContext.getOrCreate()
sc.setLogLevel("OFF")
stream = StreamingContext(sc,1)
spark = SparkSession(sc)
record = stream.socketTextStream("localhost",6100)
record.foreachRDD(lambda rdd:split_batches(rdd))
stream.start()
stream.awaitTermination()
stream.stop(stopSparkContext=True)