import numpy as np
from os.path import join
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
##############################################################################
modeling_dict = np.load(
    # ensure interoperability across different OSs
    join('Data', 'modeling_data.npz')
)
##############################################################################
# set np arrays as pd dataframes
train_df = pd.DataFrame(

    data=np.column_stack((modeling_dict['x_train'], modeling_dict['y_train']))
)

test_df = pd.DataFrame(

    data=np.column_stack((modeling_dict['x_test'], modeling_dict['y_test']))
)

##############################################################################
##############################################################################
# setup local Spark session (adapt to your hardware!)
spark = (SparkSession
         .builder
         .appName("img_classifier")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         # .config("spark.dynamicAllocation.enabled", "true")
         .config("spark.dynamicAllocation.minExecutors", "2")
         .config("spark.dynamicAllocation.maxExecutors", "2")
         # to avoid an error related to lack of memory
         .config("spark.driver.memory", "2g")
         .config("spark.executor.memory", "2g")
         # avoid an error related to using Pandas dataframes with PySpark
         .config("spark.driver.extraJavaOptions",
                 "-Dio.netty.tryReflectionSetAccessible=true")
         .config("spark.executor.extraJavaOptions",
                 "-Dio.netty.tryReflectionSetAccessible=true")
         .getOrCreate()
         )
##############################################################################
##############################################################################
# Create Spark DataFrames from Pandas DataFrames
train_df = spark.createDataFrame(train_df)
test_df = spark.createDataFrame(test_df)
##############################################################################
# rename the target variable
train_df = train_df.withColumnRenamed(train_df.columns[-1], 'label')
test_df = test_df.withColumnRenamed(test_df.columns[-1], 'label')
##############################################################################
# form a single vector out of all features
assembler = VectorAssembler(
    inputCols=train_df.columns[:-1],
    outputCol="features")

train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)
##############################################################################
# keeping only the needed columns
train_df = train_df.select('features', 'label')
test_df = test_df.select('features', 'label')
##############################################################################
# number of input features
num_inputs = modeling_dict['x_train'].shape[1]
# number of output categories
num_outputs = np.unique(modeling_dict['y_train']).shape[0]

mlp = MultilayerPerceptronClassifier(
    featuresCol='features', labelCol='label',
    # batch size
    blockSize=64,
    # 3 layers defined each with its count of neurons
    # => 1 input layer, 1 hidden layer, 1 output layer
    layers=[num_inputs, 100, num_outputs],
    # ensuring reproducible results
    seed=123)

model = mlp.fit(train_df)
##############################################################################
result = model.transform(test_df)
# obtain the predictions and select the true values (test set)
prediction_and_labels = result.select("prediction", "label")

evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
# compute the accuracy on the test set
evaluator.evaluate(prediction_and_labels)
##############################################################################
# save the trained model (and allowing overwrites)
mlp.write().overwrite().save(
    # path where to save the trained model
    join('Saved_model', 'mlp')
)
##############################################################################
spark.stop()
