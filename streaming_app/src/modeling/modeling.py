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

validation_df = pd.DataFrame(

    data=np.column_stack(
        (modeling_dict['x_validation'], modeling_dict['y_validation']))
)
##############################################################################
##############################################################################
# setup local Spark session
spark = (SparkSession
         .builder
         .appName("img_classifier")
         .config("spark.sql.execution.arrow.pyspark.enabled", "true")
         .config("spark.dynamicAllocation.enabled", "true")
         .config("spark.dynamicAllocation.minExecutors", "3")
         .config("spark.dynamicAllocation.maxExecutors", "5")
         .getOrCreate()
         )
##############################################################################
##############################################################################
# Create Spark DataFrames from Pandas DataFrames
train_df = spark.createDataFrame(train_df)
test_df = spark.createDataFrame(test_df)
validation_df = spark.createDataFrame(validation_df)
##############################################################################
# rename the target variable
train_df = train_df.withColumnRenamed(train_df.columns[-1], 'label')
test_df = train_df.withColumnRenamed(test_df.columns[-1], 'label')
validation_df = train_df.withColumnRenamed(validation_df.columns[-1], 'label')
##############################################################################
assembler = VectorAssembler(
    inputCols=train_df.columns[:-1],
    outputCol="features")

train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)
validation_df = assembler.transform(validation_df)
##############################################################################
# keeping only the needed columns
train_df = train_df.select('features', 'label')
test_df = test_df.select('features', 'label')
validation_df = validation_df.select('features', 'label')
##############################################################################
num_inputs = modeling_dict['x_train'].shape[1]
num_outputs = modeling_dict['y_train'].shape[0]

mlp = MultilayerPerceptronClassifier(
    featuresCol='features', labelCol='label',
    # batch size
    blockSize=128,
    layers=[num_inputs, 100, num_outputs],
    # ensuring reproducible results
    seed=123)


model = mlp.fit(train_df)

# compute accuracy on the test set
result = model.transform(test_df)
prediction_and_labels = result.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print(
    "Test set accuracy = {}".format(
        evaluator.evaluate(prediction_and_labels)
    )
)

mlp.save(
    # path where to save the trained model
    join('app', 'Saved_model', 'mlp')
)

spark.stop()
