from pyspark import SparkContext

from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml import Pipeline
from pyspark.ml.feature import HashingTF, Tokenizer
from pyspark.ml.feature import IDF

from pyspark.sql import SQLContext


def vectorize_data(inputStr) :
    attribute_split = inputStr.split(",")
    spam_or_ham = 0.0 if attribute_split[0] == "ham" else 1.0
    return [spam_or_ham, attribute_split[1]]

# Creating a Spark Context
spark_context = SparkContext("local", "Spam Filter")

# Reading data from the CSV file and loading it into a RDD
sms_data = spark_context.textFile("SMSSpamCollection.csv")

# Caching the data for applying actions faster
sms_data.cache()

# Print the data collected
#print sms_data.collect()

# Create a SQL context
sql_context = SQLContext(spark_context)

# Create vectors of the type [spam/ham (binary values), message]
sms_vectorized = sms_data.map(vectorize_data)

# Create the relevant Dataframe
sms_dataframe = sql_context.createDataFrame(sms_vectorized, ["label", "message"])

# Cache the data frame
sms_dataframe.cache()
#print sms_dataframe.select("label", "message").show()

# Break down the Data into two chunks - Training the Classifier to create the model & Testing the Model
(training_data, test_data) = sms_dataframe.randomSplit([0.7, 0.3])
#print training_data.count()
#print test_data.count()
#print test_data.collect()

# Tokenize each message into the words
tokenizer = Tokenizer(inputCol = "message", outputCol = "words")

# Hashing the words (from the output column of tokenizer) based on TF (term-frequency)
hashing_tf = HashingTF(inputCol = tokenizer.getOutputCol(), outputCol = "frequency")

# Inverse document frequenct (IDF) tells u the importance of the term for the document. A term that appears very often across the document (or in this case,in both spam and ham messages) will carry very little importance about the document. In a way, it is a numerical measure of how much information about the document does the term provides.
idf = IDF(inputCol = hashing_tf.getOutputCol(), outputCol = "features")

# Initialising the Naive Bayes Classifier 
naive_classifier = NaiveBayes()

# Initiate a pipeline. I could've done by individually tokenizer.fit(), hashing_tf.fit(), idf.fit() etc.. But pipeline does this in one shot
pipeline = Pipeline(stages = [tokenizer, hashing_tf, idf, naive_classifier])

# Fit the training data to the classifier to create the Naive Bayes Model
naive_model = pipeline.fit(training_data)

# Predict the values in the test_data
prediction = naive_model.transform(test_data)

# Form the confusion matrix of the data
print prediction.groupBy("label", "prediction").count().show()


"""
RESULT :-

+-----+----------+-----+                                                        
|label|prediction|count|
+-----+----------+-----+
|  1.0|       1.0|  184|
|  0.0|       0.0| 1324|
|  0.0|       1.0|   24|
|  1.0|       0.0|   37|
+-----+----------+-----+

Around 96.36% accuracy.

"""
