import pyspark

def spark_shape(self) :
    return (self.count(), len(self.columns) )

pyspark.sql.dataframe.DataFrame.shape = spark_shape
