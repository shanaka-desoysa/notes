---
title: PySpark custom shape function
date: 2024-12-06
author: Shanaka DeSoysa
description: 
---

Custom `df.shape()` function for PySpark dataframe.

```python
import pyspark

def spark_shape(self) :
    return (self.count(), len(self.columns) )

pyspark.sql.dataframe.DataFrame.shape = spark_shape

df.shape()
```

Just reminder that `.count()` could be very slow for large tables. 
