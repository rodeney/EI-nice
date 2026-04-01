package com.szubd


import org.apache.spark.ml.feature.Normalizer
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.SparkSession

object DocumentsL2Norm {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("Reddit - L2Normalization")
      .master("local[*]")
      .getOrCreate()

    // Read Parquet
    val df = spark.read.parquet("data/Reddit1000/dtm_filtered/")

    // Use Spark ML Normalizer
    val normalizer = new Normalizer()
      .setInputCol("features")
      .setOutputCol("normalized_features")
      .setP(2.0) // L2 norm (p=2.0)

    val normalizedDF = normalizer.transform(df).drop("features")

    println("L2 Normalization using MLlib Normalizer:")
    normalizedDF.show(5, false)

    // Save results
    //normalizedDF.write.mode("overwrite").parquet("data/Reddit1000/dtm_normalized/")

    spark.stop()
  }
}