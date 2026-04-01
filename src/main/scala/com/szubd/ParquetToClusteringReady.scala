package com.szubd

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.SparseVector

object ParquetToClusteringReady {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("ParquetToClusteringReady")
      .master("local[*]")
      .getOrCreate()

    val inputPath = "data/dtm20052006_filtered_newVectorSize/"
    val outputPath = "data/dtm20052006_filtered_newVectorSize_csv/"

    // Read the parquet file
    val df = spark.read.parquet(inputPath)

    df.printSchema()
    df.show(5, truncate = false)

    // Identify the vector column (usually "features")
    val vectorCol = "features" // Change this if different
    val vectorSize = df.select(col(vectorCol)).first().getAs[SparseVector](0).size

    println(s"Vector size: $vectorSize")
    println(s"Total rows: ${df.count()}")

    // Convert SparseVector to dense array for clustering
    val clusteringDF = df.withColumn("dense_features",
      udf((v: SparseVector) => {
        val denseArray = new Array[Double](v.size)
        v.indices.zip(v.values).foreach { case (idx, value) =>
          denseArray(idx) = value
        }
        denseArray
      }).apply(col(vectorCol))
    ).drop(vectorCol)  // Remove the original sparse column

    // Explode the dense array into individual columns
    val featureColumns = (0 until vectorSize).map { i =>
      col("dense_features")(i).as(s"feature_$i")
    }

    val finalDF = clusteringDF.select(col("*") +: featureColumns: _*)
      .drop("dense_features")  // Remove the array column

    // Save as Parquet (better than CSV for ML)
    finalDF.write
      .mode("overwrite")
      .parquet(outputPath)

    println(s"Clustering-ready data saved to: $outputPath")
    println(s"Final schema:")
    finalDF.printSchema()

    spark.stop()
  }
}