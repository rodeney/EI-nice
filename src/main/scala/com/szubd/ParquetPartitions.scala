package com.szubd

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object ParquetPartitions {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SimplePartitionExtract")
      .master("local[*]")
      .getOrCreate()

    // Input and output paths
    val inputPath = "data/R2012-01/"
    val outputPath = "data/R201201/Partitions/"


    //  Read Parquet file
    println("Reading Parquet file...")
    val df = spark.read.parquet(inputPath)
    println(s"Original partition count: ${df.rdd.getNumPartitions}")
    println(s"Original data count: ${df.count()}")

    //  Repartition the data
    val numPartitions = 40 // Adjust based on your data size
    println(s"Repartitioning to $numPartitions partitions...")
    val repartitionedDF = df.repartition(numPartitions)

    println(s"New partition count: ${repartitionedDF.rdd.getNumPartitions}")

    for (i <- 0 until numPartitions) {
      println(s"Writng output part...")
      repartitionedDF.filter(spark_partition_id() === i)
        .write
        .mode("overwrite")
        .parquet(s"${outputPath}partition_${i}")
    }

    //Read and extract first partition in one go
//    repartitionedDF.filter(spark_partition_id() === 0)
//      .write
//      .mode("overwrite")
//      .parquet(outputPath)


//    println("Verifying output...")
//    val writtenDF = spark.read.parquet(outputPath)
//    println(s"Written file rows: ${writtenDF.count()}")

    spark.stop()
  }
}