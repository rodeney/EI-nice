package com.szubd
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector, SparseVector, DenseVector}
import org.apache.spark.sql.types._

object ParquetTest {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("ParquetToCSVPartitions")
      .master("local[*]")
      .getOrCreate()

    try {
      // Read the parquet file
      val inputPath = "data/dtm20052006_filtered_newVectorSize_svd100/"
      val outputPath = "data/dtm20052006_filtered_newVectorSize_svd100_csv/"

      val df = spark.read.parquet(inputPath)

      // Get the number of partitions
      val numPartitions = df.rdd.getNumPartitions
      println(s"Number of partitions: $numPartitions")
      println(s"Total rows: ${df.count()}")

      // Print schema to understand the data structure
      println("Original schema:")
      df.printSchema()

      // Show sample data to understand the format
      println("Sample data:")
      df.limit(5).show(truncate = false)

      // Find array/vector columns
      val arrayColumns = df.schema.fields.collect {
        case field if field.dataType.isInstanceOf[ArrayType] => field.name
      }

      println(s"Array columns found: ${arrayColumns.mkString(", ")}")

      if (arrayColumns.isEmpty) {
        println("No array columns found. Please check your schema.")
        return
      }

      // Assuming you have one main array column
      val arrayColName = arrayColumns.head

      // Get the size of the array by sampling
      val sampleArray = df.select(col(arrayColName)).limit(1).collect()(0).getAs[Seq[Any]](0)
      val arraySize = sampleArray.size
      println(s"Array size: $arraySize")

      // Explode the array into multiple columns
      var explodedDF = df

      // Create individual columns for each array element
      (0 until arraySize).foreach { i =>
        explodedDF = explodedDF.withColumn(s"${arrayColName}_$i", col(arrayColName)(i))
      }

      // Drop the original array column
      explodedDF = explodedDF.drop(arrayColName)

      println("Schema after array explosion:")
      explodedDF.printSchema()

      // Show a sample of the exploded data
      println("Sample of exploded data:")
      explodedDF.limit(5).show()

      // Save each partition as a separate CSV file
      explodedDF.repartition(numPartitions)
        .write
        .option("header", "false")
        .mode("overwrite")
        .csv(outputPath)

      println(s"Successfully converted Parquet to CSV in directory: $outputPath")
      println(s"Created $arraySize additional columns from the array")

    } catch {
      case e: Exception =>
        println(s"Error occurred: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      // Stop SparkSession
      spark.stop()
    }
  }
}