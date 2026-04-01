package com.szubd

import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.ml.linalg.{VectorUDT, Vector => MLVector, Vectors => MLVectors}
import org.apache.spark.mllib.linalg.{Vector => MLlibVector, Vectors => MLlibVectors}
import org.apache.spark.sql.functions.{col, spark_partition_id}
import org.apache.spark.rdd.RDD

import scala.util.Try
import scala.collection.mutable
import scala.collection.mutable.WrappedArray

/**
 * Spark-specific operations for INice clustering.
 * Handles feature extraction from DataFrames and distributed clustering.
 */
object INiceSpark {

  /**
   * Extracts feature arrays from a DataFrame column and groups them by partition.
   * Returns RDD where each element represents all feature arrays from one partition.
   *
   * @param df Input DataFrame containing features
   * @param featuresCol Name of the column containing feature vectors
   * @return RDD of Lists, where each List contains all feature arrays from one partition
   * @throws IllegalArgumentException if column not found or unsupported data type
   */
  def extractFeaturesByPartition(df: DataFrame, featuresCol: String): RDD[List[Array[Double]]] = {
    require(df.columns.contains(featuresCol),
      s"Column '$featuresCol' not found. Available columns: ${df.columns.mkString(", ")}")

    df.select(featuresCol)
      .rdd
      .mapPartitions { iter =>
        // Process each partition and convert to List[Array[Double]]
        val partitionFeatures = iter.flatMap { row =>
          Try {
            val featureValue = row.get(0)

            featureValue match {
              // Case 1: WrappedArray (most common for array columns in DataFrames)
              case wrapped: WrappedArray[_] =>
                wrapped.map {
                  case d: Double => d
                  case f: Float => f.toDouble
                  case i: Int => i.toDouble
                  case l: Long => l.toDouble
                  case s: String => s.toDouble
                  case null => Double.NaN
                  case other =>
                    throw new IllegalArgumentException(s"Unsupported type in WrappedArray: ${other.getClass}")
                }.toArray

              // Case 2: ML Vector (Spark 2.0+ ML package)
              case mlVector: MLVector =>
                mlVector.toArray

              // Case 3: MLlib Vector (Spark 1.x MLlib package)
              case mllibVector: MLlibVector =>
                mllibVector.toArray

              // Case 4: Raw Array[Double] (ideal case)
              case array: Array[Double] =>
                array

              // Case 5: Generic array with mixed types
              case array: Array[_] =>
                array.map {
                  case d: Double => d
                  case f: Float => f.toDouble
                  case i: Int => i.toDouble
                  case l: Long => l.toDouble
                  case s: String => s.toDouble
                  case null => Double.NaN
                  case other =>
                    throw new IllegalArgumentException(s"Unsupported type in Array: ${other.getClass}")
                }

              // Case 6: Vector stored as Spark struct
              case struct: Row if struct.schema.fieldNames.contains("values") =>
                struct.getAs[Array[Double]]("values")

              // Case 7: Null value handling
              case null =>
                throw new IllegalArgumentException(s"Null value in column '$featuresCol'")

              // Case 8: Unknown type
              case other =>
                throw new IllegalArgumentException(
                  s"Unsupported feature type in column '$featuresCol': ${other.getClass.getName}")
            }
          }.recover {
              case e: Exception =>
                println(s"Warning: Failed to extract features from row - ${e.getMessage}")
                null
            }.toOption
            .filter(_ != null)
            .filter(_.nonEmpty)
        }.toList

        // Return the entire partition's features as a single List
        Iterator(partitionFeatures)
      }
  }

  /**
   * Original method preserved for backward compatibility
   */
  def extractFeatures(df: DataFrame, featuresCol: String): RDD[Array[Double]] = {
    extractFeaturesByPartition(df, featuresCol).flatMap(identity) // Flatten the lists
  }

   /**
   * Extracts feature arrays from a DataFrame column.
   * Handles both ML Vector struct format and Array[Double] formats.
   *
   * @param df Input DataFrame containing features
   * @param featuresCol Name of the column containing feature vectors
   * @return RDD of feature arrays ready for clustering
   */
  def extractFeatures1(df: DataFrame, featuresCol: String): RDD[Array[Double]] = {
    require(df.columns.contains(featuresCol),
      s"Column '$featuresCol' not found. Available columns: ${df.columns.mkString(", ")}")

    df.select(featuresCol)
      .rdd
      .map { row =>
        val featureValue = row.get(0)
        featureValue match {
          case vector: org.apache.spark.ml.linalg.Vector =>
            vector.toArray

          case vector: org.apache.spark.mllib.linalg.Vector =>
            vector.toArray

          case wrapped: mutable.WrappedArray[_] =>
            wrapped.map {
              case d: Double => d
              case f: Float => f.toDouble
              case i: Int => i.toDouble
              case l: Long => l.toDouble
              case other => other.toString.toDouble
            }.toArray

          case array: Array[Double] => array

          case struct: Row if struct.schema.fieldNames.contains("values") =>
            struct.getAs[Array[Double]]("values")

          case null => throw new IllegalArgumentException(s"Null value in column '$featuresCol'")
          case other => throw new IllegalArgumentException(s"Unsupported type in '$featuresCol': ${other.getClass}")
        }
      }
  }


  /**
   * Runs INice clustering on each partition of feature data.
   * Takes RDD[List[Array[Double]]] and returns RDD[List[Array[Double]]]
   * where each partition's features are replaced with cluster centers.
   *
   * @param featuresByPartitionRDD RDD where each element is a List of feature arrays from one partition
   * @param k Number of clusters to find in each partition
   * @param maxIter Maximum iterations for the clustering algorithm
   * @param minPartitionSize Minimum number of points required for clustering
   * @return RDD where each element is a List of cluster centers from one partition
   */
  def runINicePerPartition(
                            featuresByPartitionRDD: RDD[List[Array[Double]]],
                            k: Int = 5,
                            maxIter: Int = 10,
                            minPartitionSize: Int = 2
                          ): RDD[List[Array[Double]]] = {

    featuresByPartitionRDD.map { partitionFeatures =>
      if (partitionFeatures.size < minPartitionSize) {
        println(s"Skipping partition with ${partitionFeatures.size} points (min: $minPartitionSize)")
        List.empty[Array[Double]]  // Return empty list for small partitions
      } else {
        println(s"Clustering partition with ${partitionFeatures.size} points")
        try {
          val centers = INiceOps.fitMO(partitionFeatures, k, maxIter)
          println(s"Found ${centers.size} cluster centers")
          centers
        } catch {
          case e: Exception =>
            println(s"Clustering failed: ${e.getMessage}")
            List.empty[Array[Double]]  // Return empty list on failure
        }
      }
    }
  }

  /**
   * Runs INice clustering on each partition of feature data.
   *
   * @param featuresRDD RDD containing feature arrays
   * @param numObservationPoints Number of Observation Points
   * @param knn                  Number of neighbors for density estimation
   * @param minPartitionSize Minimum number of points required for clustering (default: 2)
   * @return RDD of cluster centers found in each partition
   */
  def runINicePerPartition1(
                     featuresRDD: RDD[Array[Double]],
                     numObservationPoints: Int = 5,
                     knn: Int = 10,
                     minPartitionSize: Int = 2
                   ): RDD[Array[Double]] = {

    featuresRDD.mapPartitions { iter =>
      val partitionData = iter.toList

      if (partitionData.size < minPartitionSize) {
        println(s"Skipping partition with ${partitionData.size} points (min: $minPartitionSize)")
        Iterator.empty
      } else {
        println(s"Clustering partition with ${partitionData.size} points")
        try {
          val centers = INiceOps.fitMO(partitionData, numObservationPoints, knn)
          println(s"Found ${centers.size} cluster centers")
          centers.iterator
        } catch {
          case e: Exception =>
            println(s"Clustering failed: ${e.getMessage}")
            Iterator.empty
        }
      }
    }
  }

  /**
   * Global ensemble consolidation after collecting all centers from all partitions.
   */
  def globalConsolidation(
                           centersRDD: RDD[Array[Double]],
                           percentage: Double = 0.2
                         ): Array[Array[Double]] = {
    val allCenters = centersRDD.collect().toList
    println(s"Consolidating ${allCenters.size} centers from all partitions")

    val consolidated: List[Array[Double]] = INiceOps.ensembleCenters(allCenters, percentage)
    println(s"Final consolidated centers: ${consolidated.size}")

    consolidated.toArray
  }



    /**
     * Writes RDD[List[Array[Double]]] to Parquet format.
     * Each partition's centers are stored with partition metadata.
     * ("partition_id", "center_index", "center_values")
     * @param centersRDD RDD containing lists of cluster centers per partition
     * @param outputPath Path where to save the Parquet files
     * @param spark SparkSession instance
     */
    def writeLocalCentersToParquet(
                               centersRDD: RDD[List[Array[Double]]],
                               outputPath: String,
                               spark: SparkSession
                             ): Unit = {
      import spark.implicits._

      // Convert to DataFrame with schema for Parquet storage
      val centersDF = centersRDD.zipWithIndex()
        .flatMap { case (centersList, partitionId) =>
          centersList.zipWithIndex.map { case (centerArray, centerIndex) =>
            (partitionId.toInt, centerIndex, centerArray)
          }
        }
        .toDF("partition_id", "center_index", "center_values")

      // Write to Parquet with compression
      centersDF.write
        .mode("overwrite")
        .option("compression", "snappy")  // Efficient compression
        .parquet(outputPath)

      println(s"Successfully wrote ${centersDF.count()} centers to $outputPath")
      println(s"Schema: ${centersDF.schema.treeString}")
    }

  // Get centers for specific partition (block)
  def getCentersForPartition(clusterCentersRDD: RDD[List[Array[Double]]], partitionId: Int): RDD[Array[Double]] = {
    clusterCentersRDD.zipWithIndex()
      .filter { case (_, idx) => idx == partitionId }  // Filter for specific partition
      .flatMap { case (centersList, _) => centersList }  // Flatten the list
  }

  // Get centers for the first N partitions
  def getCentersForFirstPartitions(clusterCentersRDD: RDD[List[Array[Double]]], numPartitions: Int): RDD[Array[Double]] = {
    require(numPartitions > 0, "Number of partitions must be positive")

    clusterCentersRDD.zipWithIndex()
      .filter { case (_, partitionIdx) => partitionIdx < numPartitions }
      .flatMap { case (centersList, _) => centersList }
  }

  // Get the first N partitions as RDD[List[Array[Double]]]
  def getFirstPartitions(clusterCentersRDD: RDD[List[Array[Double]]], numPartitions: Int): RDD[List[Array[Double]]] = {
    require(numPartitions > 0, "Number of partitions must be positive")

    clusterCentersRDD.zipWithIndex()
      .filter { case (_, partitionIdx) => partitionIdx < numPartitions }
      .map { case (centersList, _) => centersList }
  }



  def countElementsByPartition[T](rdd: RDD[T]): Array[(Int, Int)] = {
    rdd.mapPartitionsWithIndex { (partitionIndex, partitionData) =>
      Iterator((partitionIndex, partitionData.length))
    }.collect()
  }

  def countArraysByPartition(rdd: RDD[List[Array[Double]]]): Array[(Int, Int)] = {
    rdd.mapPartitionsWithIndex { (partitionIndex, partitionData) =>
      val arrayCount = partitionData.map(_.length).sum
      Iterator((partitionIndex, arrayCount))
    }.collect()
  }

  /**
   * Complete example: Read Parquet, extract features, run clustering, and show results.
   */
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("INiceSparkExample")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    try {
      // Use local paths and default values if no arguments provided
      val inputPath = if (args.length > 0) args(0) else "data/Reddit201201/data_normalized_svd_k100/"
      val outputPath = if (args.length > 1) args(1) else "data/Reddit201201/results5_5_100_1000/"

      val featuresCol = if (args.length > 2) args(2) else "svd_k100_features"
      val numObservationPoints = if (args.length > 3) args(3).toInt else 5
      val knn = if (args.length > 4) args(4).toInt else 5

      println("Step 1: Reading Parquet file...")
      val df = spark.read.parquet(inputPath)//.limit(10000)

      println("DataFrame schema:")
      df.printSchema()
      println(s"Total records: ${df.count()}")

      println("\nStep 2: Extracting features...")
      val featuresRDD: RDD[List[Array[Double]]] = extractFeaturesByPartition(df, featuresCol)
      println(s"Extracted ${featuresRDD.count()} Lists of feature vectors")
      println(s"#Partitions in featuresRDD: ${featuresRDD.getNumPartitions} (each partition is preserved as a list feature vectors)")

      val partitionCounts = countElementsByPartition(featuresRDD)

      println("Elements per partition:")
      partitionCounts.foreach { case (partitionIndex, count) =>
        println(s"Partition $partitionIndex: $count elements")
      }

      // Count arrays by partition
      val partitionArraysCounts = countArraysByPartition(featuresRDD)

      println("Arrays per partition:")
      partitionArraysCounts.foreach { case (partitionIndex, count) =>
        println(s"Partition $partitionIndex: $count arrays")
      }

      val totalArrays = partitionArraysCounts.map(_._2).sum
      println(s"\nTotal arrays: $totalArrays")

      println("\nAs a test, let's take 5 partitions only...")
      val featuresRDDFirst5Partitions = featuresRDD.mapPartitionsWithIndex { (index, iterator) =>
        if (index < 5) iterator else Iterator.empty
      }.filter(_.nonEmpty)

      println(s"#Partitions in featuresRDDFirst5Partitions: ${featuresRDDFirst5Partitions.getNumPartitions}")


      println("\nStep 3: Running clustering per partition...")
      val clusterCentersListRDD: RDD[List[Array[Double]]] = runINicePerPartition(featuresRDDFirst5Partitions, numObservationPoints, knn)
      val totalCenters = clusterCentersListRDD.count()
      println(s"Total cluster centers found: $totalCenters")
      println("\n=== WRITING Local Centers TO PARQUET ===")
      clusterCentersListRDD.toDF("centers_list")  // Convert to DataFrame with one column
        .write
        .mode("overwrite")
        .parquet(s"${outputPath}localCentersPerPartition")
      // Flatten the lists to get all centers in one RDD
      val clusterCentersArrayRDD: RDD[Array[Double]] = clusterCentersListRDD.flatMap(identity)
      println(s"Total centers from all partitions: ${clusterCentersArrayRDD.count()}")

      println("\nStep 4: Saving local centers from all partitions...")
      clusterCentersArrayRDD.toDF("center_array")
        .write
        .mode("overwrite")
        .parquet(s"${outputPath}localCenters")

      println("\nStep 6: Merge enters from one block as an example...")
      val oneBlockLocalCenters: RDD[Array[Double]] = getCentersForPartition(clusterCentersListRDD, 3)
      val oneBlockMergedCenters: Array[Array[Double]] = globalConsolidation(oneBlockLocalCenters, 0.3)

      println("\nStep 4: Saving merged centers from one partition...")
      val oneBlockMergedCentersRDD = spark.sparkContext.parallelize(oneBlockMergedCenters)
      oneBlockMergedCentersRDD.toDF("center_array")
        .write
        .mode("overwrite")
        .parquet(s"${outputPath}mergedCentersFromOneBlock")

      println("\nStep 5: apply ensemble to local centers from all partitions...")
      val finalCenters = globalConsolidation(clusterCentersArrayRDD, 0.3)
      println(s"Total final centers found: ${finalCenters.length}")
      println("\n Saving final centers merged from all partitions...")
      val finalCentersRDD = spark.sparkContext.parallelize(finalCenters)
      finalCentersRDD.toDF("center_array")
        .write
        .mode("overwrite")
        .parquet(s"${outputPath}mergedCentersFromAllBlocks")
    } catch {
      case e: Exception =>
        println(s"Error: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      spark.stop()
    }
  }
}