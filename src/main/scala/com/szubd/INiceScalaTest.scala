package com.szubd
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.sql.types._
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.mllib.linalg.{Vector => MLlibVector, Vectors => MLlibVectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{col, udf}

import scala.collection.JavaConverters._
import scala.collection.mutable

// INice Java class
import com.szubd.INice

/*
* Test INice in Scala
* Read data from parquet file. Data should have a features column. Check its type (ML vector or MLlib vector or WrappedArray)
* It depends on how the data was stored
* Using Spark DataFrame/RDD
* Run INice in parallel to each partition using mapPartitions
*
 */
object INiceScalaTest {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("INiceScala_Test")
      .master("local[*]")
      .getOrCreate()

    // Use local paths and default values if no arguments provided
    val inputPath = if (args.length > 0) args(0) else "data/Reddit20052006/data_nomralized_svd_k100/"
    val outputPath = if (args.length > 1) args(1) else "data/Reddit20052006/test/"
    val featuresCol = if (args.length > 1) args(1) else "svd_k100_features"

    try {
      import spark.implicits._

      // Step 1: Load data
      val df = spark.read.parquet(inputPath)

      df.printSchema()
      df.show(5, false) // Show all 30 rows

      // Verify the column exists
      require(df.schema.exists(_.name == featuresCol),
        s"Column '$featuresCol' not found in DataFrame. Available columns: ${df.columns.mkString(", ")}")

      // Step 2: Extract features as Array[Double]
      val featureRDD = df.select(featuresCol)
        .rdd
        .map { row => extractFeatures(row, featuresCol)}



      // Convert directly to Dataset[Array[Double]] ...doesn't work sometimes!
//      val outputCol = "features_array"
//
//      val featuresDS = df.select(col(featuresCol).getField("values").as(outputCol))
//        .as[Array[Double]]

      // Now you have a clean Dataset of arrays
//      val featureRDD = featuresDS.rdd

      println(s"Total records: ${featureRDD.count()}")

      // Check a sample feature
      val sampleFeature = featureRDD.first()
      println(s"First feature array length: ${sampleFeature.length}")
      println(s"Sample values: ${sampleFeature.take(5).mkString(", ")}...")


      //try directly with scala wrapper method
      // Now you can use the clean Scala interface
      val clusterCentersRDD = featureRDD.mapPartitions { iter =>
        val partitionFeatures = iter.toList

        if (partitionFeatures.isEmpty) {
          Iterator.empty
        } else {
          // Clean Scala call - no Java conversion visible
          val centers = fitMO(partitionFeatures, 5, 10)
          centers.iterator
        }
      }

      println(s"Total records in clusterCentersRDD: ${clusterCentersRDD.count()}")


      // Step: Convert each partition to Java List<double[]>
      val javaListRDD: RDD[java.util.List[Array[Double]]] = featureRDD.mapPartitions { iter =>
        // Convert current partition's data to Java List
        val javaList = iter.toList.asJava
        // Return as Iterator containing one Java List per partition
        Iterator(javaList)
      }// Now javaListRDD contains: RDD[List<double[]>] where each partition has one Java List

      println(s"Total partitions in javaListRDD: ${javaListRDD.getNumPartitions}")
      println(s"Total records in javaListRDD: ${javaListRDD.count()}")

      // Step 2: Apply Java clustering to each partition's Java List
//      val clusterCentersRDD: RDD[Array[Double]] = javaListRDD.mapPartitions { iter =>
//        iter.flatMap { javaList =>  // Each element is a Java List<double[]>
//          // Apply Java clustering method
//          val javaCenters = INice.fitMO(javaList, 5, 10)
//
//          // Convert Java results back to Scala and flatten
//          javaCenters.asScala
//        }
//      }
//
//      println(s"Total records in clusterCentersRDD: ${clusterCentersRDD.count()}")

      // Can be done together: Convert to Java and Apply Java clustering per partition
//      val clusteredRDD = featureRDD.mapPartitions { iter =>
//        // Convert iterator to list for Java method
//        val data = iter.toList
//
//        if (data.isEmpty) {
//          println("Empty partition encountered")
//          Iterator.empty
//        } else {
//          println(s"Processing partition with ${data.size} records")
//
//          // Apply Java clustering - adjust parameters as needed
//          val centers = INice.fitMO(data.asJava, 2, 2).asScala.toList
//
//          // Return rows: (center_id, center_vector, partition_size)
//          centers.zipWithIndex.iterator.map { case (arr, idx) =>
//            Row(idx, Vectors.dense(arr), data.size)
//          }
//        }
//      }
//
//      // Schema for the result DataFrame
//      val schema = StructType(Array(
//        StructField("center_id", IntegerType, nullable = false),
//        StructField("center_vector", VectorType, nullable = false),
//        StructField("partition_size", IntegerType, nullable = false)
//      ))
//
//      // Step 4: Convert clusteredRDD to DataFrame
//      val centersDF = spark.createDataFrame(clusteredRDD, schema)
//
//      println("Cluster centers (per partition):")
//      centersDF.show(false)
//
//      // Additional analysis
//      println(s"Total cluster centers found: ${centersDF.count()}")

    } finally {
      spark.stop()
    }
  }

  /**
   * Scala-friendly wrapper for INice.fitMO Java method
   *
   * @param features Scala List of feature arrays (for List<double[]> in Java)
   * @param numObservationPoints Number of Observation Points
   * @param knn Number of neighbors for density estimation.
   * @return Scala List of cluster centers
   */
  def fitMO(features: List[Array[Double]], numObservationPoints: Int, knn: Int): List[Array[Double]] = {
    // Convert Scala List[Array[Double]] to Java List<double[]>
    val javaFeatures = features.asJava

    // Call Java method
    val javaCenters = INice.fitMO(javaFeatures, numObservationPoints, knn)

    // Convert Java results back to Scala
    javaCenters.asScala.toList
  }


  /**
   * Extracts feature values from a Spark Row and converts them to Array[Double].
   *
   * This function handles multiple data types that Spark might use to store feature vectors,
   * including WrappedArray, ML vectors, MLlib vectors, and various array types.
   *
   * @param row The Spark Row containing the feature data
   * @param columnName The name of the column containing the feature vector
   * @return Array[Double] containing the feature values
   * @throws IllegalArgumentException if the feature value is null or of an unsupported type
   *
   * @example
   * // Extract features from a DataFrame row
   * val features = extractFeatures(row, "features")
   * // features: Array[Double] = Array(1.0, 2.0, 3.0, ...)
   *
   * @note This function is designed to handle common Spark data storage patterns:
   *   - WrappedArray[Double] (most common for array columns)
   *   - ML Vector (org.apache.spark.ml.linalg.Vector)
   *   - MLlib Vector (org.apache.spark.mllib.linalg.Vector)
   *   - Raw Array[Double] and other array types
   */
  def extractFeatures(row: Row, columnName: String): Array[Double] = {
    // Get the raw value from the specified column
    // Using getAs[Any] to handle multiple possible types
    val featureValue = row.getAs[Any](columnName)

    // Pattern match on the feature value to handle different storage formats
    featureValue match {

      // Case 1: WrappedArray - Spark's internal representation for array columns
      case wrapped: mutable.WrappedArray[_] =>
        wrapped.map {
          case d: Double => d        // Already Double - no conversion needed
          case f: Float => f.toDouble  // Convert Float to Double
          case i: Int => i.toDouble    // Convert Int to Double
          case l: Long => l.toDouble   // Convert Long to Double
          case s: String => s.toDouble // Convert String to Double (if stored as strings)
          case other => other.toString.toDouble // Fallback: try to convert anything else via string
        }.toArray // Convert WrappedArray to standard Scala Array

      // Case 2: ML Vector (Spark 2.0+ ML package)
      // Used by Spark ML algorithms and recommended for new code
      case vector: org.apache.spark.ml.linalg.Vector =>
        vector.toArray // Direct conversion to array

      // Case 3: MLlib Vector (Spark 1.x MLlib package - deprecated but still found in older data)
      // Used by legacy Spark MLlib algorithms
      case vector: org.apache.spark.mllib.linalg.Vector =>
        vector.toArray // Direct conversion to array

      // Case 4: Raw Array[Double] - ideal case, no conversion needed
      // This is rare in Spark DataFrames but possible in some scenarios
      case array: Array[Double] =>
        array // Return as-is - most efficient case

      // Case 5: Generic Array with mixed types - convert all elements to Double
      // Handles arrays that might contain various numeric types
      case array: Array[_] =>
        array.map {
          case d: Double => d        // Already Double
          case f: Float => f.toDouble  // Float to Double
          case i: Int => i.toDouble    // Int to Double
          case other => other.toString.toDouble // Fallback conversion
        }

      // Case 6: Null value - throw informative exception
      // Null features are typically data quality issues that should be handled upstream
      case null =>
        throw new IllegalArgumentException(s"Null value found in column '$columnName'. " +
          "Consider filtering null values before processing.")

      // Case 7: Any other unsupported type - throw descriptive exception
      // Helps with debugging unexpected data formats
      case other =>
        throw new IllegalArgumentException(
          s"Unsupported feature type in column '$columnName': ${other.getClass.getName}. " +
            s"Expected one of: WrappedArray, ML Vector, MLlib Vector, or Array. " +
            s"Actual value: $other"
        )
    }
  }

  //same without docs
  def extractFeatures1(row: Row, columnName: String): Array[Double] = {
    val featureValue = row.getAs[Any](columnName)

    featureValue match {
      case wrapped: mutable.WrappedArray[_] =>
        wrapped.map {
          case d: Double => d
          case f: Float => f.toDouble
          case i: Int => i.toDouble
          case l: Long => l.toDouble
          case s: String => s.toDouble
          case other => other.toString.toDouble
        }.toArray

      case vector: org.apache.spark.ml.linalg.Vector =>
        vector.toArray

      case vector: org.apache.spark.mllib.linalg.Vector =>
        vector.toArray

      case array: Array[Double] =>
        array

      case array: Array[_] =>
        array.map {
          case d: Double => d
          case f: Float => f.toDouble
          case i: Int => i.toDouble
          case other => other.toString.toDouble
        }

      case null =>
        throw new IllegalArgumentException(s"Null value in column $columnName")

      case other =>
        throw new IllegalArgumentException(s"Unsupported type in $columnName: ${other.getClass.getName}")
    }
  }


  //another way with UDF
  // Create a UDF that handles all the type conversions
  val extractFeaturesUDF = udf { (features: Any) =>
    features match {
      case wrapped: mutable.WrappedArray[_] =>
        wrapped.map {
          case d: Double => d
          case f: Float => f.toDouble
          case i: Int => i.toDouble
          case l: Long => l.toDouble
          case s: String => s.toDouble
          case other => other.toString.toDouble
        }.toArray

      case vector: org.apache.spark.ml.linalg.Vector =>
        vector.toArray

      case vector: org.apache.spark.mllib.linalg.Vector =>
        vector.toArray

      case array: Array[Double] => array

      case array: Array[_] =>
        array.map {
          case d: Double => d
          case f: Float => f.toDouble
          case i: Int => i.toDouble
          case other => other.toString.toDouble
        }

      case null => null
      case other => throw new IllegalArgumentException(s"Unsupported type: ${other.getClass.getName}")
    }
  }
  //Example
  // Apply directly to the column
//  val featuresDF = df.withColumn("features_array", extractFeaturesUDF(col("svd_100d_features")))
//
//  // Now use the converted column directly
//  val featureRDD = featuresDF.select("features_array")
//    .as[Array[Double]]
//    .rdd
}