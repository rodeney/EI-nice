package com.szubd

import breeze.linalg.DenseVector
import com.szubd.INiceSpark.{extractFeatures, globalConsolidation}
import org.apache.spark.SparkConf
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

import java.util.stream.Stream

//import INice Java package
import com.szubd.INice

// import LOGO Packages
import org.apache.spark.rsp.RspRDD
import org.apache.spark.sql.RspContext._
import org.apache.spark.sql.{Row, RspDataset, SparkSession}

//import other Java packages
import java.util


/**
 * INice with LOGO framework.
 *
 */


object INiceLOGOTest {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("INiceLOGO_Test")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    try {
      // Configuration
      // Use local paths and default values if no arguments provided
      val inputPath = if (args.length > 0) args(0) else "data/Reddit201201/data_normalized_svd100_scaled/"
      val outputPath = if (args.length > 1) args(1) else "data/Reddit201201/results/logo_test/"

      val featuresCol = if (args.length > 2) args(2) else "svd100_scaled_features"
      val numObservationPoints = if (args.length > 3) args(3).toInt else 5
      val knn = if (args.length > 4) args(4).toInt else 5

      println("Step 1: Reading Parquet file...")
      val inputData: RspDataset[Row] = spark.rspRead.parquet(inputPath)
      println("Data schema:")
      inputData.printSchema()
      println(s"Total partitions: ${inputData.rdd.getNumPartitions}")
      println(s"Total records: ${inputData.count()}")

      println("Step 2: Select partitions...")
      val selectedPartitions: RspDataset[Row] = inputData.getSubDataset(5)
      println("SubData:")
      selectedPartitions.printSchema()
      println(s"Total partitions: ${selectedPartitions.rdd.getNumPartitions}")
      println(s"Total records: ${selectedPartitions.count()}")

      println("Step 3： Extract Features as ...")
      val featuresRDD: RDD[Array[Double]] = extractFeatures(selectedPartitions, featuresCol)
      println(s"Extracted ${featuresRDD.count()} feature vectors")
      println(s"Extracted features has ${featuresRDD.getNumPartitions} partitions")

      // Convert each partition to a List[Array[Double]]
      val partitionsAsListsRDD: RDD[List[Array[Double]]] = featuresRDD.mapPartitions { iter =>
        Iterator(iter.toList)
      }  // Convert entire partition iterator to List
      println(s"partitionsAsListsRDD has ${partitionsAsListsRDD.count()} lists of feature vectors")
      println(s"partitionsAsListsRDD has ${partitionsAsListsRDD.getNumPartitions} partitions")

      println("Step 4： Run LO ...")
      val featuresRspRDD = new RspRDD(partitionsAsListsRDD)
      println(s"featuresRspRDD has ${featuresRspRDD.count()} lists of feature vectors")
      println(s"featuresRspRDD has ${featuresRspRDD.getNumPartitions} partitions")

      val localINiceCenters = featuresRspRDD.LO(trainDF => INiceOps.fitMO(trainDF, 2, 5))
      val localINiceCentersList = localINiceCenters.collect().toList.flatten

      println(s"localINiceCenters has ${localINiceCentersList.size} local center")

      println("Step 8： Run GO ...  ")
      val globalCenters = INiceOps.ensembleCenters(localINiceCentersList, 0.2)
      println(s"Global centers: ${globalCenters.length} ")


    }finally {
      spark.close()
    }
  }


}


//class INiceLOGO {

//add a method to do every thing with LOGO
// def runINiceLOGO()
//



//  /**
//   * Run the INice sequential algorithm on a partition of an RDD (rspRDD)
//   *
//   * @param rspPartition
//   * @return Local centroids
//   */

//  def iniceLO(rspPartition: RDD[Array[Array[Double]]], numObservationPoints: Int, k: Int): util.List[Array[Double]] = {
//    // Convert rspRDD's partition to local data structure
//
//    // Apply INice local method
//   // INice.fitMO(rspPartition, numObservationPoints , k)
//
//  }
//
//  /**
//   * Computes final global centroids from a set of local centroids represented as Spark RDD
//   * It converts the input RDD into a local data structure and call the local function: computeGlobalCentroids
//   * @param blockCentroidsRDD RDD of centroid arrays from RSP blocks
//   * @return Array of final centroids representing global clusters
//   */
//
//  def iniceGO(blockCentroidsRDD: RDD[Array[Array[Double]]], percentage: Double): util.List[Array[Double]] = {
//    // Convert RDD to local data structure
//    val blockCentroidsList: List[Array[DenseVector[Double]]] = blockCentroidsRDD
//      .collect() // Convert RDD to Array (brings all data to driver)
//      .map { arrayOfArrays =>
//        arrayOfArrays.map(arr => DenseVector(arr)) // Convert each Array[Double] to DenseVector
//      }
//      .toList // Convert Array to List
//
//    // Convert to Java data structure
//
//    // Apply local method to merge centers
//    //INice.mergeData(java.util.Arrays.asList(blockCentroidsList,percentage))
//
//
//  }

//}
