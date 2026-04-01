package com.szubd

import com.szubd.INiceSpark.globalConsolidation
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.rdd.RDD

import java.io.{File, PrintWriter}


object INiceSparkMergeExperiment {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("INiceSparkMergeExample")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    try {
      //2    01201
      val localCentersPath = if (args.length > 0) args(0) else "data/Reddit201201/results/run_10_5_100_1000_scaled/localCentersPerPartition"
      val outputPath = if (args.length > 1) args(1) else "data/Reddit201201/results/run_10_5_100_1000_scaled/"
      //2012
//      val localCentersPath = if (args.length > 0) args(0) else "data/Reddit2012/results/run_5_5_100_10000/localCentersPerPartition"
//      val outputPath = if (args.length > 1) args(1) else "data/Reddit2012/results/run_5_5_100_10000/"

      println("\nRead local centers...")
      val clusterCentersDF = spark.read.parquet(localCentersPath)
      println(s"\n#Local Centers Lists：${clusterCentersDF.count()}")
      clusterCentersDF.show(5)
      clusterCentersDF.printSchema

      // Convert back to RDD[List[Array[Double]]]
      val recoveredRDD: RDD[List[Array[Double]]] = clusterCentersDF.rdd.map { row =>
        val seqOfSeq = row.getAs[Seq[Seq[Double]]]("centers_list")
        seqOfSeq.map(_.toArray).toList  // Convert Seq[Seq[Double]] to List[Array[Double]]
      }

      println(s"Total partitions found: ${recoveredRDD.getNumPartitions}")
      println(s"Total local centers found: ${recoveredRDD.flatMap(identity).count()}")

      // Get statistics
      val (totalCenters, minCenters, maxCenters, avgCenters) = getCenterCountStats(recoveredRDD)

      // Print results
      println("Cluster Center Statistics:")
      println(s"Total centers across all partitions: $totalCenters")
      println(s"Minimum centers in a partition: $minCenters")
      println(s"Maximum centers in a partition: $maxCenters")
      println(s"Average centers per partition: ${avgCenters.formatted("%.2f")}")

      // Show the actual counts for each partition
      println("\nCenter counts per partition:")
      val partitionCounts = recoveredRDD.map(_.length).collect()
      partitionCounts.zipWithIndex.foreach { case (count, index) =>
        println(s"Partition $index: $count centers")
      }
      val localStatsDetails: List[(Int, Int, Int, Int)] = incrementalLocalStatisticsBatched(recoveredRDD)
                  println("Partition | Total | Avg ")
                  println("---------|-------|------")
      localStatsDetails.foreach { case (partition, locals, total, avg) =>
                    println(f"$partition%9d | $locals%5d| $total%5d | $avg%8d")
                  }
            println("Save local stats...")
            //saveLocalStatsAsCSV(localStatsDetails, s"${outputPath}IncLocalStats_ByBlock.csv")
            println("Save local stats...Done")

          val mergePercentage = 0.9
          val mergeStatsDetails: List[(Int, Int, Int, Int, Double, Double)] = incrementalStatisticsBatched(recoveredRDD, 1, mergePercentage)
            println("Partition | Total | Avg |Merged | Compression | Reduction %")
            println("---------|-------|------|-----|------------|------------")
            mergeStatsDetails.foreach { case (partition, total, avg, merged, compression, reduction) =>
              println(f"$partition%9d | $total%5d | $avg%8d |$merged%6d | $compression%10.3f | $reduction%10.1f%%")
            }
      println("Save merging stats...")
      saveStatsAsCSV(mergeStatsDetails, s"${outputPath}IncMergeStats_ByBatch_Threshold03.csv")
      println("Save merging stats...Done")

//      val mergeStatsDetails: List[(Int, Int, Int, Int, Double, Double)] = incrementalStatistics(recoveredRDD)
//      println("Partition | Total | Avg/Part | Merged | Compression | Reduction %")
//      println("---------|-------|----------|--------|------------|------------")
//      mergeStatsDetails.foreach { case (partition, total, avg, merged, compression, reduction) =>
//        println(f"$partition%9d | $total%5d | $avg%8d | $merged%6d | $compression%10.3f | $reduction%10.1f%%")
//      }
//      saveStatsAsCSV(mergeStatsDetails, s"${outputPath}mergeStats.csv")
      //saveIncrementalStatsAsCSV(mergeStats, s"${outputPath}mergeStats")

//      // Verify data integrity
//      println("Original first few records:")
//      recoveredRDD.take(3).foreach(println)
//
//      println("\nRecovered first few records:")
//      recoveredRDD.take(3).foreach(println)
//
//      val testMerge: Array[Array[Double]] = globalConsolidation(recoveredRDD.flatMap(identity), 0.3)
//      testMerge.take(3).foreach(println)
//      val testMergeRDD = spark.sparkContext.parallelize(testMerge)
//      testMergeRDD.toDF("center_array").show(5, false)
//
//      //test
////      val clusterCenters10 = getFirstNPartitions(clusterCentersDF, 10)
////      clusterCenters10.printSchema()
////      clusterCenters10.show(5)
////      println(s"\n#selected # parts：${clusterCenters10.rdd.getNumPartitions}")
//
//      println("\nCheck merged centers...")
//      val mergedCentersDF = spark.read.parquet(mergedCentersPath)
//      println(s"\n#Centers：${mergedCentersDF.count()}")
//      mergedCentersDF.show(5, false)
//      mergedCentersDF.printSchema


      // --------- merge only a certain number of partitions
      //val localCentersBatchList: List[Array[Double]] = getFirstNLists(recoveredRDD, 25)
      //println(s"Total local centers found in the selected lists: ${localCentersBatchList.size}")

      //val globalCentersFromBatch = INiceOps.ensembleCenters(localCentersBatchList, 0.3)
      //println(s"Total global centers found: ${globalCentersFromBatch.size}")

//      val globalCentersFromBatchRDD = spark.sparkContext.parallelize(globalCentersFromBatch)
//      globalCentersFromBatchRDD.toDF("center_array")
//        .write
//        .mode("overwrite")
//        .parquet(s"${outputPath}mergedCentersFrom20Blocks")
    }finally {
    spark.stop()
    }
  }

  /**
   * Get the first n partitions of local centers from an RDD
   * @param clusterCentersListRDD RDD containing List[Array[Double]] of local centers per partition
   * @param n Number of partitions to retrieve
   * @return RDD[List[Array[Double]]] containing only the first n partitions
   */
  def getFirstNPartitions(
                           clusterCentersListRDD: RDD[List[Array[Double]]],
                           n: Int
                         ): RDD[List[Array[Double]]] = {

    require(n >= 0, "n must be non-negative")

    // Use mapPartitionsWithIndex to filter out partitions beyond n-1
    clusterCentersListRDD.mapPartitionsWithIndex {
      case (partitionIndex, iterator) =>
        if (partitionIndex < n) {
          iterator
        } else {
          Iterator.empty
        }
    }
  }


  /**
   * Get the first n lists from the RDD and combine them into a single List[Array[Double]]
   * @param clusterCentersListRDD RDD containing List[Array[Double]] of local centers
   * @param n Number of lists to retrieve
   * @return List[Array[Double]] containing combined centers from the first n lists
   */
  def getFirstNLists(
                      clusterCentersListRDD: RDD[List[Array[Double]]],
                      n: Int
                    ): List[Array[Double]] = {

    require(n >= 0, "n must be non-negative")

    // Take the first n elements from the RDD
    val firstNLists: Array[List[Array[Double]]] = clusterCentersListRDD.take(n)

    // Combine all lists into a single list
    firstNLists.flatMap(identity).toList
  }

//  /**
//   * Calculate incremental statistics: total local centers, average local centers, and number of merged centers
//   * @param clusterCentersListRDD RDD containing List[Array[Double]] of local centers per partition
//   * @return List of (upToPartition, totalLocalCenters, avgLocalCenters, mergedCentersCount)
//   */
//  def incrementalStatistics(
//                             clusterCentersListRDD: RDD[List[Array[Double]]]): List[(Int, Int, Int, Int)] = {
//
//    // Collect the entire RDD to driver
//    val collectedData: Array[List[Array[Double]]] = clusterCentersListRDD.collect()
//    val totalPartitions = collectedData.length
//
//    // Process incrementally from partition 0 to i for each i
//    (0 until totalPartitions).map { upToPartition =>
//      // Take subsets from partition 0 to upToPartition
//      val subset = collectedData.take(upToPartition + 1).toList
//
//      // Calculate statistics
//      val totalLocalCenters = subset.flatMap(identity).size
//      val avgLocalCenters = if (subset.nonEmpty) math.round(totalLocalCenters.toDouble / subset.size).toInt else 0
//      val mergedCentersCount = ensembleMergeFunction(subset, 0.2)
//
//      (upToPartition, totalLocalCenters, avgLocalCenters, mergedCentersCount)
//    }.toList
//  }

  // Create merge function that uses ensembleCenters
  def ensembleMergeCount(allCenters: List[Array[Double]], percentage: Double = 0.5): Int = {
    //val allCentersFlatten = allCenters.flatten
    val mergedCenters = INiceOps.ensembleCenters(allCenters, percentage)
    mergedCenters.size
  }

  def ensembleMerge(allCenters: List[Array[Double]], percentage: Double = 0.5): List[Array[Double]] = {
    val mergedCenters = INiceOps.ensembleCenters(allCenters, percentage)
    mergedCenters
  }

  def getCenterCountStats(clusterCentersListRDD: RDD[List[Array[Double]]]): (Long, Int, Int, Double) = {
    val counts = clusterCentersListRDD.map(_.length)
    val total = counts.sum().toInt
    val min = counts.min()
    val max = counts.max()
    val avg = counts.mean()

    (total, min, max, avg)
  }

  /**
   * Calculate incremental statistics: total local centers, average local centers, merged centers,
   * compression ratio, and reduction percentage
   * @param clusterCentersListRDD RDD containing List[Array[Double]] of local centers per partition
   * @return List of (upToPartition, totalLocalCenters, avgLocalCenters, mergedCentersCount, compressionRatio, reductionPercentage)
   */
  def incrementalStatistics(
                             clusterCentersListRDD: RDD[List[Array[Double]]]): List[(Int, Int, Int, Int, Double, Double)] = {

    // Collect the entire RDD to driver
    val collectedData: Array[List[Array[Double]]] = clusterCentersListRDD.collect()
    val totalPartitions = collectedData.length

    // Process incrementally from partition 0 to i for each i
    (0 until totalPartitions).map { upToPartition =>
      // Take subsets from partition 0 to upToPartition
      val subset = collectedData.take(upToPartition + 1).toList
      // Calculate statistics
      val totalLocalCenters = subset.flatten.size
      println(s"Total local centers found: ${totalLocalCenters}")
      val avgLocalCenters = if (subset.nonEmpty) math.round(totalLocalCenters.toDouble / subset.size).toInt else 0
      val mergedCentersCount = ensembleMergeCount(subset.flatten, 0.2)

      // Calculate compression and reduction
      val compressionRatio = if (totalLocalCenters > 0) mergedCentersCount.toDouble / totalLocalCenters else 1.0
      val reductionPercentage = (1 - compressionRatio) * 100

      (upToPartition, totalLocalCenters, avgLocalCenters, mergedCentersCount, compressionRatio, reductionPercentage)
    }.toList
  }


  /**
   * Efficient incremental statistics that only merges new partition centers with existing global centers
   * @param clusterCentersListRDD RDD containing List[Array[Double]] of local centers per partition
   * @return List of (partitionIndex, totalLocalCenters, avgLocalCenters, currentGlobalCentersCount, compressionRatio, reductionPercentage, globalCenters)
   */
  def incrementalStatisticsEfficient(
                                      clusterCentersListRDD: RDD[List[Array[Double]]]
                                    ): List[(Int, Int, Int, Int, Double, Double)] = {

    // Collect only the partition data (not the centers themselves)
    val partitionData: Array[(Int, List[Array[Double]])] = clusterCentersListRDD.mapPartitionsWithIndex {
      case (partitionIndex, iterator) =>
        iterator.map(centersList => (partitionIndex, centersList))
    }.collect()

    // Sort by partition index to process in order
    val sortedPartitionData = partitionData.sortBy(_._1)

    var cumulativeLocalCenters = 0
    var globalCenters: List[Array[Double]] = List.empty[Array[Double]]
    val results = collection.mutable.ListBuffer[(Int, Int, Int, Int, Double, Double)]()

    // Process each partition incrementally
    for ((partitionIndex, localCenters) <- sortedPartitionData) {
      cumulativeLocalCenters += localCenters.size

      // Calculate average local centers per partition (including current partition)
      val partitionsProcessed = partitionIndex + 1
      val avgLocalCenters = if (partitionsProcessed > 0) math.round(cumulativeLocalCenters.toDouble / partitionsProcessed).toInt else 0

      // Combine existing global centers with new local centers
      val allCenters = globalCenters ++ localCenters

      // Apply merge function to the combined list
      globalCenters = ensembleMerge(allCenters, 0.2)

      val currentGlobalCount = globalCenters.size
      val compressionRatio = if (cumulativeLocalCenters > 0) currentGlobalCount.toDouble / cumulativeLocalCenters else 1.0
      val reductionPercentage = (1 - compressionRatio) * 100

      results += ((partitionIndex, cumulativeLocalCenters, avgLocalCenters, currentGlobalCount, compressionRatio, reductionPercentage))
    }

    results.toList
  }


  /**
   * Calculate incremental statistics about local centers in batches
   * @param clusterCentersListRDD RDD containing List[Array[Double]] of local centers per partition
   * @param batchSize Number of partitions to process in each batch (default: 1)
   * @return List of (upToPartition, totalLocalCenters, avgLocalCenters)
   */
  def incrementalLocalStatisticsBatched(
                                    clusterCentersListRDD: RDD[List[Array[Double]]],
                                    batchSize: Int = 1
                                  ): List[(Int, Int, Int, Int)] = {

    require(batchSize >= 1, "batchSize must be >= 1")

    // Collect the entire RDD to driver
    val collectedData: Array[List[Array[Double]]] = clusterCentersListRDD.collect()
    val totalPartitions = collectedData.length

    // Calculate number of batches
    val numBatches = math.ceil(totalPartitions.toDouble / batchSize).toInt

    // Process in batches
    (0 until numBatches).map { batchIndex =>
      println(s"Batch ${batchIndex} started:...")
      val startPartition = batchIndex * batchSize
      val endPartition = math.min((batchIndex + 1) * batchSize, totalPartitions) - 1
      val endPartitionLocalCenters = collectedData(endPartition).length
      println(s" local centers in end partition of this batch: ${endPartitionLocalCenters}")
      val partitionsInBatch = endPartition + 1
      println(s"#partitionsInBatch: ${partitionsInBatch}")
      // Take subsets from partition 0 to endPartition (cumulative)
      val subset = collectedData.take(endPartition + 1).toList

      // Calculate statistics for cumulative data up to this batch
      val totalLocalCenters = subset.flatten.size
      println(s"Total local centers found: ${totalLocalCenters}")
      val avgLocalCenters = if (subset.nonEmpty) math.round(totalLocalCenters.toDouble / subset.size).toInt else 0

      println(s"Batch ${batchIndex} completed.")
      (endPartition, endPartitionLocalCenters, totalLocalCenters, avgLocalCenters)
    }.toList
  }

  /**
   * Calculate incremental statistics about local and global centers in batches
   * @param clusterCentersListRDD RDD containing List[Array[Double]] of local centers per partition
   * @param batchSize Number of partitions to process in each batch (default: 1)
   * @return List of (upToPartition, totalLocalCenters, avgLocalCenters, mergedCentersCount, compressionRatio, reductionPercentage)
   */
  def incrementalStatisticsBatched(
                                    clusterCentersListRDD: RDD[List[Array[Double]]],
                                    batchSize: Int = 1,
                                    mergePercentage: Double = 0.1
                                  ): List[(Int, Int, Int, Int, Double, Double)] = {

    require(batchSize >= 1, "batchSize must be >= 1")

    // Collect the entire RDD to driver
    val collectedData: Array[List[Array[Double]]] = clusterCentersListRDD.collect()
    val totalPartitions = collectedData.length

    // Calculate number of batches
    val numBatches = math.ceil(totalPartitions.toDouble / batchSize).toInt

    // Process in batches
    (0 until numBatches).map { batchIndex =>
      println(s"Batch ${batchIndex} started:...")
      val startPartition = batchIndex * batchSize
      val endPartition = math.min((batchIndex + 1) * batchSize, totalPartitions) - 1
      val partitionsInBatch = endPartition + 1
      println(s"#partitionsInBatch: ${partitionsInBatch}")
      // Take subsets from partition 0 to endPartition (cumulative)
      val subset = collectedData.take(endPartition + 1).toList

      // Calculate statistics for cumulative data up to this batch
      val totalLocalCenters = subset.flatten.size
      println(s"Total local centers found: ${totalLocalCenters}")
      val avgLocalCenters = if (subset.nonEmpty) math.round(totalLocalCenters.toDouble / subset.size).toInt else 0
      println(s"#avgLocalCenters found: ${avgLocalCenters}")
      val mergedCentersCount = ensembleMergeCount(subset.flatten, mergePercentage)
      println(s"Total merged centers found: ${mergedCentersCount}")

      // Calculate compression and reduction
      val compressionRatio = if (totalLocalCenters > 0) mergedCentersCount.toDouble / totalLocalCenters else 1.0
      val reductionPercentage = (1 - compressionRatio) * 100
      println(s"Batch ${batchIndex} completed.")
      (endPartition, totalLocalCenters, avgLocalCenters, mergedCentersCount, compressionRatio, reductionPercentage)
    }.toList
  }


  def saveLocalStatsAsCSV(
                      stats: List[(Int, Int, Int, Int)],
                      filePath: String
                    ): Unit = {
    val writer = new PrintWriter(new File(filePath))

    try {
      // Write headers
      writer.println("partition,local_centers,total_local_centers,avg_local_centers")

      // Write data
      stats.foreach { case (partition, locals, total, avg) =>
        writer.println(s"$partition, $locals,$total,$avg")
      }
    } finally {
      writer.close()
    }
  }

  def saveStatsAsCSV(
                      stats: List[(Int, Int, Int, Int, Double, Double)],
                      filePath: String
                    ): Unit = {
    val writer = new PrintWriter(new File(filePath))

    try {
      // Write headers
      writer.println("partition,total_local_centers,avg_local_centers,merged_centers,compression_ratio,reduction_percentage")

      // Write data
      stats.foreach { case (partition, total, avg, merged, compression, reduction) =>
        writer.println(s"$partition,$total,$avg,$merged,${f"$compression%.6f"},${f"$reduction%.6f"}")
      }
    } finally {
      writer.close()
    }
  }


  /**
   * Save incremental statistics as CSV file with headers
   * @param stats List of (partition, totalLocalCenters, avgLocalCenters, mergedCentersCount)
   * @param filePath Path to save the CSV file
   * @param additionalInfo Optional additional information to include in the CSV
   */
  def saveIncrementalStatsAsCSV(
                                 stats: List[(Int, Int, Int, Int)],
                                 filePath: String,
                                 additionalInfo: Map[String, String] = Map.empty
                               ): Unit = {

    val writer = new PrintWriter(new File(filePath))

    try {
      // Write headers
      val headers = List("partition", "total_local_centers", "avg_local_centers",
        "merged_centers", "compression_ratio", "reduction_percentage")

      // Add additional info headers if provided
      val allHeaders = if (additionalInfo.nonEmpty) {
        headers ++ additionalInfo.keys.map("info_" + _)
      } else {
        headers
      }

      writer.println(allHeaders.mkString(","))

      // Write data rows
      stats.foreach { case (partition, total, avg, merged) =>
        val compressionRatio = if (total > 0) merged.toDouble / total else 1.0
        val reductionPercentage = (1 - compressionRatio) * 100

        val baseData = List(
          partition.toString,
          total.toString,
          avg.toString,
          merged.toString,
          f"$compressionRatio%.6f",
          f"$reductionPercentage%.6f"
        )

        // Add additional info values if provided
        val allData = if (additionalInfo.nonEmpty) {
          baseData ++ additionalInfo.values
        } else {
          baseData
        }

        writer.println(allData.mkString(","))
      }

    } finally {
      writer.close()
    }

    println(s"CSV file saved to: $filePath")
  }

}
