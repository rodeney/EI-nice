package com.szubd

import com.szubd.INiceSpark.{getCentersForFirstPartitions, getCentersForPartition, getFirstPartitions, globalConsolidation}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

object INiceSparkMergeExample {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("INiceSparkMergeExample")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    try {
      val localCentersPath = if (args.length > 0) args(0) else "data/Reddit2012/results/run_5_5_100_10000/localCentersPerPartition"
      val outputPath = if (args.length > 1) args(1) else "data/Reddit2012/results/run_5_5_100_10000/batches/"
      val numBlocks:Int = 4//if (args.length > 2) args(2) else 5

      println("\nRead local centers...")
      val clusterCentersDF = spark.read.parquet(localCentersPath)
      println(s"\n#Local Centers Lists：${clusterCentersDF.count()}")
      clusterCentersDF.show(5)
      clusterCentersDF.printSchema

      // Convert back to RDD[List[Array[Double]]]
      val recoveredRDD: RDD[List[Array[Double]]] = clusterCentersDF.rdd.map { row =>
        val seqOfSeq = row.getAs[Seq[Seq[Double]]]("centers_list")
        seqOfSeq.map(_.toArray).toList // Convert Seq[Seq[Double]] to List[Array[Double]]
      }

      println(s"Total partitions found: ${recoveredRDD.getNumPartitions}")
      println(s"Total local centers found: ${recoveredRDD.flatMap(identity).count()}")

      // select the first few partitions
      println("Select the required number of blocks...")
      val batchLocalCenters: RDD[List[Array[Double]]] = getFirstPartitions(recoveredRDD, numBlocks)
      println(s"Total partitions found in the new RDD: ${batchLocalCenters.getNumPartitions}")
      println(s"Total local centers found in the selected partitions: ${batchLocalCenters.flatMap(identity).count()}")
      val batchMergedCenters: List[Array[Double]] = INiceOps.ensembleCenters(batchLocalCenters.collect().toList.flatten, 0.3)
      println(s"Total global centers found: ${batchMergedCenters.size}")
//      batchMergedCenters.foreach { array =>
//        println(array.mkString(", "))
//      }
      println("Saving final centers merged from the selected partitions...")
      val finalCentersRDD = spark.sparkContext.parallelize(batchMergedCenters)
      finalCentersRDD.toDF("center_array").coalesce(1)
        .write
        .mode("overwrite")
        .parquet(s"${outputPath}mergedCenters_From${numBlocks}Blocks")

      //      println("apply ensemble to local centers from all partitions...")
//      val finalCenters = INiceOps.ensembleCenters(recoveredRDD.collect().toList.flatten, 0.2)
//      println("Saving final centers merged from partitions...")
//      val finalCentersRDD = spark.sparkContext.parallelize(batchMergedCenters)
//      finalCentersRDD.toDF("center_array")
//        .write
//        .mode("overwrite")
//        .parquet(s"${outputPath}mergedCenters_${}")


    }finally {
    spark.stop()
    }
  }

}
