package com.szubd

import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{count, lit, posexplode, spark_partition_id, udf}

object ClusterStatsInPartitions {

  def main(args: Array[String]): Unit = {

    // Use local paths and default values if no arguments provided
    val preditionPath = if (args.length > 0) args(0) else "data/Reddit201201/results/run_10_5_100_1000_scaled/mllib_kmeans_preds/predictions/"
    val outputPath = if (args.length > 1) args(1) else "data/Reddit201201/results/run_10_5_100_1000_scaled/mllib_kmeans_preds/"
    val dictionaryPath = if (args.length > 2) args(2) else "data/Reddit201201/valid_terms/"
    //val allTermsDictionaryPath = if (args.length > 3) args(3) else "data/Reddit20052006/Cleaned_column_names_with_coding.parquet"

    val svd_featuresCol = "svd100_scaled_features" // this is used for training and prediction
    val normalized_featuresCol = "normalized_features" // this is used for cluster stats

    // Create Spark session
    val spark = SparkSession.builder()
      .appName("ClusterStatsInPartitions")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    try {

      //read prediction data
      val resultDF = spark.read.parquet(preditionPath)

      resultDF.printSchema()
      // Show cluster stats
      println(s"\n#Partitions:${resultDF.rdd.getNumPartitions}")
      resultDF.show(10)



      // Get stats per partition
      // Add partition ID to the DataFrame
      val resultDFWithPartition = resultDF
          .withColumn("partition_id", spark_partition_id())

      // Calculate cluster stats per partition
      val partitionClusterStats = resultDFWithPartition
        .groupBy("partition_id", "cluster_label")
        .agg(
          count("*").alias("cluster_size"),
          Summarizer.mean($"normalized_features").alias("tfidf_mean")
        )
        .orderBy("partition_id", "cluster_label")

      partitionClusterStats.printSchema()
      println(s"\n#Rows in Output:${partitionClusterStats.count()}")
      partitionClusterStats.select("partition_id", "cluster_label","cluster_size").show(10)
      //partitionClusterStats.show(1)

      println(s"\nSaving cluster stats of partitions to CSV (without tf-idf means): ")
      partitionClusterStats.select("partition_id", "cluster_label", "cluster_size")
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(s"${outputPath}partition_cluster_size.csv")
      println(s"\nDone. ")

      println(s"\nSaving cluster stats of partitions to parquet (with tfidf means):")
      partitionClusterStats.coalesce(1)
        .write
        .mode("overwrite")
        .parquet(s"${outputPath}partition_cluster_stats")
      println(s"\nDone. ")



            //Extract Top Terms from Mean Vectors

      // Step 1: Read the dictionary from Parquet file
      val dictionaryDF = spark.read.parquet(dictionaryPath).withColumnRenamed("code", "term_code")
      dictionaryDF.show(5)
      val numTerms = dictionaryDF.count
      println(s"#Terms in the valid terms dictionary: ${numTerms}")

      // Assuming the parquet file has columns: term_index (Int) and term_code (String)
      // Convert to Map for efficient lookup
      val termMap = dictionaryDF.rdd
        .map(row => (row.getAs[Int]("term_index"), row.getAs[String]("term_code")))
        .collect()
        .toMap

      // Broadcast the term map to all executors
      val broadcastTermMap = spark.sparkContext.broadcast(termMap)

      // Step 3: Extract top N terms with their mean values for each cluster
      val topN = 5 // Number of top terms to extract for word cloud

      // UDF to extract top N terms from mean vector, respecting sparse structure
      val getTopTermsWithMeansUDF = udf { (meanVector: org.apache.spark.ml.linalg.Vector, n: Int) =>
        val termDictionary = broadcastTermMap.value

        meanVector match {
          case sparse: SparseVector =>
            // For sparse vectors, only process non-zero entries
            sparse.values.zip(sparse.indices)
              .sortBy { case (value, _) => -value } // Sort descending by value
              .take(n)
              .map { case (value, idx) =>
                val termCode = termDictionary.getOrElse(idx, s"unknown_$idx")
                (termCode, value)
              }

          case dense: org.apache.spark.ml.linalg.DenseVector =>
            // Handle dense vectors (convert to sparse-like processing)
            dense.toArray.zipWithIndex
              .filter { case (value, _) => value > 0 } // Filter zeros for efficiency
              .sortBy { case (value, _) => -value }
              .take(n)
              .map { case (value, idx) =>
                val termCode = termDictionary.getOrElse(idx, s"unknown_$idx")
                (termCode, value)
              }
        }
      }

      val clusterTopTermsDF = partitionClusterStats
        .withColumn("top_terms", getTopTermsWithMeansUDF($"tfidf_mean", lit(topN)))

      clusterTopTermsDF.show(5)

      // Step 4: Explode the array to get one row per term per cluster
      val explodedDF = clusterTopTermsDF
        .select($"cluster_label", posexplode($"top_terms").as(Seq("rank", "term_data")))
        .withColumn("term_code", $"term_data".getItem("_1"))
        .withColumn("weight", $"term_data".getItem("_2"))
        .drop("term_data")
        .withColumn("rank", $"rank" + 1) // Convert 0-based to 1-based ranking
        .orderBy("cluster_label", "rank")

      // Show the results
      println("Top terms for each cluster (ready for word cloud):")
      explodedDF.show(5)

    } catch {
      case e: Exception =>
        println(s"\n❌ Error occurred: ${e.getMessage}")
        e.printStackTrace()
    } finally {
      spark.stop()
      println(s"\nSpark session closed.")
    }
  }
}


