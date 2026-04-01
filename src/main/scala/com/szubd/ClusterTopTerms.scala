package com.szubd

import org.apache.spark.mllib.clustering.{KMeans => MLlibKMeans, KMeansModel => MLlibKMeansModel}
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLlibVector}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, count, lit, posexplode, udf}
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.stat.Summarizer

import scala.collection.mutable

/*
Use cluster stats to get top terms in each cluster
 */

object ClusterTopTerms {
  def main(args: Array[String]): Unit = {

    // Use local paths and default values if no arguments provided

    val preditionPath = if (args.length > 0) args(0) else "data/Reddit201201/results/run_10_5_100_1000_scaled/mllib_kmeans_preds/predictions/"
    val dictionaryPath = if (args.length > 1) args(1) else "data/Reddit201201/valid_terms/"
    val allTermsDictionaryPath = if (args.length > 2) args(2) else "data/Reddit20052006/Cleaned_column_names_with_coding.parquet"

    val svd_featuresCol = "svd100_scaled_features" // this is used for training and prediction
    val normalized_featuresCol = "normalized_features" // this is used for cluster stats

    // Create Spark session
    val spark = SparkSession.builder()
      .appName("KMeansWithINiceClusters")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    try {

      val resultDF = spark.read.parquet(preditionPath)

      resultDF.printSchema()

      // Show cluster stats
            println(s"\n11. Sample cluster stats:")
      resultDF.show(10)


      // Group by label and calculate mean features for each group
      val clusterStats = resultDF.groupBy("cluster_label")
        .agg(count("*").alias("cluster_size"), Summarizer.mean($"normalized_features").alias("tfidf_mean"))
        .orderBy("cluster_label")

      clusterStats.printSchema()
      //clusterStats.show(10)


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

      val clusterTopTermsDF = clusterStats
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

      //save
//      explodedDF
//        .coalesce(1)
//        .write
//        .option("header", "true")
//        .csv(s"${outputPath}cluster_top_terms.csv")


//      //another method
//      // Step 1: Read both dictionary files
//      val indexToCodeDictDF = spark.read.parquet(dictionaryPath)
//      val codeToWordDictDF = spark.read.parquet(allTermsDictionaryPath).withColumnRenamed("code", "term_code")
//
//      // Show the dictionary structures
//      println("Index to Code Dictionary:")
//      indexToCodeDictDF.show(5)
//
//      println("Code to Word Dictionary:")
//      codeToWordDictDF.show(5)
//
//      // Step 2: Merge the two dictionaries
//      val mergedDictionaryDF = indexToCodeDictDF
//        .join(codeToWordDictDF, "term_code") // Join on term_code column
//        .select("term_index", "term_code", "original_word")
//        .orderBy("term_index")
//
//      println("Merged Dictionary:")
//      mergedDictionaryDF.show(5)
//
//      // Step 3: Create a combined mapping (term_index -> original_word)
//      val wordMap = mergedDictionaryDF.rdd
//        .map(row => (row.getAs[Int]("term_index"), row.getAs[String]("original_word")))
//        .collect()
//        .toMap
//
//      // Broadcast the combined dictionary
//      val broadcastwordMap = spark.sparkContext.broadcast(wordMap)
//
//      // Step 4: Compute mean vector for each cluster (label)
//      //val meanDF = df.groupBy("cluster_label")
//      // .agg(Summarizer.mean($"normalized_features").alias("tfidf_mean"))
//
//      // Step 5: UDF to extract top N terms with original words
//      val getTopWordsWithMeansUDF = udf { (meanVector: org.apache.spark.ml.linalg.Vector, n: Int) =>
//        val termDictionary = broadcastwordMap.value
//
//        meanVector match {
//          case sparse: org.apache.spark.ml.linalg.SparseVector =>
//            // For SparseVector, use the actual indices and values
//            sparse.values.zip(sparse.indices)
//              .filter { case (value, _) => value > 0 } // Only keep positive values
//              .sortBy { case (value, _) => -value }    // Sort descending by mean value
//              .take(n)
//              .map { case (value, idx) =>
//                val originalWord = termDictionary.getOrElse(idx, s"unknown_$idx")
//                (originalWord, value) // (original_word, mean_value)
//              }
//
//          case dense: org.apache.spark.ml.linalg.DenseVector =>
//            // Fallback for dense vectors
//            dense.toArray.zipWithIndex
//              .filter { case (value, _) => value > 0 }
//              .sortBy { case (value, _) => -value }
//              .take(n)
//              .map { case (value, idx) =>
//                val originalWord = termDictionary.getOrElse(idx, s"unknown_$idx")
//                (originalWord, value)
//              }
//        }
//      }
//
//      val clusterTopWordsDF = clusterStats
//        .withColumn("top_terms", getTopWordsWithMeansUDF($"tfidf_mean", lit(topN)))
//
//      // Step 6: Explode the array to get one row per term per cluster
//      val explodedWordsDF = clusterTopWordsDF
//        .select($"cluster_label", posexplode($"top_terms").as(Seq("rank", "term_data")))
//        .withColumn("original_word", $"term_data".getItem("_1"))
//        .withColumn("weight", $"term_data".getItem("_2"))
//        .drop("term_data")
//        .withColumn("rank", $"rank" + 1)
//        .orderBy("cluster_label", "rank")

      // Step 7: Join back with the merged dictionary to get term_code as well (optional)
      //      val finalDF = explodedWordsDF
      //        .join(mergedDictionaryDF, explodedWordsDF("original_word") === mergedDictionaryDF("original_word"), "left")
      //        .select(
      //          $"cluster_label",
      //          $"term_index",
      //          $"term_code",
      //          $"original_word",
      //          $"weight",
      //          $"rank"
      //        )
      //        .orderBy("cluster_label", "rank")

//      println("Final Results with Original Words:")
//      explodedWordsDF.show(false)

      // Step 8: Export for word cloud visualization
//      explodedWordsDF
//        .coalesce(1)
//        .write
//        .option("header", "true")
//        .csv(s"${outputPath}cluster_top_terms_with_original_words")



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