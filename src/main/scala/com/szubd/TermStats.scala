package com.szubd

import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.clustering.{KMeans => MLlibKMeans, KMeansModel => MLlibKMeansModel}
import org.apache.spark.mllib.linalg.{Vectors, Vector => MLlibVector}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, count, lit, posexplode, udf}
import org.apache.spark.rdd.RDD
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.linalg.SparseVector
import org.apache.spark.ml.stat.Summarizer
import org.apache.spark.ml.linalg.{VectorUDT, Vector => MLVector, Vectors => MLVectors}


object Terms_Stats {
  def main(args: Array[String]): Unit = {

    // Create Spark session
    val spark = SparkSession.builder()
      .appName("TermStats")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._ // For $-notation

    val inputPath = "data/Reddit20052006/data_normalized_svd"
    val mergedCentersPath = "data/Reddit20052006/results5_5_100_1000/mergedCentersFromAllBlocks/"
    val preditionsPath = "data/Reddit20052006/results/mllib_kmeans_results5_5_100_1000/predictions"
    val outputPath = "data/Reddit20052006/results/term_stats/"

    val dictionaryPath = "data/Reddit20052006/all_terms"
    val allTermsDictionaryPath = "data/Reddit20052006/Cleaned_column_names_with_coding.parquet"

    val svd_featuresCol = "svd_100d_features" // this is used for training and prediction
    val normalized_featuresCol = "normalized_features" // this is used for cluster stats


    //read predictions parquet
    val predsDF:DataFrame = spark.read.parquet(preditionsPath)
    predsDF.printSchema()
    predsDF.show(5)


    //get means of terms
    //val clusterStats = predsDF.select(Summarizer.mean($"normalized_features").alias("normalized_features_means"))

    val clusterStats = predsDF.select(Summarizer.mean($"normalized_features").alias("tfidf_mean"))
    clusterStats.printSchema()
    clusterStats.show(5)

//    val finalDF = predsDF.select($"tfidf_mean").as[MLVector]
//      .flatMap { meanVector =>
//        meanVector.toArray.zipWithIndex.map {
//          case (meanValue, index) => (index, meanValue)
//        }
//      }
//      .toDF("feature_index", "mean_value")
//      .orderBy(desc("mean_value")) // Sort descending by mean value
//
//    finalDF.show(20, truncate = false)


    // Step 1: Read the dictionary from Parquet file
    val dictionaryDF = spark.read.parquet(dictionaryPath)
    dictionaryDF.show(5)

    // Assuming the parquet file has columns: term_index (Int) and term_code (String)
    // Convert to Map for efficient lookup
    val termMap = dictionaryDF.rdd
      .map(row => (row.getAs[Int]("term_index"), row.getAs[String]("term_code")))
      .collect()
      .toMap

    // Broadcast the term map to all executors
    val broadcastTermMap = spark.sparkContext.broadcast(termMap)

    // Step 3: Extract top N terms with their mean values for each cluster
    val topN = 100 // Number of top terms to extract for word cloud

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
      .select(posexplode($"top_terms").as(Seq("rank", "term_data")))
      .withColumn("term_code", $"term_data".getItem("_1"))
      .withColumn("weight", $"term_data".getItem("_2"))
      .drop("term_data")
      .withColumn("rank", $"rank" + 1) // Convert 0-based to 1-based ranking
      .orderBy("rank")

    // Show the results
    println("Top terms for each cluster (ready for word cloud):")
    explodedDF.show(5)


    //===============another method with them mein dictionary ==========================

    // Step 1: Read both dictionary files
    val indexToCodeDictDF = spark.read.parquet(dictionaryPath)
    val codeToWordDictDF = spark.read.parquet(allTermsDictionaryPath).withColumnRenamed("code", "term_code")

    // Show the dictionary structures
    println("Index to Code Dictionary:")
    indexToCodeDictDF.show(5)

    println("Code to Word Dictionary:")
    codeToWordDictDF.show(5)

    // Step 2: Merge the two dictionaries
    val mergedDictionaryDF = indexToCodeDictDF
      .join(codeToWordDictDF, "term_code") // Join on term_code column
      .select("term_index", "term_code", "original_word")
      .orderBy("term_index")

    println("Merged Dictionary:")
    mergedDictionaryDF.show(5)

    // Step 3: Create a combined mapping (term_index -> original_word)
    val wordMap = mergedDictionaryDF.rdd
      .map(row => (row.getAs[Int]("term_index"), row.getAs[String]("original_word")))
      .collect()
      .toMap

    // Broadcast the combined dictionary
    val broadcastwordMap = spark.sparkContext.broadcast(wordMap)

    // Step 4: Compute mean vector for each cluster (label)
    //val meanDF = df.groupBy("cluster_label")
    // .agg(Summarizer.mean($"normalized_features").alias("tfidf_mean"))

    // Step 5: UDF to extract top N terms with original words
    val getTopWordsWithMeansUDF = udf { (meanVector: org.apache.spark.ml.linalg.Vector, n: Int) =>
      val termDictionary = broadcastwordMap.value

      meanVector match {
        case sparse: org.apache.spark.ml.linalg.SparseVector =>
          // For SparseVector, use the actual indices and values
          sparse.values.zip(sparse.indices)
            .filter { case (value, _) => value > 0 } // Only keep positive values
            .sortBy { case (value, _) => -value }    // Sort descending by mean value
            .take(n)
            .map { case (value, idx) =>
              val originalWord = termDictionary.getOrElse(idx, s"unknown_$idx")
              (originalWord, value) // (original_word, mean_value)
            }

        case dense: org.apache.spark.ml.linalg.DenseVector =>
          // Fallback for dense vectors
          dense.toArray.zipWithIndex
            .filter { case (value, _) => value > 0 }
            .sortBy { case (value, _) => -value }
            .take(n)
            .map { case (value, idx) =>
              val originalWord = termDictionary.getOrElse(idx, s"unknown_$idx")
              (originalWord, value)
            }
      }
    }

    val clusterTopWordsDF = clusterStats
      .withColumn("top_terms", getTopWordsWithMeansUDF($"tfidf_mean", lit(topN)))

    // Step 6: Explode the array to get one row per term per cluster
    val explodedWordsDF = clusterTopWordsDF
      .select(posexplode($"top_terms").as(Seq("rank", "term_data")))
      .withColumn("original_word", $"term_data".getItem("_1"))
      .withColumn("weight", $"term_data".getItem("_2"))
      .drop("term_data")
      .withColumn("rank", $"rank" + 1)
      .orderBy("rank")

    explodedWordsDF.show(false)

    // Step 8: Export for word cloud visualization
//    explodedWordsDF
//      .coalesce(1)
//      .write
//      .option("header", "true")
//      .csv(s"${outputPath}corpus_top_terms_with_original_words")

    //docs stats

    // Add document length column
//    val dfWithLength = predsDF.withColumn("doc_length", size($"normalized_features"))
//
//    dfWithLength.printSchema()
//    dfWithLength.show(5)

    // Define a UDF to calculate the number of non-zero elements (document length)
    val countNonZeroUDF = udf { (v: SparseVector) =>
      v.numNonzeros // This counts non-zero elements (actual terms in document)
    }

    // Alternative: UDF to get total vector size (including zeros)
    val vectorSizeUDF = udf { (v: SparseVector) =>
      v.size // This gives the total vocabulary size, not document length
    }

    // Add document length as a new column
    val dfWithLength = predsDF.withColumn("doc_length", countNonZeroUDF($"normalized_features"))

    // Show the result with document lengths
    dfWithLength.select($"doc_length").show(10, truncate = false)

    // Calculate summary statistics
    dfWithLength.select(
      count($"doc_length").as("total_docs"),
      mean($"doc_length").as("avg_length"),
      stddev($"doc_length").as("stddev_length"),
      min($"doc_length").as("min_length"),
      max($"doc_length").as("max_length")
    ).show()

    //  Export for  visualization
    dfWithLength.select($"doc_length")
          .coalesce(1)
          .write
          .option("header", "true")
          .csv(s"${outputPath}docs_length")
  }
}
