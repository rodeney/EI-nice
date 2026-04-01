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
/*
* In case we have the merged centers and we want to predict on new data
* Apply KMeans using the estimated number of clusters and merged cluster centers from inice
* We should use svd_features (the low-dimension data)
* Gel lables (int assigned to each document, representing its cluster)
* Join labels with the pre-svd version of the data (normalized_features)
* Calculate average TF-IDF for each term per cluster
* Get Top terms in each cluster , each term with its average tf-idf value
* Use the dictionary to get the real words (for visulization in the word cloud)
 */
* We have merged centers from RSP blocks, stored in csv file.
* Input data: submission_id, normalized_features, svd_features
* Apply Spark MLlib KMeans with initial model using Inice centers
* Predict
* Collect cluster stats inlcuding terms means and top terms
* Output:
 */

// Example: we ran kmeans with merged clusters from 10 blocks from year 2005-2006
//val inputPath = "data/dtm20052006_filtered_newVectorSize_svd100/" //args(0)
//val initialClustersPath = "data/svd100_results/10blocks_mergedCenters.csv/" //args(1)
//val outputPath = "data/svd100_results/mllib_kmeans_results/"//args(2)
// Total samples: 82932
//   Number of clusters: 172
//   WSSE: 49.9252022763193
//Largest cluster: 14 (36884 samples)
//Smallest cluster: 16 (1 samples)
//Average cluster size: 482.16 samples
//  Cluster size ratio (max/min): 36884.00

object KMeansWithINiceClusters_201201 {
  def main(args: Array[String]): Unit = {

    // Use local paths and default values if no arguments provided

    val inputPath = if (args.length > 0) args(0) else "data/Reddit201201/data_normalized_svd100_scaled"
    val outputPath = if (args.length > 1) args(1) else "data/Reddit201201/results/run_10_5_100_1000_scaled/mllib_kmeans_preds/"
    val mergedCentersPath = if (args.length > 2) args(2) else "data/Reddit201201/results/run_10_5_100_1000_scaled/batches/mergedCentersFromAllBlocks/"


    val dictionaryPath = if (args.length > 3) args(3) else "data/Reddit201201/valid_terms"
    val allTermsDictionaryPath = if (args.length > 4) args(4) else "data/Reddit20052006/Cleaned_column_names_with_coding.parquet"
    val initialClustersPath = mergedCentersPath

    val svd_featuresCol = "svd100_scaled_features" // this is used for training and prediction
    val normalized_featuresCol = "normalized_features" // this is used for cluster stats

    // Create Spark session
    val spark = SparkSession.builder()
      .appName("KMeansWithINiceClusters")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    try {
      println("=" * 60)
      println("K-MEANS CLUSTERING WITH CSV INITIAL CENTERS")
      println("=" * 60)

      // 1. Read initial clusters
      println(s"\n1. Reading initial clusters from: $initialClustersPath")
      val initialClustersDF = spark.read.parquet(initialClustersPath)

      //      val initialClustersDF = spark.read
      //        .option("header", "false")
      //        .option("inferSchema", "true")
      //        .csv(initialClustersPath)

      println(s"Initial clusters schema:")
      initialClustersDF.printSchema()
      println(s"Initial clusters data:")
      initialClustersDF.show(10, truncate = false)

      val initialCenters = initialClustersDF.rdd.map { row =>
        Vectors.dense(row.getAs[Seq[Double]]("center_array").toArray)
      }.collect()

      //      val initialCenters = initialClustersDF.rdd.map { row =>
      //        val values = row.toSeq.map {
      //          case d: Double => d
      //          case i: Int => i.toDouble
      //          case f: Float => f.toDouble
      //          case other => throw new IllegalArgumentException(s"Unsupported type: $other")
      //        }.toArray
      //        Vectors.dense(values)
      //      }.collect()

      //      val initialCenters = initialClustersDF.rdd.map { row =>
      //        val values = row.toSeq.map {
      //          // Handle regular Array types
      //          case array: Array[_] =>
      //            array.map {
      //              case d: Double => d
      //              case i: Int => i.toDouble
      //              case f: Float => f.toDouble
      //              case l: Long => l.toDouble
      //              case other => throw new IllegalArgumentException(s"Unsupported array element type: $other")
      //            }
      //
      //          // Handle WrappedArray (common in Spark DataFrames)
      //          case wrapped: mutable.WrappedArray[_] =>
      //            wrapped.map {
      //              case d: Double => d
      //              case i: Int => i.toDouble
      //              case f: Float => f.toDouble
      //              case l: Long => l.toDouble
      //              case other => throw new IllegalArgumentException(s"Unsupported WrappedArray element type: $other")
      //            }.toArray
      //
      //          // Handle individual numeric values
      //          case d: Double => Array(d)
      //          case i: Int => Array(i.toDouble)
      //          case f: Float => Array(f.toDouble)
      //          case l: Long => Array(l.toDouble)
      //
      //          // Handle unexpected types
      //          case other => throw new IllegalArgumentException(s"Unsupported row element type: ${other.getClass.getName}")
      //        }
      //
      //        // Flatten the array of arrays and create dense vector
      //        val flattenedValues = values.flatten
      //        Vectors.dense(flattenedValues)
      //      }.collect()

      println(s"✓ Successfully loaded ${initialCenters.length} initial cluster centers")
      println("Initial cluster centers:")
      initialCenters.zipWithIndex.foreach { case (center, idx) =>
        println(s"  Cluster $idx: $center")
      }

      // 2. Read training data and handle array features
      println(s"\n2. Reading training data from: $inputPath")
      val df = spark.read.parquet(inputPath)

      println(s"Training data schema:")
      df.printSchema()
      println(s"Sample training data (first 5 rows):")
      df.show(5, truncate = false)

      val totalSamples = df.count()
      println(s"✓ Total training samples: $totalSamples")

      // Debug: inspect features column type
      println(s"\n3. Inspecting features column type...")
      val featuresSample = df.select(svd_featuresCol).limit(3).collect()
      featuresSample.foreach { row =>
        val value = row.get(0)
        println(s"  Features type: ${value.getClass.getName}, value: $value")
      }

      val featuresRDD = df.select(svd_featuresCol)
        .rdd
        .map { row =>
          val features = row.get(0)
          features match {
            case array: Seq[_] =>
              val doubleArray = array.map {
                case d: Double => d
                case i: Int => i.toDouble
                case f: Float => f.toDouble
                case other => throw new IllegalArgumentException(s"Unsupported array element: $other")
              }.toArray
              Vectors.dense(doubleArray)
            case vector: org.apache.spark.ml.linalg.Vector =>
              Vectors.fromML(vector)
            case other =>
              throw new IllegalArgumentException(s"Unsupported features type: ${other.getClass}")
          }
        }
        .cache()

      println(s"✓ Successfully converted features to MLlib vectors")

      // 3. Train model
      println(s"\n4. Training K-means model...")
      println(s"   Number of clusters (K): ${initialCenters.length}")
      println(s"   Maximum iterations: 100")

      // Create initial model from centers
      val initialModel = new MLlibKMeansModel(initialCenters)

      val model = new MLlibKMeans()
        .setK(initialCenters.length)
        .setMaxIterations(100)
        .setSeed(42L)
        .setInitialModel(initialModel)
        .run(featuresRDD)

      println(s"✓ Model training completed successfully!")

      // 4. Calculate clustering statistics
      println(s"\n5. Calculating clustering statistics...")

      val finalCost = model.computeCost(featuresRDD)
      println(s"   Within Set Sum of Squared Errors (WSSE): $finalCost")

      val featureDimension = if (featuresRDD.count() > 0) featuresRDD.first().size else 0
      println(s"   Feature dimension: $featureDimension")

      println(s"\n   Final cluster centers:")
      model.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
        println(s"   Cluster $idx: $center")
      }

      // 5. Create prediction UDF
      println(s"\n6. Creating prediction function...")
      val broadcastModel = spark.sparkContext.broadcast(model)
      val predictUDF = udf { (features: Any) =>
        val vector = features match {
          case array: Seq[_] =>
            Vectors.dense(array.map(_.toString.toDouble).toArray)
          case vector: org.apache.spark.ml.linalg.Vector =>
            Vectors.fromML(vector)
          case other =>
            throw new IllegalArgumentException(s"Unsupported type: ${other.getClass}")
        }
        broadcastModel.value.predict(vector)
      }

      // 6. Add predictions and calculate cluster distribution
      println(s"7. Making predictions and analyzing results...")
      val resultDF = df.withColumn("cluster_label", predictUDF($"svd100_scaled_features"))

      // Calculate cluster distribution
      val clusterDistribution = resultDF
        .groupBy("cluster_label")
        .count()
        .orderBy("cluster_label")
        .collect()


      println(s"\n8. CLUSTERING RESULTS SUMMARY")
      println(s"   " + "=" * 40)
      println(s"   Total samples: $totalSamples")
      println(s"   Number of clusters: ${initialCenters.length}")
      println(s"   WSSE: $finalCost")
      println(s"   " + "-" * 40)
      println(s"   CLUSTER DISTRIBUTION:")

      clusterDistribution.foreach { row =>
        val clusterId = row.getInt(0)
        val count = row.getLong(1)
        val percentage = (count.toDouble / totalSamples) * 100
        println(f"   Cluster $clusterId: $count samples ($percentage%.2f%%)")
      }

      // Calculate some basic statistics
      val maxCluster = clusterDistribution.maxBy(_.getLong(1))
      val minCluster = clusterDistribution.minBy(_.getLong(1))
      val avgClusterSize = totalSamples.toDouble / initialCenters.length

      println(s"   " + "-" * 40)
      println(s"   Largest cluster: ${maxCluster.getInt(0)} (${maxCluster.getLong(1)} samples)")
      println(s"   Smallest cluster: ${minCluster.getInt(0)} (${minCluster.getLong(1)} samples)")
      println(f"   Average cluster size: $avgClusterSize%.2f samples")

      val imbalanceRatio = maxCluster.getLong(1).toDouble / minCluster.getLong(1)
      println(f"   Cluster size ratio (max/min): $imbalanceRatio%.2f")

      if (imbalanceRatio > 5.0) {
        println(s"   ⚠️  Warning: Significant cluster imbalance detected!")
      }

      // 7. Save results
      println(s"\n9. Saving results to: $outputPath")
      resultDF.write.mode("overwrite").parquet(s"${outputPath}predictions")


      // Show final sample with predictions
      println(s"\n10. Sample results with predictions:")
      resultDF.select(normalized_featuresCol, "cluster_label").show(10, truncate = false)

      println(s"\n✅ K-means clustering completed successfully!")
      println(s"✅ Results saved to: $outputPath")
      println(s"✅ Total processing time: ${System.currentTimeMillis()} ms")

      // Terms stats
      resultDF.printSchema()
      // Group by label and calculate mean features for each group
      val clusterStats = resultDF.groupBy("cluster_label")
        .agg(count("*").alias("cluster_size"), Summarizer.mean($"normalized_features").alias("tfidf_mean"))
        .orderBy("cluster_label")

      //val termStats = resultDF.select(Summarizer.mean($"normalized_features").alias("tfidf_mean"))

      clusterStats.printSchema()

      // Show cluster stats
//      println(s"\n11. Sample cluster stats:")
//      clusterStats.show(10)

      println(s"\n9. Saving cluster stats to CSV (only cluster label and cluster size): ")
      clusterStats.select("cluster_label", "cluster_size")
        .coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(s"${outputPath}cluster_size.csv")


      println(s"\n10. Saving cluster stats to Parquet: label, size, tfidf means ")
      clusterStats.select("cluster_label", "cluster_size", "tfidf_mean")
        .coalesce(1)
        .write
        .mode("overwrite")
        .parquet(s"${outputPath}cluster_stats")


      //=============================================================================================
      //Extract Top Terms from Mean Vectors

      // Step 1: Read the dictionary from Parquet file
      val dictionaryDF = spark.read.parquet(dictionaryPath)
      //dictionaryDF.show(5)

      // Assuming the parquet file has columns: term_index (Int) and term_code (String)
      // Convert to Map for efficient lookup
      val termMap = dictionaryDF.rdd
        .map(row => (row.getAs[Int]("term_index"), row.getAs[String]("original_word")))
        .collect()
        .toMap

      // Broadcast the term map to all executors
      val broadcastTermMap = spark.sparkContext.broadcast(termMap)

      // Step 3: Extract top N terms with their mean values for each cluster
      val topN = 20 // Number of top terms to extract for word cloud

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

      println("clusterTopTermsDF:")
      clusterTopTermsDF.show(5)

      // Step 4: Explode the array to get one row per term per cluster
      val explodedDF = clusterTopTermsDF
        .select($"cluster_label", posexplode($"top_terms").as(Seq("rank", "term_data")))
        .withColumn("original_word", $"term_data".getItem("_1"))
        .withColumn("weight", $"term_data".getItem("_2"))
        .drop("term_data")
        .withColumn("rank", $"rank" + 1) // Convert 0-based to 1-based ranking
        .orderBy("cluster_label", "rank")

      // Show the results
      println("Top terms for each cluster (ready for word cloud):")
       explodedDF.show(5)

      //save
      explodedDF
        .coalesce(1)
        .write
        .option("header", "true")
        .csv(s"${outputPath}cluster_top_terms.csv")


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
//
//      // Step 7: Join back with the merged dictionary to get term_code as well (optional)
//      //      val finalDF = explodedWordsDF
//      //        .join(mergedDictionaryDF, explodedWordsDF("original_word") === mergedDictionaryDF("original_word"), "left")
//      //        .select(
//      //          $"cluster_label",
//      //          $"term_index",
//      //          $"term_code",
//      //          $"original_word",
//      //          $"weight",
//      //          $"rank"
//      //        )
//      //        .orderBy("cluster_label", "rank")
//
//      println("Final Results with Original Words:")
//      explodedWordsDF.show(false)
//
//      // Step 8: Export for word cloud visualization
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