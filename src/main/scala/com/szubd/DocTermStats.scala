package com.szubd

import com.szubd.TermFiltering_VectorColumn.{calculateDocFrequencies, calculateTermVariances, filterTerms, findPercentiles, findPercentilesWithDetails, getTermCounts}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, row_number}
import org.apache.spark.ml.linalg.{DenseVector => MLDenseVector, SparseVector => MLSparseVector, Vector => MLVector, Vectors => MLVectors}

/*
* Calculate term statistics for exploring the data before applying the filtering step.
* DocFreq
* TermVariance
 */
object DocTermStats {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("DocTermStats")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Use local paths and default values if no arguments provided
    val inputDataPath = if (args.length > 0) args(0) else "data/R2012-01/"
    val outputPath = if (args.length > 1) args(1) else "data/RedditTest/201201/"
//    val dictPath = if (args.length > 1) args(1) else "data/RedditDictionary/"
//    val validDictPath = if (args.length > 2) args(2) else "data/RedditTest/201201/valid_terms/"


    val featuresCol = "features"//if (args.length > 2) args(2) else "features"

    try {
      // Load your data
      val df = spark.read.parquet(inputDataPath)
      df.printSchema()
      df.show(5, false)
      val totalDocs = df.count()
      println(s"Total records: ${totalDocs}")

//#all terms
// First, get the vector size from the first non-null row
  val vectorSize = df.select(col(featuresCol))
    .filter(col(featuresCol).isNotNull)
    .head(1) // Get first row
    .headOption
    .map { row =>
      row.getAs[Any](0) match {
        case sv: MLSparseVector => sv.size
        case dv: MLDenseVector => dv.values.length
        case _ => 0
      }
    }
    .getOrElse(0)

      println(s"#All Terms: '$vectorSize'")

      //docFreq with zero freq too
      val docFreqsWithZeros: RDD[(Int, Long)] = calculateDocFrequenciesWithZeros(spark, df, featuresCol, vectorSize)
      val docFreqsWithZerosDF = docFreqsWithZeros.sortByKey().toDF("term_index", "doc_freq")
      val numTerms = docFreqsWithZeros.count()

      println(s"#Terms (including zero-freqs): '$numTerms'")
      docFreqsWithZerosDF.printSchema()
      docFreqsWithZerosDF.show(100)

      //Save filtered dataframe using the original num of partitions
      println("[INFO] Saving docFreqsWithZerosDF  ...")
      docFreqsWithZerosDF.sort("term_index").coalesce(1).write.mode("overwrite").parquet(s"${outputPath}term_docfreq/")
      println("[INFO] Saving docFreqsWithZerosDF completed ...")

     // ====== Term Variance ==========
      val termStatsDF  = computeTermStatistics(df, featuresCol)
      val termCount = termStatsDF.count()
      println(s"#Terms (with zeros too): '$termCount'")
      termStatsDF.printSchema()
      termStatsDF.sort("term_index").show(100)


      //Save term variance
      println("[INFO] Saving termStatsDF  ...")
      termStatsDF.sort("term_index").coalesce(1).write.mode("overwrite").parquet(s"${outputPath}term_stats/")
      println("[INFO] Saving termStatsDF  completed ...")

    }finally {
      spark.stop()
    }
  }
/*
* Similar to calculateDocFrequencies, but keeps terms with zero freqs
 */
  def calculateDocFrequenciesWithZeros(
                                        spark: SparkSession,
                                        df: DataFrame,
                                        featuresCol: String = "features",
                                        vocabSize: Int // Need to know the total number of terms
                                      ): RDD[(Int, Long)] = {

    val existingFreqs = calculateDocFrequencies(df, featuresCol)

    // Create RDD with all term indices (0 to vocabSize-1) with frequency 0
    val allIndices = spark.sparkContext.parallelize(0 until vocabSize).map(_ -> 0L)

    // Union and reduce to get frequencies (existing terms keep their counts, others get 0)
    allIndices.union(existingFreqs).reduceByKey(_ + _)
  }

  /**
   * Computes comprehensive term statistics for ALL terms in the vocabulary, including zero-count terms.
   * Returns a DataFrame with complete coverage of all term indices.
   *
   * @param df Input DataFrame containing feature vectors
   * @param featuresCol Name of the column storing feature vectors (default: "features")
   * @param vocabSize Optional vocabulary size to ensure all terms are included
   * @return DataFrame with statistics for all terms, including those with zero occurrences:
   *         - term_index: Int (the term identifier, 0 to vocabSize-1)
   *         - mean: Double (0.0 for zero-count terms)
   *         - variance: Double (0.0 for zero-count terms)
   *         - std_dev: Double (0.0 for zero-count terms)
   *         - sum_values: Double (0.0 for zero-count terms)
   *         - sum_squares: Double (0.0 for zero-count terms)
   *         - count: Long (0 for zero-count terms)
   *
   * @note This function guarantees that the output DataFrame contains all term indices
   *       from 0 to vocabSize-1, ensuring complete vocabulary coverage.
   */
  def computeTermStatistics(
                                     df: DataFrame,
                                     featuresCol: String = "features",
                                     vocabSize: Option[Int] = None
                                   ): DataFrame = {
    import df.sparkSession.implicits._

    // Step 1: Calculate term variance statistics for non-zero terms
    val nonZeroStatsRDD: RDD[(Int, (Double, Double, Double, Double, Double, Long))] = calculateTermVariances(df, featuresCol)
      .map { case (termIndex, (sum, sumSq, count)) =>
        val mean = if (count > 0) sum / count else 0.0
        val variance = if (count > 1) {
          val numerator = sumSq - (sum * sum) / count
          math.max(0.0, numerator) / (count - 1)
        } else {
          0.0
        }
        val stdDev = if (variance >= 0) math.sqrt(variance) else 0.0

        (termIndex, (mean, variance, stdDev, sum, sumSq, count))
      }

    // Step 2: Determine vocabulary size if not provided
    val actualVocabSize = vocabSize.getOrElse {
      // Find maximum term index from data or vector sizes
      val maxIndexFromData = if (nonZeroStatsRDD.isEmpty()) {
        0
      } else {
        nonZeroStatsRDD.keys.max() + 1  // +1 because indices are 0-based
      }

      // Also check vector sizes from the data
      val maxVectorSize = df.select(col(featuresCol))
        .rdd
        .map { row =>
          row.getAs[MLVector](0) match {
            case sv: MLSparseVector => sv.size
            case dv: MLDenseVector => dv.values.length
          }
        }
        .max()

      math.max(maxIndexFromData, maxVectorSize)
    }

    // Step 3: Create zero entries for all terms not present in nonZeroStatsRDD
    val zeroStatsRDD = df.sparkSession.sparkContext
      .parallelize(0 until actualVocabSize)
      .map(termIndex => (termIndex, (0.0, 0.0, 0.0, 0.0, 0.0, 0L)))

    // Step 4: Union and reduce to get complete statistics
    val completeStatsRDD = nonZeroStatsRDD
      .union(zeroStatsRDD)
      .reduceByKey { case ((m1, v1, sd1, s1, sq1, c1), (m2, v2, sd2, s2, sq2, c2)) =>
        // For terms that appear in both, non-zero stats should win
        if (c1 > 0) (m1, v1, sd1, s1, sq1, c1)
        else if (c2 > 0) (m2, v2, sd2, s2, sq2, c2)
        else (0.0, 0.0, 0.0, 0.0, 0.0, 0)
      }

    // Step 5: Convert to DataFrame
    completeStatsRDD
      .map { case (termIndex, (mean, variance, stdDev, sum, sumSq, count)) =>
        TermStats(
          term_index = termIndex,
          mean = mean,
          variance = variance,
          std_dev = stdDev,
          sum_values = sum,
          sum_squares = sumSq,
          count = count
        )
      }
      .toDF()
  }

  /**
   * Case class representing comprehensive term statistics including document frequency.
   */
  case class TermStats(
                             term_index: Int,
                             mean: Double,
                             variance: Double,
                             std_dev: Double,
                             sum_values: Double,
                             sum_squares: Double,
                             count: Long
                           )


}
