package com.szubd

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector => MLDenseVector, SparseVector => MLSparseVector, Vector => MLVector, Vectors => MLVectors}
import org.apache.spark.mllib.linalg.{Vector => MLLibVector, Vectors => MLLibVectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.ml.linalg.VectorUDT
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions._

object TermFiltering_VectorColumn {


  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("TermFiltering")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Use local paths and default values if no arguments provided
    val inputDataPath = if (args.length > 0) args(0) else "data/R2012-01/"
    val dictPath = if (args.length > 1) args(1) else "data/RedditDictionary/"
    val validDictPath = if (args.length > 2) args(2) else "data/RedditTest/201201/valid_terms/"
    val outputDataPath = if (args.length > 3) args(3) else "data/RedditTest/201201/data_filtered"

    val featuresCol = "features"//if (args.length > 2) args(2) else "features"

    try {
      // Load your data
      val df = spark.read.parquet(inputDataPath)
      df.printSchema()
      df.show(5, false)
      val totalDocs = df.count()
      println(s"Total records: ${totalDocs}")

      //find optimal percentiles
//      val docFreqs = calculateDocFrequencies(df, featuresCol)
//      val percentilesDetails = findPercentilesWithDetails(docFreqs, totalDocs)
//      println(s"findOptimalPercentiles: ${percentilesDetails.toString()}")
//
//      val (bestMin, bestMax) = findPercentiles(docFreqs, 0.05, 0.95)
//      println(s"bestMin: $bestMin, bestMax: $bestMax")
//
//      val (kept, removed) = getTermCounts(docFreqs, bestMin, bestMax)
//      println(s"Terms kept: $kept, Terms removed: $removed")

      //then set minDocFreq and maxDocFreq as the vlaues of these percentiles

      // Term Filtering Process::
      println("[INFO] Start of Term Filtering...")
//      val (filteredDF, validTermsIndicesDF) = filterTerms(df,
//        featuresCol = featuresCol,
//        minDocFreq = bestMin,
//        maxDocFreq = bestMax,
//        minVariance = 1e-3
//      )
      val (filteredDF, validTermsIndicesDF) = filterTerms1(df,
        featuresCol = featuresCol,
        minDocFreq = 1000,
        maxDocFreqRatio = 0.5,
        minVariance = 1e-6
      )

      println("[INFO] End of Term Filtering.")
      filteredDF.printSchema()
      filteredDF.show(5, false)
      val validTermsCount = validTermsIndicesDF.count()
      println(s"[INFO] #Terms (after filtering) = $validTermsCount")
      val validDocsCount = filteredDF.count()
      println(s"[INFO] #Docs (after filtering) = $validDocsCount")
      val filteredPartitions = filteredDF.rdd.getNumPartitions
      println(s"[INFO] #Partitions (after filtering) = $filteredPartitions")

      //Load dictionary
      val dictDF = spark.read.parquet(dictPath)
        .distinct()
        .withColumn("term_index", row_number().over(Window.orderBy("original_word")) - 1)
      dictDF.printSchema()
      dictDF.show(5, false)

       //Get valid terms: indices and columns names
      val validTermsDF = dictDF.join(validTermsIndicesDF, "term_index").orderBy("term_index")
      validTermsDF.printSchema()
      validTermsDF.show(5, false)

      //save valid terms: indices and names
      println("[INFO] Saving valid terms dictionary...")
      validTermsDF.coalesce(1).write.mode("overwrite").parquet(validDictPath)

      //Save filtered dataframe using the original num of partitions
      println("[INFO] Saving output DataFrame ...")
      filteredDF.write.mode("overwrite").parquet(outputDataPath)

    }finally {
      spark.stop()
    }
  }

  def loadAndIndexDictionary(spark: SparkSession, dictionaryPath: String): (Broadcast[Map[Int, String]], Int) = {
    import spark.implicits._

    println("[INFO] 正在加载和索引字典...")

    val dictDF = spark.read.parquet(dictionaryPath)
      .select("original_word")
      .distinct()
      .withColumn("index", row_number().over(Window.orderBy("original_word")) - 1)

    val wordToIndexMap = dictDF.collect().map(row =>
      row.getAs[String]("original_word") -> row.getAs[Int]("index")
    ).toMap

    val indexToWordMap = wordToIndexMap.map(_.swap)

    val vocabularySize = wordToIndexMap.size
    println(s"[INFO] 字典加载完成: ${vocabularySize} 个唯一词汇")

    val broadcastIndexToWordMap = spark.sparkContext.broadcast(indexToWordMap)

    (broadcastIndexToWordMap, vocabularySize)
  }


  def getOrCreateFeatureVector(df: DataFrame, featuresCol: String = "features"): DataFrame = {

    // Simple check: if column exists, assume it's valid (we can add more validation if needed)
    if (df.columns.contains(featuresCol)) {
      println(s"Using column '$featuresCol' (assuming it contains vectors)")
      return df
    }

    // Find all numeric (Double) columns to use for vector assembly
    val numericCols = df.columns.filter(c => df.schema(c).dataType.typeName == "double")

    if (numericCols.isEmpty) {
      throw new IllegalArgumentException(
        s"No numeric (Double) columns found to create feature vector '$featuresCol'. " +
          s"Available columns: ${df.columns.mkString(", ")}"
      )
    }

    println(s"➤ Creating feature vector '$featuresCol' from columns: ${numericCols.mkString(", ")}")
    // Use VectorAssembler to create the feature vector
    val assembler = new VectorAssembler()
      .setInputCols(numericCols)
      .setOutputCol(featuresCol)

    assembler.transform(df)
  }


  /**
   * Term filtering method
   *
   * @param dtmDF Input DataFrame containing feature vectors
   * @param featuresCol Name of the column containing feature vectors
   * @param minDocFreq Minimum document frequency (default: 2)
   * @param maxDocFreq Maximum documents a term can appear in (use Long.MaxValue for no limit)
   * @param minVariance Minimum variance threshold (default: 1e-6)
   * @return DataFrame with near-zero variance features filtered out,
   *         and another DataFrame containing valid terms (indices)
   */
  def filterTerms(
                    dtmDF: DataFrame,
                    featuresCol: String = "features",
                    minDocFreq: Long = 10,
                    maxDocFreq: Long = Long.MaxValue,
                    minVariance: Double = 1e-6
                  ): (DataFrame, DataFrame) = {

    require(minVariance >= 0, "minVariance must be >= 0")

    import dtmDF.sparkSession.implicits._
    val totalDocs = dtmDF.count()

    // Phase 1 - Document Frequency: Calculate how many documents each term appears in
    val docFreqs: RDD[(Int, Long)] = calculateDocFrequencies(dtmDF, featuresCol)

    // Phase 2 - Term Statistics: Calculate term statistics needed for variance calculation
    val termStats: RDD[(Int, (Double, Double, Long))] = calculateTermVariances(dtmDF, featuresCol)

    // Phase 3 - Filter Terms: Filter terms based on document frequency and variance thresholds
    val validTerms: RDD[Int] = filterValidTerms(
      docFreqs,
      termStats,
      totalDocs,
      minDocFreq,
      maxDocFreq,
      minVariance
    )
    // get the indices of valid terms as dataframe
    val validTermsDF = validTerms.toDF("term_index")

    // Phase 4: Apply the filtering to the original vectors
    val filteredDF = applyTermFilter(dtmDF, featuresCol, validTerms)

    (filteredDF, validTermsDF)
  }


  /**
   * Term filtering method
   *
   * @param dtmDF Input DataFrame containing feature vectors
   * @param featuresCol Name of the column containing feature vectors
   * @param minDocFreq Minimum document frequency (default: 2)
   * @param maxDocFreqRatio Maximum document frequency ratio (default: 0.9)
   * @param minVariance Minimum variance threshold (default: 1e-6)
   * @return DataFrame with near-zero variance features filtered out,
   *         and another DataFrame containing valid terms (indices)
   */
  def filterTerms1(
                   dtmDF: DataFrame,
                   featuresCol: String = "features",
                   minDocFreq: Int = 5,
                   maxDocFreqRatio: Double = 0.95,
                   minVariance: Double = 1e-6
                 ): (DataFrame, DataFrame) = {
    require(minDocFreq >= 1, "minDocFreq must be >= 1")
    require(maxDocFreqRatio > 0 && maxDocFreqRatio <= 1,
      "maxDocFreqRatio must be between 0 and 1")
    require(minVariance >= 0, "minVariance must be >= 0")

    import dtmDF.sparkSession.implicits._
    val totalDocs = dtmDF.count()

    // Phase 1 - Document Frequency: Calculate how many documents each term appears in
    val docFreqs: RDD[(Int, Long)] = calculateDocFrequencies(dtmDF, featuresCol)

    // Phase 2 - Term Statistics: Calculate term statistics needed for variance calculation
    val termStats: RDD[(Int, (Double, Double, Long))] = calculateTermVariances(dtmDF, featuresCol)

    // Phase 3 - Filter Terms: Filter terms based on document frequency and variance thresholds
    val validTerms: RDD[Int] = filterValidTermsWithRatio(
      docFreqs,
      termStats,
      totalDocs,
      minDocFreq,
      maxDocFreqRatio,
      minVariance
    )
    // get the indices of valid terms as dataframe
    val validTermsDF = validTerms.toDF("term_index")

    // Phase 4: Apply the filtering to the original vectors
    val filteredDF = applyTermFilter(dtmDF, featuresCol, validTerms)

    (filteredDF, validTermsDF)
  }



  /**
   * Phase 1:
   * Calculates document frequencies for terms in a DataFrame column of feature vectors.
   *
   * Document frequency is defined as the number of documents in which a term (feature)
   * appears **at least once**.
   *
   * @param df            Input DataFrame containing feature vectors.
   * @param featuresCol   Name of the column storing the feature vectors (default: "features").
   * @return              An RDD of `(termIndex, documentFrequency)` pairs, where:
   *                      - `termIndex` (Int): Index of the term in the feature vector.
   *                      - `documentFrequency` (Long): Number of documents containing this term.
   *
   * @note This method handles both `SparseVector` and `DenseVector`:
   *       - For `SparseVector`, only non-zero indices are counted (efficient by design).
   *       - For `DenseVector`, zeros are explicitly filtered out.
   *       - Each term is counted **once per document**, even if it appears multiple times.
   *
   * @example
   * Input DataFrame:
   * +---------------------+
   * | features            |
   * +---------------------+
   * | (5,[1,3],[1.0,2.0]) | // SparseVector: term 1 and 3 appear
   * | [0.0,1.0,0.0,3.0]   | // DenseVector: term 1 and 3 appear
   * +---------------------+
   *
   * Output RDD:
   * (1, 2L)  // Term 1 appears in 2 documents
   * (3, 2L)  // Term 3 appears in 2 documents
   */
   def calculateDocFrequencies(
                                       df: DataFrame,
                                       featuresCol: String = "features"
                                     ): RDD[(Int, Long)] = {
    df.select(col(featuresCol))
      .rdd
      .flatMap { row =>
        row.getAs[MLVector](0) match {
          // SparseVector: Indices are guaranteed to represent non-zero terms
          case sv: MLSparseVector => sv.indices.map(_ -> 1L)
          // DenseVector: Explicitly filter out zero values
          case dv: MLDenseVector =>
            dv.values.zipWithIndex.filter(_._1 != 0).map(_._2 -> 1L)
        }
      }
      .reduceByKey(_ + _)  // Aggregate counts across documents
  }

  /**
   * Phase 2: Calculate term statistics needed for variance computation.
   *
   * For each term (feature), computes three metrics across all documents:
   * 1. Sum of values
   * 2. Sum of squared values
   * 3. Total count of non-zero occurrences.
   *
   * These statistics enable efficient variance calculation without storing all values:
   * Variance = (sumSq - (sum²)/count) / (count - 1)
   *
   * @param df Input DataFrame containing feature vectors
   * @param featuresCol Name of the column storing feature vectors (default: "features")
   * @return RDD of tuples where:
   *         - Key: Term index (Int)
   *         - Value: Tuple of (sum: Double, sumSquares: Double, count: Long)
   *
   * @note Handles both sparse and dense vectors:
   *       - SparseVector: Only processes non-zero entries (efficient)
   *       - DenseVector: Explicitly filters zero values
   *       - Each term's statistics are aggregated across all documents
   *
   * @example For a term appearing in two documents with values 3.0 and 5.0:
   *          Output: (termIndex, (8.0, 34.0, 2L)) // sum=8, sumSq=34, count=2
   */
   def calculateTermVariances(
                                      df: DataFrame,
                                      featuresCol: String = "features"
                                    ): RDD[(Int, (Double, Double, Long))] = {
    df.select(col(featuresCol))
      .rdd
      .flatMap { row =>
        row.getAs[MLVector](0) match {
          // Process SparseVector:
          case sv: MLSparseVector =>
            sv.indices.zip(sv.values).map { case (idx, value) =>
              // Output: (termIndex, (value, value², 1L))
              (idx, (value, value * value, 1L))
            }

          // Process DenseVector: need to filter out zero values
          case dv: MLDenseVector =>
            dv.values.zipWithIndex
              .filter(_._1 != 0)  // Only keep non-zero values
              .map { case (value, idx) =>
                // Output: (termIndex, (value, value², 1L))
                (idx, (value, value * value, 1L))
              }
        }
      }
      // Aggregate statistics across all documents
      .reduceByKey { case ((s1, sq1, c1), (s2, sq2, c2)) =>
        (s1 + s2,  // Sum of values
          sq1 + sq2, // Sum of squared values
          c1 + c2)   // Total count
      }
  }

// ------------ Percentiles ---------------------

  /**
   * Finds optimal percentile-based thresholds for term filtering.
   *
   * This function analyzes the distribution of document frequencies across terms
   * and returns the 5th and 95th percentile values, which are commonly used as
   * optimal thresholds for removing rare and overly common terms.
   *
   * @param docFreqs RDD of (termId, documentFrequency) pairs
   * @param totalDocs Total number of documents in the corpus (for context)
   * @return Map containing optimal thresholds and diagnostic information
   *
   * @note The 5th percentile (p5) removes the bottom 5% rarest terms
   * @note The 95th percentile (p95) removes the top 5% most common terms
   * @note This approach automatically adapts to dataset size and term distribution
   *
   * @example For a dataset with 1M documents and 60K terms:
   *          p5 might be 12 (remove terms in <12 docs)
   *          p95 might be 85,000 (remove terms in >85,000 docs)
   */
  def findPercentilesWithDetails(docFreqs: RDD[(Int, Long)],
                             totalDocs: Long): Map[String, Any] = {

    val freqs = docFreqs.map(_._2).collect().sorted
    val totalTerms = freqs.length

    if (totalTerms == 0) {
      return Map(
        "error" -> "No term frequencies found",
        "minDocFreq" -> 2L,
        "maxDocFreq" -> totalDocs
      )
    }

    val p5Index = math.max(0, math.min(totalTerms - 1, (totalTerms * 0.05).toInt))
    val p95Index = math.max(0, math.min(totalTerms - 1, (totalTerms * 0.95).toInt))

    val p5Value = freqs(p5Index)
    val p95Value = freqs(p95Index)

    val termsKept = freqs.count(f => f >= p5Value && f <= p95Value)
    val reductionPct = (1 - termsKept.toDouble / totalTerms) * 100

    val minFreq = freqs.head
    val maxFreq = freqs.last
    val medianFreq = freqs(totalTerms / 2)

    // Create the rationale string first, then use it in the Map
    val rationale = s"This will remove the bottom 5% rarest and top 5% most common terms, " +
      s"keeping ${reductionPct.formatted("%.1f")}% of the original vocabulary"

    Map(
      "minDocFreq" -> p5Value,
      "maxDocFreq" -> p95Value,
      "minPercentile" -> 0.05,
      "maxPercentile" -> 0.95,
      "minPercentileValue" -> p5Value,
      "maxPercentileValue" -> p95Value,
      "totalTerms" -> totalTerms,
      "termsKept" -> termsKept,
      "termsRemoved" -> (totalTerms - termsKept),
      "reductionPercentage" -> reductionPct,
      "minFrequency" -> minFreq,
      "maxFrequency" -> maxFreq,
      "medianFrequency" -> medianFreq,
      "totalDocuments" -> totalDocs,
      "recommendation" -> s"Use minDocFreq = $p5Value (p5) and maxDocFreq = $p95Value (p95)",
      "rationale" -> rationale  // Use the pre-built string here
    )
  }

  // Alternative version that returns just the thresholds for direct use
  def findPercentiles5And95(docFreqs: RDD[(Int, Long)]): (Long, Long) = {
    val freqs = docFreqs.map(_._2).collect().sorted
    val totalTerms = freqs.length

    if (totalTerms == 0) (2L, Long.MaxValue) // Fallback values

    val p5Index = math.min(totalTerms - 1, (totalTerms * 0.10).toInt)
    val p95Index = math.min(totalTerms - 1, (totalTerms * 0.90).toInt)

    (freqs(p5Index), freqs(p95Index))
  }

  /**
   * Calculates document frequency values at specified percentiles.
   *
   * @param docFreqs RDD of (termId, documentFrequency) pairs
   * @param lowPercentile The lower percentile for minimum document frequency (e.g., 0.05 for 5th percentile)
   * @param highPercentile The higher percentile for maximum document frequency (e.g., 0.95 for 95th percentile)
   * @return Tuple of (minDocFreqValue, maxDocFreqValue) at the specified percentiles
   *
   * @example findPercentiles(docFreqs, 0.05, 0.95) // Returns (p5 value, p95 value)
   * @example findPercentiles(docFreqs, 0.10, 0.90) // Returns (p10 value, p90 value)
   */
  def findPercentiles(docFreqs: RDD[(Int, Long)],
                      lowPercentile: Double = 0.05,
                      highPercentile: Double = 0.95): (Long, Long) = {

    // Validate percentile inputs
    require(lowPercentile >= 0.0 && lowPercentile <= 1.0,
      s"lowPercentile must be between 0.0 and 1.0, got $lowPercentile")
    require(highPercentile >= 0.0 && highPercentile <= 1.0,
      s"highPercentile must be between 0.0 and 1.0, got $highPercentile")
    require(lowPercentile <= highPercentile,
      s"lowPercentile ($lowPercentile) must be <= highPercentile ($highPercentile)")

    val freqs = docFreqs.map(_._2).collect().sorted
    val totalTerms = freqs.length

    if (totalTerms == 0) (2L, Long.MaxValue) // Fallback values

    // Calculate indices with bounds checking
    val lowIndex = math.max(0, math.min(totalTerms - 1, (totalTerms * lowPercentile).toInt))
    val highIndex = math.max(0, math.min(totalTerms - 1, (totalTerms * highPercentile).toInt))

    (freqs(lowIndex), freqs(highIndex))
  }

  /**
   * Calculates both kept and removed term counts.
   *
   * @param docFreqs RDD of (termId, documentFrequency) pairs
   * @param minDocFreq Minimum document frequency threshold (inclusive)
   * @param maxDocFreq Maximum document frequency threshold (inclusive)
   * @return (termsKept, termsRemoved)
   */
  def getTermCounts(docFreqs: RDD[(Int, Long)],
                    minDocFreq: Long,
                    maxDocFreq: Long): (Long, Long) = {
    val totalTerms = docFreqs.count()

    val termsKept = docFreqs.map(_._2)
      .filter(freq => freq >= minDocFreq && freq <= maxDocFreq)
      .count()
    val termsRemoved = totalTerms - termsKept
    (termsKept, termsRemoved)
  }

  /**
   * Phase 3: Filters terms based on document frequency and variance thresholds.
   * Optimized single-pass term filtering combining document frequency and variance checks.
   *
   * Filters terms based on:
   * 1. Document frequency (min/max thresholds)
   * 2. Sample variance (minimum threshold)
   *
   * Performs filtering during the join result processing to avoid materializing
   * intermediate data for discarded terms.
   *
   * @param docFreqs RDD of (termIndex, documentFrequency) pairs from Phase 1
   * @param termStats RDD of (termIndex, (sum, sumSquares, count)) from Phase 2
   * @param totalDocs Total documents in the corpus (for context/logging)
   * @param minDocFreq Minimum documents a term must appear in (default: 2)
   * @param maxDocFreq Maximum documents a term can appear in (use Long.MaxValue for no limit)
   * @param minVariance Minimum sample variance required (default: 1e-6)
   * @return RDD[Int] Indices of terms passing all filters
   *
   * @note Uses Bessel's correction (n-1 denominator) for unbiased sample variance
   * @example For termIndex=5 appearing in 10/100 docs (sum=8.0, sumSq=34.0, count=5):
   *          - variance = (34 - (8²)/5)/4 = 1.3
   *          - Kept if: 2 <= 10 <= maxDocFreq && 1.3 >= 1e-6
   */
  private def filterValidTerms(
                                docFreqs: RDD[(Int, Long)],
                                termStats: RDD[(Int, (Double, Double, Long))],
                                totalDocs: Long,
                                minDocFreq: Long,
                                maxDocFreq: Long,
                                minVariance: Double
                              ): RDD[Int] = {

    docFreqs.join(termStats).mapPartitions { iter =>
      iter.flatMap { case (termIndex, (df, (sum, sumSq, count))) =>
        // --- Variance Calculation ---
        // Sample variance with Bessel's correction (n-1 denominator)
        // Guard against count <=1 to avoid division by zero
        val variance = if (count > 1) {
          (sumSq - (sum * sum) / count) / (count - 1)
        } else 0.0

        // --- Combined Filtering ---
        // All conditions checked in one pass:
        // 1. Term appears in enough documents (minDocFreq)
        // 2. Term doesn't appear in too many documents (maxDocFreq)
        // 3. Term has sufficient variability (minVariance)
        if (df >= minDocFreq && df <= maxDocFreq && variance >= minVariance) {
          Some(termIndex)
        } else {
          None  // Discard term immediately
        }
      }
    }
  }



  /**
   * Phase 3: Filters terms based on document frequency and variance thresholds.
   * Optimized single-pass term filtering combining document frequency and variance checks.
   *
   * Filters terms based on:
   * 1. Document frequency (min/max thresholds)
   * 2. Sample variance (minimum threshold)
   *
   * Performs filtering during the join result processing to avoid materializing
   * intermediate data for discarded terms.
   *
   * @param docFreqs RDD of (termIndex, documentFrequency) pairs from Phase 1
   * @param termStats RDD of (termIndex, (sum, sumSquares, count)) from Phase 2
   * @param totalDocs Total documents in the corpus
   * @param minDocFreq Minimum documents a term must appear in (default: 2)
   * @param maxDocFreqRatio Maximum ratio of documents a term can appear in (default: 0.9)
   * @param minVariance Minimum sample variance required (default: 1e-6)
   * @return RDD[Int] Indices of terms passing all filters
   *
   * @note Uses Bessel's correction (n-1 denominator) for unbiased sample variance
   * @example For termIndex=5 appearing in 10/100 docs (sum=8.0, sumSq=34.0, count=5):
   *          - maxDocFreq = 100 * 0.9 = 90
   *          - variance = (34 - (8²)/5)/4 = 1.3
   *          - Kept if: 2 <= 10 <= 90 && 1.3 >= 1e-6
   */
  private def filterValidTermsWithRatio(
                                docFreqs: RDD[(Int, Long)],
                                termStats: RDD[(Int, (Double, Double, Long))],
                                totalDocs: Long,
                                minDocFreq: Int,
                                maxDocFreqRatio: Double,
                                minVariance: Double
                              ): RDD[Int] = {
    // Calculate absolute max document frequency from ratio
    val maxDocFreq = (totalDocs * maxDocFreqRatio).toLong

    docFreqs.join(termStats).mapPartitions { iter =>
      iter.flatMap { case (termIndex, (df, (sum, sumSq, count))) =>
        // --- Variance Calculation ---
        // Sample variance with Bessel's correction (n-1 denominator)
        // Guard against count <=1 to avoid division by zero
        val variance = if (count > 1) {
          (sumSq - (sum * sum) / count) / (count - 1)
        } else 0.0

        // --- Combined Filtering ---
        // All conditions checked in one pass:
        // 1. Term appears in enough documents (minDocFreq)
        // 2. Term doesn't appear in too many documents (maxDocFreq)
        // 3. Term has sufficient variability (minVariance)
        if (df >= minDocFreq && df <= maxDocFreq && variance >= minVariance) {
          Some(termIndex)
        } else {
          None  // Discard term immediately
        }
      }
    }
  }


  /**
   * Applies term filtering to the original feature vectors using broadcasted valid terms.
   *
   * Efficiently removes near-zero variance terms by:
   * 1. Broadcasting the set of valid term indices to all workers
   * 2. Filtering each vector to retain only valid terms
   * 3. Preserving the original vector size for downstream compatibility
   *
   * @param df Input DataFrame containing raw feature vectors
   * @param featuresCol Name of the features column (default: "features")
   * @param validTerms RDD of term indices that passed all filters
   * @return DataFrame with filtered feature vectors
   *
   * @note Maintains original vector dimensions (size) for model compatibility
   * @example Input vector: [0.0, 1.5, 0.0, 3.0] with validTerms=[1,3]
   *          Output: SparseVector(4, [1,3], [1.5, 3.0])
   */
  private def applyTermFilter_KeepEmptyRows(
                                             df: DataFrame,
                                             featuresCol: String = "features",
                                             validTerms: RDD[Int]
                                           ): DataFrame = {
    val validTermsSet = validTerms.collect().toSet
    if (validTermsSet.isEmpty) {
      // Return empty DataFrame or throw informative error
      return df.sparkSession.emptyDataFrame
      // OR: throw new IllegalArgumentException("No valid terms to filter")
    }

    // Broadcast the valid term indices for efficient distributed lookup
    val validTermsBC = df.sparkSession.sparkContext.broadcast(validTermsSet) // Materialize as Set for O(1) lookups


    // Define UDF for vector filtering
    val filterUDF = udf { (vector: MLVector) =>
      vector match {
        case sv: MLSparseVector =>
          // --- Sparse Vector Optimization ---
          // Only check validTerms against existing indices (already non-zero)
          val newIndices = sv.indices.filter(validTermsBC.value.contains)
          val newValues = newIndices.map(idx => sv.values(sv.indices.indexOf(idx)))
          MLVectors.sparse(sv.size, newIndices, newValues)

        case dv: MLDenseVector =>
          // --- Dense Vector Handling ---
          // Convert to sparse first for efficient filtering
          val sv = dv.toSparse
          val newIndices = sv.indices.filter(validTermsBC.value.contains)
          val newValues = newIndices.map(idx => sv.values(sv.indices.indexOf(idx)))
          MLVectors.sparse(sv.size, newIndices, newValues)
      }
    }

    // Apply filtering and return new DataFrame
    df.withColumn(featuresCol, filterUDF(col(featuresCol)))
  }


  /**
   * Applies term filtering and removes rows where all terms were filtered out.
   *
   * @param df Input DataFrame with feature vectors
   * @param featuresCol Name of the features column (default: "features")
   * @param validTerms RDD of term indices that passed all filters
   * @return DataFrame with:
   *         1. Filtered feature vectors (only valid terms retained)
   *         2. Rows with empty vectors removed
   * @note Preserves original vector dimensions but drops empty results
   */

  private def applyTermFilter(
                               df: DataFrame,
                               featuresCol: String = "features",
                               validTerms: RDD[Int]
                             ): DataFrame = {
    // 1. Prepare term mapping
    val validTermsArray = validTerms.distinct().collect().sorted
    if (validTermsArray.isEmpty) {
      return df.sparkSession.emptyDataFrame
    }

    // - termMap.keys = validTermsSet
    // - termMap.values = contiguous indices
    val termMap = validTermsArray.zipWithIndex.toMap
    val globalSize = validTermsArray.length

    // Broadcast just the map
    val broadcastTermMap = df.sparkSession.sparkContext.broadcast(termMap)

    // 2. UDF using only the map
    val filterUDF = udf { vector: MLVector =>
      val localTermMap = broadcastTermMap.value

      vector match {
        case sv: MLSparseVector =>
          val newIndices = sv.indices
            .filter(localTermMap.contains) // Check existence (replaces validTermsSet)
            .map(term => localTermMap(term)) // Get new index

          val newValues = newIndices.map(idx =>
            sv.values(sv.indices.indexOf(localTermMap.find(_._2 == idx).get._1))
          )

          if (newIndices.isEmpty) null
          else MLVectors.sparse(globalSize, newIndices, newValues)

        case dv: MLDenseVector =>
          val sv = dv.toSparse
          val newIndices = sv.indices
            .filter(localTermMap.contains)
            .map(localTermMap)

          val newValues = newIndices.map(idx =>
            sv.values(sv.indices.indexOf(localTermMap.find(_._2 == idx).get._1))
          )

          if (newIndices.isEmpty) null
          else MLVectors.sparse(globalSize, newIndices, newValues)
      }
    }

    // 3. Apply transformations
    df.withColumn(featuresCol, filterUDF(col(featuresCol)))
      .filter(col(featuresCol).isNotNull)
  }


  private def applyTermFilter_SameVectorSize(
                                              df: DataFrame,
                                              featuresCol: String = "features",
                                              validTerms: RDD[Int]
                                            ): DataFrame = {

    // 1. Collect valid terms and broadcast
    val validTermsSet = validTerms.collect().toSet
    if (validTermsSet.isEmpty) {
      return df.sparkSession.createDataFrame(
        df.sparkSession.sparkContext.emptyRDD[Row],
        df.schema
      )
    }
    val validTermsBC = df.sparkSession.sparkContext.broadcast(validTermsSet)

    // 2. UDF to filter vectors and return null if empty
    val filterUDF = udf { (vector: MLVector) =>
      vector match {
        case sv: MLSparseVector =>
          val newIndices = sv.indices.filter(validTermsBC.value.contains)
          if (newIndices.isEmpty) null
          else MLVectors.sparse(sv.size, newIndices, newIndices.map(idx => sv.values(sv.indices.indexOf(idx))))

        case dv: MLDenseVector =>
          val sv = dv.toSparse
          val newIndices = sv.indices.filter(validTermsBC.value.contains)
          if (newIndices.isEmpty) null
          else MLVectors.sparse(sv.size, newIndices, newIndices.map(idx => sv.values(sv.indices.indexOf(idx))))
      }
    }

    // 3. Apply filtering and remove null/empty vectors
    df.withColumn(featuresCol, filterUDF(col(featuresCol)))
      .filter(col(featuresCol).isNotNull)  // Remove rows with null vectors
  }


  /**
   * Counts rows where a vector column is empty (all zeros or no active elements).
   *
   * @param df Input DataFrame
   * @param vectorCol Name of the vector column (default: "features")
   * @return Count of rows where the vector is:
   *         - Truly zero (all elements = 0) OR
   *         - Sparse with no active elements OR
   *         - Null
   */
  def countEmptyVectorRows(df: DataFrame, vectorCol: String = "features"): Long = {
    val spark = df.sparkSession

    // Validate column exists and is a vector
    require(df.schema.exists(_.name == vectorCol),
      s"Column $vectorCol not found in DataFrame")
    require(df.schema(vectorCol).dataType.typeName.contains("vector"),
      s"Column $vectorCol is not a vector type")

    // UDF to check for empty vectors
    val isEmptyVector = udf { v: MLVector =>
      v == null || v.numActives == 0
    }

    df.filter(isEmptyVector(col(vectorCol))).count()
  }

  def countStrictZeroRows(df: DataFrame): Long = {
    val doubleCols = df.schema.fields
      .filter(_.dataType.typeName == "double")
      .map(_.name)

    if (doubleCols.isEmpty) 0L
    else df.filter(doubleCols.map(c => col(c) === 0.0).reduce(_ && _)).count()
  }



}
