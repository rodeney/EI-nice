package com.szubd
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.linalg.{Vector => MLVector}
import org.apache.spark.mllib.linalg.{Vector => MLLibVector, Vectors => MLLibVectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD

object SVD_EstimateK {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SVD_EstimateK")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // Use local paths if no arguments provided
    val inputPath = if (args.length > 0) args(0) else "data/RedditTest/201201/data_filtered/"
    val outputPath = if (args.length > 1) args(1) else "data/Reddit20052006/test/"
    val varianceThreshold = if (args.length > 2) args(2).toDouble else 0.95

    // Load data and convert to indexed RDD immediately
    val df = spark.read.parquet(inputPath)//.limit(10000)

    val indexedRDD: RDD[(Long, String, MLVector)] = df
      .select("submission_id", "normalized_features")
      .rdd
      .zipWithIndex()
      .map { case (row, index) =>
        val submissionId = row.getAs[String]("submission_id")
        val mlVector = row.getAs[MLVector]("normalized_features")
        (index, submissionId, mlVector)
      }
      .persist() // Cache to avoid recomputation

    println(s"Indexed RDD count: ${indexedRDD.count()}")

    // Extract just the vectors for SVD (maintain same partitioning)
    val vectorsRDD: RDD[MLLibVector] = indexedRDD.map { case (index, submissionId, mlVector) =>
      MLLibVectors.fromML(mlVector)
    }

    // Create RowMatrix
    val rowMatrix = new RowMatrix(vectorsRDD)

    //Estimate K by Variance:
    val optimalK = estimateKByVarianceWithLimit(rowMatrix, varianceThreshold, maxK = 2000)
    println(s"optimalK with variance ${varianceThreshold}： ${optimalK}")


    //Estimate k with elbow mehtod
    // Run scree plot analysis
    val result = comprehensiveScreeAnalysis(
      matrix = rowMatrix,
      maxK = 2000 // Safe limit for our high-dimension feature matrix
    )

    //compute SVD with optimal k
//    val k = if (optimalK > 1) optimalK else 100
//
//    val svd = rowMatrix.computeSVD(k, computeU = true)
//
//    // Create result with perfect index alignment
//    val resultRDD = indexedRDD.map { case (index, submissionId, _) =>
//      (index, submissionId)
//    }.zip(svd.U.rows).map { case ((index, submissionId), uVector) =>
//      (submissionId, uVector.asML.toArray)
//    }
//
//    val resultDF = df.join(resultRDD.toDF("submission_id", s"svd_${k}d_features"), "submission_id") // keep both normalized and svd features
//    println(s"Final result count: ${resultDF.count()}")
//
//    //save
//    //val outputDir = s"$outputPath/data_normalized_svd_k${optimalK}"
//    resultDF.write.mode("overwrite").parquet(outputPath)

    spark.stop()
  }


  /**
   * Estimates the optimal number of singular values (k) to retain based on explained variance threshold.
   *
   * This method computes the minimum number of singular values required to capture at least the specified
   * percentage of the total variance in the data matrix. It's based on the property that the squared
   * singular values represent the amount of variance captured by each principal component.
   *
   * Mathematical foundation:
   * - Total variance = sum(singular_value_i²) for all i
   * - Explained variance by first k components = sum(singular_value_i²) for i=1 to k
   * - We find the smallest k such that (explained variance / total variance) >= threshold
   *
   * @param matrix            The RowMatrix containing the data for SVD computation
   * @param varianceThreshold The minimum proportion of total variance to be explained by the retained components.
   *                         Must be between 0 and 1. Typical values: 0.90 (90%), 0.95 (95%), or 0.99 (99%).
   *                         Higher values retain more dimensions but may include noise.
   * @return The optimal number of components k to retain. Always returns at least 1.
   *
   * @throws IllegalArgumentException if varianceThreshold is not between 0 and 1
   * @throws ArithmeticException if matrix is empty or contains invalid values
   *
   * @example
   * // For a matrix where first 5 singular values explain 95% of variance:
   * val k = estimateKByVariance(matrix, 0.95) // Returns 5
   *
   * @note This method computes the full SVD which can be computationally expensive for large matrices.
   *       Consider using incremental SVD or randomized SVD for very large datasets.
   */
  def estimateKByVariance(matrix: RowMatrix, varianceThreshold: Double = 0.95, maxK: Int = 500): Int = {
    // Validate input parameters
    require(varianceThreshold > 0 && varianceThreshold <= 1.0,
      s"Variance threshold must be between 0 and 1, got $varianceThreshold")

    // Get the number of features (columns) in the matrix
    val numFeatures = matrix.numCols().toInt

    // Compute the full SVD without the U matrix since we only need singular values for k estimation
    // computeU = false saves memory and computation time as we don't need the left singular vectors
    val svd = matrix.computeSVD(numFeatures, computeU = false)

    // Extract singular values from the SVD result and convert to array
    // Singular values are returned in descending order of magnitude
    val singularValues = svd.s.toArray

    // Calculate total variance: sum of squares of all singular values
    // This represents the total "information content" in the matrix
    val totalVariance = singularValues.map(math.pow(_, 2)).sum

    // Validate that we have a meaningful matrix (non-zero variance)
    require(totalVariance > 0, "Matrix has zero variance - cannot compute meaningful SVD")

    // Initialize cumulative variance tracker and component counter
    var cumulativeVariance = 0.0  // Running total of explained variance
    var k = 0                     // Number of components to retain

    // Iterate through singular values until we reach the variance threshold
    // We process singular values in descending order (most important first)
    while (cumulativeVariance < varianceThreshold && k < numFeatures) {
      // Calculate variance contributed by the k-th component (squared singular value)
      val componentVariance = math.pow(singularValues(k), 2)

      // Add this component's variance to the cumulative total
      cumulativeVariance += componentVariance / totalVariance

      // Move to the next component
      k += 1

      // Optional: Add debug logging for development (remove in production)
      println(f"Component $k: explains ${componentVariance/totalVariance*100}%.2f%% (cumulative: ${cumulativeVariance*100}%.2f%%)")
    }

    // Ensure we return at least 1 component even if threshold is very low
    // math.max(1, k) guarantees we never return 0, which would be meaningless
    math.max(1, k)
  }


  /**
   * Estimates the optimal number of singular values (k) to retain based on explained variance threshold,
   * with a safety limit to prevent memory issues on large matrices.
   *
   * This method computes the minimum number of singular values required to capture at least the specified
   * percentage of the total variance in the data matrix, while respecting a maximum allowed k value to
   * prevent out-of-memory errors on large datasets.
   *
   * @param matrix            The RowMatrix containing the data for SVD computation
   * @param varianceThreshold The minimum proportion of total variance to be explained by the retained components.
   *                         Must be between 0 and 1. Typical values: 0.90 (90%), 0.95 (95%), or 0.99 (99%).
   *                         Higher values retain more dimensions but may include noise.
   * @param maxK              The maximum number of components to consider. This is a safety parameter to
   *                         prevent memory overflow when working with large matrices. Should be set based
   *                         on available memory and matrix dimensions.
   * @return The optimal number of components k to retain. Returns at most maxK, even if variance threshold
   *         is not reached. Always returns at least 1.
   *
   * @throws IllegalArgumentException if varianceThreshold is not between 0 and 1 or if maxK < 1
   *
   * @example
   * // For a large matrix, limit k to 1000 to avoid memory issues
   * val k = estimateKByVariance(largeMatrix, 0.95, 1000)
   *
   * @note For matrices with many features (e.g., > 5000), set maxK to a reasonable value (e.g., 1000-2000)
   *       to prevent OutOfMemoryError. The method will return the best k within the maxK limit.
   */
  def estimateKByVarianceWithLimit(
                                    matrix: RowMatrix,
                                    varianceThreshold: Double = 0.95,
                                    maxK: Int = Int.MaxValue  // Default: no limit (original behavior)
                                  ): Int = {

    // Validate input parameters
    require(varianceThreshold > 0 && varianceThreshold <= 1.0,
      s"Variance threshold must be between 0 and 1, got $varianceThreshold")
    require(maxK >= 1, s"maxK must be at least 1, got $maxK")

    // Get the number of features (columns) in the matrix
    val numFeatures = matrix.numCols().toInt

    // Determine the actual maximum k to compute based on matrix dimensions and maxK parameter
    val actualMaxK = math.min(maxK, numFeatures)

    // Log the computation parameters for debugging
    println(s"📊 SVD k estimation parameters:")
    println(s"   Matrix dimensions: ${matrix.numRows()} x $numFeatures")
    println(s"   Variance threshold: ${varianceThreshold * 100}%")
    println(s"   Maximum k allowed: $maxK")
    println(s"   Actual k to compute: $actualMaxK")

    // Compute partial SVD - only up to actualMaxK components to save memory
    val svd = matrix.computeSVD(actualMaxK, computeU = false)  // computeU = false saves memory
    println(s"SVD computed successfully for actualMaxK=$actualMaxK")
    // Extract singular values from the SVD result
    val singularValues = svd.s.toArray

    // Calculate total variance from the computed components
    // Note: This is the variance explained by the first actualMaxK components, not total matrix variance
    val computedVariance = singularValues.map(math.pow(_, 2)).sum

    // Initialize cumulative variance tracker and component counter
    var cumulativeVariance = 0.0
    var k = 0

    // Iterate through singular values until we reach variance threshold or max components
    while (cumulativeVariance < varianceThreshold && k < actualMaxK) {
      // Calculate variance contributed by the k-th component
      val componentVariance = math.pow(singularValues(k), 2)

      // Add this component's variance to the cumulative total
      cumulativeVariance += componentVariance / computedVariance

      // Log progress for every 10% of components or every 100 components for large k
      if (k % math.max(1, actualMaxK / 10) == 0 || k % 100 == 0) {
        println(f"   Component ${k + 1}: ${cumulativeVariance * 100}%.1f%% variance explained")
      }

      k += 1
    }

    // Final result and logging
    val finalK = math.max(1, k)

    if (cumulativeVariance >= varianceThreshold) {
      println(f"✅ Optimal k found: $finalK (explains ${cumulativeVariance * 100}%.1f%% variance)")
    } else {
      println(f"⚠️  Variance threshold not reached: k=$finalK explains ${cumulativeVariance * 100}%.1f%% variance")
      println(s"💡 Consider increasing maxK parameter or accepting lower variance")
    }

    finalK
  }

  /**
   * Computes scree plot data for optimal k determination using elbow method
   *
   * @param matrix The RowMatrix for SVD computation
   * @param maxK Maximum number of components to consider (for memory safety)
   * @return DataFrame with scree plot data and estimated elbow k
   */
  def computeScreePlotData(
                            matrix: RowMatrix,
                            maxK: Int = 1000
                          ): (DataFrame, Int) = {

    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val numFeatures = matrix.numCols().toInt
    val actualMaxK = math.min(maxK, numFeatures)

    println(s"📊 Computing scree plot data for k=1 to $actualMaxK")

    // Compute SVD up to maxK components
    val svd = matrix.computeSVD(actualMaxK, computeU = false)
    println(s"📊 SVD computed successfully for actualMaxk= $actualMaxK")
    val singularValues = svd.s.toArray
    val totalVariance = singularValues.map(math.pow(_, 2)).sum

    // Calculate variance metrics for each k
    val screeData = (1 to actualMaxK).map { k =>
      println(s"📊 variance metrics for k= $k")
      val variance = math.pow(singularValues(k-1), 2)
      val individualVarianceRatio = variance / totalVariance
      val cumulativeVariance = singularValues.take(k).map(v => math.pow(v, 2)).sum / totalVariance

      (k, variance, individualVarianceRatio, cumulativeVariance)
    }

    // Convert to DataFrame
    val screeDF = screeData.toDF("k", "variance", "individual_variance_ratio", "cumulative_variance_ratio")

    screeDF.show(5)
    // Find elbow point using second differences
    val elbowK = findElbowPoint(screeDF)

    // Save to CSV for external plotting
    //saveScreeDataToCSV(screeDF, outputPath, elbowK)

    (screeDF, elbowK)
  }

  /**
   * Finds the elbow point using second differences method
   *
   * The elbow point is where the second derivative of explained variance is maximized,
   * indicating the point of maximum curvature where additional components provide
   * diminishing returns.
   *
   * @param screeDF DataFrame with k and individual_variance_ratio columns
   * @return Estimated elbow point k value
   */
  private def findElbowPoint(screeDF: DataFrame): Int = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    // Collect data to driver for processing (small dataset)
    val varianceData = screeDF.select("k", "individual_variance_ratio")
      .as[(Int, Double)]
      .collect()
      .sortBy(_._1)

    val kValues = varianceData.map(_._1)
    val varianceRatios = varianceData.map(_._2)

    // Calculate second differences to find point of maximum curvature
    val secondDifferences = (2 until varianceRatios.length).map { i =>
      val diff1 = varianceRatios(i-1) - varianceRatios(i-2)  // first derivative at i-1
      val diff2 = varianceRatios(i) - varianceRatios(i-1)    // first derivative at i
      val secondDiff = diff2 - diff1                         // second derivative
      (kValues(i), secondDiff)
    }

    // Find point with maximum second difference (elbow point)
    val elbowPoint = secondDifferences.maxBy(_._2)._1

    println(s"📈 Elbow point detected at k = $elbowPoint")
    elbowPoint
  }

  /**
   * Alternative elbow detection using cumulative variance curvature
   */
  private def findElbowByCumulativeVariance(screeDF: DataFrame): Int = {
    val spark = SparkSession.builder().getOrCreate()
    import spark.implicits._

    val cumData = screeDF.select("k", "cumulative_variance_ratio")
      .as[(Int, Double)]
      .collect()
      .sortBy(_._1)

    val kValues = cumData.map(_._1)
    val cumVariance = cumData.map(_._2)

    // Calculate angles between consecutive segments
    val angles = (1 until cumVariance.length - 1).map { i =>
      val dx1 = 1.0
      val dy1 = cumVariance(i) - cumVariance(i-1)
      val dx2 = 1.0
      val dy2 = cumVariance(i+1) - cumVariance(i)

      val angle = math.atan2(dy2, dx2) - math.atan2(dy1, dx1)
      (kValues(i), math.abs(angle))
    }

    // Find point with maximum angle change (elbow)
    angles.maxBy(_._2)._1
  }

  /**
   * Enhanced scree plot analysis with multiple elbow detection methods
   */
  def comprehensiveScreeAnalysis(
                                  matrix: RowMatrix,
                                  maxK: Int = 1000
                                ): Map[String, Any] = {

    val (screeDF, elbow1) = computeScreePlotData(matrix, maxK)
    val elbow2 = findElbowByCumulativeVariance(screeDF)

    // Choose the more conservative (smaller) elbow point
    val finalElbow = math.min(elbow1, elbow2)

    println(s"🔍 Comprehensive scree analysis:")
    println(s"   Method 1 (second differences): k = $elbow1")
    println(s"   Method 2 (cumulative curvature): k = $elbow2")
    println(s"   Selected elbow point: k = $finalElbow")

    Map(
      "scree_data" -> screeDF,
      "elbow_k_second_diff" -> elbow1,
      "elbow_k_cumulative" -> elbow2,
      "final_elbow_k" -> finalElbow
    )
  }

  def findRobustElbow(singularValues: Array[Double]): Int = {
    val variances = singularValues.map(math.pow(_, 2))
    val totalVariance = variances.sum
    val explained = variances.map(_ / totalVariance)

    // Method 1: Second differences
    val secondDiffs = (2 until explained.length).map { i =>
      val diff1 = explained(i-1) - explained(i-2)
      val diff2 = explained(i) - explained(i-1)
      (i, diff2 - diff1)
    }

    // Method 2: Percentage of max variance
    val maxVar = explained.max
    val threshold = maxVar * 0.1 // 10% of max

    // Method 3: Cumulative variance reach
    val cumulatives = explained.scanLeft(0.0)(_ + _).tail

    // Combine methods
    val elbow1 = secondDiffs.maxBy(_._2)._1
    val elbow2 = explained.indexWhere(_ < threshold)
    val elbow3 = cumulatives.indexWhere(_ > 0.8) // 80% cumulative

    // Choose the most conservative (smallest)
    List(elbow1, elbow2, elbow3).min
  }

}
