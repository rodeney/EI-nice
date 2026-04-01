package com.szubd

import scala.collection.JavaConverters._

import com.szubd.INice

/**
 * Scala-friendly operations wrapper for the Java INice clustering algorithm.
 *
 * Provides clean Scala interfaces for Java INice methods by handling
 * Java-Scala conversions internally. This object contains static methods
 * for clustering operations.
 *
 * ==Usage==
 * {{{
 * import INiceOps._
 *
 * val features: List[Array[Double]] = ... // Your feature data
 * val centers = fitMO(features, numObservationPoints = 5, knn = 5)
 * }}}
 *
 */

object INiceOps {

  /**
   * Validates clustering parameters before calling Java method.
   *
   * @param features             the feature data to validate
   * @param numObservationPoints Number of Observation Points
   * @param knn                  Number of neighbors for density estimation.
   * @throws IllegalArgumentException if any parameter is invalid
   */
  private def validateParameters(features: List[Array[Double]], numObservationPoints: Int, knn: Int): Unit = {
    require(features.nonEmpty, "Features list cannot be empty - clustering requires data")
    require(numObservationPoints > 0, s"Number of Observation Points must be positive, got: $numObservationPoints")
    require(knn > 0, s"umber of neighbors for density estimation must be positive, got: $knn")

    // Validate that all feature arrays have the same dimension
    if (features.length > 1) {
      val firstSize = features.head.length
      require(features.tail.forall(_.length == firstSize),
        "All feature arrays must have the same dimension for clustering")
    }
  }

  /**
   * Perform clustering using INice.fitMO algorithm with specified parameters.
   *
   * Wraps the Java INice.fitMO method, handling all Java-Scala conversions
   * automatically. This is the main entry point for clustering operations.
   *
   * @param features             the feature data to cluster, represented as a List of feature arrays.
   *                             Each inner array represents one data point's features.
   * @param numObservationPoints Number of Observation Points (default: 5)
   * @param knn                  Number of neighbors for density estimation (default: 5)
   * @return a List of cluster centers, each represented as an Array[Double] of same dimension as input features
   * @throws IllegalArgumentException if features list is empty, numObservationPoints ≤ 0, knn ≤ 0,
   *                                  or feature arrays have inconsistent dimensions
   * @throws RuntimeException         if the underlying Java clustering algorithm fails
   *
   */
  def fitMO(features: List[Array[Double]], numObservationPoints: Int = 5, knn: Int = 5): List[Array[Double]] = {
    validateParameters(features, numObservationPoints, knn)

    try {
      // Convert Scala List[Array[Double]] to Java List<double[]>
      val javaFeatures = features.asJava

      // Call underlying Java method
      val javaCenters = INice.fitMO(javaFeatures, numObservationPoints, knn)

      // Convert Java results back to Scala
      javaCenters.asScala.toList

    } catch {
      case e: IllegalArgumentException =>
        // Re-throw validation errors as-is
        throw e
      case e: Exception =>
        throw new RuntimeException(s"INice clustering failed: ${e.getMessage}", e)
    }
  }


  /**
   * Wrapper for INice.ensembleCenters Java method.
   * Performs ensemble selection on cluster centers.
   *
   * @param centersAll List of all cluster centers from multiple partitions or runs.
   * @param percentage The percentile value (0-1) used to determine the merging threshold.
   * @return List of selected ensemble centers
   * @throws IllegalArgumentException if centersAll is empty, percentage is not in [0,1],
   *                                  or centers have inconsistent dimensions
   *
   */
  def ensembleCenters(centersAll: List[Array[Double]], percentage: Double): List[Array[Double]] = {
    require(centersAll.nonEmpty, "Centers list cannot be empty for ensemble selection")
    require(percentage >= 0.0 && percentage <= 1.0,
      s"Percentage must be between 0.0 and 1.0, got: $percentage")

    // Validate consistent dimensions
    if (centersAll.length > 1) {
      val firstSize = centersAll.head.length
      require(centersAll.tail.forall(_.length == firstSize),
        "All cluster centers must have the same dimension for ensemble selection")
    }

    try {
      val javaCenters = centersAll.asJava
      val javaEnsemble = INice.ensembleCenters(javaCenters, percentage) //INice.mergeData(javaCenters, percentage)
      javaEnsemble.asScala.toList
    } catch {
      case e: IllegalArgumentException => throw e
      case e: Exception => throw new RuntimeException(s"Ensemble selection failed: ${e.getMessage}", e)
    }
  }


}



