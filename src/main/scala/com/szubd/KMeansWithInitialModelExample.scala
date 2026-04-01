package com.szubd

import org.apache.spark.mllib.clustering.{KMeans => MLlibKMeans, KMeansModel => MLlibKMeansModel}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession


object KMeansWithInitialModelExample {
  def main(args: Array[String]): Unit = {
    // Create Spark session
    val spark = SparkSession.builder()
      .appName("KMeansWithInitialModel")
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext

    try {
      // Create sample data - 3D points that form 3 clusters
      val data = Seq(
        // Cluster 1: points around (1, 1, 1)
        Vectors.dense(1.0, 1.2, 0.9),
        Vectors.dense(0.8, 1.1, 1.3),
        Vectors.dense(1.3, 0.9, 1.0),
        Vectors.dense(0.9, 1.0, 1.1),

        // Cluster 2: points around (5, 5, 5)
        Vectors.dense(5.1, 4.9, 5.2),
        Vectors.dense(4.8, 5.3, 5.0),
        Vectors.dense(5.2, 5.1, 4.8),
        Vectors.dense(4.9, 5.0, 5.1),

        // Cluster 3: points around (9, 9, 9)
        Vectors.dense(9.2, 8.9, 9.1),
        Vectors.dense(8.8, 9.3, 9.0),
        Vectors.dense(9.1, 9.0, 8.8),
        Vectors.dense(8.9, 9.1, 9.2)
      )

      val parsedData: RDD[org.apache.spark.mllib.linalg.Vector] = sc.parallelize(data)
      parsedData.cache() // Cache for multiple iterations

      // Create initial cluster centers (can be good guesses or from previous run)
      val initialCenters = Array(
        Vectors.dense(0.5, 0.5, 0.5),  // Initial guess for cluster 1
        Vectors.dense(4.5, 4.5, 4.5),  // Initial guess for cluster 2
        Vectors.dense(8.5, 8.5, 8.5)   // Initial guess for cluster 3
      )

      // Create initial model from centers
      val initialModel = new MLlibKMeansModel(initialCenters)

      // Set K-means parameters
      val numClusters = 3
      val numIterations = 100
      val epsilon = 1e-4  // Convergence threshold

      println("Initial Cluster Centers:")
      initialModel.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
        println(s"Cluster $idx: $center")
      }

      // Calculate initial cost
      val initialCost = initialModel.computeCost(parsedData)
      println(s"Initial cost: $initialCost")
      println("-" * 50)

      // Configure and run K-means with initial model
      val kmeans = new MLlibKMeans()
        .setK(numClusters)
        .setMaxIterations(numIterations)
        .setEpsilon(epsilon)
        .setSeed(42L) // For reproducibility
        .setInitialModel(initialModel)

      // Train starting from initial model
      val finalModel = kmeans.run(parsedData)

      // Results
      println("Final Cluster Centers:")
      finalModel.clusterCenters.zipWithIndex.foreach { case (center, idx) =>
        println(s"Cluster $idx: $center")
      }

      val finalCost = finalModel.computeCost(parsedData)
      println(s"Final cost: $finalCost")
      println(s"Cost improvement: ${initialCost - finalCost}")
      println(s"Number of iterations actually run: ???") // Note: This info isn't directly available

      println("\nCluster assignments:")
      val predictions = parsedData.map { point =>
        val clusterId = finalModel.predict(point)
        (point, clusterId)
      }

      predictions.collect().foreach { case (point, clusterId) =>
        println(s"Point $point -> Cluster $clusterId")
      }

      // Count points per cluster
      val clusterCounts = predictions.map(_._2)
        .countByValue()
        .toSeq
        .sortBy(_._1)

      println("\nPoints per cluster:")
      clusterCounts.foreach { case (clusterId, count) =>
        println(s"Cluster $clusterId: $count points")
      }

    } finally {
      spark.stop()
    }
  }
}