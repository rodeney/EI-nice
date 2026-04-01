package com.szubd

import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.StandardScaler


object SVD_StandardScaler {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("StandardScalerTest")
      .master("local[*]")
      .getOrCreate()

    try {
      import spark.implicits._

      // Use local paths and default values if no arguments provided
      val inputPath = if (args.length > 0) args(0) else "data/Reddit20052006/data_normalized_svd_k100/"
      val outputPath = if (args.length > 1) args(1) else "data/Reddit20052006/test/data_normalized_svd100_scaled/"
      val featuresCol = if (args.length > 2) args(2).toInt else "svd_k100_features"

      // Load svd version
      val df = spark.read.parquet(inputPath)
      df.printSchema()
      df.show(5)


      // Create StandardScaler
      val scaler = new StandardScaler()
        .setInputCol("svd_k100_features")
        .setOutputCol("svd100_scaled_features")
        .setWithStd(true)    // Scale to unit standard deviation
        .setWithMean(true)   // Center data with mean
      // Fit and transform
      val scalerModel = scaler.fit(df)
      val scaledDF = scalerModel.transform(df)

      scaledDF.printSchema()
      scaledDF.show(5)

      //write the output dataframe without the original svd features
      //we keep the  normalized high deminion features and the new scaled svd features. We will need both when assigning clusters
      println("Saving scaledDF  ...")
      scaledDF.drop("data_normalized_svd_k100").write.mode("overwrite").parquet(outputPath)
      println("Done.")
    }finally {
      spark.stop()
    }
  }

}
