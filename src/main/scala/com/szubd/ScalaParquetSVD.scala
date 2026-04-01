package com.szubd

import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{Vector => MLVector, Vectors => MLVectors, SparseVector=> MLSparseVector, DenseVector=> MLDenseVector}
import org.apache.spark.mllib.linalg.{Vector => MLLibVector, Vectors => MLLibVectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._

object ScalaParquetSVD {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("DirectSVDOnDataFrame")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._

    // 1. Load your data
    val df = spark.read.parquet("data/RedditTest/201201/data_filtered/")
    println(s"num of cols: ${df.columns.length}")
    println(s"cols names: ${df.columns.mkString("Array(", ", ", ")")}")
    df.printSchema()
    df.show(5, truncate = false)


    val selectedColumns = df.columns.slice(1, 10000)


   // 4. Convert to RDD[MLLibVector] for SVD computation.
    val vectorsRDD = df.select("features").rdd.map { row =>
      val mlVector = row.getAs[MLVector](0)
      MLLibVectors.fromML(mlVector)
    } // .cache() // Cache if will be used more than one time

    println(s"vectorsRDD.getClass: ${vectorsRDD.getClass}")

    // 4. Alternative: Direct conversion to MLlib sparse vectors
    val sparseVectorsRDD = df.select("features").rdd.map { row =>
      val mlVector = row.getAs[MLVector](0)
      // Force conversion to sparse in the most efficient way
      val sparse = mlVector match {
        case v: org.apache.spark.ml.linalg.SparseVector => v
        case v: org.apache.spark.ml.linalg.DenseVector => v.toSparse
      }
      MLLibVectors.sparse(sparse.size, sparse.indices, sparse.values)
    }//.cache()

    println(s"sparseVectorsRDD.getClass: ${sparseVectorsRDD.getClass}")


    val rowMatrix = new RowMatrix(vectorsRDD)
    println(s"rowMatrix.getClass: ${rowMatrix.getClass}")
    println(s"rowMatrix.numRows: ${rowMatrix.numRows()}")
    println(s"rowMatrix.numCols: ${rowMatrix.numCols()}")


      // 5. Compute SVD
      val k = 10 // number of components to keep
      val svd = rowMatrix.computeSVD(k, computeU = true)

      println("SVD::" + svd.toString)

      // 6. Get U matrix as DataFrame (converting back to ML vectors)
       val uMatrixSparse = svd.U.rows.map { mllibVector =>
      mllibVector.asML.toArray // Convert to ML vector and then to array
      }.toDF("u_vector")

      //Or Convert to ML dense vector directly
      //val uMatrixDense = svd.U.rows.map { mllibVector => mllibVector.toDense.toArray}.toDF("features")

    //or explode to multipel columsn of type double
    // More efficient: avoid intermediate vector column
//    val uMatrixExplodedDF = svd.U.rows.map { mllibVector =>
//      val array = mllibVector.toArray
//      // Create a Row with each component as separate field
//      (array(0), array(1), array(2)) // Add more fields up to k
//    }.toDF((1 to k).map(i => s"u_component_$i"): _*)


      uMatrixSparse.printSchema()
      uMatrixSparse.show(5, truncate = false)


      //7. Save results
      //resultDF.write.parquet("path/to/save/svd_results.parquet")



    spark.stop()
  }



}
