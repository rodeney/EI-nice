package com.szubd

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.{VectorUDT, Vector => MLVector, Vectors => MLVectors}
import org.apache.spark.mllib.linalg.{Vector => MLlibVector, Vectors => MLlibVectors}
import org.apache.spark.ml.linalg.SQLDataTypes.VectorType
import org.apache.spark.mllib.linalg.VectorUDT

object VectorTypeAnalyzer {

  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("VectorTypeAnalyzer")
      .master("local[*]")
      .getOrCreate()

    import spark.implicits._
    val inputPath = if (args.length > 0) args(0) else "data/R2012-01/"
    val featuresCol = if (args.length > 1) args(1) else "features"

    // Read Parquet file
    val df = spark.read.parquet(inputPath)
    val columnName = featuresCol

    // 1. Check schema and basic column type
    println("=" * 60)
    println("SCHEMA ANALYSIS")
    println("=" * 60)
    df.printSchema()

    val column = df.schema(columnName)
    println(s"\nColumn: $columnName")
    println(s"Data type: ${column.dataType}")
    println(s"Nullable: ${column.nullable}")

    // 2. Detect vector type
    println("\n" + "=" * 60)
    println("VECTOR TYPE DETECTION")
    println("=" * 60)

    detectVectorType(df, columnName)

    // 3. Analyze vector contents
    println("\n" + "=" * 60)
    println("VECTOR CONTENT ANALYSIS")
    println("=" * 60)

    analyzeVectorContents(df, columnName)

    // 4. Sample data with detailed info
    println("\n" + "=" * 60)
    println("SAMPLE VECTOR DATA")
    println("=" * 60)

    showSampleVectors(df, columnName)

    spark.stop()
  }

  def detectVectorType(df: DataFrame, columnName: String): Unit = {
    val columnType = df.schema(columnName).dataType

    columnType match {
      case _ if columnType.isInstanceOf[VectorUDT] =>
        println("✓ ML Vector (spark.ml.linalg.Vector)")

      case _ if columnType.isInstanceOf[org.apache.spark.mllib.linalg.VectorUDT] =>
        println("✓ MLlib Vector (spark.mllib.linalg.Vector)")

      case ArrayType(DoubleType, _) | ArrayType(FloatType, _) =>
        println("✓ Array of numeric values (could be converted to vector)")
        val elementType = columnType.asInstanceOf[ArrayType].elementType
        println(s"  Element type: $elementType")

      case _ =>
        println("✗ Not a recognized vector type")
        println(s"  Actual type: ${columnType.typeName}")
    }
  }

  def analyzeVectorContents(df: DataFrame, columnName: String): Unit = {
    val columnType = df.schema(columnName).dataType

    // Register UDFs for different vector types
    val getSizeUDF = udf { vector: Any =>
      vector match {
        case v: MLVector => v.size
        case v: MLlibVector => v.size
        case arr: Seq[_] => arr.length
        case null => -1
        case _ => -2 // Unknown type
      }
    }

    val getTypeUDF = udf { vector: Any =>
      vector match {
        case _: MLVector => "MLVector"
        case _: MLlibVector => "MLlibVector"
        case _: Seq[_] => "Array"
        case null => "Null"
        case _ => "Unknown"
      }
    }

    val getElementTypeUDF = udf { vector: Any =>
      vector match {
        case v: MLVector =>
          if (v.numNonzeros > 0) v(0).getClass.getSimpleName else "Empty"
        case v: MLlibVector =>
          if (v.size > 0) v(0).getClass.getSimpleName else "Empty"
        case arr: Seq[_] =>
          if (arr.nonEmpty) arr.head.getClass.getSimpleName else "Empty"
        case null => "Null"
        case _ => "Unknown"
      }
    }

    // Analyze vector statistics
    val analysisDF = df.select(
      col(columnName),
      getTypeUDF(col(columnName)).as("vector_type"),
      getSizeUDF(col(columnName)).as("vector_size"),
      getElementTypeUDF(col(columnName)).as("element_type")
    )

    println("Vector Type Distribution:")
    analysisDF.groupBy("vector_type").count().show()

    println("Vector Size Statistics:")
    analysisDF.describe("vector_size").show()

    println("Element Type Distribution:")
    analysisDF.groupBy("element_type").count().show()

    // Check for null values
    val nullCount = df.filter(col(columnName).isNull).count()
    println(s"Null values count: $nullCount")
  }

  def showSampleVectors(df: DataFrame, columnName: String): Unit = {
    // Sample some rows
    val sample = df.select(columnName).limit(10).collect()

    sample.zipWithIndex.foreach { case (row, index) =>
      println(s"\nSample #${index + 1}:")
      val vector = row.get(0)

      vector match {
        case v: MLVector =>
          println(s"Type: ML Vector (${v.getClass.getSimpleName})")
          println(s"Size: ${v.size}")
          println(s"Num non-zero: ${v.numNonzeros}")
          println(s"Values: ${v.toArray.take(10).mkString(", ")}${if (v.size > 10) "..." else ""}")
          println(s"Value types: ${v.toArray.map(_.getClass.getSimpleName).distinct.mkString(", ")}")

        case v: MLlibVector =>
          println(s"Type: MLlib Vector (${v.getClass.getSimpleName})")
          println(s"Size: ${v.size}")
          println(s"Values: ${v.toArray.take(10).mkString(", ")}${if (v.size > 10) "..." else ""}")
          println(s"Value types: ${v.toArray.map(_.getClass.getSimpleName).distinct.mkString(", ")}")

        case arr: Seq[_] =>
          println(s"Type: Array (${arr.getClass.getSimpleName})")
          println(s"Length: ${arr.length}")
          println(s"Values: ${arr.take(10).mkString(", ")}${if (arr.length > 10) "..." else ""}")
          println(s"Element types: ${arr.map(_.getClass.getSimpleName).distinct.mkString(", ")}")

        case null =>
          println("Type: Null")

        case other =>
          println(s"Type: Unknown (${other.getClass.getName})")
          println(s"Value: $other")
      }
    }
  }

  // Additional utility function to convert arrays to vectors if needed
  def convertArrayToMLVector(df: DataFrame, arrayColumn: String, outputColumn: String): DataFrame = {
    df.withColumn(outputColumn, udf { arr: Seq[Double] =>
      MLVectors.dense(arr.toArray)
    }.apply(col(arrayColumn)))
  }

  def convertArrayToMLlibVector(df: DataFrame, arrayColumn: String, outputColumn: String): DataFrame = {
    df.withColumn(outputColumn, udf { arr: Seq[Double] =>
      MLlibVectors.dense(arr.toArray)
    }.apply(col(arrayColumn)))
  }
}