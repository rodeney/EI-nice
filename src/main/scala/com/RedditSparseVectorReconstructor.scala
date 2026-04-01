import scala.collection.mutable.ListBuffer
import org.apache.spark.ml.linalg.{SparseVector, Vectors}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.broadcast.Broadcast
import java.time.LocalDate
import java.time.format.DateTimeFormatter

object RedditSparseVectorReconstructor {

  def main(args: Array[String]): Unit = {
    // 验证参数数量
    if (args.length < 4) {
      println("Usage: spark-submit --class RedditSparseVectorReconstructor --master yarn [jar] <startYear> <startMonth> <endYear> <endMonth> [batchSize]")
      System.exit(1)
    }

    // 解析命令行参数
    val startYear = args(0).toInt
    val startMonth = args(1).toInt
    val endYear = args(2).toInt
    val endMonth = args(3).toInt
    val batchSize = if (args.length > 4) args(4).toInt else 0 // 0表示不使用分批处理

    val spark = SparkSession.builder()
      .appName(s"Reddit Sparse Vector Reconstruction: ${startYear}-${startMonth} to ${endYear}-${endMonth}")
      .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
      .config("spark.driver.extraJavaOptions", "-Xss20m")
      .config("spark.executor.extraJavaOptions", "-Xss10m")
      .config("spark.debug.maxToStringFields", "100000")
      .config("spark.sql.optimizer.maxIterations", "20")
      .config("spark.sql.adaptive.enabled", "true")
      .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
      .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    // 生成年月列表
    val yearMonths = generateYearMonthList(startYear, startMonth, endYear, endMonth)
    val totalStartTime = System.currentTimeMillis()

    println(s"=== 启动稀疏向量重构 ===")
    println(s"• 时间范围: ${yearMonths.mkString(", ")}")

    yearMonths.foreach { ym =>
      val startTime = System.currentTimeMillis()
      val inputPath = s"hdfs://172.31.238.105:8020/data/RedditDataCleansing/R$ym/finalTable_wide"
      val outputPath = s"hdfs://172.31.238.105:8020/data/RedditData_sparse_vectors/R$ym"

      println(s"\n=== 处理 $ym 月份数据 ===")
      val success = if (batchSize > 0) {
        println(s"[MODE] 使用分批处理模式，批大小: $batchSize")
        reconstructSparseVectorsBatched(spark, inputPath, outputPath, ym, batchSize)
      } else {
        println(s"[MODE] 使用标准处理模式")
        reconstructSparseVectors(spark, inputPath, outputPath, ym)
      }

      val elapsed = (System.currentTimeMillis() - startTime) / 1000.0
      val status = if (success) "成功" else "失败"
      println(f"[$ym] 重构完成! 耗时: $elapsed%.2f 秒, 状态: $status")
    }

    val totalTime = (System.currentTimeMillis() - totalStartTime) / 1000.0
    println(f"\n=== 所有任务完成 ===")
    println(f"总耗时: $totalTime%.2f 秒")

    spark.stop()
  }

  def generateYearMonthList(startYear: Int, startMonth: Int, endYear: Int, endMonth: Int): List[String] = {
    val start = LocalDate.of(startYear, startMonth, 1)
    val end = LocalDate.of(endYear, endMonth, 1)

    var current = start
    val list = ListBuffer[String]()
    val formatter = DateTimeFormatter.ofPattern("yyyy-MM")

    while (!current.isAfter(end)) {
      list += current.format(formatter)
      current = current.plusMonths(1)
    }

    list.toList
  }

  def loadDictionary(spark: SparkSession): (Map[String, Int], Int) = {
    import spark.implicits._

    val dictPath = "hdfs://172.31.238.105:8020/data/RedditData_column_name_coding/dictionaries/Cleaned_column_names_with_coding.parquet"

    println(s"[DICT] 加载字典文件: $dictPath")
    val dictDF = spark.read.parquet(dictPath)

    // 显示字典结构
    println(s"[DICT] 字典文件结构:")
    dictDF.printSchema()
    println(s"[DICT] 字典样本数据:")
    dictDF.show(10, false)

    // 创建单词到连续索引的映射，而不使用原始的T编码
    val dictData = dictDF.select("original_word", "code").collect().map { row =>
      val word = row.getString(0)
      val code = row.getString(1)
      val originalIndex = code.substring(1).toInt // 原始索引（仅用于排序）
      (word, originalIndex)
    }.sortBy(_._2) // 按原始索引排序

    // 创建连续的索引映射：0, 1, 2, 3, ...
    val wordToIndexMap = dictData.zipWithIndex.map { case ((word, _), newIndex) =>
      (word, newIndex)
    }.toMap

    val vocabularySize = wordToIndexMap.size // 向量维度为字典总数

    // 统计原始索引范围
    val originalIndices = dictData.map(_._2)
    val minOriginalIndex = originalIndices.min
    val maxOriginalIndex = originalIndices.max

    println(s"[DICT] 字典统计:")
    println(s"  - 词汇总数: ${wordToIndexMap.size}")
    println(s"  - 向量维度: $vocabularySize")
    println(s"  - 原始索引范围: $minOriginalIndex - $maxOriginalIndex")
    println(s"  - 重新映射到: 0 - ${vocabularySize-1}")

    // 显示一些映射示例
    println(s"[DICT] 映射示例:")
    wordToIndexMap.take(10).foreach { case (word, index) =>
      println(s"  '$word' -> $index")
    }

    (wordToIndexMap, vocabularySize)
  }

  def reconstructSparseVectors(
                                spark: SparkSession,
                                inputPath: String,
                                outputPath: String,
                                yearMonth: String
                              ): Boolean = {
    import spark.implicits._

    try {
      // 1. 加载字典
      val (wordToIndexMap, vocabularySize) = loadDictionary(spark)
      val wordToIndexBroadcast: Broadcast[Map[String, Int]] = spark.sparkContext.broadcast(wordToIndexMap)

      // 2. 获取所有块目录
      val hdfs = org.apache.hadoop.fs.FileSystem.get(spark.sparkContext.hadoopConfiguration)
      val inputPathObj = new org.apache.hadoop.fs.Path(inputPath)

      if (!hdfs.exists(inputPathObj)) {
        println(s"[ERROR] 输入路径不存在: $inputPath")
        return false
      }

      val chunkDirs = hdfs.listStatus(inputPathObj)
        .filter(_.isDirectory)
        .map(_.getPath.getName)
        .filter(_.startsWith("chunk_"))
        .sorted
        .toList

      if (chunkDirs.isEmpty) {
        println(s"[ERROR] 未找到任何块目录在: $inputPath")
        return false
      }

      println(s"[DEBUG] 发现 ${chunkDirs.length} 个块目录: ${chunkDirs.mkString(", ")}")

      // 3. 为每个块创建部分稀疏向量，避免合并操作
      val chunkSparseVectors = chunkDirs.zipWithIndex.map { case (chunkDir, idx) =>
        val chunkPath = s"$inputPath/$chunkDir"
        println(s"[CHUNK] 处理块 $idx: $chunkPath")

        val chunkDF = spark.read.parquet(chunkPath)
        val chunkFeatures = chunkDF.columns.filter(_ != "submission_id")

        // 过滤出在字典中存在的特征列，并按字典索引排序
        val validChunkFeatures = chunkFeatures
          .filter(wordToIndexMap.contains)
          .sortBy(wordToIndexMap(_)) // 按字典索引排序

        val invalidFeatures = chunkFeatures.filterNot(wordToIndexMap.contains)

        println(s"[CHUNK] 块 $idx:")
        println(s"  - 总特征数: ${chunkFeatures.length}")
        println(s"  - 有效特征数: ${validChunkFeatures.length}")
        println(s"  - 无效特征数: ${invalidFeatures.length}")

        // 为当前块创建部分稀疏向量的UDF
        val createChunkSparseVectorUDF = udf { row: org.apache.spark.sql.Row =>
          val wordToIndex = wordToIndexBroadcast.value
          val indices = ListBuffer[Int]()
          val values = ListBuffer[Double]()

          // 跳过submission_id列(索引0)，处理所有特征列
          for (i <- 1 until row.length) {
            val columnName = row.schema.fields(i).name

            wordToIndex.get(columnName) match {
              case Some(dictIndex) =>
                val value = row.get(i) match {
                  case f: Float => f.toDouble
                  case d: Double => d
                  case i: Int => i.toDouble
                  case l: Long => l.toDouble
                  case null => 0.0
                  case other => 0.0
                }

                if (value != 0.0) {
                  indices += dictIndex
                  values += value
                }
              case None =>
              // 忽略不在字典中的列
            }
          }

          // 由于列已按索引排序，indices应该已经有序
          (indices.toArray, values.toArray)
        }

        // 只选择有效的特征列并创建部分向量
        val filteredChunkDF = chunkDF.select(
          col("submission_id") +: validChunkFeatures.map(col): _*
        )

        val chunkVectorDF = filteredChunkDF
          .withColumn("chunk_vector", createChunkSparseVectorUDF(struct(filteredChunkDF.columns.map(col): _*)))
          .select("submission_id", "chunk_vector")
          .persist(StorageLevel.MEMORY_AND_DISK_SER)

        println(s"[CHUNK] 块 $idx 向量创建完成")
        chunkVectorDF
      }

      // 4. 合并所有块的部分向量
      println("[MERGE] 合并块向量...")
      var mergedVectorDF = chunkSparseVectors.head

      chunkSparseVectors.tail.zipWithIndex.foreach { case (chunkVectorDF, idx) =>
        println(s"[MERGE] 合并块 ${idx + 1}")
        mergedVectorDF = mergedVectorDF.join(chunkVectorDF, "submission_id")
      }

      // 5. 创建最终稀疏向量的UDF
      val combineVectorsUDF = udf { row: org.apache.spark.sql.Row =>
        val allIndices = ListBuffer[Int]()
        val allValues = ListBuffer[Double]()

        // 收集所有块的indices和values
        for (i <- 1 until row.length) { // 跳过submission_id
          val chunkVector = row.get(i).asInstanceOf[org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema]
          val indices = chunkVector.get(0).asInstanceOf[scala.collection.mutable.WrappedArray[Int]].toArray
          val values = chunkVector.get(1).asInstanceOf[scala.collection.mutable.WrappedArray[Double]].toArray

          allIndices ++= indices
          allValues ++= values
        }

        // 必须排序，因为不同块的索引可能交错
        if (allIndices.isEmpty) {
          // 空向量
          Vectors.sparse(vocabularySize, Array.empty[Int], Array.empty[Double]).asInstanceOf[SparseVector]
        } else {
          // 按索引排序并合并重复索引（如果有的话）
          val sortedPairs = allIndices.zip(allValues).sortBy(_._1)

          // 合并重复索引
          val mergedMap = scala.collection.mutable.Map[Int, Double]()
          sortedPairs.foreach { case (index, value) =>
            mergedMap(index) = mergedMap.getOrElse(index, 0.0) + value
          }

          // 转换为排序的数组
          val finalSorted = mergedMap.toSeq.sortBy(_._1)
          val finalIndices = finalSorted.map(_._1).toArray
          val finalValues = finalSorted.map(_._2).toArray

          // 验证索引是严格递增的
          if (finalIndices.length > 1) {
            for (i <- 1 until finalIndices.length) {
              if (finalIndices(i) <= finalIndices(i-1)) {
                throw new IllegalArgumentException(s"索引不是严格递增的: ${finalIndices(i-1)} >= ${finalIndices(i)}")
              }
            }
          }

          Vectors.sparse(vocabularySize, finalIndices, finalValues).asInstanceOf[SparseVector]
        }
      }

      // 6. 应用最终合并UDF
      println("[DEBUG] 创建最终稀疏向量...")
      val finalSparseVectorDF = mergedVectorDF
        .withColumn("tfidf_vector", combineVectorsUDF(struct(mergedVectorDF.columns.map(col): _*)))
        .select("submission_id", "tfidf_vector")
        .persist(StorageLevel.MEMORY_AND_DISK_SER)

      // 清理中间结果
      chunkSparseVectors.foreach(_.unpersist())

      // 7. 显示统计信息
      val totalValidFeatures = chunkSparseVectors.length match {
        case 0 => 0
        case _ =>
          // 重新计算总的有效特征数
          chunkDirs.map { chunkDir =>
            val chunkPath = s"$inputPath/$chunkDir"
            val chunkDF = spark.read.parquet(chunkPath)
            chunkDF.columns.filter(_ != "submission_id").count(wordToIndexMap.contains)
          }.sum
      }

      println(s"[DEBUG] 合并统计:")
      println(s"  - 有效特征数: $totalValidFeatures")
      println(s"  - 字典词汇表大小: ${wordToIndexMap.size}")
      println(s"  - 向量维度: $vocabularySize")
      println(s"  - 合并后数据行数: ${finalSparseVectorDF.count()}")

      val sampleVectors = finalSparseVectorDF.take(5)
      println(s"[SAMPLE] 前5个稀疏向量统计:")
      sampleVectors.zipWithIndex.foreach { case (row, idx) =>
        val vector = row.getAs[SparseVector]("tfidf_vector")
        val nonZeros = vector.numNonzeros
        val density = nonZeros.toDouble / vector.size * 100
        println(f"  样本${idx+1}: 非零元素=$nonZeros, 密度=${density}%.4f%%, 向量大小=${vector.size}")
      }

      // 8. 保存结果
      println(s"[SAVING] 保存稀疏向量到: $outputPath")
      finalSparseVectorDF
        .repartition(200, col("submission_id"))
        .write
        .mode("overwrite")
        .parquet(outputPath)

      // 9. 验证输出
      val savedDF = spark.read.parquet(outputPath)
      val savedCount = savedDF.count()
      println(s"[VERIFY] 保存的记录数: $savedCount")

      // 清理
      finalSparseVectorDF.unpersist()
      wordToIndexBroadcast.unpersist()

      println(s"[SUCCESS] 成功重构 $yearMonth 月份的稀疏向量")
      true

    } catch {
      case e: Exception =>
        println(s"[ERROR] 处理 $yearMonth 时出错: ${e.getMessage}")
        e.printStackTrace()
        false
    }
  }

  // 替代方案：如果内存不足，使用分批处理
  def reconstructSparseVectorsBatched(
                                       spark: SparkSession,
                                       inputPath: String,
                                       outputPath: String,
                                       yearMonth: String,
                                       batchSize: Int = 10000
                                     ): Boolean = {
    import spark.implicits._

    try {
      // 1. 加载字典
      val (wordToIndexMap, vocabularySize) = loadDictionary(spark)
      val wordToIndexBroadcast: Broadcast[Map[String, Int]] = spark.sparkContext.broadcast(wordToIndexMap)

      // 2. 获取所有块目录
      val hdfs = org.apache.hadoop.fs.FileSystem.get(spark.sparkContext.hadoopConfiguration)
      val chunkDirs = hdfs.listStatus(new org.apache.hadoop.fs.Path(inputPath))
        .filter(_.isDirectory)
        .map(_.getPath.getName)
        .filter(_.startsWith("chunk_"))
        .sorted
        .toList

      if (chunkDirs.isEmpty) {
        println(s"[ERROR] 未找到任何块目录")
        return false
      }

      // 3. 首先获取所有submission_id
      val firstChunkPath = s"$inputPath/${chunkDirs.head}"
      val submissionIds = spark.read.parquet(firstChunkPath)
        .select("submission_id")
        .rdd
        .map(_.getString(0))
        .collect()
        .toList

      println(s"[DEBUG] 总记录数: ${submissionIds.length}")

      // 4. 分批处理
      val batches = submissionIds.grouped(batchSize).toList
      println(s"[DEBUG] 分为 ${batches.length} 批，每批 $batchSize 条记录")

      batches.zipWithIndex.foreach { case (batch, batchIdx) =>
        println(s"[BATCH] 处理第 ${batchIdx + 1} 批")

        // 读取当前批次的所有块数据
        var batchDF: DataFrame = null

        chunkDirs.foreach { chunkDir =>
          val chunkPath = s"$inputPath/$chunkDir"
          val chunkDF = spark.read.parquet(chunkPath)
            .filter(col("submission_id").isin(batch: _*))

          // 过滤出在字典中存在的列，并按字典索引排序
          val chunkFeatures = chunkDF.columns.filter(_ != "submission_id")
          val validChunkFeatures = chunkFeatures
            .filter(wordToIndexMap.contains)
            .sortBy(wordToIndexMap(_)) // 按字典索引排序

          val filteredChunkDF = chunkDF.select(
            col("submission_id") +: validChunkFeatures.map(col): _*
          )

          if (batchDF == null) {
            batchDF = filteredChunkDF
          } else {
            batchDF = batchDF.join(filteredChunkDF, "submission_id")
          }
        }

        // 创建稀疏向量UDF
        val createSparseVectorUDF = udf { row: org.apache.spark.sql.Row =>
          val wordToIndex = wordToIndexBroadcast.value
          val indices = ListBuffer[Int]()
          val values = ListBuffer[Double]()

          for (i <- 1 until row.length) {
            val columnName = row.schema.fields(i).name

            wordToIndex.get(columnName) match {
              case Some(dictIndex) =>
                val value = row.get(i) match {
                  case f: Float => f.toDouble
                  case d: Double => d
                  case _ => 0.0
                }

                if (value != 0.0) {
                  indices += dictIndex
                  values += value
                }
              case None =>
              // 忽略不在字典中的列
            }
          }

          // 创建稀疏向量前需要按索引排序
          val sortedPairs = indices.zip(values).sortBy(_._1)
          val sortedIndices = sortedPairs.map(_._1).toArray
          val sortedValues = sortedPairs.map(_._2).toArray

          Vectors.sparse(vocabularySize, sortedIndices, sortedValues).asInstanceOf[SparseVector]
        }

        val batchSparseDF = batchDF
          .withColumn("tfidf_vector", createSparseVectorUDF(struct(batchDF.columns.map(col): _*)))
          .select("submission_id", "tfidf_vector")

        // 保存批次
        val batchOutputPath = s"$outputPath/batch_$batchIdx"
        batchSparseDF
          .write
          .mode("overwrite")
          .parquet(batchOutputPath)

        println(s"[BATCH] 第 ${batchIdx + 1} 批处理完成")
      }

      // 5. 合并所有批次
      println("[MERGE] 合并所有批次...")
      val allBatchPaths = (0 until batches.length).map(i => s"$outputPath/batch_$i")
      val finalDF = spark.read.parquet(allBatchPaths: _*)

      finalDF
        .repartition(200, col("submission_id"))
        .write
        .mode("overwrite")
        .parquet(s"$outputPath/final")

      // 6. 清理临时批次文件
      allBatchPaths.foreach { path =>
        hdfs.delete(new org.apache.hadoop.fs.Path(path), true)
      }

      // 清理广播变量
      wordToIndexBroadcast.unpersist()

      println(s"[SUCCESS] 分批重构完成: $yearMonth")
      true

    } catch {
      case e: Exception =>
        println(s"[ERROR] 分批处理失败: ${e.getMessage}")
        e.printStackTrace()
        false
    }
  }
}