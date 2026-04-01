package com.szubd;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.mllib.linalg.SingularValueDecomposition;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RspContext;
import org.apache.spark.sql.RspDataset;
import org.apache.spark.sql.SparkSession;

import javax.swing.*;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 *创建项目时在pom.xml文件引入依赖，将spark-rsp的包加入maven仓库，即可引入依赖
 *
 */
public class App_text
{
    public static void main( String[] args )
    {

        SparkConf conf = new SparkConf().setAppName("java RDD").setMaster("local[*]");
//        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder()
                .config(conf)
                .getOrCreate();
//        可以调用rspRead读取数据，但是需要将读取的数据类型进行转换
        RspDataset<Row> st = RspContext.SparkSessionFunc(spark).rspRead().csv("data/textdata_demo.csv");
//        将rspRDD转换为JavaRDD
        JavaRDD<Row> rdd = st.javaRDD().repartition(2);
//        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5);
//        JavaRDD<Integer> rdd = sc.parallelize(data);
//        System.out.println(rdd.collect());
        // 调用LO算子前需要 将每个分区的数据存入一个list
//        JavaRspRDD<List<List<Object>>> rspRDD = apply(rdd);
//        System.out.println(rspRDD.LO(convertList()).collect());

        JavaRspRDD<List<double[]>> listJavaRspRDD = apply_data(rdd);
        System.out.println(listJavaRspRDD.rdd().getNumPartitions());

        JavaRDD<Vector> vectorRDD = listJavaRspRDD.rdd()
                .flatMap(list -> list.stream()
                        .map(arr -> Vectors.dense(arr))
                        .iterator());
        RowMatrix mat = new RowMatrix(vectorRDD.rdd());
        int k = 10;
        SingularValueDecomposition<RowMatrix, Matrix> svd = mat.computeSVD(k, true, 1.0E-9d);
        RowMatrix U = svd.U(); // 左奇异向量
//        Vector s = svd.s(); // 奇异值
//        Matrix V = svd.V(); // 右奇异向量

        // U.rows() 返回的是 RDD<Vector>
        JavaRDD<Vector> uRows = U.rows().toJavaRDD();
        JavaRDD<List<double[]>> SVDRDD = uRows.map(vector -> {
            double[] arr = vector.toArray();        // Vector -> double[]
            List<double[]> list = new ArrayList<>();
            list.add(arr);                          // 包一层 List
            return list;
        });
        List<List<double[]>> collect = SVDRDD.collect();
        List<double[]> flatList = collect.stream()
                .flatMap(innerList -> innerList.stream())
                .collect(Collectors.toList());
        saveToCSV(flatList,"data/SVDdata1.csv");
//        System.out.println(s);
// 你可以 collect() 到 driver 打印（仅适用于数据不大）
//        List<Vector> uList = uRows.take(10);//collect();
//        for (Vector vec : uList) {
//            System.out.println(vec);
//        }

        RspDataset<Row> svddataread = RspContext.SparkSessionFunc(spark).rspRead().csv("data/SVDdata1.csv");
        JavaRDD<Row> svdrdd = svddataread.javaRDD().repartition(2);
        JavaRspRDD<List<double[]>> listJavaRspSVDRDD = apply_data(svdrdd);

        //JavaRDD<List<double[]>> subPartitions = listJavaRspRDD.getSubPartitions(2);
        //I_niceMOF参数：numberOfObservers表示观测点的数量，k表示合并聚类中心时的近邻数，percentage表示计算合并阈值的比例。
        // LO
        JavaRDD<List<double[]>> center = listJavaRspSVDRDD.LO(I_niceMOF(10, 5, 0.1));
        // 打印每个分区的内容
//
        JavaRspRDD<List<double[]>> listJavaRspRDDcenter = new JavaRspRDD<>(center);
        //I_niceMOGO参数：percentage表示计算合并阈值的比例。
        JavaRspRDD<List<double[]>> goResult = listJavaRspRDDcenter.GO(I_niceMOGO(spark));
        System.out.println("预测的簇数："+goResult.rdd().flatMap((List<double[]> list) -> list.iterator()) // flatten 成 double[]
                .collect().size());
//       goResult.rdd().repartition(1).foreachPartition(partitionIterator -> {
//            // 每个分区是一个迭代器
//            System.out.println("New Partition:");
//            while (partitionIterator.hasNext()) {
//                List<double[]> partitionData = partitionIterator.next();
//                // 打印该分区中的每个 List<double[]> 数据
//                for (double[] array : partitionData) {
//                    System.out.println(Arrays.toString(array));  // 打印每个 double[] 数组
//                }
//            }
//        });
        //可视化

        List<double[]> centermeage = goResult.rdd().flatMap((List<double[]> list) -> list.iterator()) // flatten 成 double[]
                .collect();
        List<double[]> datavector = listJavaRspSVDRDD.rdd().flatMap((List<double[]> list) -> list.iterator()) // flatten 成 double[]
                .collect();
        JFrame frame = new JFrame("2D Scatter Plot");
        ScatterPlot plot = new ScatterPlot(datavector, centermeage,4, 15); // 红点 15px，蓝点 7px
        frame.add(plot);
        frame.setSize(500, 500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

    /**
     * 将每个分区的数据存入一个list
     * 需要double就将Object类型改为Double
     * @param inputData
     * @return
     */
    public static JavaRspRDD<List<List<Object>>> apply (JavaRDD<Row> inputData) {

        JavaRDD<List<Object>> mappedRDD = inputData.map(row-> {
//            WrappedArray<Integer> wrappedArray = (WrappedArray<Integer>) row.get(0);
            List<Object> array = new ArrayList<>(row.size());
            for (int i = 0; i < row.size(); i++) {
                array.add(row.get(i));
            }
            return array;
        });
        JavaRDD<List<List<Object>>> glomRDD = mappedRDD.glom();
        return new JavaRspRDD<>(glomRDD);
    }

    // LO 阶段的函数需要进一步封装，不能像scala直接调用
    public static Function<List<List<Object>>, List<Object>> convertList () {
        return new Function<List<List<Object>>, List<Object>>() {
            @Override
            public List<Object> call(List<List<Object>> lists) throws Exception {
                return lists.stream().flatMap(List::stream).collect(Collectors.toList());
            }
        };
    }


    public static Function<JavaRDD<List<double[]>>, JavaRDD<List<double[]>>> I_niceMOGO(SparkSession spark) {
        return rdd -> {
            // 将所有分区的数据收集到一个大的列表中，并合并内部的 List<double[]> 到一个单一的 List<double[]>
            List<double[]> allData = rdd
                    .collect() // 收集所有分区的数据
                    .stream()  // 转换为流
                    .flatMap(List::stream)  // 将 List<List<double[]>> 展平为 List<double[]>
                    .collect(Collectors.toList()); // 收集到一个 List<double[]> 中
            // 使用 mergeDataGO 方法来合并数据
            List<double[]> mergedData = I_niceMO_Observers.mergeDataGO(allData);
            // 将 mergedData 转换为 List<List<double[]>> 的形式
            List<List<double[]>> wrappedData = new ArrayList<>();
            wrappedData.add(mergedData);
            // 从 SparkSession 获取 JavaSparkContext
            JavaSparkContext jsc = new JavaSparkContext(spark.sparkContext());
            // 使用 parallelize 将 wrappedData 转换成 JavaRDD<List<double[]>>
            JavaRDD<List<double[]>> javaRDD = jsc.parallelize(wrappedData);
            // 返回生成的 JavaRDD<List<double[]>>
            return javaRDD;
        };
    }

    public static Function<List<double[]>, List<double[]>> I_niceMOF(int numberOfObservers, int k, double percentage) {
        return new Function<List<double[]>, List<double[]>>() {
            @Override
            public List<double[]> call(List<double[]> dataVectors) throws Exception {
                List<double[]> result = I_niceMO_Observers.generateExpandedData(dataVectors, 20);
                List<double[]> centersAll = new ArrayList<>();

                for (int i = 0; i < numberOfObservers; i++) {
                    double[] observer = result.get(i);
                    List<double[]> centers = I_niceMO_Observers.GMMtest(dataVectors, observer, k);
                    centersAll.addAll(centers);
                }
                return I_niceMO_Observers.mergeData(centersAll, percentage);
            }
        };
    }

    public static JavaRspRDD<List<double[]>> apply_data(JavaRDD<Row> inputData) {
        // 将 Row -> double[]
        JavaRDD<double[]> vectorRDD = inputData.map(row -> {
            double[] vector = new double[row.size()];
            for (int i = 0; i < row.size(); i++) {
                Object val = row.get(i);
                if (val instanceof Number) {
                    vector[i] = ((Number) val).doubleValue();
                } else if (val instanceof String) {
                    vector[i] = Double.parseDouble(((String) val).trim());
                } else {
                    throw new IllegalArgumentException("Unsupported type at column " + i + ": " + val.getClass());
                }
            }
            return vector;
        });
        // 每个 partition -> List<double[]>
        JavaRDD<List<double[]>> glomRDD = vectorRDD.glom();
        return new JavaRspRDD<>(glomRDD);
    }

    public static void saveToCSV(List<double[]> centermerge, String filename) {
        try (FileWriter writer = new FileWriter(filename)) {
            for (double[] row : centermerge) {
                for (int i = 0; i < row.length; i++) {
                    writer.append(Double.toString(row[i]));
                    if (i < row.length - 1) {
                        writer.append(',');
                    }
                }
                writer.append('\n'); // 换行
            }
            System.out.println("保存成功: " + filename);
        } catch (IOException e) {
            System.err.println("写入CSV文件时出错: " + e.getMessage());
        }
    }
}
