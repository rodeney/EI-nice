package com.szubd;

//数据太少了，没有用repartition
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.linalg.SparseVector;
import org.apache.spark.sql.Row;
import java.util.Arrays;
import java.util.List;

import javax.swing.*;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 *创建项目时在pom.xml文件引入依赖，将spark-rsp的包加入maven仓库，即可引入依赖
 *
 */
public class INiceLOGO
{
    private static final Logger log = LoggerFactory.getLogger(INiceLOGO.class);

    public static void main(String[] args )
    {

        SparkConf conf = new SparkConf().setAppName("java RDD").setMaster("local[*]");

        SparkSession spark = SparkSession.builder()
                .config(conf)
                .getOrCreate();

//      可以调用rspRead读取数据，但是需要将读取的数据类型进行转换
        RspDataset<Row> st = RspContext.SparkSessionFunc(spark).rspRead().parquet("data/dtm1000_filtered_newVectorSize/");//csv("data/DS1.csv");
        st.printSchema();
        System.out.println(st.count());

        JavaRspRDD<List<double[]>> listJavaRspRDD = apply_data(st.javaRDD());

        JavaRDD<List<double[]>> listJavaRDD = SparseVectorConverter(st.javaRDD());

        System.out.println(listJavaRDD.count());
        //System.out.println(listJavaRDD.rdd().take(3));


        //listJavaRspRDD.getSubPartitions(5)

        System.out.println(listJavaRspRDD.rdd().getNumPartitions());
        System.out.println(listJavaRspRDD.rdd().take(1));

//        JavaRDD<List<double[]>> centers = listJavaRspRDD.LO(part -> {
//            // Convert Java List<Vector> to required input format for INice.fitMO()
//            List<double[]> converted = new ArrayList<>();
//            // Convert Vector to double[]
//            converted.addAll(part);
//            return INice.fitMO(converted, 2, 5);
//        });

        JavaRDD<List<double[]>> centers1 = listJavaRspRDD.LO(
                part -> INice.fitMO(part, 2, 5)
        );
        System.out.printf("Centers Count:: "+centers1.collect());
        //List<List<double[]>> centersList = centers1.collect().getClass();

       List<double[]> allCenters = centers1.collect().stream()
                .flatMap(List::stream)
                .collect(Collectors.toList());


//  Print all centers
        System.out.println("\nIdentified Cluster Centers:");
        for (int i = 0; i < allCenters.size(); i++) {
            System.out.printf("Center %2d: %s%n",
                    i+1,
                    Arrays.toString(allCenters.get(i)));
        }


//merge
        List<double[]>  mergedClusters = INice.mergeData(allCenters, 0.15);


        System.out.printf("Final Centers:: "+ mergedClusters);

        System.out.println("\n Cluster Centers (after distance-based merge):");
        for (int i = 0; i < mergedClusters.size(); i++) {
            System.out.printf("Center %2d: %s%n",
                    i+1,
                    Arrays.toString(mergedClusters.get(i)));
        }


    }


        public static JavaRDD<List<double[]>> SparseVectorConverter(JavaRDD<Row> sparseVectorRDD) {
            return sparseVectorRDD.map(row -> {
                // Get the SparseVector from the Row (assuming it's the first column)
                SparseVector sparseVector = (SparseVector) row.get(0);

                // Convert SparseVector to dense double[]
                double[] denseArray = new double[sparseVector.size()];

                // Initialize all values to 0.0
                Arrays.fill(denseArray, 0.0);

                // Fill in the non-zero values
                int[] indices = sparseVector.indices();
                double[] values = sparseVector.values();
                for (int i = 0; i < indices.length; i++) {
                    denseArray[indices[i]] = values[i];
                }

                // Wrap in a List (though you might just want the double[])
                return Arrays.asList(denseArray);
            });
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
