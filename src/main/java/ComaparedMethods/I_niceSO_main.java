package ComaparedMethods;

import com.szubd.JavaRspRDD;
import com.szubd.KMeans;
import com.szubd.PurityCalculator;
import com.szubd.ScatterPlot;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.*;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

/**
 *创建项目时在pom.xml文件引入依赖，将spark-rsp的包加入maven仓库，即可引入依赖
 *
 */
public class I_niceSO_main
{
    public static void main( String[] args )
    {
        long startTime = System.currentTimeMillis(); // 开始时间（毫秒）
        SparkConf conf = new SparkConf().setAppName("java RDD").setMaster("local[*]");
//        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder()
                .config(conf)
                .getOrCreate();
        List<Double> ClusterRuns = new ArrayList<>();
        List<Double> PurityRuns = new ArrayList<>();
        List<Double> ariRuns = new ArrayList<>();
        int repeatTimes = 5;
        for (int iterAll = 0; iterAll < repeatTimes; iterAll++){
//        可以调用rspRead读取数据，但是需要将读取的数据类型进行转换
        Dataset<Row> st = spark.read().csv("data/letter.csv");
        //保证数据的顺序
        Dataset<Row> dataSVD = spark.read().csv("data/letter.csv");
        JavaRspRDD<List<double[]>> listJavaRspRDD1 = apply_data(dataSVD.javaRDD());
        System.out.println("分区数"+listJavaRspRDD1.rdd().getNumPartitions());
        List<double[]> collect = listJavaRspRDD1.rdd().flatMap((List<double[]> list) -> list.iterator()) // flatten 成 double[]
                .collect();
//        使用java spark读取数据
//        Dataset<Row> st = spark.read().option("header",true).option("inferSchema", "true").csv("file:///C://li//program//workplace//test1//data//train2.csv");
        st.show(10);
//        将rspRDD转换为JavaRDD
//        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5);
//        JavaRDD<Integer> rdd = sc.parallelize(data);
//        System.out.println(rdd.collect());
        // 调用LO算子前需要 将每个分区的数据存入一个list
//        JavaRspRDD<List<List<Object>>> rspRDD = apply(rdd);
//        System.out.println(rspRDD.LO(convertList()).collect());

        List<double[]> dataVectors = apply_data(st.javaRDD()).rdd().flatMap((List<double[]> list) -> list.iterator()) // flatten 成 double[]
                .collect();

        List<double[]> result = ComaparedMethods.I_niceMO_Observers.uniformSample(dataVectors, 20);
        List<double[]> centersMax = new ArrayList<>();
        int numberOfObservers = 10;
        int k = 10;
        int flag = 0;
        for (int i = 0; i < numberOfObservers; i++) {
            double[] observer = result.get(i);
            List<double[]> centers = ComaparedMethods.I_niceMO_Observers.GMMtest(dataVectors, observer, k);
            if(i>0&& centers.size()>flag){
                centersMax = new ArrayList<>();
                centersMax.addAll(centers);
                flag = centers.size();
            }
        }

        ClusterRuns.add((double) centersMax.size());


        //可视化

//        List<double[]> centermeage = goResult.rdd().flatMap((List<double[]> list) -> list.iterator()) // flatten 成 double[]
//                .collect();
//        saveToCSV(centermeage,"data/centerallU100.csv");
//        List<double[]> datavector = listJavaRspRDD1.rdd().flatMap((List<double[]> list) -> list.iterator()) // flatten 成 double[]
//                .collect();
//        JFrame frame = new JFrame("2D Scatter Plot");
//        ScatterPlot plot = new ScatterPlot(datavector, centersMax,4, 15); // 红点 15px，蓝点 7px
//        frame.add(plot);
//        frame.setSize(500, 500);
//        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame.setVisible(true);
//
        //聚类结果
        // 将合并后的聚类中心作为初始聚类中心，用kmeans的方法获得最终的聚类结果
        KMeans kmeans = new KMeans(centersMax);
        int[] labels = kmeans.fit(collect);
//        System.out.println("Array elements:");
//        for (int i = 0; i < labels.length; i++) {
//            System.out.print(labels[i] + " ");
//        }

            try {
                int[] labels_true = Functions.readLabels("data/letterlabel.csv");
//            System.out.println(Arrays.toString(labels));
                double purity = com.szubd.PurityCalculator.calculatePurity(labels_true, labels);
                System.out.println("聚类纯度 (Purity): " + purity);
                PurityRuns.add(purity);

                double ari = PurityCalculator.computeARI(labels_true, labels);
                System.out.println("调整兰德系数(Adjusted Rand Index (ARI)): " + ari);
                ariRuns.add(ari);

            } catch (IOException e) {
                System.err.println("读取文件失败: " + e.getMessage());
            }

    }

    long endTime = System.currentTimeMillis();   // 结束时间
    long durationInSeconds = (endTime - startTime) / 1000; // 秒

        System.out.println("程序运行时间：" + durationInSeconds/repeatTimes + " 秒");

        double meanCluster = Functions.computeMean(ClusterRuns);
        double stdCluster = Functions.computeStdDev(ClusterRuns, meanCluster);
        System.out.println("Cluster list: " + ClusterRuns);
        System.out.printf("Cluster: %.4f\n", meanCluster);
        System.out.printf("STD Cluster: %.6f\n", stdCluster);

        double meanPurity = Functions.computeMean(PurityRuns);
        double stdPurity = Functions.computeStdDev(PurityRuns, meanPurity);
        System.out.println("Purity list: " + PurityRuns);
        System.out.printf("Purity: %.4f\n", meanPurity);
        System.out.printf("STD Purity: %.6f\n", stdPurity);

        double meanARI = Functions.computeMean(ariRuns);
        double stdARI = Functions.computeStdDev(ariRuns, meanARI);
        System.out.println("ARI list: " + ariRuns);
        System.out.printf("ARI: %.4f\n", meanARI);
        System.out.printf("STD ARI: %.6f\n", stdARI);

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
