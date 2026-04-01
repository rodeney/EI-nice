package com.szubd;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.sql.*;

import javax.swing.*;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

import java.io.*;

import java.util.stream.Collectors;

/**
 *创建项目时在pom.xml文件引入依赖，将spark-rsp的包加入maven仓库，即可引入依赖
 *
 */
public class App {
    public static void main(String[] args) {
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
        for (int iterAll = 0; iterAll < repeatTimes; iterAll++) {
//        可以调用rspRead读取数据，但是需要将读取的数据类型进行转换
            RspDataset<Row> st = RspContext.SparkSessionFunc(spark).rspRead().csv("data/letter.csv");
            //保证数据的顺序
            Dataset<Row> dataSVD = spark.read().csv("data/letter.csv");
            JavaRspRDD<List<double[]>> listJavaRspRDD1 = apply_data(dataSVD.javaRDD());
            System.out.println("分区数" + listJavaRspRDD1.rdd().getNumPartitions());
            List<double[]> collect = listJavaRspRDD1.rdd().flatMap((List<double[]> list) -> list.iterator()) // flatten 成 double[]
                    .collect();


//        使用java spark读取数据
//        Dataset<Row> st = spark.read().option("header",true).option("inferSchema", "true").csv("file:///C://li//program//workplace//test1//data//train2.csv");
            st.show(10);
//        将rspRDD转换为JavaRDD
            //////方式一：数据分区比较少的情况，直接计算所有分区的数据//////
//            JavaRDD<Row> rdd = st.javaRDD().repartition(5);
            //////方式二：数据分区数比较多的情况下，选择一部分分区做计算//////
//            // 随机选择5个分区索引
            int numberofPartition = 20;
            JavaRDD<Row> rddoriginal = st.javaRDD().repartition(numberofPartition);
            Random rand = new Random();
            Set<Integer> selectedPartitions = new HashSet<>();
            while (selectedPartitions.size() < 5) {
                selectedPartitions.add(rand.nextInt(numberofPartition)); // 0 到 9 之间的分区索引
            }
            System.out.println("Selected partitions: " + selectedPartitions);
            // 从这些分区中抽取数据
            JavaRDD<Row> sampledRDD = rddoriginal.mapPartitionsWithIndex(
                    new Function2<Integer, Iterator<Row>, Iterator<Row>>() {
                        @Override
                        public Iterator<Row> call(Integer index, Iterator<Row> iterator) {
                            if (selectedPartitions.contains(index)) {
                                return iterator;
                            } else {
                                return Arrays.<Row>asList().iterator(); // 空迭代器
                            }
                        }
                    }, true);
            JavaRDD<Row> rdd = sampledRDD.repartition(5);
//            ////// 数据分区数比较多的情况下，选择一部分分区做计算//////

//        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5);
//        JavaRDD<Integer> rdd = sc.parallelize(data);
//        System.out.println(rdd.collect());
            // 调用LO算子前需要 将每个分区的数据存入一个list
//        JavaRspRDD<List<List<Object>>> rspRDD = apply(rdd);
//        System.out.println(rspRDD.LO(convertList()).collect());

            JavaRspRDD<List<double[]>> listJavaRspRDD = apply_data(rdd);
            System.out.println(listJavaRspRDD.rdd().getNumPartitions());

//            List<double[]> data = listJavaRspRDD.rdd().flatMap((List<double[]> list) -> list.iterator()).collect();
//            List<double[]> result = I_niceMO_Observers.uniformSample(data, 20);
//            saveToCSV(result, "data/centerallU100.csv");

            //JavaRDD<List<double[]>> subPartitions = listJavaRspRDD.getSubPartitions(2);
            //I_niceMOF参数：numberOfObservers表示观测点的数量，k表示合并聚类中心时的近邻数，percentage表示计算合并阈值的比例。
            // LO
            JavaRDD<List<double[]>> center = listJavaRspRDD.LO(I_niceMOF(10, 10));
            // 打印每个分区的内容


            JavaRspRDD<List<double[]>> listJavaRspRDDcenter = new JavaRspRDD<>(center);
            //I_niceMOGO参数：percentage表示计算合并阈值的比例。
            JavaRspRDD<List<double[]>> goResult = listJavaRspRDDcenter.GO(I_niceMOGO(spark,0.3));
            System.out.println("预测的簇数：" + goResult.rdd().flatMap((List<double[]> list) -> list.iterator()) // flatten 成 double[]
                    .collect().size());

            ClusterRuns.add((double) goResult.rdd().flatMap((List<double[]> list) -> list.iterator()) // flatten 成 double[]
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
////        saveToCSV(centermeage, "data/centerallU100.csv");
//            List<double[]> datavector = listJavaRspRDD.rdd().flatMap((List<double[]> list) -> list.iterator()) // flatten 成 double[]
//                    .collect();
//            JFrame frame = new JFrame("2D Scatter Plot");
//            ScatterPlot plot = new ScatterPlot(datavector, centermeage, 4, 15); // 红点 15px，蓝点 7px
//            frame.add(plot);
//            frame.setSize(500, 500);
//            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//            frame.setVisible(true);

//        //聚类结果
            // 将合并后的聚类中心作为初始聚类中心，用kmeans的方法获得最终的聚类结果
            KMeans kmeans = new KMeans(centermeage);
            int[] labels = kmeans.fit(collect);
            System.out.println("Array elements:");
            for (int i = 0; i < labels.length; i++) {
                System.out.print(labels[i] + " ");
            }

            try {
                int[] labels_true = readLabels("data/letterlabel.csv");
//            System.out.println(Arrays.toString(labels));
                double purity = PurityCalculator.calculatePurity(labels_true, labels);
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

        System.out.println("程序运行时间：" + durationInSeconds / repeatTimes + " 秒");

        double meanCluster = computeMean(ClusterRuns);
        double stdCluster = computeStdDev(ClusterRuns, meanCluster);

        System.out.printf("Cluster: %.4f\n", meanCluster);
        System.out.printf("STD Cluster: %.6f\n", stdCluster);

        double meanPurity = computeMean(PurityRuns);
        double stdPurity = computeStdDev(PurityRuns, meanPurity);

        System.out.printf("Purity: %.4f\n", meanPurity);
        System.out.printf("STD Purity: %.6f\n", stdPurity);

        double meanARI = computeMean(ariRuns);
        double stdARI = computeStdDev(ariRuns, meanARI);

        System.out.printf("ARI: %.4f\n", meanARI);
        System.out.printf("STD ARI: %.6f\n", stdARI);
    }



    // === 平均值计算 ===
    public static double computeMean(List<Double> values) {
        double sum = 0;
        for (double v : values) {
            sum += v;
        }
        return sum / values.size();
    }

    // === 标准差计算 ===
    public static double computeStdDev(List<Double> values, double mean) {
        double sumSq = 0;
        for (double v : values) {
            sumSq += Math.pow(v - mean, 2);
        }
        return Math.sqrt(sumSq / values.size());
    }

    public static int[] readLabels(String filePath) throws IOException {
        List<Integer> labelList = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty()) {
                    labelList.add(Integer.parseInt(line));
                }
            }
        }

        // 转换为 int[]
        return labelList.stream().mapToInt(i -> i).toArray();
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


    public static Function<JavaRDD<List<double[]>>, JavaRDD<List<double[]>>> I_niceMOGO(SparkSession spark,double percentage) {
        return rdd -> {
            // 将所有分区的数据收集到一个大的列表中，并合并内部的 List<double[]> 到一个单一的 List<double[]>
            List<double[]> allData = rdd
                    .collect() // 收集所有分区的数据
                    .stream()  // 转换为流
                    .flatMap(List::stream)  // 将 List<List<double[]>> 展平为 List<double[]>
                    .collect(Collectors.toList()); // 收集到一个 List<double[]> 中
            // 使用 mergeDataGO 方法来合并数据
            List<double[]> mergedData = I_niceMO_Observers.mergeDataGOMO(allData,percentage);
//            List<double[]> mergedData = I_niceMO_Observers.mergeDataGO(allData);
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

    public static Function<List<double[]>, List<double[]>> I_niceMOF(int numberOfObservers, int k) {
        return new Function<List<double[]>, List<double[]>>() {
            @Override
            public List<double[]> call(List<double[]> dataVectors) throws Exception {
//                List<double[]> result = new ArrayList<>();
//                if(dataVectors.get(0).length<3){
//                    result = I_niceMO_Observers.generateExpandedData(dataVectors, 20);}
//                else {
//                    result = I_niceMO_Observers.uniformSample(dataVectors, 20);
//                }
                List<double[]> result = I_niceMO_Observers.uniformSample(dataVectors, 20);
//                saveToCSV(result, "data/centerallU100.csv");
                List<double[]> centersAll = new ArrayList<>();
                for (int i = 0; i < numberOfObservers; i++) {
                    double[] observer = result.get(i);
                    List<double[]> centers = I_niceMO_Observers.GMMtest(dataVectors, observer, k);
                    centersAll.addAll(centers); //每个观测点的结果直接加在聚类中心集合里面。
                }
                return centersAll;
//                return I_niceMO_Observers.mergeData(centersAll, percentage);
            }
        };
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
