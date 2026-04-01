package ComaparedMethods;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.kdtree.KDTree;
import pers.hongweiye.gammaem.mixtool.GammamixEM;
import pers.hongweiye.gammaem.mixtool.MixEM;
import pers.hongweiye.gammaem.mixtool.util.GammaUtil;

import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

public class I_niceMO_Observers {
    // I_niceMO函数
    public static  List<double[]> I_niceMOF( List<double[]> dataVectors,int numberOfObservors,int k,double percentage) {
        List<double[]> result = uniformSample(dataVectors,numberOfObservors);
        List<double[]> centersAll = new ArrayList<>();
        for (int i = 0; i < numberOfObservors; i++) {
//            int randomNumber = (int) (Math.random() * result.size()) + 1; //随机选点
            double[] observer = result.get(i);
            List<double[]> centers = GMMtest(dataVectors, observer, k);
            for (int j = 0; j < centers.size(); j++) {
                centersAll.add(centers.get(j));
            }
        }
        // 进行数据合并，采用每个样本先合并初步的中心，再合并所有中心的方法
//        List<double[]> mergedCenters = mergeDataGO1(centersAll);//mergeData(centersAll,percentage);
//        return mergedCenters;
        // 采用一步合并所有的中心的方法。
        return centersAll;
    }

    public static  List<double[]> I_niceMOFcheck( List<double[]> dataVectors,int numberOfObservors,int k) {
        List<double[]> result = generateExpandedData(dataVectors,numberOfObservors);
        List<double[]> centersAll = new ArrayList<>();
        for (int i = 0; i < numberOfObservors; i++) {
//            int randomNumber = (int) (Math.random() * result.size()) + 1; //随机选点
            double[] observer = result.get(i);
            List<double[]> centers = GMMtest(dataVectors, observer, k);
            for (int j = 0; j < centers.size(); j++) {
                centersAll.add(centers.get(j));
            }
        }
        // 进行数据合并
        return centersAll;
    }

    // 生成观测点，在样本张成的空间里面均匀抽样
    public static List<double[]> uniformSample(List<double[]> inputData, int numPoints) {
        if (numPoints > inputData.size() || numPoints < 0) {
            throw new IllegalArgumentException("numPoints must be between 0 and inputData.size()");
        }

        // Create a list of indices and shuffle it
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < inputData.size(); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random());

        // Select the first numPoints indices
        List<double[]> sampledData = new ArrayList<>();
        for (int i = 0; i < numPoints; i++) {
            sampledData.add(inputData.get(indices.get(i)));
        }

        return sampledData;
    }
    public static List<double[]> generateExpandedData(List<double[]> inputData, int numPoints) {
        if (inputData != null && !inputData.isEmpty()) {
            int dimensions = ((double[])inputData.get(0)).length;
            double[] minValues = new double[dimensions];
            double[] maxValues = new double[dimensions];
            Arrays.fill(minValues, Double.MAX_VALUE);
            Arrays.fill(maxValues, Double.MIN_VALUE);
            Iterator var5 = inputData.iterator();

            while(var5.hasNext()) {
                double[] point = (double[])var5.next();

                for(int d = 0; d < dimensions; ++d) {
                    minValues[d] = Math.min(minValues[d], point[d]);
                    maxValues[d] = Math.max(maxValues[d], point[d]);
                }
            }
            Set<String> uniquePoints = new HashSet();
            List<double[]> expandedData = new ArrayList();
            Random random = new Random();
            int gridSize = (int)Math.ceil(Math.pow((double)numPoints, 1.0 / (double)dimensions));
            double[] steps = new double[dimensions];

            for(int d = 0; d < dimensions; ++d) {
                steps[d] = gridSize > 1 ? (maxValues[d] - minValues[d]) / (double)(gridSize - 1) : 0.0;
            }
            generateGridPoints(expandedData, uniquePoints, minValues, steps, new double[dimensions], 0, gridSize, numPoints);
            while(expandedData.size() < numPoints) {
                double[] newPoint = new double[dimensions];
                for(int d = 0; d < dimensions; ++d) {
                    newPoint[d] = minValues[d] + (maxValues[d] - minValues[d]) * random.nextDouble();
                }
                String key = Arrays.toString(newPoint);
                if (!uniquePoints.contains(key)) {
                    expandedData.add(newPoint);
                    uniquePoints.add(key);
                }
            }
            Collections.shuffle(expandedData);

            return expandedData;
        } else {
            throw new IllegalArgumentException("输入数据不能为空！");
        }
    }
    private static void generateGridPoints(List<double[]> expandedData, Set<String> uniquePoints, double[] minValues, double[] steps, double[] currentPoint, int dimIndex, int gridSize, int numPoints) {
        if (dimIndex == minValues.length) {
            String key = Arrays.toString(currentPoint);
            if (!uniquePoints.contains(key)) {
                expandedData.add((double[])currentPoint.clone());
                uniquePoints.add(key);
            }
        } else {
            for(int i = 0; i < gridSize && expandedData.size() < numPoints; ++i) {
                currentPoint[dimIndex] = minValues[dimIndex] + (double)i * steps[dimIndex];
                generateGridPoints(expandedData, uniquePoints, minValues, steps, currentPoint, dimIndex + 1, gridSize, numPoints);
            }

        }
    }
    public static double euclideanDistance(double[] p1, double[] p2) {
        double sum = 0.0;

        for(int i = 0; i < p1.length; ++i) {
            sum += Math.pow(p1[i] - p2[i], 2.0);
        }

        return Math.sqrt(sum);
    }
    public static double[][] computeDistanceMatrix(List<double[]> centersAll) {
        int n = centersAll.size();
        double[][] distanceMatrix = new double[n][n];
        for(int i = 0; i < n; ++i) {
            for(int j = i + 1; j < n; ++j) {
                double dist = euclideanDistance((double[])centersAll.get(i), (double[])centersAll.get(j));
                distanceMatrix[i][j] = dist;
                distanceMatrix[j][i] = dist;
            }
        }
        return distanceMatrix;
    }
    public static double computeThreshold(double[][] distanceMatrix, double percentage) {
        List<Double> distances = new ArrayList();
        int n = distanceMatrix.length;
        int numThreshold;
        for(numThreshold = 0; numThreshold < n; ++numThreshold) {
            for(int j = numThreshold + 1; j < n; ++j) {
                distances.add(distanceMatrix[numThreshold][j]);
            }
        }
        Collections.sort(distances);
        numThreshold = Math.max(1, (int)((double)distances.size() * percentage));
        double sum = 0.0;
        for(int i = 0; i < numThreshold; ++i) {
            sum += (Double)distances.get(i);
        }
        return sum / (double)numThreshold;
    }
    // 合并I-nice的聚类中心
    public static List<double[]> mergeData(List<double[]> centersAll, double percentage) {
        double[][] distanceMatrix = computeDistanceMatrix(centersAll);
        double threshold = computeThreshold(distanceMatrix, percentage);
        List<double[]> mergedCenters = new ArrayList(centersAll);
        boolean[] merged = new boolean[centersAll.size()];
        int j;
        for(int i = 0; i < centersAll.size(); ++i) {
            if (!merged[i]) {
                for(j = i + 1; j < centersAll.size(); ++j) {
                    if (!merged[j] && euclideanDistance((double[])centersAll.get(i), (double[])centersAll.get(j)) < threshold) {
                        double[] newPoint = new double[((double[])centersAll.get(i)).length];
                        for(int d = 0; d < newPoint.length; ++d) {
                            newPoint[d] = (((double[])centersAll.get(i))[d] + ((double[])centersAll.get(j))[d]) / 2.0; //平均值的方法
                        }
                        mergedCenters.set(i, newPoint);
                        merged[j] = true;
                    }
                }
            }
        }
        List<double[]> finalCenters = new ArrayList();
        for(j = 0; j < mergedCenters.size(); ++j) {
            if (!merged[j]) {
                finalCenters.add((double[])mergedCenters.get(j));
            }
        }
        return finalCenters;
    }

    public static List<double[]> mergeDataGO1(List<double[]> centersAll) {
        //谱聚类的GO方法
//        // 计算相似度矩阵
//        Matrix similarityMatrix = computeSimilarityMatrix(centersAll);
//        System.out.println(centersAll.size());
//
//        // 计算拉普拉斯矩阵
//        Matrix laplacianMatrix = computeLaplacianMatrix(similarityMatrix);
//
//        // 设置阈值
//        //double threshold = 0.000001;
//
//        // 获取特征值小于阈值的特征向量
//        Matrix eigenvectors = getEigenvectors(laplacianMatrix, threshold);
//        System.out.println("行数: " + eigenvectors.getRowDimension());
//        System.out.println("列数: " + eigenvectors.getColumnDimension());
//
//        // 将谱聚类特征向量转换为 JavaML 数据集
//        Dataset dataset = convertToDataset(eigenvectors);
//
//
//        // 使用 JavaML 的 KMeans 聚类
//        KMeans kmeans = new KMeans(eigenvectors.getColumnDimension());  // 直接在构造器中指定簇数
//        // 聚类簇数由特征值小于阈值的数量决定
//        int k = eigenvectors.getColumnDimension();
//        System.out.println("聚类簇数（特征值小于 " + threshold + " 的数量）: " + k);
//
//        // 进行聚类
//        Dataset[] clusters = kmeans.cluster(dataset);
//        System.out.println("簇数: " + clusters.length);
//        // 获取每个数据点的簇标签并存储在 labels 数组中
//        int[] labels = new int[dataset.size()];
//        for (int i = 0; i < dataset.size(); i++) {
//            Instance instance = dataset.get(i);
//            int clusterIndex = -1;
//            for (int j = 0; j < clusters.length; j++) {
//                if (clusters[j].contains(instance)) {
//                    clusterIndex = j;
//                    break;
//                }
//            }
//            labels[i] = clusterIndex; // 保存每个数据点的簇标签
//        }
//
//        // 输出 labels 数组
//        System.out.print("聚类结果（标签数组）： ");
//        for (int label : labels) {
//            System.out.print(label + " ");
//        }
//        System.out.println();
//
//
//        List<double[]> finalCenters = calculateClusterCenters(centersAll, labels);
//        return finalCenters;

        //根据距离的GO方法
        double[][] distanceMatrix = computeDistanceMatrix(centersAll);
        double[] vector = convertToVector(distanceMatrix);
        // 设置距离变化检查窗口大小
        int windowSize = 200;
        // 如果数据长度小于窗口大小，直接退出
        if (vector.length < windowSize) {
            System.out.println("数据长度小于窗口大小！");
            return centersAll;
        }
        // 对整个 vector 进行排序
        double[] sortedVector = Arrays.copyOf(vector, vector.length);
        Arrays.sort(sortedVector);
        // 计算排序后的变化最快的位置
        int changeIndex = findFastestChangeIndex(sortedVector, windowSize);
        System.out.println("距离开始"+changeIndex);
        // 计算变化最快位置的平均距离
        double threshold = calculateAverageDistance(sortedVector, changeIndex, windowSize);
        // 保存 CSV 的文件路径
//        String filePath = "data/distanceMatrix.csv";
//        try (FileWriter writer = new FileWriter(filePath)) {
//            for (int i = 0; i < distanceMatrix.length; i++) {
//                for (int j = 0; j < distanceMatrix[i].length; j++) {
//                    writer.append(String.valueOf(distanceMatrix[i][j]));
//                    if (j < distanceMatrix[i].length - 1) {
//                        writer.append(","); // 每个值后添加逗号
//                    }
//                }
//                writer.append("\n"); // 每行结束后添加换行符
//            }
//            System.out.println("CSV 文件保存成功!");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        double threshold = computeThreshold(distanceMatrix, percentage);  //手动设置阈值
        List<double[]> mergedCenters = new ArrayList(centersAll);
        boolean[] merged = new boolean[centersAll.size()];
        int j;
        for(int i = 0; i < centersAll.size(); ++i) {
            if (!merged[i]) {
                for(j = i + 1; j < centersAll.size(); ++j) {
                    if (!merged[j] && euclideanDistance((double[])centersAll.get(i), (double[])centersAll.get(j)) < threshold) {
                        double[] newPoint = new double[((double[])centersAll.get(i)).length];

                        for(int d = 0; d < newPoint.length; ++d) {
                            newPoint[d] = ((double[])centersAll.get(i))[d];
                        }
                        mergedCenters.set(i, newPoint);
                        merged[j] = true;
                    }
                }
            }
        }
        List<double[]> finalCenters = new ArrayList();
        for(j = 0; j < mergedCenters.size(); ++j) {
            if (!merged[j]) {
                finalCenters.add((double[])mergedCenters.get(j));
            }
        }
        return finalCenters;
    }

    public static double[][] findKNearestNeighbors(List<double[]> dataList, int k) {
        int n = dataList.size();
        if (n <= 1 || k <= 0) {
            throw new IllegalArgumentException("Data must contain at least 2 points, and k must be > 0.");
        }

        if (k >= n) {
            System.err.println("Warning: k >= number of points. Reducing k to n - 1.");
            k = n - 1;
        }

        double[][] distances = new double[n][k];

        for (int i = 0; i < n; i++) {
            double[] current = dataList.get(i);
            // 存储所有其他点的距离
            List<Double> distanceList = new ArrayList<>();

            for (int j = 0; j < n; j++) {
                if (i == j) continue; // 跳过自己
                double dist = euclideanDistance(current, dataList.get(j));
                distanceList.add(dist);
            }

            // 排序并取前 k 个最小距离
            distanceList.sort(Double::compareTo);
            for (int d = 0; d < k; d++) {
                distances[i][d] = distanceList.get(d);
            }
        }

        return distances;
    }

    // 在GO里面合并聚类中心
    public static List<double[]> mergeDataGO(List<double[]> centersAll) {
        //谱聚类的GO方法
//        // 计算相似度矩阵
//        Matrix similarityMatrix = computeSimilarityMatrix(centersAll);
//        System.out.println(centersAll.size());
//
//        // 计算拉普拉斯矩阵
//        Matrix laplacianMatrix = computeLaplacianMatrix(similarityMatrix);
//
//        // 设置阈值
//        //double threshold = 0.000001;
//
//        // 获取特征值小于阈值的特征向量
//        Matrix eigenvectors = getEigenvectors(laplacianMatrix, threshold);
//        System.out.println("行数: " + eigenvectors.getRowDimension());
//        System.out.println("列数: " + eigenvectors.getColumnDimension());
//
//        // 将谱聚类特征向量转换为 JavaML 数据集
//        Dataset dataset = convertToDataset(eigenvectors);
//
//
//        // 使用 JavaML 的 KMeans 聚类
//        KMeans kmeans = new KMeans(eigenvectors.getColumnDimension());  // 直接在构造器中指定簇数
//        // 聚类簇数由特征值小于阈值的数量决定
//        int k = eigenvectors.getColumnDimension();
//        System.out.println("聚类簇数（特征值小于 " + threshold + " 的数量）: " + k);
//
//        // 进行聚类
//        Dataset[] clusters = kmeans.cluster(dataset);
//        System.out.println("簇数: " + clusters.length);
//        // 获取每个数据点的簇标签并存储在 labels 数组中
//        int[] labels = new int[dataset.size()];
//        for (int i = 0; i < dataset.size(); i++) {
//            Instance instance = dataset.get(i);
//            int clusterIndex = -1;
//            for (int j = 0; j < clusters.length; j++) {
//                if (clusters[j].contains(instance)) {
//                    clusterIndex = j;
//                    break;
//                }
//            }
//            labels[i] = clusterIndex; // 保存每个数据点的簇标签
//        }
//
//        // 输出 labels 数组
//        System.out.print("聚类结果（标签数组）： ");
//        for (int label : labels) {
//            System.out.print(label + " ");
//        }
//        System.out.println();
//
//
//        List<double[]> finalCenters = calculateClusterCenters(centersAll, labels);
//        return finalCenters;

        //根据距离的GO方法
        double[][] distanceMatrix = computeDistanceMatrix(centersAll);
        double[] vector = convertToVector(distanceMatrix);
        // 设置距离变化检查窗口大小
        int windowSize = 200;
        // 如果数据长度小于窗口大小，直接退出
        if (vector.length < windowSize) {
            System.out.println("数据长度小于窗口大小！");
            return centersAll;
        }
        // 对整个 vector 进行排序
        double[] sortedVector = Arrays.copyOf(vector, vector.length);
        Arrays.sort(sortedVector);
        // 计算排序后的变化最快的位置
        int changeIndex = findFastestChangeIndex(sortedVector, windowSize);
        System.out.println("距离开始"+changeIndex);
        // 计算变化最快位置的平均距离
        double threshold = calculateAverageDistance(sortedVector, changeIndex, windowSize);
        // 保存 CSV 的文件路径
//        String filePath = "data/distanceMatrix.csv";
//        try (FileWriter writer = new FileWriter(filePath)) {
//            for (int i = 0; i < distanceMatrix.length; i++) {
//                for (int j = 0; j < distanceMatrix[i].length; j++) {
//                    writer.append(String.valueOf(distanceMatrix[i][j]));
//                    if (j < distanceMatrix[i].length - 1) {
//                        writer.append(","); // 每个值后添加逗号
//                    }
//                }
//                writer.append("\n"); // 每行结束后添加换行符
//            }
//            System.out.println("CSV 文件保存成功!");
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//        double threshold = computeThreshold(distanceMatrix, percentage);  //手动设置阈值
        List<double[]> mergedCenters = new ArrayList(centersAll);
        boolean[] merged = new boolean[centersAll.size()];
        int j;
        for(int i = 0; i < centersAll.size(); ++i) {
            if (!merged[i]) {
                for(j = i + 1; j < centersAll.size(); ++j) {
                    if (!merged[j] && euclideanDistance((double[])centersAll.get(i), (double[])centersAll.get(j)) < threshold) {
                        double[] newPoint = new double[((double[])centersAll.get(i)).length];

                        for(int d = 0; d < newPoint.length; ++d) {
                            newPoint[d] = ((double[])centersAll.get(i))[d];
                        }
                        mergedCenters.set(i, newPoint);
                        merged[j] = true;
                    }
                }
            }
        }
        List<double[]> finalCenters = new ArrayList();
        for(j = 0; j < mergedCenters.size(); ++j) {
            if (!merged[j]) {
                finalCenters.add((double[])mergedCenters.get(j));
            }
        }
        return finalCenters;
    }
    //////寻找距离变化最快的部分函数///////
    // 将二维矩阵转换为一维向量
    public static double[] convertToVector(double[][] distanceMatrix) {
        int rows = distanceMatrix.length;
        int cols = distanceMatrix[0].length;
        double[] vector = new double[rows * cols];
        int index = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                vector[index++] = distanceMatrix[i][j];
            }
        }
        return vector;
    }
    // 查找变化最快的位置：窗口内的差值总和最大的位置
    public static int findFastestChangeIndex(double[] vector, int windowSize) {
        double maxChange = Double.NEGATIVE_INFINITY;
        int bestIndex = 0;
        // 遍历每个窗口，计算变化量
        for (int i = 0; i <= (vector.length)/2 - windowSize; i++) {
            double change = 0;
            // 计算窗口内的变化
            for (int j = i + 1; j < i + windowSize; j++) {
                change += Math.abs(vector[j] - vector[j - 1]);
            }
            // 更新最大变化和对应的起始位置
            if (change > maxChange) {
                maxChange = change;
                bestIndex = i;
            }
        }
        return bestIndex;
    }
    // 计算指定窗口内的平均距离
    public static double calculateAverageDistance(double[] vector, int startIndex, int windowSize) {
        double sum = 0;
        for (int i = startIndex; i < startIndex + windowSize; i++) {
            sum += vector[i];
        }
        return sum/windowSize;//vector[startIndex+windowSize];
    }
    //////寻找距离变化最快的部分///////
    public static double gammaPdf(double x, double alpha, double beta) {
        return x <= 0.0 ? 0.0 : GammaUtil.dgamma(x, alpha, beta);
    }
    // 根据伽马混合分布结果确定数据属于那个类
    public static int classifyComponent(double x, double[] pi, double[] alpha, double[] beta) {
        int k = pi.length;
        double[] probabilities = new double[k];
        double sumProb = 0.0;

        int bestComponent;
        for(bestComponent = 0; bestComponent < k; ++bestComponent) {
            probabilities[bestComponent] = pi[bestComponent] * gammaPdf(x, alpha[bestComponent], beta[bestComponent]);
            sumProb += probabilities[bestComponent];
        }
        bestComponent = 0;
        double maxProb = 0.0;
        for(int i = 0; i < k; ++i) {
            probabilities[i] /= sumProb;
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                bestComponent = i;
            }
        }
        return bestComponent;
    }
    public static int[] classifyVector(double[] data, double[] pi, double[] alpha, double[] beta) {
        int[] labels = new int[data.length];

        for(int i = 0; i < data.length; ++i) {
            labels[i] = classifyComponent(data[i], pi, alpha, beta);
        }
        return labels;
    }
    public static List<List<double[]>> clusterData(List<double[]> dataVectors, int[] labels, int k) {
        List<List<double[]>> clusters = new ArrayList(k);

        int i;
        for(i = 0; i < k; ++i) {
            clusters.add(new ArrayList());
        }

        for(i = 0; i < dataVectors.size(); ++i) {
            int label = labels[i];
            ((List)clusters.get(label)).add((double[])dataVectors.get(i));
        }
        return clusters;
    }
    // 找数据的的K近邻
//    public static double[][] findKNearestNeighbors(List<double[]> dataList, int k) {
//        int n = dataList.size();
//        KDTree kdTree = new KDTree(((double[])dataList.get(0)).length);
//        Iterator var4 = dataList.iterator();
//
//        while(var4.hasNext()) {
//            double[] point = (double[])var4.next();
//            kdTree.insert(point, point);
//        }
//
//        double[][] distances = new double[n][k];
//        for(int i = 0; i < n; ++i) {
//            Object[] neighbors = kdTree.nearest((double[])dataList.get(i), k + 1);
//            double[] dist = new double[k];
//            int index = 0;
//            Object[] var9 = neighbors;
//            int var10 = neighbors.length;
//            for(int var11 = 0; var11 < var10; ++var11) {
//                Object neighbor = var9[var11];
//                double[] neighborPoint = (double[])neighbor;
//                if (!Arrays.equals(neighborPoint, (double[])dataList.get(i))) {
//                    dist[index++] = euclideanDistance((double[])dataList.get(i), neighborPoint);
//                    if (index >= k) {
//                        break;
//                    }
//                }
//            }
//            distances[i] = dist;
//        }
//        return distances;
//    }

//    private static double euclideanDistance(double[] p1, double[] p2) {
//        double sum = 0.0;
//
//        for(int i = 0; i < p1.length; ++i) {
//            sum += Math.pow(p1[i] - p2[i], 2.0);
//        }
//
//        return Math.sqrt(sum);
//    }
    // 找最高密度的点
    public static int findHighestDensityPoint(List<double[]> dataList, int k) {
        double[][] distances = findKNearestNeighbors(dataList, k);
        double maxDensity = -1.0;
        int maxIndex = -1;
        for(int i = 0; i < distances.length; ++i) {
            double avgDistance = Arrays.stream(distances[i]).average().orElse(Double.MAX_VALUE);
            double density = 1.0 / avgDistance;
            if (density > maxDensity) {
                maxDensity = density;
                maxIndex = i;
            }
        }
        return maxIndex;
    }
    // 计算 data_selected 与 dataVectors 中每个点的欧几里得距离，并返回 double[] 数组
    public static double[] calculateDistances(List<double[]> dataVectors, double[] data_selected) {
        int size = dataVectors.size();
        double[] distances = new double[size];
        // 遍历 dataVectors 中的每个数据点，计算与 data_selected 的欧几里得距离
        for (int i = 0; i < size; i++) {
            distances[i] = euclideanDistance(dataVectors.get(i), data_selected);
        }
        return distances;
    }

    public static void saveDoubleArrayAsCSV(double[] data_vec, String filePath) {
        try (FileWriter writer = new FileWriter(filePath)) {
            // 将数组写为一行，逗号分隔
            for (int i = 0; i < data_vec.length; i++) {
                writer.write(Double.toString(data_vec[i]));
                if (i < data_vec.length - 1) {
                    writer.write(",");
                }
            }
            writer.write("\n"); // 换行，结束一行
            System.out.println("CSV 写入成功: " + filePath);
        } catch (IOException e) {
            System.err.println("写入 CSV 文件失败: " + e.getMessage());
        }
    }
    // 遍历不同的伽马分量，确定一个最好的模型。
    public static List<double[]> GMMtest(List<double[]> dataVectors, double[] data_selected, int k){
        double[] data_vec = calculateDistances(dataVectors,data_selected);
//        saveDoubleArrayAsCSV(data_vec, "data/distance.csv");
        for (int i = 0; i < data_vec.length; i++) {
            if (data_vec[i] == 0) {
                data_vec[i] = 0.1;
            }
        }
        //component数
        int n_components=25;
        int N = data_vec.length;
        //初始化 lambda
        System.out.println(Arrays.toString(data_vec));
        //迭代
        List<Double> aiccList = new ArrayList<>();
        int consecutiveCount = 0;      // 连续满足条件的计数器
        double historicalMinAICc = Double.POSITIVE_INFINITY; // 历史最小AICc初始值设为正无穷
        for (int i = 1; i <= 1000; i++) {
            //
            double[] lambda=new double[i];
            Arrays.fill(lambda, 1.0 / i);
            double[][]out=GammamixEM.gammamixInit(data_vec,lambda,null,null,lambda.length);
            System.out.println("alpha:"+Arrays.toString(out[0]));
            System.out.println("beta:"+Arrays.toString(out[1]));
            MixEM em=GammamixEM.gammamixEM(data_vec,lambda,out[0],out[1],i,0.1,1000,20,true);
            // 计算 AICc 值（添加括号明确运算顺序，避免整数除法）
            double numerator = 2 * 3 * i * N;
            double denominator = N - 3 * i - 1;
            // 防止分母非正的健壮性检查
            if (denominator <= 0) {
                System.err.println("警告：分母非正，跳过 i=" + i);
                continue;
            }
            double AICc = -2 * em.loglik + numerator / denominator;
            if(i==1){
                historicalMinAICc = AICc;
            }
            else{
            // 核心逻辑：判断是否满足连续条件
            if (AICc > historicalMinAICc) {
                consecutiveCount++;          // 更新计数器
            } else {
                consecutiveCount = 0;        // 不满足则重置计数器
                historicalMinAICc = AICc;    // 更新历史最小值
            }}
            // 终止条件检查
            if (consecutiveCount >= 6) {
                System.out.printf("在第 %d 次迭代后连续 5 次 AICc 下降，终止循环 ", i);
                break;
            }
            aiccList.add(AICc);  // 仅在未满足终止条件时才保留该记录
        }
        // 获取分类标签
        n_components = aiccList.size()-5;
        System.out.println(n_components);
        n_components = aiccList.size()-5;
        double[] lambda=new double[n_components];
        Arrays.fill(lambda, 1.0 / n_components);
        double[][]out=GammamixEM.gammamixInit(data_vec,lambda,null,null,lambda.length);
        System.out.println("alpha:"+Arrays.toString(out[0]));
        System.out.println("beta:"+Arrays.toString(out[1]));
        MixEM em=GammamixEM.gammamixEM(data_vec,lambda,out[0],out[1],n_components,0.1,1000,20,true);
        System.out.print("lambda: ");
        System.out.println(Arrays.toString(em.lambda));
        int[] labels = classifyVector(data_vec, em.lambda, em.alpha, em.beta);
        List<List<double[]>> clusteredData = clusterData(dataVectors, labels,em.lambda.length);
        List<double[]> centers = new ArrayList<>();
        //KNN的方法
        // 输出每个类别的簇数据
        for (int i = 0; i < em.lambda.length; i++) {
            List<double[]> temp = clusteredData.get(i);
            if(temp.size()>0){
            int highestDensityIndex = findHighestDensityPoint(temp, k);
            centers.add(temp.get(highestDensityIndex));}
        }
        return centers;
    }

    // 计算拉普拉斯矩阵
    public static Matrix computeLaplacianMatrix(Matrix similarityMatrix) {
        int n = similarityMatrix.getRowDimension();
        // 计算度矩阵 D (对角矩阵，元素是每行相似度矩阵的行和)
        Matrix D = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = 0; j < n; j++) {
                sum += similarityMatrix.get(i, j);
            }
            D.set(i, i, sum);
        }
        // 计算拉普拉斯矩阵 L = D - S
        return D.minus(similarityMatrix);
    }
    // 计算相似度矩阵，假设是基于欧氏距离计算的
    public static Matrix computeSimilarityMatrix(List<double[]> data) {
        int n = data.size();
        Matrix similarityMatrix = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double dist = 0;
                for (int k = 0; k < data.get(0).length; k++) {
                    dist += Math.pow(data.get(i)[k] - data.get(j)[k], 2);
                }
                similarityMatrix.set(i, j, Math.exp(-dist)); // Gaussian kernel
            }
        }
        return similarityMatrix;
    }

    // 获取特征值小于阈值的特征向量
    public static Matrix getEigenvectors(Matrix laplacianMatrix, double threshold) {
        EigenvalueDecomposition eig = new EigenvalueDecomposition(laplacianMatrix);
        double[] eigenvalues = eig.getRealEigenvalues();
        Matrix eigenvectors = eig.getV();
        // 统计小于阈值的特征值数量
        int n = eigenvalues.length;
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (eigenvalues[i] < threshold) {
                count++;
            }
        }
        // 创建一个新矩阵来存储这些特征向量
        Matrix filteredEigenvectors = new Matrix(n, count);
        int colIdx = 0;
        for (int i = 0; i < n; i++) {
            if (eigenvalues[i] < threshold) {
                // 将第i个特征向量复制到filteredEigenvectors的第colIdx列
                for (int row = 0; row < n; row++) {
                    filteredEigenvectors.set(row, colIdx, eigenvectors.get(row, i));
                }
                colIdx++;
            }
        }
        return filteredEigenvectors;
    }
    // 将谱聚类的特征向量转换为 JavaML 数据集
    public static Dataset convertToDataset(Matrix eigenvectors) {
        int numRows = eigenvectors.getRowDimension();
        int numCols = eigenvectors.getColumnDimension();
        Dataset dataset = new DefaultDataset(); // 使用 DefaultDataset 代替 Dataset
        for (int i = 0; i < numRows; i++) {
            double[] rowData = new double[numCols];
            for (int j = 0; j < numCols; j++) {
                rowData[j] = eigenvectors.get(i, j);
            }
            Instance instance = new DenseInstance(rowData);
            dataset.add(instance);
        }
        return dataset;
    }
    public static List<double[]> calculateClusterCenters(List<double[]> data, int[] labels) {
        // 使用 Map 来存储每个类别的点集合
        Map<Integer, List<double[]>> clusters = new HashMap<>();
        // 按类别对数据进行分组
        for (int i = 0; i < data.size(); i++) {
            int label = labels[i];
            clusters.putIfAbsent(label, new ArrayList<>());
            clusters.get(label).add(data.get(i));
        }
        // 计算每个类别的中心点
        List<double[]> clusterCenters = new ArrayList<>();
        for (Map.Entry<Integer, List<double[]>> entry : clusters.entrySet()) {
            int clusterLabel = entry.getKey();
            List<double[]> clusterPoints = entry.getValue();
            // 计算该类别的中心点
            double[] center = new double[clusterPoints.get(0).length]; // 初始化中心点数组
            // 累加该类别所有点的值
            for (double[] point : clusterPoints) {
                for (int j = 0; j < point.length; j++) {
                    center[j] += point[j];
                }
            }
            // 计算平均值（中心点）
            for (int j = 0; j < center.length; j++) {
                center[j] /= clusterPoints.size();
            }
            // 添加到结果列表
            clusterCenters.add(center);
        }
        return clusterCenters;
    }
}