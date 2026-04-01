package com.szubd;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

// 将合并后的聚类中心作为初始聚类中心，然后进行kmeans聚类
public class KMeans {
    private List<double[]> centers;
    private int maxIterations = 100;
    private double tolerance = 1e-4;

    public KMeans(List<double[]> initialCenters) {
        this.centers = new ArrayList<>(initialCenters);
    }

    public int[] fit(List<double[]> dataPoints) {
        int k = centers.size();
        int n = dataPoints.size();
        int[] labels = new int[n];

        for (int iter = 0; iter < maxIterations; iter++) {
            boolean changed = false;

            // 重新分配点到最近的中心
            for (int i = 0; i < n; i++) {
                int newLabel = findClosestCenter(dataPoints.get(i));
                if (labels[i] != newLabel) {
                    labels[i] = newLabel;
                    changed = true;
                }
            }

            // 计算新中心
            List<double[]> newCenters = new ArrayList<>(Collections.nCopies(k, null));
            int[] clusterSizes = new int[k];

            for (int i = 0; i < k; i++) {
                newCenters.set(i, new double[dataPoints.get(0).length]);
            }

            for (int i = 0; i < n; i++) {
                int label = labels[i];
                double[] point = dataPoints.get(i);
                double[] newCenter = newCenters.get(label);
                for (int d = 0; d < point.length; d++) {
                    newCenter[d] += point[d];
                }
                clusterSizes[label]++;
            }

            for (int i = 0; i < k; i++) {
                if (clusterSizes[i] > 0) {
                    for (int d = 0; d < newCenters.get(i).length; d++) {
                        newCenters.get(i)[d] /= clusterSizes[i];
                    }
                } else {
                    newCenters.set(i, centers.get(i)); // 防止空簇
                }
            }

            if (!changed) break; // 如果所有点都未改变分类，则停止迭代
            centers = newCenters;
        }

        return labels;
    }

    private int findClosestCenter(double[] point) {
        int bestIndex = 0;
        double minDist = Double.MAX_VALUE;
        for (int i = 0; i < centers.size(); i++) {
            double dist = euclideanDistance(point, centers.get(i));
            if (dist < minDist) {
                minDist = dist;
                bestIndex = i;
            }
        }
        return bestIndex;
    }

    private double euclideanDistance(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    public static void main(String[] args) {
        List<double[]> mergedCenters = Arrays.asList(
                new double[]{1.0, 2.0}, new double[]{5.0, 6.0}, new double[]{8.0, 8.0}
        );
        System.out.println("预测的簇数："+mergedCenters.size());
        List<double[]> dataPoints = Arrays.asList(
                new double[]{1.1, 2.1}, new double[]{0.9, 1.9},
                new double[]{5.2, 6.1}, new double[]{4.9, 5.8},
                new double[]{8.3, 8.1}, new double[]{7.8, 8.2}
        );

        KMeans kmeans = new KMeans(mergedCenters);
        int[] labels = kmeans.fit(dataPoints);

        System.out.println("Data point labels: " + Arrays.toString(labels));
    }
}
