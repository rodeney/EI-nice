package ComaparedMethods;

import java.util.*;

public class PurityCalculator {
    public static void main(String[] args) {
        // 示例数据（真实标签 和 聚类标签）
        // 假设共 10 个样本
        int[] trueLabels =    {0, 0, 1, 1, 2, 2, 0, 1, 2, 2}; // 真实类别
        int[] clusterLabels = {2, 2, 2, 2, 2, 2, 1, 0, 2, 2}; // 聚类分组

        double purity = calculatePurity(trueLabels, clusterLabels);
        System.out.println("聚类纯度 (Purity): " + purity);

        double ari = computeARI(trueLabels, clusterLabels);
        System.out.println("Adjusted Rand Index (ARI): " + ari);
    }

    public static double computeARI(int[] trueLabels, int[] clusterLabels) {
        if (trueLabels.length != clusterLabels.length) {
            throw new IllegalArgumentException("标签数组长度不一致！");
        }

        int n = trueLabels.length;

        // 构建 contingency matrix：Map<true, Map<cluster, count>>
        Map<Integer, Map<Integer, Integer>> contingency = new HashMap<>();

        // 所有标签集合
        Set<Integer> trueSet = new HashSet<>();
        Set<Integer> clusterSet = new HashSet<>();

        for (int i = 0; i < n; i++) {
            int trueLabel = trueLabels[i];
            int clusterLabel = clusterLabels[i];

            trueSet.add(trueLabel);
            clusterSet.add(clusterLabel);

            contingency.putIfAbsent(trueLabel, new HashMap<>());
            Map<Integer, Integer> clusterCount = contingency.get(trueLabel);
            clusterCount.put(clusterLabel, clusterCount.getOrDefault(clusterLabel, 0) + 1);
        }

        // 计算组合数量
        double sumCombC = 0.0;
        for (Map<Integer, Integer> clusterCount : contingency.values()) {
            for (int nij : clusterCount.values()) {
                sumCombC += comb2(nij);
            }
        }

        // 真实标签中每类的组合
        Map<Integer, Integer> trueCounts = new HashMap<>();
        for (int label : trueLabels) {
            trueCounts.put(label, trueCounts.getOrDefault(label, 0) + 1);
        }

        double sumCombA = 0.0;
        for (int ni : trueCounts.values()) {
            sumCombA += comb2(ni);
        }

        // 聚类标签中每类的组合
        Map<Integer, Integer> clusterCounts = new HashMap<>();
        for (int label : clusterLabels) {
            clusterCounts.put(label, clusterCounts.getOrDefault(label, 0) + 1);
        }

        double sumCombB = 0.0;
        for (int nj : clusterCounts.values()) {
            sumCombB += comb2(nj);
        }

        double expectedIndex = (sumCombA * sumCombB) / comb2(n);
        double maxIndex = 0.5 * (sumCombA + sumCombB);

        double ari = (sumCombC - expectedIndex) / (maxIndex - expectedIndex);
        return ari;
    }

    // 计算组合数 C(n, 2)
    private static double comb2(int n) {
        return n < 2 ? 0 : n * (n - 1) / 2.0;
    }

    public static double calculatePurity(int[] trueLabels, int[] clusterLabels) {
        if (trueLabels.length != clusterLabels.length) {
            throw new IllegalArgumentException("标签数组长度不一致！");
        }

        int n = trueLabels.length;
        // Map<clusterID, Map<trueLabel, count>>
        Map<Integer, Map<Integer, Integer>> clusterLabelCounts = new HashMap<>();

        for (int i = 0; i < n; i++) {
            int cluster = clusterLabels[i];
            int label = trueLabels[i];

            clusterLabelCounts.putIfAbsent(cluster, new HashMap<>());
            Map<Integer, Integer> labelCount = clusterLabelCounts.get(cluster);
            labelCount.put(label, labelCount.getOrDefault(label, 0) + 1);
        }

        int totalCorrect = 0;
        for (Map<Integer, Integer> labelCount : clusterLabelCounts.values()) {
            int max = Collections.max(labelCount.values()); // 选最多的那个类
            totalCorrect += max;
        }

        return (double) totalCorrect / n;
    }
}