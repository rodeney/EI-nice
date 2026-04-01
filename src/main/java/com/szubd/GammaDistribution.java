package com.szubd;

import java.util.*;

public class GammaDistribution {

    // Gamma分布概率密度函数
    public static double gammaPdf(double x, double alpha, double beta) {
        if (x < 0) return 0; // Gamma分布仅定义在 x >= 0

        double gammaFunc = gamma(alpha); // 伽马函数的计算
        return Math.pow(x / beta, alpha - 1) * Math.exp(-x / beta) / (gammaFunc * Math.pow(beta, alpha));
    }

    // 计算伽马函数的近似值（使用斯特林近似）
    public static double gamma(double n) {
        if (n == 1 || n == 2) return 1;
        double result = 1;
        for (double i = 1; i < n; i++) {
            result *= i;
        }
        return result;
    }

    // 求概率密度最高的数据对应的index
    public static int getMaxDensityIndex(List<double[]> data, double alpha, double beta) {
        double maxDensity = Double.NEGATIVE_INFINITY;
        int maxIndex = -1;

        for (int i = 0; i < data.size(); i++) {
            double[] point = data.get(i);
            double x = point[0]; // 获取数据点
            double density = gammaPdf(x, alpha, beta); // 计算该数据点的概率密度

            if (density > maxDensity) {
                maxDensity = density;
                maxIndex = i;
            }
        }

        return maxIndex; // 返回最大概率密度的索引
    }

    public static void main(String[] args) {
        // 示例数据
        List<double[]> data = new ArrayList<>();
        data.add(new double[]{1.5});
        data.add(new double[]{2.5});
        data.add(new double[]{3.5});
        data.add(new double[]{4.5});
        data.add(new double[]{5.5});

        // 伽马分布的形状参数 alpha 和 尺度参数 beta
        double alpha = 3;
        double beta = 2;

        // 找到最大概率密度对应的索引
        int maxIndex = getMaxDensityIndex(data, alpha, beta);
        System.out.println("概率密度最高的点的索引是: " + maxIndex);
    }
}