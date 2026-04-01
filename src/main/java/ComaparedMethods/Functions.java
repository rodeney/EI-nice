package ComaparedMethods;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Functions {

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
}
