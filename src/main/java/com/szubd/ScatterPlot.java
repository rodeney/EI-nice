package com.szubd;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;

// 计算结果可视化
public class ScatterPlot extends JPanel {
    private final List<double[]> redPoints;
    private final List<double[]> bluePoints;
    private final int redSize;
    private final int blueSize;

    public ScatterPlot(List<double[]> redPoints, List<double[]> bluePoints, int redSize, int blueSize) {
        this.redPoints = redPoints;
        this.bluePoints = bluePoints;
        this.redSize = redSize;
        this.blueSize = blueSize;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2d = (Graphics2D) g;

        // 设置绘制区域的缩放比例
        int width = getWidth();
        int height = getHeight();
        double xMin = Double.MAX_VALUE, xMax = Double.MIN_VALUE;
        double yMin = Double.MAX_VALUE, yMax = Double.MIN_VALUE;

        // 计算数据的边界
        for (double[] p : redPoints) {
            xMin = Math.min(xMin, p[0]);
            xMax = Math.max(xMax, p[0]);
            yMin = Math.min(yMin, p[1]);
            yMax = Math.max(yMax, p[1]);
        }
        for (double[] p : bluePoints) {
            xMin = Math.min(xMin, p[0]);
            xMax = Math.max(xMax, p[0]);
            yMin = Math.min(yMin, p[1]);
            yMax = Math.max(yMax, p[1]);
        }

        // 防止数据点过于贴近边缘
        xMin -= 1; xMax += 1;
        yMin -= 1; yMax += 1;

        // 绘制红色大点
        g2d.setColor(Color.BLUE);
        for (double[] p : redPoints) {
            int x = (int) ((p[0] - xMin) / (xMax - xMin) * width);
            int y = (int) ((p[1] - yMin) / (yMax - yMin) * height);
            g2d.fillOval(x - redSize / 2, height - y - redSize / 2, redSize, redSize);
        }

        // 绘制蓝色小点
        g2d.setColor(Color.RED);
        for (double[] p : bluePoints) {
            int x = (int) ((p[0] - xMin) / (xMax - xMin) * width);
            int y = (int) ((p[1] - yMin) / (yMax - yMin) * height);
            g2d.fillOval(x - blueSize / 2, height - y - blueSize / 2, blueSize, blueSize);
        }
    }

    public static void main(String[] args) {
        // 示例数据
        List<double[]> redPoints = new ArrayList<>();
        redPoints.add(new double[]{1.2, 3.4});
        redPoints.add(new double[]{2.5, 4.1});
        redPoints.add(new double[]{3.7, 2.8});

        List<double[]> bluePoints = new ArrayList<>();
        bluePoints.add(new double[]{4.0, 1.5});
        bluePoints.add(new double[]{5.2, 3.9});
        bluePoints.add(new double[]{6.1, 2.3});

        JFrame frame = new JFrame("2D Scatter Plot");
        ScatterPlot plot = new ScatterPlot(redPoints, bluePoints, 15, 7); // 红点 15px，蓝点 7px
        frame.add(plot);
        frame.setSize(500, 500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}
