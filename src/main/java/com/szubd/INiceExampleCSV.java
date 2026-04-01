package com.szubd;

import javax.swing.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.io.BufferedWriter;

/*
- load one block
- run inice
- apply kmeans (just as a test. this should be run with merged centers after GO)
 */
public class INiceExampleCSV {

    public static void main(String[] args) throws IOException {
        // 1. Read data from CSV file
        String csvFile = "data/dtm20052006_filtered_newVectorSize_svd100_csv/part-00009-4e7ec99b-a9d0-4d05-abd6-770cce99e6d2-c000.csv"; // Replace with your file path
        List<double[]> data = readCSV(csvFile);

        if (data.isEmpty()) {
            System.err.println("No data loaded from CSV file!");
            return;
        }

        System.out.println("Loaded " + data.size() + " data points with "
                + data.get(0).length + " dimensions");

        // 2. Set algorithm parameters
        int numObservationPoints = 2;  // Number of observation points
        int k = 5;                    // Number of neighbors for density estimation

        // 3. Run clustering
        List<double[]> clusterCenters = INice.fitMO(data, numObservationPoints, k);


        // 4. Print all centers
        System.out.println("\nIdentified Cluster Centers:");
        for (int i = 0; i < clusterCenters.size(); i++) {
            System.out.printf("Center %2d: %s%n",
                    i+1,
                    Arrays.toString(clusterCenters.get(i)));
        }

        //save local center
        saveToCSV(clusterCenters, "data/svd100_results/part9_centers.csv");

        // 将合并后的聚类中心作为初始聚类中心，用kmeans的方法获得最终的聚类结果
        KMeans kmeans = new KMeans(clusterCenters);
        int[] labels = kmeans.fit(data);
        System.out.println("KMeans Array elements:");
        for (int i = 0; i < labels.length; i++) {
            System.out.print(labels[i] + " ");
        }
        saveIntArrayToCsv(labels, "data/svd100_results/part9_labels.csv");


        //  Test merging clusters (this should be done as a GO operation on centers from multiple RSP blocks in production)
        List<double[]> mergedClusterCenters = INice.mergeData(clusterCenters, 0.3);
        System.out.println("\n Cluster Centers (after distance-based merge):");
        for (int i = 0; i < mergedClusterCenters.size(); i++) {
            System.out.printf("Center %2d: %s%n",
                    i+1,
                    Arrays.toString(mergedClusterCenters.get(i)));
        }
        //save mergedclusters
        saveToCSV(mergedClusterCenters, "data/svd100_results/part9_mergedCenters.csv");
        //apply kmeans
        KMeans kmeansMerged = new KMeans(clusterCenters);
        int[] labelsMerged = kmeansMerged.fit(data);
        System.out.println("kmeansMerged Array elements:");
        for (int i = 0; i < labelsMerged.length; i++) {
            System.out.print(labelsMerged[i] + " ");
        }
        saveIntArrayToCsv(labelsMerged, "data/svd100_results/part9_mergedLabels.csv");



        // save labels

        //plot (for 2D)
//        JFrame frame = new JFrame("2D Scatter Plot");
//        ScatterPlot plot = new ScatterPlot(data, mergedClusterCenters,4, 15); // 红点 15px，蓝点 7px
//        frame.add(plot);
//        frame.setSize(500, 500);
//        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame.setVisible(true);
    }


    // Helper method to read CSV file (assuming no header and numeric values only)
    private static List<double[]> readCSV(String filename) {
        List<double[]> data = new ArrayList<>();

        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                double[] point = new double[values.length];

                for (int i = 0; i < values.length; i++) {
                    point[i] = Double.parseDouble(values[i].trim());
                }
                data.add(point);
            }
        } catch (IOException e) {
            System.err.println("Error reading CSV file: " + e.getMessage());
        } catch (NumberFormatException e) {
            System.err.println("Error parsing number in CSV: " + e.getMessage());
        }

        return data;
    }

    /**
     * Saves an int array to a CSV file with one column
     * @param data the integer array to save
     * @param filename the output CSV filename
     * @throws IOException if file cannot be written
     */
    public static void saveIntArrayToCsv(int[] data, String filename) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            for (int value : data) {
                writer.write(String.valueOf(value));
                writer.newLine();
            }
        }
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
