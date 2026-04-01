package com.szubd;


import javax.swing.*;
import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import javax.swing.*;
import java.util.*;
import java.util.stream.*;

/*
- read local centers stored in files - each file contains local centers from one block
- merge local centers (as we do in the GO operation)
- save merged centers (to use later to cluster the whole data)
 */

public class INiceRedditMergeDemo {


    public static void main(String[] args) throws IOException {
        String rspCentersDir = "data/svd100_localCenters/";

        // read rsp centers
        List<List<double[]>> rspCenters = readCSVFilesFromDirectory(rspCentersDir);

        // Merge all blocks (for plotting purposes only)
        List<double[]> allCenters = rspCenters.stream()
                .flatMap(List::stream)
                .collect(Collectors.toList());



        System.out.printf("%nTotal centers before merging: %d%n", allCenters.size());
        //printCenters(allCenters, "All Local Centers");
        for (int i = 0; i < allCenters.size(); i++) {
            System.out.printf("Center %2d: %s%n",
                    i+1,
                    Arrays.toString(allCenters.get(i)));
        }

        // Merge centers
        double mergePercentile = 0.3;
        List<double[]> mergedCenters = INice.mergeData(allCenters, mergePercentile);

        System.out.printf("After merging: %d final centers%n%n", mergedCenters.size());
        //printCenters(mergedCenters, "Final Merged Centers");
        System.out.println("\n Cluster Centers (after distance-based merge):");
        for (int i = 0; i < mergedCenters.size(); i++) {
            System.out.printf("Center %2d: %s%n",
                    i+1,
                    Arrays.toString(mergedCenters.get(i)));
        }

        //saveToCSV(mergedCenters, "data/svd100_results/10blocks_mergedCenters.csv");

        //try ensmeble without averaging
        List<double[]> ensembleClusterCenters = INice.ensembleCenters(allCenters, 0.3);
        System.out.println("\n Cluster Centers (after distance-based ensemble):");
        for (int i = 0; i < ensembleClusterCenters.size(); i++) {
            System.out.printf("Center %2d: %s%n",
                    i+1,
                    Arrays.toString(ensembleClusterCenters.get(i)));
        }

        //saveToCSV(mergedCenters, "data/svd100_results/10blocks_mergedCenters.csv");

        //apply kmeans
        // load test data
        String csvFile = "data/dtm20052006_filtered_newVectorSize_svd100_csv/part-00009-4e7ec99b-a9d0-4d05-abd6-770cce99e6d2-c000.csv"; // Replace with your file path
        List<double[]> data = readCSV(csvFile);
        KMeans kmeansMerged = new KMeans(ensembleClusterCenters);
        int[] labelsMerged = kmeansMerged.fit(data);
        System.out.println("kmeansMerged Array elements:");
        for (int i = 0; i < labelsMerged.length; i++) {
            System.out.print(labelsMerged[i] + " ");
        }
        saveIntArrayToCsv(labelsMerged, "data/svd100_results/10blocks_mergedLabels.csv");
    }

    /**
     * Reads all CSV files from a directory where each file is a separate RSP block
     * @param directoryPath Path to the directory containing CSV files
     * @return List where each file's data is kept as a separate list
     */
    private static List<List<double[]>> readCSVFilesFromDirectory(String directoryPath) {
        List<List<double[]>> blocks = new ArrayList<>();

        File directory = new File(directoryPath);
        if (!directory.exists() || !directory.isDirectory()) {
            System.err.println("Error: Directory does not exist or is not a directory: " + directoryPath);
            return blocks;
        }

        // Get all CSV files from the directory
        File[] csvFiles = directory.listFiles((dir, name) ->
                name.toLowerCase().endsWith(".csv"));

        if (csvFiles == null || csvFiles.length == 0) {
            System.out.println("No CSV files found in directory: " + directoryPath);
            return blocks;
        }

        // Sort files by name for consistent ordering
        Arrays.sort(csvFiles);

        System.out.printf("Found %d CSV files in directory:%n", csvFiles.length);
        for (File file : csvFiles) {
            System.out.println("  - " + file.getName());
        }

        // Read each CSV file
        for (File csvFile : csvFiles) {
            List<double[]> fileData = readCSV(csvFile.getAbsolutePath());
            if (!fileData.isEmpty()) {
                blocks.add(fileData);
                System.out.printf("Successfully read %d data points from %s%n",
                        fileData.size(), csvFile.getName());
            } else {
                System.out.printf("Warning: No data read from %s%n", csvFile.getName());
                blocks.add(new ArrayList<>()); // Add empty block to maintain order
            }
        }

        return blocks;
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
}



