package com.szubd;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;
import javax.swing.*;
import java.util.*;
import java.util.stream.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;

public class INiceRedditRSPDemo {


    public static void main(String[] args) {
        String rspDir = "data/dtm20052006_filtered_newVectorSize_svd100_csv/";

        // 1. read rsp blocks
        List<List<double[]>> rspBlocks = readCSVFilesFromDirectory(rspDir);

//        // 2. Merge all blocks (for plotting purposes only)
//        List<double[]> globalData = rspBlocks.stream()
//                .flatMap(List::stream)
//                .collect(Collectors.toList());

        // 3. Process each RSP block separately and store results
        //INice inice = new INice();
        int numObservationPoints = 3;
        int kNeighbors = 3;

        // Store both the centers and their source block
        List<BlockResult> blockResults = new ArrayList<>();

        for (int i = 0; i < rspBlocks.size(); i++) {
            List<double[]> block = rspBlocks.get(i);
            List<double[]> centers = INice.fitMO(block, numObservationPoints, kNeighbors);
            blockResults.add(new BlockResult(i, centers));

            System.out.printf("Block %d: Found %d centers%n", i, centers.size());
            printCenters(centers, "Block " + i + " centers");

            //plot
//            JFrame frame = new JFrame("Block:"+i);
//            ScatterPlot plot = new ScatterPlot(block, centers,4, 15); // 红点 15px，蓝点 7px
//            frame.add(plot);
//            frame.setSize(500, 500);
//            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//            frame.setVisible(true);
        }

        // 4. Combine all centers for merging
        List<double[]> allCenters = blockResults.stream()
                .flatMap(br -> br.centers.stream())
                .collect(Collectors.toList());

        System.out.printf("%nTotal centers before merging: %d%n", allCenters.size());

        // 5. Merge centers
        double mergePercentile = 0.3;
        List<double[]> mergedCenters = INice.mergeData(allCenters, mergePercentile);

        System.out.printf("After merging: %d final centers%n%n", mergedCenters.size());
        printCenters(mergedCenters, "Final Merged Centers");

        //plot
//        JFrame frame = new JFrame("All Data");
//        ScatterPlot plot = new ScatterPlot(globalData, mergedCenters,4, 15); // 红点 15px，蓝点 7px
//        frame.add(plot);
//        frame.setSize(500, 500);
//        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
//        frame.setVisible(true);
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

    private static class BlockResult {
        int blockId;
        List<double[]> centers;

        public BlockResult(int blockId, List<double[]> centers) {
            this.blockId = blockId;
            this.centers = centers;
        }
    }


    private static void printCenters(List<double[]> centers, String title) {
        System.out.println("\n" + title + ":");
        System.out.println("-----------------------");
        for (int i = 0; i < centers.size(); i++) {
            System.out.printf("Center %d: [%.2f, %.2f]%n",
                    i + 1, centers.get(i)[0], centers.get(i)[1]);
        }
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
}


