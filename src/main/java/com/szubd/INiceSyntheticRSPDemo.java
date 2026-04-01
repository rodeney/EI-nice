package com.szubd;

import javax.swing.*;
import java.util.*;
import java.util.stream.*;

public class INiceSyntheticRSPDemo {
    private static final int TOTAL_DATA_POINTS = 1000;
    private static final int RSP_BLOCK_SIZE = 200;
    private static final int TRUE_CLUSTERS = 4;

    public static void main(String[] args) {
        // 1. Generate global dataset
        List<double[]> globalData = generateGlobalDataset(TOTAL_DATA_POINTS, TRUE_CLUSTERS);

        // 2. Create RSP blocks
        List<List<double[]>> rspBlocks = createRSPBlocks(globalData, RSP_BLOCK_SIZE);
        System.out.printf("Created %d RSP blocks of size %d%n",
                rspBlocks.size(), RSP_BLOCK_SIZE);

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
            JFrame frame = new JFrame("Block:"+i);
            ScatterPlot plot = new ScatterPlot(block, centers,4, 15); // 红点 15px，蓝点 7px
            frame.add(plot);
            frame.setSize(500, 500);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setVisible(true);
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
        JFrame frame = new JFrame("All Data");
        ScatterPlot plot = new ScatterPlot(globalData, mergedCenters,4, 15); // 红点 15px，蓝点 7px
        frame.add(plot);
        frame.setSize(500, 500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }

    private static class BlockResult {
        int blockId;
        List<double[]> centers;

        public BlockResult(int blockId, List<double[]> centers) {
            this.blockId = blockId;
            this.centers = centers;
        }
    }

    private static List<double[]> generateGlobalDataset(int numPoints, int numClusters) {
        List<double[]> data = new ArrayList<>();
        Random rand = new Random(42);
        double clusterSpread = 10.0;

        for (int i = 0; i < numPoints; i++) {
            int cluster = i % numClusters;
            double[] center = {cluster * clusterSpread, cluster * clusterSpread};

            data.add(new double[] {
                    center[0] + rand.nextGaussian() * 1.5,
                    center[1] + rand.nextGaussian() * 1.5
            });
        }
        return data;
    }

    private static List<List<double[]>> createRSPBlocks(List<double[]> globalData, int blockSize) {
        Collections.shuffle(globalData);
        List<List<double[]>> blocks = new ArrayList<>();

        for (int i = 0; i < globalData.size(); i += blockSize) {
            int end = Math.min(i + blockSize, globalData.size());
            blocks.add(new ArrayList<>(globalData.subList(i, end)));
        }
        return blocks;
    }

    private static void printCenters(List<double[]> centers, String title) {
        System.out.println("\n" + title + ":");
        System.out.println("-----------------------");
        for (int i = 0; i < centers.size(); i++) {
            System.out.printf("Center %d: [%.2f, %.2f]%n",
                    i + 1, centers.get(i)[0], centers.get(i)[1]);
        }
    }
}

