package com.szubd;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class INiceExampleEnsemble {

    public static void main(String[] args) {
        // 1. Generate 5 synthetic datasets with 2-3 clusters each
        List<List<double[]>> allDatasets = generateSyntheticDatasets(8);

        // 2. Configure I-nice parameters
        int numObservationPoints = 3;  // Observation points per dataset
        int kNeighbors = 3;            // Neighbors for density estimation
        double mergePercentage = 0.2;  // Aggressiveness of merging (20th percentile)

        // 3. Process each dataset with I-nice-MO
        List<double[]> allCenters = new ArrayList<>();
        //INiceMO inice = new INiceMO();

        System.out.println("Processing datasets...");
        for (int i = 0; i < allDatasets.size(); i++) {
            List<double[]> dataset = allDatasets.get(i);
            List<double[]> centers = INice.fitMO(dataset, numObservationPoints, kNeighbors);
            allCenters.addAll(centers);
            System.out.printf("Dataset %d: Found %d centers%n", i+1, centers.size());

            //plot
            JFrame frame = new JFrame("2D Scatter Plot");
            ScatterPlot plot = new ScatterPlot(dataset, centers,4, 15); // 红点 15px，蓝点 7px
            frame.add(plot);
            frame.setSize(500, 500);
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setVisible(true);
        }

        // 4. Merge all centers
        System.out.printf("%nBefore merging: %d centers%n", allCenters.size());
        List<double[]> mergedCenters = INice.mergeData(allCenters, mergePercentage);
        System.out.printf("After merging: %d centers%n", mergedCenters.size());

        // 5. Print final centers
        System.out.println("\nFinal merged centers:");
        for (int i = 0; i < mergedCenters.size(); i++) {
            System.out.printf("Center %d: %s%n",
                    i+1, Arrays.toString(mergedCenters.get(i)));
        }


    }

    // Helper: Generate synthetic datasets with 2-3 Gaussian clusters
    private static List<List<double[]>> generateSyntheticDatasets(int numDatasets) {
        List<List<double[]>> datasets = new ArrayList<>();
        Random rand = new Random(42); // Fixed seed for reproducibility

        for (int i = 0; i < numDatasets; i++) {
            List<double[]> dataset = new ArrayList<>();
            int numClusters = 2 + rand.nextInt(2); // 2 or 3 clusters

            for (int c = 0; c < numClusters; c++) {
                double centerX = 10 * c + rand.nextDouble() * 2;
                double centerY = 10 * c + rand.nextDouble() * 2;

                // Generate 20 points per cluster
                for (int p = 0; p < 20; p++) {
                    double[] point = {
                            centerX + rand.nextGaussian(),
                            centerY + rand.nextGaussian()
                    };
                    dataset.add(point);
                }
            }
            datasets.add(dataset);
        }
        return datasets;
    }
}
