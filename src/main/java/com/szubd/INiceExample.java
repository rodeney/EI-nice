package com.szubd;

import javax.swing.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class INiceExample {
    public static void main(String[] args) {
        // Create synthetic data with 3 clusters
        List<double[]> data = new ArrayList<>();

        // Cluster 1 (around [1, 1])
        data.add(new double[]{1.0, 1.0});
        data.add(new double[]{1.1, 0.9});
        data.add(new double[]{0.9, 1.1});
        data.add(new double[]{1.2, 0.8});

        // Cluster 2 (around [4, 4])
        data.add(new double[]{4.0, 4.0});
        data.add(new double[]{4.1, 3.9});
        data.add(new double[]{3.9, 4.1});

        // Cluster 3 (around [8, 1])
        data.add(new double[]{8.0, 1.0});
        data.add(new double[]{8.2, 1.1});
        data.add(new double[]{7.9, 0.9});


        // Set parameters
        int numObservationPoints = 3;  // Use 3 observation points
        int k = 3;  // Number of neighbors for density estimation

        // Run the clustering
        List<double[]> clusterCenters = INice.fitMO(data, numObservationPoints, k);

        // Print results
        System.out.println("Identified Cluster Centers:");
        for (int i = 0; i < clusterCenters.size(); i++) {
            System.out.printf("Center %d: %s%n",
                    i+1,
                    Arrays.toString(clusterCenters.get(i)));
        }

        JFrame frame = new JFrame("2D Scatter Plot");
        ScatterPlot plot = new ScatterPlot(data, clusterCenters,4, 15); // 红点 15px，蓝点 7px
        frame.add(plot);
        frame.setSize(500, 500);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}
