package com.szubd;


import java.util.*;
import java.util.stream.Collectors;

public class INiceResultExample {
    public static void main(String[] args) {
        // Generate synthetic 2D clustering data (3 clusters, 50 points each)
        List<double[]> syntheticData = generateSyntheticClusters(3, 50);

        // Execute clustering
        List<INice.INiceResult> results = INice.fitMO_WithDetails(syntheticData, 5, 10);

        // 4. Display results using the enhanced toString()
        System.out.println("=== INiceMO Clustering Results ===");
        System.out.printf("Analyzed %d data points\n\n", syntheticData.size());

        for (int i = 0; i < results.size(); i++) {
            System.out.println("----- Observation " + (i+1) + " -----");
            System.out.println(results.get(i));  // Uses our modified toString()
            System.out.println();
        }

        // Show final ensemble centers
        System.out.println("=== Final Consensus Centers ===");
        List<double[]> finalCenters = INice.mergeData(INice.getAllCenters(results), 1.0);  // 1.0 distance threshold
        finalCenters.forEach(center ->
                System.out.println(" • " + Arrays.toString(center)));
    }


    // Helper methods
    /**
     * Generates synthetic cluster data as List<double[]>
     * @param numClusters Number of clusters to generate
     * @param pointsPerCluster Points per cluster
     * @return List of data points where each point is a double[]
     */
    private static List<double[]> generateSyntheticClusters(int numClusters, int pointsPerCluster) {
        Random rand = new Random();
        List<double[]> data = new ArrayList<>();

        // Define cluster centroids
        double[][] centroids = {
                {2.0, 2.0},  // Cluster 1 center
                {8.0, 3.0},  // Cluster 2 center
                {5.0, 8.0}   // Cluster 3 center
        };

        // Generate points around each centroid
        for (double[] centroid : centroids) {
            for (int j = 0; j < pointsPerCluster; j++) {
                double[] point = new double[]{
                        centroid[0] + rand.nextGaussian() * 0.8,
                        centroid[1] + rand.nextGaussian() * 0.8
                };
                data.add(point);
            }
        }

        // Shuffle the data points
        Collections.shuffle(data);
        return data;
    }

}