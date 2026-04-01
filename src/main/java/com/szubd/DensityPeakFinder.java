package com.szubd;

import org.apache.commons.math3.analysis.function.Gaussian;
import org.apache.commons.math3.stat.StatUtils;
import java.util.ArrayList;
import java.util.List;

public class DensityPeakFinder {

    /**
     * Estimates the initial number of clusters in a dataset by analyzing density peaks
     * in the distribution of pairwise distances. This implementation uses Kernel Density
     * Estimation (KDE) with automatic bandwidth selection.
     *
     * @param distances Array of pairwise distances between data points
     * @return Estimated number of clusters (count of density peaks)
     *
     */
    public static int estimateInitialClusterCount(double[] distances) {
        // --- Step 1: Prepare sorted data ---
        // Create defensive copy to avoid modifying input array
        double[] sortedDistances = distances.clone();
        // Sorting enables reliable peak detection in subsequent steps
        java.util.Arrays.sort(sortedDistances);

        // --- Step 2: Calculate optimal smoothing ---
        double bandwidth = calculateBandwidth(sortedDistances);

        // --- Step 3: Compute Kernel Density Estimation ---
        // Applies Gaussian kernel smoothing to reveal underlying distribution
        double[] kde = computeKDE(sortedDistances, bandwidth);

        // --- Step 4. Find peaks in the density curve ---
        List<Double> peaks = findPeaks(kde, sortedDistances);

        // The number of stable peaks indicates the likely number of clusters
        return peaks.size();
    }

    /**
     * Calculates the optimal bandwidth for Kernel Density Estimation (KDE)
     * using Silverman's rule of thumb for Gaussian kernels.
     *
     * Formula:
     *    h = 1.06 * σ * n^(-1/5)
     *
     * Where:
     *    h  : Optimal bandwidth (smoothing parameter)
     *    σ  : Standard deviation of the input data
     *    n  : Number of data points
     *    1.06: Scaling constant optimal for Gaussian distributions
     *    n^(-1/5): Sample size adjustment factor (decays as n increases)
     *
     * Reference: Silverman, B.W. (1986). Density Estimation for Statistics and Data Analysis.
     *            https://doi.org/10.1007/978-1-4899-3324-9
     */
    private static double calculateBandwidth(double[] data) {
        // Calculate standard deviation (σ) using Apache Commons Math
        double stdDev = Math.sqrt(StatUtils.variance(data)); // σ = sqrt(variance)

        // Apply Silverman's rule: h = 1.06 * σ * n^(-0.2)
        return 1.06 * stdDev * Math.pow(data.length, -0.2); // n^(-1/5) = n^(-0.2)
    }

    /**
     * Computes Kernel Density Estimation (KDE) using a Gaussian kernel.
     *
     * Formula:
     *   f̂(x) = (1/(n*h)) * Σ K((x - x_i)/h)  for i=1 to n
     *
     * Where:
     *   f̂(x) : Estimated density at point x (output)
     *   n    : Number of data points (distances.length)
     *   h    : Bandwidth (smoothing parameter)
     *   K    : Gaussian kernel function K(u) = (1/sqrt(2π)) * e^(-u²/2)
     *   x_i  : ith data point in input distances array
     *   (x - x_i)/h: Scaled distance
     *
     * @param distances Input data points ( distances)
     * @param bandwidth Smoothing parameter (h)
     * @return Array of density estimates f̂(x_i) for each input point
     *
     */
    private static double[] computeKDE(double[] distances, double bandwidth) {
        Gaussian gaussian = new Gaussian(0, 1); // Standard normal kernel
        double[] kde = new double[distances.length];
        double normalizer = distances.length * bandwidth; // n * h

        for (int i = 0; i < distances.length; i++) {
            double sum = 0;
            for (double x : distances) {
                double u = (distances[i] - x) / bandwidth; // (x - x_i)/h
                sum += gaussian.value(u); // K((x - x_i)/h)
            }
            kde[i] = sum / normalizer; // (1/(n*h)) * sum
        }
        return kde;
    }

    /**
     * Identifies local maxima (peaks) in a Kernel Density Estimation (KDE) curve
     * and returns pairs of distance values and their corresponding densities.
     *
     * @param kde       Array of density values from KDE
     * @param distances Sorted array of input distance values
     * @return          List of entries (each represented as Map.Entry) containing both distance (key)
     *                  and density (value) pairs.
     *
     * @throws IllegalArgumentException if array lengths don't match
     */
    private static List<Double> findPeaks(double[] kde, double[] distances) {
        // Validate input arrays
        if (kde.length != distances.length) {
            throw new IllegalArgumentException(
                    "KDE and distances arrays must have equal length");
        }

        List<Double> peaks = new ArrayList<>();

        // Iterate through interior points (skip boundaries)
        for (int i = 1; i < kde.length - 1; i++) {
            // Check if current point is a local maximum
            if (kde[i] > kde[i-1] && kde[i] > kde[i+1]) {
                // Add corresponding distance value (not the density value)
                peaks.add(distances[i]);
            }
        }
        return peaks;
    }


    public static void main(String[] args) {
        double[] testData = {1.2, 1.5, 1.7, 2.1, 2.3, 3.5, 3.8, 4.0, 4.5, 5.1};

        int clusterCount = estimateInitialClusterCount(testData);
        System.out.println("Estimated clusters: " + clusterCount);

        // If you also want to see the peak locations:
        double[] sorted = testData.clone();
        java.util.Arrays.sort(sorted);
        List<Double> peaks = findPeaks(
                computeKDE(sorted, calculateBandwidth(sorted)),
                sorted
        );
        System.out.println("Peak locations: " + peaks);
    }
}