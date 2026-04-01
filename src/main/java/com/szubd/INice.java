package com.szubd;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;
import org.apache.commons.math3.analysis.function.Gaussian;
import org.apache.commons.math3.stat.StatUtils;
import pers.hongweiye.gammaem.mixtool.GammamixEM;
import pers.hongweiye.gammaem.mixtool.MixEM;
import pers.hongweiye.gammaem.mixtool.util.GammaUtil;

/**
 * Implements I-nice-MO algorithm
 * Key Features:
 * 1. Uses multiple observation points to estimate cluster centers.
 * 2. Estimates the initial number of clusters by analyzing density peaks in the distribution of pairwise distances.
 * 3. Fits Gamma Mixture Models (GMMs) to distance distributions.
 * 4. Selects optimal GMM for each observation point via AICc (corrected Akaike Information Criterion).
 * 5. Identifies cluster centers via density peaks with KNN.
 */

public class INice {

    /**
     * Constants for the I-nice clustering algorithm.
     */
    // Observation Points
    private static final int DEFAULT_OBSERVATION_POINTS = 5;    // umber of observation points
    //Kernel Density Estimation (KDE)
    private static final double KDE_MIN_BANDWIDTH  = 1e-5; // Prevent division by zero in Kernel Density Estimation (KDE)
    private static final double PEAK_SIGNIFICANCE_RATIO = 1.25; // Peak must be 25% higher than neighbors
    //GammaMM EM
    private final static int maxComponents = 500;        // Max Gamma components to test (maxM)
    private final static int delta = 5;   //search window for number of components
    private static final int EM_MAX_ITERATIONS = 1000;          // Prevent infinite loops
    //Distance Processing
    private static final double ZERO_REPLACEMENT = 0.1;         // Replace 0 distances (GammaMM requirement)
    private static final double MERGE_THRESHOLD_PERCENTILE = 0.15; // For center merging (15th percentile)

    /**
     * Fit the I-nice-MO model to the data (inice with multiple observation points)
     * @param data Input data points (each as double[]).
     * @param k Number of neighbors for density estimation.
     * @return List of cluster centers (double[]). TODO: return INiceModel
     */
    public static List<double[]> fitMO(List<double[]> data, int numObservationPoints, int k) {
        List<double[]> allCenters = new ArrayList<>();

        // Generate P observation points via simple random sampling
        List<double[]> observationPoints = uniformSample(data, numObservationPoints);

        // For each observation point, claculate distances, find initial M, fit GMMs, select best model and find centers
        for (double[] point : observationPoints) {
            // Calculate distance vector with respect to the observation point and all data points
            double[] distances = calculateDistances(data, point);
            // Validate and preprocess
            double[] processedDistances = validateAndPrepareDistances(distances);

            System.out.println("Distance stats - Min: " + Arrays.stream(processedDistances).min()
                    + " Max: " + Arrays.stream(distances).max()
                    + " Mean: " + Arrays.stream(distances).average());

            // Call fitSO: it runs iNice algorithm
            List<double[]> clusterCentersSO = fitSO(data,processedDistances, point, k);

            //add identified centers to the list of all centers
            allCenters.addAll(clusterCentersSO);

        }

        // return all centers identified from all observation points
        return allCenters;

    }

    /**
     * Fit the I-nice model to the data (with a single observer)
     * @param data Input data points.
     * @param distances distance vector.
     * @param observationPoint //may not need this since distances already.
     * @param k Number of neighbors for density estimation with Knn.
     * @return List of cluster centers (double[]).
     */
    public static List<double[]> fitSO(List<double[]> data, double[] distances, double[] observationPoint, int k) {

        List<MixEM> emModels = new ArrayList<>();
        List<Double> aiccList = new ArrayList<>();
        List<double[]> clusterCenters = new ArrayList<>();

        // Calculate the initial number of Gamma mixture components (initial M)：
        // Calculate the density values of the distances vector with a kernel density function and find the number of density peaks
        int mInitial = estimateInitialClusterCount(distances);
        System.out.println("mInitial:"+mInitial);

        // Set search range ( to avoid M < 1 or large M)
//        int minM = Math.max(1, mInitial - delta);
//        int maxM = Math.min(maxComponents, mInitial + delta);


        int minM;
        int maxM;

        if (Math.max(1, mInitial - delta) >= maxComponents) {
            // This case handles when even the minimum possible value exceeds maxComponents
            minM = Math.max(1, maxComponents - 2 * delta);
            maxM = maxComponents;
        } else {
            // Normal bounded range
            minM = Math.max(1, mInitial - delta);
            maxM = Math.min(maxComponents, minM + 2 * delta);
        }

        System.out.println("Min M:"+minM);
        System.out.println("Max M:"+maxM);

        // Model Fitting and AICc Calculation
        for (int m = minM; m <= maxM; m++) {
            // Fit a GMM model with m components to the distances vector
            // Initialize parameters (lambda, alpha, beta) for the Gamma Mixture Model
            double[] lambda = new double[m];
            Arrays.fill(lambda, 1.0 / m); // Uniform initial weights
            double[][] out = GammamixEM.gammamixInit(distances, lambda, null, null, lambda.length);
            System.out.println("alpha:"+Arrays.toString(out[0]));
            System.out.println("beta:"+Arrays.toString(out[1]));
            MixEM em = GammamixEM.gammamixEM(distances, lambda, out[0], out[1], m, 0.1, 1000, 10, false);
            emModels.add(em);

            // AICc Calculation
            double AICc = calculateAICc(em.loglik, m, distances.length);
            aiccList.add(AICc);
        }
        // Model Selection: Choose the best GMM model (the one with the smallest AICc)
        if (! aiccList.isEmpty()) {
            MixEM bestEM = emModels.get(aiccList.indexOf(Collections.min(aiccList)));

            // Cluster data and find cluster centers: identify cluster centers in the selected model using Knn
            int[] labels = classifyVector(distances, bestEM.lambda, bestEM.alpha, bestEM.beta);
            List<List<double[]>> clusteredData = clusterData(data, labels, bestEM.lambda.length);
            //KNN的方法
            // 输出每个类别的簇数据
            for (int i = 0; i < bestEM.lambda.length; i++) {
                List<double[]> temp = clusteredData.get(i);
                if (temp.size() > k + 1) {
                    int highestDensityIndex = findHighestDensityPoint(temp, k);
                    clusterCenters.add(temp.get(highestDensityIndex));
                    //here we may need to keep indices of the identified centers (to find them in the original data)
                }
            }
            //return cluster centers (identified from the best GMM trained for the input observation point)
            return clusterCenters;
        }
        return clusterCenters;
    }

//=== same fitSO and fitMO methods but return all details as INiceResult obkects ===
    /**
     * Fit the I-nice-MO model to the data (inice with multiple observation points). Similar to fitMO but returns all details as INiceResult.
     * @param data Input data points (each as double[]).
     * @param k Number of neighbors for density estimation.
     * @return List of INiceResult.
     */
    public static List<INiceResult> fitMO_WithDetails(List<double[]> data, int numObservationPoints, int k) {
        List<double[]> allCenters = new ArrayList<>();

        // Generate P observation points via simple random sampling
        List<double[]> observationPoints = uniformSample(data, numObservationPoints);

        List<INiceResult> results = new ArrayList<>();

        // For each observation point, claculate distances, find initial M, fit GMMs, select best model and find centers
        for (double[] point : observationPoints) {
            // Calculate distance vector with respect to the observation point and all data points
            double[] distances = calculateDistances(data, point);
            // Validate and preprocess
            double[] processedDistances = validateAndPrepareDistances(distances);

            System.out.println("Distance stats - Min: " + Arrays.stream(processedDistances).min()
                    + " Max: " + Arrays.stream(distances).max()
                    + " Mean: " + Arrays.stream(distances).average());

            // Call fitSO: it runs iNice algorithm
            INiceResult resultSO = fitSO_WithDetails(data,processedDistances, point, k);

            // Store result for this observation point
            results.add(resultSO);

        }

        // return all results from all observation points
        return results;

    }

    /**
     * Fit the I-nice model to the data (with a single observer). Similar to fitSO, but returns all details as INiceResult.
     * @param data Input data points.
     * @param distances distance vector.
     * @param observationPoint //may not need this since distances already.
     * @param k Number of neighbors for density estimation with Knn.
     * @return INiceResult.
     */
    public static INiceResult fitSO_WithDetails(List<double[]> data, double[] distances, double[] observationPoint, int k) {

        List<MixEM> emModels = new ArrayList<>();
        List<Double> aiccList = new ArrayList<>();
        List<double[]> clusterCenters = new ArrayList<>();

        // Calculate the initial number of Gamma mixture components (initial M)：
        // Calculate the density values of the distances vector with a kernel density function and find the number of density peaks
        int mInitial = estimateInitialClusterCount(distances);
        System.out.println("mInitial:"+mInitial);

        // Set search range ( to avoid M < 1 or large M)
        int minM = Math.max(1, mInitial - delta);
        int maxM = Math.min(maxComponents, mInitial + delta);
        //int M_max = mInitial + Delta;
        System.out.println("Min M:"+minM);
        System.out.println("Max M:"+maxM);

        // Search for best model around initial estimate
        MixEM bestModel = null;
        double bestAicc = Double.POSITIVE_INFINITY;
        int bestM = mInitial;

        // Model Fitting and AICc Calculation
        for (int m = minM; m <= maxM; m++) {
            // Fit a GMM model with m components to the distances vector
            // Initialize parameters (lambda, alpha, beta) for the Gamma Mixture Model
            double[] lambda = new double[m];
            Arrays.fill(lambda, 1.0 / m); // Uniform initial weights
            double[][] out = GammamixEM.gammamixInit(distances, lambda, null, null, lambda.length);
            System.out.println("alpha:"+Arrays.toString(out[0]));
            System.out.println("beta:"+Arrays.toString(out[1]));
            MixEM em = GammamixEM.gammamixEM(distances, lambda, out[0], out[1], m, 0.1, 100, 10, false);
            emModels.add(em);

            // AICc Calculation
            double aicc = calculateAICc(em.loglik, m, distances.length);
            aiccList.add(aicc);

            //find if this is the best so far
            if (aicc < bestAicc) {
                bestAicc = aicc;
                bestModel = em;
                bestM = m;
            }

        }
        // Model Selection: Choose the best GMM model (the one with the smallest AICc)
        //MixEM bestEM = emModels.get(aiccList.indexOf(Collections.min(aiccList)));

        // Cluster data and find cluster centers: identify cluster centers in the selected model using Knn
        int[] labels = classifyVector(distances, bestModel.lambda, bestModel.alpha, bestModel.beta);
        List<List<double[]>> clusteredData = clusterData(data, labels,bestModel.lambda.length);
        //KNN的方法
        // 输出每个类别的簇数据
        for (int i = 0; i < bestModel.lambda.length; i++) {
            List<double[]> temp = clusteredData.get(i);
            if(temp.size()>k+1){
                int highestDensityIndex = findHighestDensityPoint(temp, k);
                clusterCenters.add(temp.get(highestDensityIndex));}
        }
        //return cluster centers (identified from the best GMM trained for the input observation point)
        return new INiceResult(clusterCenters, bestM, observationPoint, bestModel, bestAicc);
    }

    // ==== Observation Points ====
    // 生成观测点，在样本张成的空间里面均匀抽样
    public static List<double[]> uniformSample(List<double[]> inputData, int numPoints) {
        if (numPoints > inputData.size() || numPoints < 0) {
            throw new IllegalArgumentException("numPoints must be between 0 and inputData.size()");
        }

        // Create a list of indices and shuffle it
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < inputData.size(); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, new Random());

        // Select the first numPoints indices
        List<double[]> sampledData = new ArrayList<>();
        for (int i = 0; i < numPoints; i++) {
            sampledData.add(inputData.get(indices.get(i)));
        }

        return sampledData;
    }

    // ==== Distances Vector ====

    /**
     * Validates and preprocesses distance arrays before cluster estimation.
     * Ensures:
     * 1. No null/negative values
     * 2. At least 3 distances
     * 3. All zeros replaced with 0.1
     * 4. Final sorted array
     *
     * @param distances Input distance array
     * @return Validated, preprocessed, and sorted distance array
     * @throws IllegalArgumentException for invalid inputs
     */
    private static double[] validateAndPrepareDistances(double[] distances) {
        // --- Input Validation ---
        if (distances == null) {
            throw new IllegalArgumentException("Distance array cannot be null");
        }
        if (distances.length < 3) {
            throw new IllegalArgumentException(
                    "At least 3 distances required for density estimation");
        }

        // --- Create defensive copy ---
        double[] processed = distances.clone();

        // --- Zero Handling ---
        for (int i = 0; i < processed.length; i++) {
            if (processed[i] < 0) {
                throw new IllegalArgumentException(
                        "Distances must be non-negative. Found: " + processed[i]);
            }
            if (processed[i] == 0) {
                processed[i] = ZERO_REPLACEMENT;
            }
        }

        // --- Finally, Sort distances into ascending order (this is required for peak detection and GammaMixtureModel EM)  ---
        Arrays.sort(processed);

        return processed;
    }

    // 计算 data_selected 与 dataVectors 中每个点的欧几里得距离，并返回 double[] 数组
    public static double[] calculateDistances(List<double[]> dataVectors, double[] data_selected) {
        int size = dataVectors.size();
        double[] distances = new double[size];
        // 遍历 dataVectors 中的每个数据点，计算与 data_selected 的欧几里得距离
        for (int i = 0; i < size; i++) {
            distances[i] = euclideanDistance(dataVectors.get(i), data_selected);
        }
        return distances;
    }

    public static double euclideanDistance(double[] p1, double[] p2) {
        double sum = 0.0;

        for(int i = 0; i < p1.length; ++i) {
            sum += Math.pow(p1[i] - p2[i], 2.0);
        }

        return Math.sqrt(sum);
    }


// ==== Find Initial Number of Clusters （GMM Components）: mInitial ====

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

        // --- Step 1: Calculate optimal smoothing ---
        double bandwidth = calculateBandwidthEnhanced(distances);

        // --- Step 2: Compute Kernel Density Estimation ---
        // Applies Gaussian kernel smoothing to reveal underlying distribution
        double[] kde = computeKDE(distances, bandwidth);

        // --- Step 3. Find peaks in the density curve ---
        List<Double> peaks = findPeaks(kde, distances);

        // The number of stable peaks indicates the initial number of clusters (GMM components)
        return Math.max(1, peaks.size());//ensure at least one cluster
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
     * Calculates optimal KDE bandwidth using Silverman's enhanced rule:
     *    h = 0.9 * min(σ, IQR/1.34) * n^(-1/5)
     *
     * More robust to outliers than basic Silverman's rule.
     */
    private static double calculateBandwidthEnhanced(double[] data) {

        // --- Calculate Spread Estimates ---
        double stdDev = Math.sqrt(StatUtils.variance(data));
        double iqr = StatUtils.percentile(data, 75) - StatUtils.percentile(data, 25);
        double spreadEstimate = Math.min(stdDev, iqr/1.34);

        // --- Apply Silverman's Rule ---
        double h = 0.9 * spreadEstimate * Math.pow(data.length, -0.2);

        // --- Ensure Numerical Stability by using a minimum bandwidth to prevent division by zero in the KDE---
        return Math.max(h, 1e-5);
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


//  ==== AICc: ====
    /**
     * Calculates the corrected Akaike Information Criterion (AICc) for a Gamma Mixture Model.
     *
     * @param loglik Log-likelihood of the model.
     * @param numComponents Number of components (m) in the mixture model.
     * @param sampleSize Number of data points (n): length of the distance vector in our case.
     * @return AICc value. Returns Double.POSITIVE_INFINITY if denominator is invalid.
     */
    private static double calculateAICc(double loglik, int numComponents, int sampleSize) {
        int numParams = 3 * numComponents; // alpha, beta, lambda for each component
        double numerator = 2 * numParams * sampleSize;
        double denominator = sampleSize - numParams - 1;

        if (denominator <= 0) {
            System.err.println("Warning: Invalid AICc denominator (N=" + sampleSize + ", k=" + numComponents + ")");
            return Double.POSITIVE_INFINITY;
        }
        return -2 * loglik + (numerator / denominator);
    }

    // ==== Final Stage: Find Cluster Centers in GMM components using KNN ====

    public static double[][] findKNearestNeighbors(List<double[]> dataList, int k) {
        int n = dataList.size();
        if (n <= 1 || k <= 0) {
            throw new IllegalArgumentException("Data must contain at least 2 points, and k must be > 0.");
        }

        if (k >= n) {
            System.err.println("Warning: k >= number of points. Reducing k to n - 1.");
            k = n - 1;
        }

        double[][] distances = new double[n][k];

        for (int i = 0; i < n; i++) {
            double[] current = dataList.get(i);
            // 存储所有其他点的距离
            List<Double> distanceList = new ArrayList<>();

            for (int j = 0; j < n; j++) {
                if (i == j) continue; // 跳过自己
                double dist = euclideanDistance(current, dataList.get(j));
                distanceList.add(dist);
            }

            // 排序并取前 k 个最小距离
            distanceList.sort(Double::compareTo);
            for (int d = 0; d < k; d++) {
                distances[i][d] = distanceList.get(d);
            }
        }

        return distances;
    }

    // 找最高密度的点
    public static int findHighestDensityPoint(List<double[]> dataList, int k) {
        double[][] distances = findKNearestNeighbors(dataList, k);
        double maxDensity = -1.0;
        int maxIndex = -1;
        for(int i = 0; i < distances.length; ++i) {
            double avgDistance = Arrays.stream(distances[i]).average().orElse(Double.MAX_VALUE);
            double density = 1.0 / avgDistance;
            if (density > maxDensity) {
                maxDensity = density;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    //////寻找距离变化最快的部分///////
    public static double gammaPdf(double x, double alpha, double beta) {
        return x <= 0.0 ? 0.0 : GammaUtil.dgamma(x, alpha, beta);
    }

    // 根据伽马混合分布结果确定数据属于那个类
    public static int classifyComponent(double x, double[] pi, double[] alpha, double[] beta) {
        int k = pi.length;
        double[] probabilities = new double[k];
        double sumProb = 0.0;

        int bestComponent;
        for(bestComponent = 0; bestComponent < k; ++bestComponent) {
            probabilities[bestComponent] = pi[bestComponent] * gammaPdf(x, alpha[bestComponent], beta[bestComponent]);
            sumProb += probabilities[bestComponent];
        }
        bestComponent = 0;
        double maxProb = 0.0;
        for(int i = 0; i < k; ++i) {
            probabilities[i] /= sumProb;
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                bestComponent = i;
            }
        }
        return bestComponent;
    }

    public static int[] classifyVector(double[] data, double[] pi, double[] alpha, double[] beta) {
        int[] labels = new int[data.length];

        for(int i = 0; i < data.length; ++i) {
            labels[i] = classifyComponent(data[i], pi, alpha, beta);
        }
        return labels;
    }
    public static List<List<double[]>> clusterData(List<double[]> dataVectors, int[] labels, int k) {
        List<List<double[]>> clusters = new ArrayList(k);

        int i;
        for(i = 0; i < k; ++i) {
            clusters.add(new ArrayList());
        }

        for(i = 0; i < dataVectors.size(); ++i) {
            int label = labels[i];
            ((List)clusters.get(label)).add((double[])dataVectors.get(i));
        }
        return clusters;
    }


    // ==== Merge Clusters ====

    public static double[][] computeDistanceMatrix(List<double[]> centersAll) {
        int n = centersAll.size();
        double[][] distanceMatrix = new double[n][n];
        for(int i = 0; i < n; ++i) {
            for(int j = i + 1; j < n; ++j) {
                double dist = euclideanDistance((double[])centersAll.get(i), (double[])centersAll.get(j));
                distanceMatrix[i][j] = dist;
                distanceMatrix[j][i] = dist;
            }
        }
        return distanceMatrix;
    }

    /**
     * Computes a merging threshold based on a specified percentile of pairwise distances.
     *
     * <p>The method:
     * 1. Extracts all unique pairwise distances from the upper triangle of the distance matrix
     * 2. Sorts the distances in ascending order
     * 3. Calculates the threshold as the average of the smallest P% distances,
     *    where P is the specified percentage
     *
     * @param distanceMatrix N x N symmetric matrix of pairwise distances.
     * @param percentage The percentile threshold (0.0 to 1.0) specifying how many
     *                   of the smallest distances to consider. For example:
     *                   - 0.1 = average of smallest 10% of distances
     *                   - 0.5 = average of smallest 50% of distances
     * @return The computed threshold value for merging cluster centers
     *
     */
    public static double computeThreshold(double[][] distanceMatrix, double percentage) {
        // Validate input percentage range
        if (percentage < 0.0 || percentage > 1.0) {
            throw new IllegalArgumentException("Percentage must be between 0.0 and 1.0");
        }

        // Step 1: Extract unique pairwise distances from upper triangle of matrix
        // (excluding diagonal and lower triangle to avoid duplicates)
        List<Double> distances = new ArrayList<>();
        int n = distanceMatrix.length;

        for (int i = 0; i < n; ++i) {
            // Only process j > i to avoid diagonal and duplicate distances
            for (int j = i + 1; j < n; ++j) {
                distances.add(distanceMatrix[i][j]);
            }
        }

        // Step 2: Sort distances in ascending order to find percentile cutoff
        Collections.sort(distances);

        // Step 3: Determine how many distances to include based on percentage
        // Ensure at least 1 distance is considered (Math.max(1, ...))
        int numThreshold = Math.max(1, (int)(distances.size() * percentage));

        // Step 4: Calculate average of the smallest 'numThreshold' distances
        double sum = 0.0;
        for (int i = 0; i < numThreshold; ++i) {
            sum += distances.get(i);
        }

        return sum / numThreshold;
    }

    /**
     * Merges similar cluster centers identified from multiple observation points in the I-nice algorithm.
     *
     * <p>This method:
     * 1. Computes pairwise distances between all cluster centers
     * 2. Determines a merging threshold based on a specified percentile of distances
     * 3. Merges centers closer than the threshold by averaging their coordinates
     * 4. Returns consolidated centers
     *
     * @param centersAll List of all candidate cluster centers from multiple observations,
     *                   where each center is a double[] array of coordinates
     * @param percentage The percentile value (0-1) used to determine the merging threshold.
     *                   Lower values result in more aggressive merging (e.g., 0.1 merges more centers than 0.3)
     * @return List of merged cluster centers with averaged coordinates
     *
     */
    public static List<double[]> mergeData(List<double[]> centersAll, double percentage) {
        // Validate input parameters
        if (centersAll == null || centersAll.isEmpty()) {
            return new ArrayList<>();
        }
        if (percentage < 0 || percentage > 1) {
            throw new IllegalArgumentException("Percentage must be between 0 and 1");
        }

        // Step 1: Compute pairwise distance matrix between all centers
        // matrix[i][j] = Euclidean distance between centersAll[i] and centersAll[j]
        double[][] distanceMatrix = computeDistanceMatrix(centersAll);

        // Step 2: Calculate merging threshold as specified percentile of all distances
        // Example: percentage=0.15 uses the 15th percentile distance as threshold
        double threshold = computeThreshold(distanceMatrix, percentage);

        // Initialize working structures
        List<double[]> mergedCenters = new ArrayList<>(centersAll); // Copy of original centers
        boolean[] merged = new boolean[centersAll.size()];          // Tracks merged status

        // Step 3: Perform center merging
        for (int i = 0; i < centersAll.size(); ++i) {
            if (!merged[i]) {  // Only process unmerged centers
                double[] centerA = centersAll.get(i);

                for (int j = i + 1; j < centersAll.size(); ++j) {
                    if (!merged[j]) {  // Only compare with unmerged centers
                        double[] centerB = centersAll.get(j);

                        // Check if centers should be merged
                        if (euclideanDistance(centerA, centerB) < threshold) {
                            // Create new averaged center
                            double[] newCenter = new double[centerA.length];
                            for (int d = 0; d < newCenter.length; ++d) {
                                // Coordinate-wise averaging
                                newCenter[d] = (centerA[d] + centerB[d]) / 2.0;
                            }

                            // Update the centers list
                            mergedCenters.set(i, newCenter);
                            merged[j] = true;  // Mark as merged
                        }
                    }
                }
            }
        }

        // Step 4: Compile final list of unmerged centers
        List<double[]> finalCenters = new ArrayList<>();
        for (int j = 0; j < mergedCenters.size(); ++j) {
            if (!merged[j]) {
                finalCenters.add(mergedCenters.get(j));
            }
        }

        return finalCenters;
    }

    //similar to mergeData, but doesn't merge the centers, it takes one of the close centers as the final one
    public static List<double[]> ensembleCenters(List<double[]> centersAll, double percentage) {
        // Validate input parameters
        if (centersAll == null || centersAll.isEmpty()) {
            return new ArrayList<>();
        }
        if (percentage < 0 || percentage > 1) {
            throw new IllegalArgumentException("Percentage must be between 0 and 1");
        }

        // Step 1: Compute pairwise distance matrix between all centers
        // matrix[i][j] = Euclidean distance between centersAll[i] and centersAll[j]
        double[][] distanceMatrix = computeDistanceMatrix(centersAll);

        // Step 2: Calculate merging threshold as specified percentile of all distances
        // Example: percentage=0.15 uses the 15th percentile distance as threshold
        double threshold = 0.3;//computeThreshold(distanceMatrix, percentage);
        System.out.println("Merging threshold: "+threshold);

        // Initialize working structures
        List<double[]> mergedCenters = new ArrayList<>(centersAll); // Copy of original centers
        boolean[] merged = new boolean[centersAll.size()];          // Tracks merged status

        // Step 3: Perform center merging
        for (int i = 0; i < centersAll.size(); ++i) {
            if (!merged[i]) {  // Only process unmerged centers
                double[] centerA = centersAll.get(i);

                for (int j = i + 1; j < centersAll.size(); ++j) {
                    if (!merged[j]) {  // Only compare with unmerged centers
                        double[] centerB = centersAll.get(j);

                        // Check if centers should be merged
                        if (euclideanDistance(centerA, centerB) < threshold) {
                            // Take the first center as the final one and Update the centers list
                            // mergedCenters.set(i, centerA);
                            merged[j] = true;  // Mark as merged
                        }
                    }
                }
            }
        }

        // Step 4: Compile final list of unmerged centers
        List<double[]> finalCenters = new ArrayList<>();
        for (int j = 0; j < centersAll.size(); ++j) {
            if (!merged[j]) {
                finalCenters.add(centersAll.get(j));
            }
        }

        return finalCenters;
    }


    //==== Methods for working with INiceResult from multiple observation points ===
    /**
     * Extracts all cluster centers from multiple INiceResult (multiple observations)
     * @param results List of INiceResult objects
     * @return List containing all cluster centers from all results
     */
    public static List<double[]> getAllCenters(List<INice.INiceResult> results) {
        return results.stream()
                .flatMap(r -> r.getClusterCenters().stream())
                .collect(Collectors.toList());
    }



    /**
     * Inner class representing the learned model (cluster centers + metadata) from a single observation point.
     * TODO: improve fields so that we keep all calculated values that may be used for later analysis
     * TODO: modify toString
     */

    /**
     * Static nested class representing the result of I-nice algorithm execution
     */
    public static class INiceResult {
        private final List<double[]> clusterCenters;
        private final int numClusters;
        private final double[] observationPoint;
        private final MixEM gmm;
        private final double AICcValue;

        /**
         * Constructor
         * @param clusterCenters List of cluster center coordinates
         * @param numClusters Number of clusters found
         * @param observationPoint The observation point used for this result
         * @param gmm The fitted Gamma mixture model
         * @param AICcValue The AICc value of the model
         */
        public INiceResult(List<double[]> clusterCenters,
                           int numClusters,
                           double[] observationPoint,
                           MixEM gmm,
                           double AICcValue) {
            this.clusterCenters = clusterCenters; // May consider making a defensive copy
            this.numClusters = numClusters;
            this.observationPoint = Arrays.copyOf(observationPoint, observationPoint.length);
            this.gmm = gmm; //
            this.AICcValue = AICcValue;
        }

        // Getters
        public List<double[]> getClusterCenters() {
            return clusterCenters;
        }

        public int getNumClusters() {
            return numClusters;
        }

        public double[] getObservationPoint() {
            return Arrays.copyOf(observationPoint, observationPoint.length);
        }

        public MixEM getGmm() {
            return gmm;
        }

        public double getAICcValue() {
            return AICcValue;
        }



        /**
         * Overridden toString() method showing key information
         */
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();

            // Header with basic info
            sb.append(String.format("Found %d clusters with AICc=%.2f\n",
                    numClusters, AICcValue));
            sb.append(String.format("Observation Point: %s\n",
                    Arrays.toString(observationPoint)));

            // Cluster centers with distances
            sb.append("Cluster Centers:\n");
            for (int i = 0; i < clusterCenters.size(); i++) {
                double dist = euclideanDistance(observationPoint, clusterCenters.get(i));
                sb.append(String.format("%d. %s (distance=%.2f)\n",
                        i+1,
                        Arrays.toString(clusterCenters.get(i)),
                        dist));
            }

            // GMM info
            sb.append(String.format("Gamma Mix Model: %s", gmm.toString()));

            return sb.toString();
        }
    }

}
