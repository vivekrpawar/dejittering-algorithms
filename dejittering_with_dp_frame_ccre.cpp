#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <limits>

using namespace std;
using namespace cv;

// Function to compute CCRE (Cross Cumulative Residual Entropy)
double computeCCRE(const vector<uchar>& row1, const vector<uchar>& row2) {
    if (row1.size() != row2.size()) {
        throw runtime_error("Rows must have the same length.");
    }

    int min1 = *min_element(row1.begin(), row1.end());
    int max1 = *max_element(row1.begin(), row1.end());
    int min2 = *min_element(row2.begin(), row2.end());
    int max2 = *max_element(row2.begin(), row2.end());

    int N1 = max1 - min1 + 1;
    int N2 = max2 - min2 + 1;

    vector<vector<double>> jointHist(N1, vector<double>(N2, 0.0));
    for (size_t x = 0; x < row1.size(); x++) {
        int i = row1[x] - min1;
        int j = row2[x] - min2;
        jointHist[i][j] += 1.0;
    }

    // Marginal histograms
    vector<double> h1(N1, 0), h2(N2, 0);
    for (int i = 0; i < N1; i++) {
        for (int j = 0; j < N2; j++) {
            h1[i] += jointHist[i][j];
            h2[j] += jointHist[i][j];
        }
    }

    double total = row1.size();
    for (int i = 0; i < N1; i++) h1[i] /= total;
    for (int j = 0; j < N2; j++) h2[j] /= total;
    for (int i = 0; i < N1; i++)
        for (int j = 0; j < N2; j++)
            jointHist[i][j] /= total;

    // Cumulative distribution for h1
    vector<double> c1(N1, 0);
    c1[0] = h1[0];
    for (int k = 1; k < N1; k++) {
        c1[k] = c1[k-1] + h1[k];
    }

    // Compute CCRE
    double ccre = 0.0;
    for (int j = 0; j < N1 - 1; j++) {
        if (c1[j] <= 0) continue;
        for (int i = 0; i < N2; i++) {
            if (jointHist[j][i] > 0 && h2[i] > 0) {
                ccre -= jointHist[j][i] * log2(jointHist[j][i] / (c1[j] * h2[i]));
            }
        }
    }
    return ccre;
}

// Compute intensity difference after shifting rows
double computeIntensityDiff(const vector<uchar>& row1, const vector<uchar>& row2, int shift1, int shift2, int maxShift) {
    int rowLength = row1.size() - 2 * maxShift;
    int s_max = max(abs(shift1), abs(shift2));
    int validLength = rowLength - 2 * s_max;

    vector<uchar> shifted1(row1.begin() + maxShift + shift1, row1.begin() + maxShift + shift1 + validLength);
    vector<uchar> shifted2(row2.begin() + maxShift + shift2, row2.begin() + maxShift + shift2 + validLength);

    return computeCCRE(shifted1, shifted2);
}

// Main jitter removal function
Mat removeLineJitter(const Mat& image, int maxShift, double alpha, double lambda) {
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    int H = gray.rows, W = gray.cols;

    Mat padded;
    copyMakeBorder(gray, padded, 0, 0, maxShift, maxShift, BORDER_CONSTANT, Scalar(0));

    vector<vector<double>> cost(H, vector<double>(2 * maxShift + 1, numeric_limits<double>::infinity()));
    vector<vector<int>> shifts(H, vector<int>(2 * maxShift + 1, 0));

    for (int s = -maxShift; s <= maxShift; s++) {
        cost[0][s + maxShift] = 0;
    }

    for (int i = 1; i < H; i++) {
        for (int s = -maxShift; s <= maxShift; s++) {
            for (int s_prev = -maxShift; s_prev <= maxShift; s_prev++) {
                if (cost[i-1][s_prev + maxShift] == numeric_limits<double>::infinity()) continue;

                vector<uchar> row = padded.row(i);
                vector<uchar> prevRow = padded.row(i-1);

                double diff_i_minus_1 = computeIntensityDiff(row, prevRow, s, s_prev, maxShift);

                double diffTotal = diff_i_minus_1;
                if (i >= 2) {
                    vector<uchar> twoRowsBack = padded.row(i-2);
                    int s_prev_prev = shifts[i-1][s_prev + maxShift];
                    double diff_i_minus_2 = computeIntensityDiff(row, twoRowsBack, s, s_prev_prev, maxShift);
                    diffTotal = 0.7 * diff_i_minus_1 + 0.3 * diff_i_minus_2;
                }

                double penalty = lambda * abs(s - s_prev);
                double currentCost = cost[i-1][s_prev + maxShift] + diffTotal + penalty;

                if (currentCost < cost[i][s + maxShift]) {
                    cost[i][s + maxShift] = currentCost;
                    shifts[i][s + maxShift] = s_prev;
                }
            }
        }
    }

    vector<int> correctedShifts(H, 0);
    correctedShifts[H-1] = (int)(min_element(cost[H-1].begin(), cost[H-1].end()) - cost[H-1].begin()) - maxShift;

    for (int i = H-2; i >= 0; i--) {
        correctedShifts[i] = shifts[i+1][correctedShifts[i+1] + maxShift];
    }

    Mat corrected(H, W, image.type(), Scalar(0));
    for (int i = 0; i < H; i++) {
        int shift = correctedShifts[i];
        for (int j = 0; j < W; j++) {
            int srcJ = j + shift;
            if (srcJ >= 0 && srcJ < W) {
                corrected.at<Vec3b>(i, j) = image.at<Vec3b>(i, srcJ);
            }
        }
    }

    return corrected;
}

// Main function
int main(int argc, char** argv) {
    if (argc < 5) {
        cerr << "Usage: " << argv[0] << " input_file output_file alpha lambda" << endl;
        return -1;
    }

    string inputFile = argv[1];
    string outputFile = argv[2];
    double alpha = atof(argv[3]);
    double lambda = atof(argv[4]);

    Mat inputImage = imread(inputFile);
    if (inputImage.empty()) {
        cerr << "Failed to load image: " << inputFile << endl;
        return -1;
    }

    rotate(inputImage, inputImage, ROTATE_180);

    auto start = chrono::high_resolution_clock::now();
    Mat correctedImage = removeLineJitter(inputImage, 15, alpha, lambda);
    auto end = chrono::high_resolution_clock::now();
    cout << "Processing time: " << chrono::duration_cast<chrono::milliseconds>(end - start).count() << " ms" << endl;

    rotate(correctedImage, correctedImage, ROTATE_180);
    imwrite(outputFile, correctedImage);

    return 0;
}
