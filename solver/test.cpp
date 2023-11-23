#include <iostream>
#include <vector>

using namespace std;

struct CSRMatrix {
    vector<double> values;
    vector<int> columns;
    vector<int> row_ptr;
};

vector<double> conjugateGradient(const CSRMatrix& A, const vector<double>& b, const vector<double>& x0, int maxIterations, double tolerance) {
    int n = x0.size();
    vector<double> x = x0;
    vector<double> r = b;
    vector<double> p = r;

    for (int k = 0; k < maxIterations; ++k) {
        // Calculate Ap
        vector<double> Ap(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j) {
                Ap[i] += A.values[j] * p[A.columns[j]];
            }
        }

        // Calculate alpha
        double alpha = 0.0;
        double rDot = 0.0;
        for (int i = 0; i < n; ++i) {
            alpha += r[i] * r[i];
            rDot += r[i] * Ap[i];
        }
        alpha /= rDot;

        // Update solution
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }

        // Check for convergence
        double residual = 0.0;
        for (int i = 0; i < n; ++i) {
            residual += r[i] * r[i];
        }
        if (residual < tolerance * tolerance) {
            cout << "Converged in " << k + 1 << " iterations.\n";
            break;
        }

        // Calculate beta
        double beta = 0.0;
        double rDotNew = 0.0;
        for (int i = 0; i < n; ++i) {
            beta += r[i] * r[i];
            rDotNew += r[i] * r[i];
        }
        beta /= rDotNew / rDot;

        // Update search direction
        for (int i = 0; i < n; ++i) {
            p[i] = r[i] + beta * p[i];
        }
    }

    return x;
}

int main() {
    // Example usage
    CSRMatrix A = {
            {4.0, 1.0, -1.0, 1.0, -1.0},
            {0, 1, 3, 4, 5},
            {0, 2, 5}
    };

    vector<double> b = {8.0, 2.0, 4.0};
    vector<double> x0 = {0.0, 0.0, 0.0};

    int maxIterations = 1000;
    double tolerance = 1e-8;

    vector<double> solution = conjugateGradient(A, b, x0, maxIterations, tolerance);

    cout << "Solution: ";
    for (double val : solution) {
        cout << val << " ";
    }
    cout << endl;

    return 0;
}
