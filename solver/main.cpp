// https://www.articleshub.net/2023/04/metod-sopryazhennikh-gradientov-python.html
// http://www.machinelearning.ru/wiki/index.php?title=Метод_сопряжённых_градиентов
// https://studfile.net/preview/5793014/page:6/
// https://ru.wikipedia.org/wiki/Метод_сопряжённых_градиентов_(для_решения_СЛАУ)
// https://en.wikipedia.org/wiki/Conjugate_gradient_method
// https://basegroup.ru/community/articles/conjugate
// https://math.nist.gov/MatrixMarket/formats.html#MMformat
// https://www.programmersought.com/article/5090926579/
// https://cplusplus.com/forum/general/65804/
// https://github.com/cwpearson/matrix-market
// https://gitlab.com/libeigen/eigen
// https://www.quantstart.com/articles/Eigen-Library-for-Matrix-Algebra-in-C/
// https://cplusplus.com/forum/general/222617/
// https://people.math.sc.edu/Burkardt/cpp_src/cg/cg.html

#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cmath>

using namespace std;

struct CSRMatrix {
    vector<double> values;
    vector<int> columns;
    vector<int> row_ptr;
};

vector<double> matrixTimesVector(const CSRMatrix &A, const vector<double> &x) {
    int n = A.row_ptr.size() - 1;
    vector<double> result(n, 0.0);

    for (int i = 0; i < n; i++) {
        for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
            result[i] += A.values[k] * x[A.columns[k]];
        }
    }

    return result;
}

vector<double> vectorCombination(double a, const vector<double> &x, double b, const vector<double> &y) {
    int n = x.size();
    vector<double> result(n);

    for (int i = 0; i < n; i++) {
        result[i] = a * x[i] + b * y[i];
    }

    return result;
}

double innerProduct(const vector<double> &x, const vector<double> &y) {
    double result = 0.0;
    int n = x.size();

    for (int i = 0; i < n; i++) {
        result += x[i] * y[i];
    }

    return result;
}

double vectorNorm(const vector<double> &x) {
    return sqrt(innerProduct(x, x));
}

vector<double> solveLinearSystem(const CSRMatrix &A, const vector<double> &b, double tol = 1e-10) {
    double TOLERANCE = 1.0e-10;

    int n = A.row_ptr.size() - 1;
    vector<double> X(n, 0.0);

    vector<double> R = b;
    vector<double> P = R;
    int k = 0;

    while (k < n) {
        vector<double> Rold = R;
        vector<double> AP = matrixTimesVector(A, P);

        double alpha = innerProduct(R, R) / max(innerProduct(P, AP), TOLERANCE);
        X = vectorCombination(1.0, X, alpha, P);
        R = vectorCombination(1.0, R, -alpha, AP);

        if (vectorNorm(R) < tol) break;

        double beta = innerProduct(R, R) / max(innerProduct(Rold, Rold), TOLERANCE);
        P = vectorCombination(1.0, R, beta, P);
        k++;
    }

    return X;
}

CSRMatrix convertToCSRFormat(const vector<vector<double>> &matrix) {
    CSRMatrix csrMatrix;
    int num_rows = matrix.size();
    int num_cols = matrix[0].size();

    csrMatrix.row_ptr.push_back(0);

    for (int i = 0; i < num_rows; i++) {
        int nonzero_count = 0;
        for (int j = 0; j < num_cols; j++) {
            if (matrix[i][j] != 0.0) {
                csrMatrix.values.push_back(matrix[i][j]);
                csrMatrix.columns.push_back(j);
                nonzero_count++;
            }
        }
        csrMatrix.row_ptr.push_back(csrMatrix.row_ptr.back() + nonzero_count);
    }

    return csrMatrix;
}

void readMatrixFromFile(const string &filename, CSRMatrix &csrMatrix) {
    ifstream file(filename);

    if (!file.is_open()) {
        throw runtime_error("Error: Unable to open the file.");
    }

    int num_rows, num_cols, num_elements;

    while (file.peek() == '%') {
        file.ignore(2048, '\n');
    }

    file >> num_rows >> num_cols >> num_elements;

    if (num_rows <= 0 || num_cols <= 0) {
        throw runtime_error("Error: Invalid matrix dimensions.");
    }

    csrMatrix.row_ptr.push_back(0);

    for (int i = 0; i < num_elements; i++) {
        int row, col;
        double value;
        file >> row >> col >> value;
        if (row < 1 || row > num_rows || col < 1 || col > num_cols) {
            throw runtime_error("Error: Matrix entry out of bounds.");
        }
        csrMatrix.values.push_back(value);
        csrMatrix.columns.push_back(col - 1);

        if (i < num_elements - 1) {
            int next_row;
            file >> next_row;
            if (next_row != row) {
                csrMatrix.row_ptr.push_back(csrMatrix.values.size());
            }
            file.unget();
        }
    }

    while (csrMatrix.row_ptr.size() < num_rows+1) {
        csrMatrix.row_ptr.push_back(csrMatrix.values.size()-2);
    }

    file.close();
}

void printMatrix(const CSRMatrix &csrMatrix) {
    int num_rows = csrMatrix.row_ptr.size() - 1;
    int num_cols = 0;

    for (int i = 0; i < num_rows; i++) {
        for (int k = csrMatrix.row_ptr[i]; k < csrMatrix.row_ptr[i + 1]; k++) {
            num_cols = max(num_cols, csrMatrix.columns[k] + 1);
        }
    }

    cout << "Matrix in CSR Format:" << endl;
    for (int i = 0; i < num_rows; i++) {
        int column_index = 0;
        for (int k = csrMatrix.row_ptr[i]; k < csrMatrix.row_ptr[i + 1]; k++) {
            while (column_index < csrMatrix.columns[k]) {
                cout << "0.0\t";
                column_index++;
            }
            cout << csrMatrix.values[k] << "\t";
            column_index++;
        }

        while (column_index < num_cols) {
            cout << "0.0\t";
            column_index++;
        }

        cout << endl;
    }
}


int main() {
    string filename = "matrix.mtx";
    CSRMatrix csrMatrix;

    try {
        readMatrixFromFile(filename, csrMatrix);

        int num_rows = csrMatrix.row_ptr.size() - 1;

        printMatrix(csrMatrix);

        vector<double> b(num_rows);

        cout << "Enter the vector of right-hand sides (debug - b):" << endl;
        for (int i = 0; i < num_rows; i++) {
            cin >> b[i];
        }

        vector<double> x = solveLinearSystem(csrMatrix, b);

        cout << "Solution:" << endl;
        for (int i = 0; i < num_rows; i++) {
            cout << "x[" << i << "] = " << x[i] << endl;
        }
    } catch (const runtime_error &e) {
        cerr << e.what() << endl;
    }

    return 0;
}


/* 
Параллельное программирование `OpenMP`
Вять решение СЛАУ, распараллелить и сравнить по скорости работы.
*/