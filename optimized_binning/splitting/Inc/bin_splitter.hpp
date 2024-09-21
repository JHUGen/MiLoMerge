#ifndef binSplitter
#define binSplitter

#include <unordered_map>
#include <vector>
#include <utility>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <Eigen/StdVector>

#if EIGEN_VERSION_AT_LEAST(3,4,0)
    #define Run
#endif

#ifndef Run
#error Bin_splitter needs Eigen >= 3.4.0
#endif

typedef Eigen::Array<long double, Eigen::Dynamic, Eigen::Dynamic> ArrayXXld;
typedef Eigen::Array<long double, Eigen::Dynamic, 1> ArrayXld;

class bin_splitter{
    private:
        std::vector<Eigen::ArrayXXd> data;
        int nObservables;
        int nPoints;
        int nHypotheses;
        std::vector<std::string> encodedFinalStrings;
        std::vector<std::vector<double>> finalBinCounts;
        std::vector<int> hypoList;
        std::vector<int> observablesList;
        std::vector<std::pair<double,double>> maximaAndMinima;
        std::unordered_map<std::string, std::vector<std::vector<int>>> bins;
        std::unordered_map<std::string, double> previousCalculations;
        
        void initialize(
            std::vector<std::vector<std::vector<double>>>& data,
            std::vector<std::vector<double>>& weights
        );
        
        void score(
            std::vector<std::vector<int>>& b1, 
            std::vector<std::vector<int>>& b2,
            long double& metricVal,
            bool compareToFirstOnly
        );

    public:
        bin_splitter(
            std::vector<std::vector<std::vector<double>>>& data
        );

        bin_splitter(
            std::vector<std::vector<std::vector<double>>>& data,
            std::vector<double>& weight
        );

        bin_splitter(
            std::vector<std::vector<std::vector<double>>>& data,
            std::vector<std::vector<double>>& weights
        );

        Eigen::MatrixXd getData(int h) const;
        std::vector<std::vector<double>> getFinalBinCounts() const;
        std::vector<std::string> getEncodedStrings() const;
        std::vector<std::pair<double,double>> getMinimaAndMaxima() const;
        std::vector<double> getMinima() const;
        std::vector<double> getMaxima() const;
        int getNObservables() const;
        int getNPoints() const;
        int getNHypotheses() const;

        void split(
            int nBinsDesired,
            size_t granularity,
            double statLimit,
            bool log=true
        );

        static std::string decode(std::vector<std::string>& names, std::string& leafNode);
        std::vector<std::string> decodeCuts(std::vector<std::string>& names);

        void reset();

        ~bin_splitter() noexcept;
};

#endif
