#include <bin_splitter.hpp>
#include <vector>
#include <iostream>
#include "TFile.h"
#include "TTree.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"

int main(int argc, char const *argv[])
{
    std::vector<std::vector<double>> data(5);
    std::vector<std::vector<double>> weight(2);

    TFile* dataFile = TFile::Open("../../test_data/all_reweighted_2e2mu.root");
    TTreeReader myReader("eventTree", dataFile);

    TTreeReaderValue<float> costheta1d(myReader, "costheta1d");
    TTreeReaderValue<float> costheta2d(myReader, "costheta2d");
    TTreeReaderValue<float> Phid(myReader, "Phid");
    TTreeReaderValue<float> MZ1(myReader, "MZ1");
    TTreeReaderValue<float> MZ2(myReader, "MZ2");

    TTreeReaderValue<double> h1(myReader, "p_Gen_GG_SIG_ghg2_1_ghz1_1_JHUGen");
    TTreeReaderValue<double> h2(myReader, "p_Gen_GG_SIG_ghg2_1_ghz4_1_JHUGen");


    size_t N = 10000000;
    size_t n = 0;
    double h1Sum=0;
    double h2Sum=0;
    while(myReader.Next() && n < N){
        data[0].push_back(*MZ1);
        data[1].push_back(*MZ2);

        data[2].push_back(*costheta1d);
        data[3].push_back(*costheta2d);
        data[4].push_back(*Phid);

        weight[0].push_back((*h1)/(*h1));
        h1Sum += ((*h1)/(*h1));
        weight[1].push_back((*h2)/(*h1));
        h2Sum += (*h2)/(*h1);
        n++;
    }

    // std::cout << "hSums " << h1Sum << " " << h2Sum << std::endl;

    for(int i = 0; i < n; i++){
        weight[0][i] /= h1Sum;
        weight[1][i] /= h2Sum; //norm to 1
    }

    std::vector<std::vector<std::vector<double>>> data2;

    data2.push_back(data);
    data2.push_back(data);

    bin_splitter b = bin_splitter(
        data2,
        weight
    );

    //std::cout << b.getData(0) << std::endl << std::endl;
   // std::cout << b.getData(1) << std::endl << std::endl;

    b.split(
        100,
        10,
        10000,
        false
    );

    std::vector<std::string> names = {"MZ1", "MZ2", "c1", "c2", "Phi"};

    for(auto it : b.decodeCuts(names)){
        std::cout << it << std::endl;
    }

    return 0;
}
