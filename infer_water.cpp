#include "deepmd/DeepPot.h"

int main(){
    deepmd::DeepPot model("./examples/water/se_e2_a/compressed_model_water.pth");
    std::vector<double > coord ={2.,2.,2.,2.95,2.,2.,1.6800001, 2.9,2.};
    std::vector<int > atype ={0, 1, 1};
    std::vector<double > cell={12.444661,0.,0.,0.,12.444661,0.,0.,0.,12.444661};
    double e;
    std::vector<double > f, v;
    model.compute(e, f, v, coord, atype, cell);
    std::cout << e << std::endl;
}