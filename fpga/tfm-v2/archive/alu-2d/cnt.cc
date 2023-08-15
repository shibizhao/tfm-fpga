#include <iostream>

int N = 16384;
int log2n = 16;

int main(){
    int cnt = 0;
    for (unsigned int s = 1; s <= log2n; ++s){
        unsigned int m = 1 << s;
        for (unsigned int k = 0; k < N; k += m){
            for (unsigned int j = 0; j < m / 2; ++j){
                cnt += 1;
            }
        }
    }
    std::cout << cnt << std::endl;
}