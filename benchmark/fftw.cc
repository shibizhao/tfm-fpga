#include <iostream>
#include <fftw3.h>
#include <chrono>

// 定义数组的大小
#define A 1000
#define B 4096

// 创建输入数组
fftw_complex mat[A][B] = {
    {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}},
    {{5.0, 0.0}, {6.0, 0.0}, {7.0, 0.0}, {8.0, 0.0}},
    {{9.0, 0.0}, {10.0, 0.0}, {11.0, 0.0}, {12.0, 0.0}}
};

// 创建输出数组
fftw_complex out[A][B];

fftw_complex out2[A][B];
int main() {


    // 创建输入和输出变量
    fftw_complex* in = reinterpret_cast<fftw_complex*>(mat);
    fftw_complex* out1 = reinterpret_cast<fftw_complex*>(out);
    fftw_plan fft, ifft;

    // 创建FFT计划
    fft  = fftw_plan_dft_1d(B, in, out[0], FFTW_FORWARD, FFTW_ESTIMATE);
    ifft = fftw_plan_dft_1d(B, out1, out2[0], FFTW_FORWARD, FFTW_ESTIMATE);
    // 启用多线程计算
    fftw_init_threads();
    fftw_plan_with_nthreads(4);  // 设置为使用4个线程

    // 开始计时
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    // 执行一维FFT
    for (int i = 0; i < A; i++) {
        fftw_execute_dft(fft, in + i * B, out[i]);
    }
    for (int i = 0; i < A; i++) {
        fftw_execute_dft(ifft, out1 + i * B, out2[i]);
    }


    // 停止计时
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    // 打印结果
    // for (int i = 0; i < A; i++) {
    //     for (int j = 0; j < B; j++) {
    //         std::cout << out[i][j][0] << " + " << out[i][j][1] << "i\t";
    //     }
    //     std::cout << std::endl;
    // }

    // 计算执行时间（毫秒）
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

    // 销毁计划和释放内存
    fftw_destroy_plan(fft);
    fftw_destroy_plan(ifft);
    fftw_cleanup_threads();

    return 0;
}
