#pragma once
#include <emmintrin.h> // 包含SSE2指令集的头文件

using namespace std;

double dot_product(size_t n, const double* a, const double* b) {
    __m128d sum = _mm_setzero_pd(); // 初始化为0
    size_t i;
    for (i = 0; i < n - (n % 2); i += 2) {
        __m128d ai = _mm_load_pd(a + i); // 加载a[i]到a[i+1]
        __m128d bi = _mm_load_pd(b + i); // 加载b[i]到b[i+1]
        __m128d prod = _mm_mul_pd(ai, bi); // 计算a[i]*b[i]到a[i+1]*b[i+1]
        sum = _mm_add_pd(sum, prod); // 累加结果
    }
    // 将2个double相加得到最终结果
    sum = _mm_add_sd(sum, _mm_unpackhi_pd(sum, sum));
    double total = _mm_cvtsd_f64(sum);
    // 处理剩余的元素
    for (; i < n; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

void hadamard_product(double* a, double* b, double* result, int size) {
    int i = 0;
    for (; i < size - (size % 4); i += 4) {
        __m128d vector1a = _mm_load_pd(a + i); // 加载对齐的double精度浮点数
        __m128d vector1b = _mm_load_pd(a + i + 2); // 加载对齐的double精度浮点数
        __m128d vector2a = _mm_load_pd(b + i); // 加载对齐的double精度浮点数
        __m128d vector2b = _mm_load_pd(b + i + 2); // 加载对齐的double精度浮点数
        __m128d resa = _mm_mul_pd(vector1a, vector2a); // 计算Hadamard乘积
        __m128d resb = _mm_mul_pd(vector1b, vector2b); // 计算Hadamard乘积
        _mm_store_pd(result + i, resa); // 存储结果
        _mm_store_pd(result + i + 2, resb); // 存储结果
    }
    // 处理剩余的元素
    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void scale_product(int size, double* arr, double scalar) {
    __m128d factor = _mm_set1_pd(scalar); // set scalar to both parts of a vector
    int i;
    for (i = 0; i <= size - 2; i += 2) {
        __m128d vec = _mm_load_pd(&arr[i]); // load two elements of arr
        __m128d result = _mm_mul_pd(vec, factor); // multiply them by scalar
        _mm_store_pd(&arr[i], result); // store the result back into arr
    }

    // Handle the case where size is not a multiple of 2
    if (i < size) {
        arr[i] *= scalar;
    }
}

void add_arrays(int n, double* a, double* b) {
    int i;
    // 处理可以被4整除的部分
    for (i = 0; i + 3 < n; i += 4) {
        __m128d va1 = _mm_load_pd(a + i);
        __m128d va2 = _mm_load_pd(a + i + 2);
        __m128d vb1 = _mm_load_pd(b + i);
        __m128d vb2 = _mm_load_pd(b + i + 2);
        vb1 = _mm_add_pd(va1, vb1);
        vb2 = _mm_add_pd(va2, vb2);
        _mm_store_pd(b + i, vb1);
        _mm_store_pd(b + i + 2, vb2);
    }
    // 处理剩余的元素
    for (; i < n; ++i) {
        b[i] += a[i];
    }
}