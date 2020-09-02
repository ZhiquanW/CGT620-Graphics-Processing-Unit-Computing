#include <iostream>
#include <chrono> 
#include <vector>
#include <omp.h>
using namespace std::chrono;

int main() {
	bool enable_omp = true;
	auto mat1 = std::vector<std::vector<float> >();
	auto mat2 = std::vector<std::vector<float> >();
	auto result = std::vector<std::vector<float> >();
	// init mats
	int mat_size[2] = { 10000, 10000 };
	for (int i = 0; i < mat_size[0]; ++i) {
		auto tmp_vec = std::vector<float>();
		for (int j = 0; j < mat_size[1]; ++j) {
			tmp_vec.push_back(i % 10);
		}
		mat1.emplace_back(tmp_vec);
		mat2.emplace_back(tmp_vec);
		result.emplace_back(std::vector<float>(mat_size[1]));
	}

	// add
	/*#pragma omp parallel
	#pragma omp for*/
	for (int tt = 0; tt < 20; ++tt) {
		auto start = high_resolution_clock::now();
		if (enable_omp){
			int nthreads, tid;
			#pragma omp parallel
			{
				if (omp_get_thread_num() == 0)
					nthreads = omp_get_num_threads(); //# of threads
			}
			long unsigned int i;
			int t_size = mat_size[0] * mat_size[1];
			#pragma omp parallel private(tid, i)
			{
				tid = omp_get_thread_num();
				for (i = 0 + tid; i < t_size; i += nthreads) {
					int x = i / mat_size[0];
					int y = i % mat_size[1];
					result[x][y] = mat1[x][y] + mat2[x][y];
				}
			}
		/*	#pragma omp parallel
			#pragma omp for
			for (int i = 0; i < mat_size[0]; ++i) {
				for (int j = 0; j < mat_size[1]; ++j) {
					result[i][j] = mat1[i][j] + mat2[i][j];
				}
			}*/
		} else {
			// single thread
			for (int i = 0; i < mat_size[0]; ++i) {
				for (int j = 0; j < mat_size[1]; ++j) {
					result[i][j] = mat1[i][j] + mat2[i][j];
				}
			}
		}
		auto duration = duration_cast<microseconds>(high_resolution_clock::now() - start);
		std::cout << duration.count() << std::endl;

	}
	
	
}