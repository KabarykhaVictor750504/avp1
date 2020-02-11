//
//#include <iostream>
//#include <variant>
//#include <functional>
//
//#include "Timer.h"
//
//
//template<class T>
//class Matrix;
//
//template<class T>
//class IGenerator
//{
//public:
//	virtual T generate() = 0;
//};
//
//template<class T>
//class RandomGenerator
//{
//public:
//	T generate() { return 0; }
//};
//
//
//template<>
//class Matrix<int>
//{
//private:
//	int* ptr;
//
//	const size_t Ncol;
//	const size_t Nrow;
//
//	const size_t length;
//
//private:
//	void FillValues(const std::function<int()>& generator)
//	{
//		for (size_t i = 0; i < length; i++)
//		{
//			*(ptr + i) = generator();
//		}
//	}
//
//public:
//	Matrix(const size_t nrow, const size_t ncol, const std::function<int()>& generator) :Ncol(ncol), Nrow(nrow), length(ncol*nrow)
//	{
//		ptr = new int[Ncol*Nrow];
//
//		FillValues(generator);
//	};
//
//	Matrix<int> SlowMultiply(const Matrix<int>& other)
//	{
//		if (Nrow != other.Ncol)
//		{
//			exit(0);
//		}
//
//		Matrix<int> ret(Nrow, Ncol, []() {return 0; });
//
//		for (size_t iRow = 0; iRow < Nrow; iRow++)
//		{
//			for (size_t iCol = 0; iCol < other.Ncol; iCol++)
//			{
//				int sum = 0;
//				for (size_t k = 0; k < Ncol; k++)
//				{
//					sum += (*(this->ptr + iRow * Ncol + k))*(*(other.ptr + k * Ncol + iCol));
//				}
//
//				*(ret.ptr + iRow * Ncol + iCol) = sum;
//			}
//		}
//		return ret;
//	}
//
//	Matrix<int> FastMultiply(const Matrix<int>& other)
//	{
//		if (Nrow != other.Ncol)
//		{
//			exit(0);
//		}
//
//		Matrix<int> ret(Nrow, Ncol, []() {return 0; });
//
//#pragma loop(no_vector)
//		for (size_t iRow = 0; iRow < Nrow; iRow++)
//		{
//#pragma loop(no_vector)
//			for (size_t iCol = 0; iCol < other.Ncol; iCol++)
//			{
//				int sum = 0;
//#pragma loop(no_vector)
//				for (size_t k = 0; k < Ncol; k++)
//				{
//					sum += (*(this->ptr + iRow * Ncol + k))*(*(other.ptr + k * Ncol + iCol));
//				}
//
//				*(ret.ptr + iRow * Ncol + iCol) = sum;
//			}
//		}
//		return ret;
//	}
//
//	void DebugPrintAll()
//	{
//		for (size_t i = 0; i < Nrow; i++)
//		{
//			for (size_t j = 0; j < Ncol; j++)
//			{
//				std::cout << "\t" << *(ptr + i * Ncol + j);
//			}
//			std::cout << std::endl;
//		}
//	}
//
//	~Matrix()
//	{
//		//delete[] ptr;
//	}
//};
//
//int main() {
//
//	Timer tSlow;
//	Matrix<int> m1(500, 500, []() {static int i = 109; i++; return i; });
//	Matrix<int> m2(500, 500, []() {static int i = 1111; i++; return i; });
//	
//	auto a = m1.SlowMultiply(m2);
//	long long timeeeSloW = tSlow.TimeFromBegin();
//	std::cout << timeeeSloW;
//
//
//	Timer tFast;
//	Matrix<int> mfast1(500, 500, []() {static int i = 109; i++; return i; });
//	Matrix<int> mfast2(500, 500, []() {static int i = 1111; i++; return i; });
//
//	auto aaa = mfast1.FastMultiply(mfast2);
//	long long fastTime = tFast.TimeFromBegin();
//	std::cout << std::endl<< fastTime;
//
//	system("pause");
//	return 0;
//}

#include <iostream>
#include <xmmintrin.h>
#include <immintrin.h>
#include <ctime>
#include <chrono>
#include <Windows.h>
#include <new>


constexpr size_t MS = 32;
constexpr size_t KS = 16;

#define OUTER_N_ROW 2
#define OUTER_N_COL 2

#define INNER_N_ROW_FIRST 8
#define INNER_N_COL_FIRST 16

#define INNER_N_ROW_SECOND 16
#define INNER_N_COL_SECOND 4

void mat_out(const float const *const *const *const *const a)
{
	for (size_t i = 0; i < OUTER_N_ROW; ++i)
	{
		for (size_t k = 0; k < INNER_N_ROW_FIRST; ++k)
		{
			for (size_t j = 0; j < OUTER_N_COL; ++j)
			{
				for (size_t m = 0; m < INNER_N_COL_SECOND; ++m)
					std::cout << a[i][j][k][m] << ' ';
				std::cout << '\t';
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}
}

void mat_mul(const float const *const *const *const *const  matA, const  float const *const*const*const*const  matB,  float *const*const*const*const matC)
{
	for (int i = 0; i < OUTER_N_ROW; i++)
	{
		for (int j = 0; j < OUTER_N_COL; j++)
		{
			for (int k = 0; k < OUTER_N_COL; k++)
			{
				for (int l = 0; l < INNER_N_ROW_FIRST; l++)
				{
					for (int x = 0; x < INNER_N_COL_SECOND; x++)
					{
						float tmp = matA[i][k][l][x];
						for (int m = 0; m < INNER_N_ROW_SECOND; m++)
						{
							matC[i][j][l][m] += tmp * matB[k][j][x][m];
						}
					}
				}
			}
		}
	}
}

void mat_mul_intrin(float**** const matA, float**** const matB, float**** matC)
{
	for (int i = 0; i < OUTER_N_ROW; i++)
	{
		for (int j = 0; j < OUTER_N_COL; j++)
		{
			for (int k = 0; k < OUTER_N_COL; k++)
			{
				for (int l = 0; l < INNER_N_ROW_FIRST; l++)
				{
					__m128* c = (__m128*)(matC[i][j][l]);

					__m128 a = _mm_load_ps1(&matA[i][k][l][0]);
					__m128* b = (__m128*)(matB[k][j][0]);
					*(c + 0) = _mm_add_ps(*(c + 0), _mm_mul_ps(a, *(b + 0)));
					*(c + 1) = _mm_add_ps(*(c + 1), _mm_mul_ps(a, *(b + 1)));
					*(c + 2) = _mm_add_ps(*(c + 2), _mm_mul_ps(a, *(b + 2)));
					*(c + 3) = _mm_add_ps(*(c + 3), _mm_mul_ps(a, *(b + 3)));

					a = _mm_load_ps1(&matA[i][k][l][1]);
					b = (__m128*)(matB[k][j][1]);
					*(c + 0) = _mm_add_ps(*(c + 0), _mm_mul_ps(a, *(b + 0)));
					*(c + 1) = _mm_add_ps(*(c + 1), _mm_mul_ps(a, *(b + 1)));
					*(c + 2) = _mm_add_ps(*(c + 2), _mm_mul_ps(a, *(b + 2)));
					*(c + 3) = _mm_add_ps(*(c + 3), _mm_mul_ps(a, *(b + 3)));

					a = _mm_load_ps1(&matA[i][k][l][2]);
					b = (__m128*)(matB[k][j][2]);
					*(c + 0) = _mm_add_ps(*(c + 0), _mm_mul_ps(a, *(b + 0)));
					*(c + 1) = _mm_add_ps(*(c + 1), _mm_mul_ps(a, *(b + 1)));
					*(c + 2) = _mm_add_ps(*(c + 2), _mm_mul_ps(a, *(b + 2)));
					*(c + 3) = _mm_add_ps(*(c + 3), _mm_mul_ps(a, *(b + 3)));

					a = _mm_load_ps1(&matA[i][k][l][3]);
					b = (__m128*)(matB[k][j][3]);
					*(c + 0) = _mm_add_ps(*(c + 0), _mm_mul_ps(a, *(b + 0)));
					*(c + 1) = _mm_add_ps(*(c + 1), _mm_mul_ps(a, *(b + 1)));
					*(c + 2) = _mm_add_ps(*(c + 2), _mm_mul_ps(a, *(b + 2)));
					*(c + 3) = _mm_add_ps(*(c + 3), _mm_mul_ps(a, *(b + 3)));
				}
			}
		}
	}
}

void mat_mul_avx(float**** const matA, float**** const matB, float**** matC)
{
	for (int i = 0; i < MS; i++)
	{
		for (int j = 0; j < MS; j++)
		{
			for (int k = 0; k < MS; k++)
			{
				for (int l = 0; l < KS; l++)
				{
					__m256* c = (__m256*)(matC[i][j][l]);
					for (int x = 0; x < KS; x++)
					{
						__m256 a = _mm256_broadcast_ss(&matA[i][k][l][x]);
						__m256* b = (__m256*)(matB[k][j][x]);

						*(c + 0) = _mm256_add_ps(*(c + 0), _mm256_mul_ps(a, *(b + 0)));
						*(c + 1) = _mm256_add_ps(*(c + 1), _mm256_mul_ps(a, *(b + 1)));
						//for (int m = 0; m < KS; m += 4)
						//{
						//	/*__m128 c = _mm_loadu_ps(&matC[i][j][l][m]);
						//	__m128 b = _mm_loadu_ps(&matB[k][j][x][m]);
						//	__m128 ab = _mm_mul_ps(a, b);
						//	c = _mm_add_ps(c, ab);
						//	_mm_storeu_ps(&matC[i][j][l][m], c);*/

						//	//*(__m128*)(matC[i][j][l] + m) = _mm_add_ps(*(__m128*)(matC[i][j][l]+m), _mm_mul_ps(a, *(__m128*)(matB[i][j][l] + m)));

						//	//matC[i][j][l][m] += matA[i][k][l][x] * matB[k][j][x][m];
						//}
					}
				}
			}
		}
	}
}

void mat_mul_no_vec(float**** const matA, float**** const matB, float**** const matC)
{
	for (int i = 0; i < OUTER_N_ROW; i++)
	{
		for (int j = 0; j < OUTER_N_COL; j++)
		{
			for (int k = 0; k < OUTER_N_COL; k++)
			{
				for (int l = 0; l < INNER_N_ROW_FIRST; l++)
				{
					for (int x = 0; x < INNER_N_COL_SECOND; x++)
					{
#pragma loop( no_vector )	
						for (int m = 0; m < INNER_N_ROW_SECOND; m++)
						{
							matC[i][j][l][m] += matA[i][k][l][x] * matB[k][j][x][m];
						}
					}
				}
			}
		}
	}
}

bool mat_eq(float**** a, float**** b)
{
	for (int i = 0; i < MS; i++)
	{
		for (int j = 0; j < MS; j++)
		{
			for (int k = 0; k < KS; k++)
			{
				for (int l = 0; l < KS; l++)
				{
					if (a[i][j][k][l] != b[i][j][k][l])
						return false;
				}
			}
		}
	}
	return true;
}

int main()
{
	srand(time(NULL));

	float**** a = new float***[MS];
	float**** b = new float***[MS];
	float**** c = new float***[MS];
	float**** d = new float***[MS];
	float**** e = new float***[MS];
	float**** f = new float***[MS];

	for (int i = 0; i < MS; i++)
	{
		a[i] = new float**[MS];
		b[i] = new float**[MS];
		c[i] = new float**[MS];
		d[i] = new float**[MS];
		e[i] = new float**[MS];
		f[i] = new float**[MS];
		for (int j = 0; j < MS; j++)
		{
			a[i][j] = new float*[KS];
			b[i][j] = new float*[KS];
			c[i][j] = new float*[KS];
			d[i][j] = new float*[KS];
			e[i][j] = new float*[KS];
			f[i][j] = new float*[KS];
			for (int k = 0; k < KS; k++)
			{
				a[i][j][k] = (float*) operator new[](sizeof(float)*KS, std::align_val_t{ 32 });
				b[i][j][k] = (float*) operator new[](sizeof(float)*KS, std::align_val_t{ 32 });
				c[i][j][k] = (float*) operator new[](sizeof(float)*KS, std::align_val_t{ 32 });
				d[i][j][k] = (float*) operator new[](sizeof(float)*KS, std::align_val_t{ 32 });
				e[i][j][k] = (float*) operator new[](sizeof(float)*KS, std::align_val_t{ 32 });
				f[i][j][k] = (float*) operator new[](sizeof(float)*KS, std::align_val_t{ 32 });
				for (int l = 0; l < KS; l++)
				{
					a[i][j][k][l] = rand() % 10;
					b[i][j][k][l] = rand() % 10;
					c[i][j][k][l] = 0.0f;
					d[i][j][k][l] = 0.0f;
					e[i][j][k][l] = 0.0f;
					f[i][j][k][l] = 0.0f;
				}
			}
		}
	}

	SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);


	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	start = std::chrono::high_resolution_clock::now();

	mat_mul(a, b, c);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time    vec: " << (end - start).count() << " ns" << std::endl;


	start = std::chrono::high_resolution_clock::now();

	mat_mul_no_vec(a, b, d);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time no vec: " << (end - start).count() << " ns" << std::endl;


	start = std::chrono::high_resolution_clock::now();

	mat_mul_intrin(a, b, e);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time    sse: " << (end - start).count() << " ns" << std::endl;


	start = std::chrono::high_resolution_clock::now();

	mat_mul_avx(a, b, f);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time    avx: " << (end - start).count() << " ns" << std::endl;


	if (!(mat_eq(c, d) && mat_eq(d, e) && mat_eq(e, f)))
		std::cout << "Not equal. Not Xdd" << std::endl;


	mat_out(a);

	std::cout << "-----------------------------" << std::endl;

	mat_out(b);

	std::cout << "-----------------------------" << std::endl;

	mat_out(c);

	std::cout << "-----------------------------" << std::endl;

	mat_out(d);

	std::cout << "-----------------------------" << std::endl;

	mat_out(e);

	/*for (int i = 0; i < MS; i++)
	{
		for (int j = 0; j < MS; j++)
		{
			for (int k = 0; k < KS; k++)
			{
				::operator delete[](a[i][j][k], std::align_val_t{ 16 });
				::operator delete[](b[i][j][k], std::align_val_t{ 16 });
				::operator delete[](c[i][j][k], std::align_val_t{ 16 });
				::operator delete[](d[i][j][k], std::align_val_t{ 16 });
			}
		}
	}
*/
	system("pause");
	return 0;
}