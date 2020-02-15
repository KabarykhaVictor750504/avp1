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
#include <functional>
#include <chrono>
#include <Windows.h>
#include <new>

#define OUTER_N_ROW 200
#define OUTER_N_COL 200

#define INNER_N_ROW_FIRST 8
#define INNER_N_COL_FIRST 4

#define INNER_N_ROW_SECOND 4
#define INNER_N_COL_SECOND 64

class MatrixProcessor
{
	using FltMxMxIn =  float  ****;
	using FltMxMxOut = float ****;
public:
	static void Print(FltMxMxIn cpcpcpcpA, const size_t nOutRow, const size_t nOutCol, const size_t nInRow, const size_t nInCol)
	{
		for (size_t iOuterRow = 0; iOuterRow < nOutRow; ++iOuterRow)
		{
			for (size_t kInnerRow = 0; kInnerRow < nInRow; ++kInnerRow)
			{
				for (size_t jOuterCol = 0; jOuterCol < nOutCol; ++jOuterCol)
				{
					for (size_t mInnerCol = 0; mInnerCol < nInCol; ++mInnerCol)
						std::cout << cpcpcpcpA[iOuterRow][jOuterCol][kInnerRow][mInnerCol] << ' ';
					std::cout << '\t';
				}
				std::cout << std::endl;
			}
			std::cout << std::endl;
		}
	}
	static void pizdecfunc(float**** inA, float**** inB, float**** out,int i) {
		for (int j = 0; j < OUTER_N_COL; j++)
		{
			for (int k = 0; k < OUTER_N_COL; k++)
			{
				for (int l = 0; l < INNER_N_ROW_FIRST; l++)
				{
					float* pOut = out[i][j][l];
					for (int x = 0; x < INNER_N_COL_FIRST; x++)
					{
						const float* pb = inB[k][j][x];
						const float tmp = inA[i][k][l][x];
						for (int m = 0; m < INNER_N_COL_SECOND; ++m) {
							pOut[m] +=tmp* pb[m];
						}
					}
				}
			}
		}
	}



	static void Mult(FltMxMxIn inA, FltMxMxIn inB, FltMxMxOut out)
	{
		for (int i = 0; i < OUTER_N_ROW; i++)
		{
			pizdecfunc(inA, inB, out, i);
		}
		
	}



	static void MullIntrinNoUnroll(FltMxMxIn inA, FltMxMxIn inB, FltMxMxOut out)
	{
		for (int i = 0; i < OUTER_N_ROW; ++i)
		{
			for (int j = 0; j < OUTER_N_COL;++j)
			{
				for (int k = 0; k < OUTER_N_COL; ++k)
				{
					for (int l = 0; l < INNER_N_ROW_FIRST; ++l)
					{
						float* pOut = out[i][j][l];
						for (int x = 0; x < INNER_N_COL_FIRST; ++x)
						{
							const __m256 a = _mm256_set1_ps(inA[i][k][l][x]);
							const float* pB = inB[k][j][x];
							for (int ms = 0; ms <INNER_N_COL_SECOND; ms+=16)
							{
								_mm256_storeu_ps(pOut + ms, _mm256_fmadd_ps(a,
									_mm256_loadu_ps(pB + ms + 0), _mm256_loadu_ps(pOut + ms)));
								_mm256_storeu_ps(pOut + ms + 8, _mm256_fmadd_ps(a,
									_mm256_loadu_ps(pB + ms + 8), _mm256_loadu_ps(pOut + ms + 8)));
							}
						}
					}
				}
			}
		}
	}
	static void MultIntrinUnroll(FltMxMxIn inA, FltMxMxIn inB, FltMxMxOut out)
	{
		for (int i = 0; i < OUTER_N_ROW; i++)
		{
			for (int j = 0; j < OUTER_N_COL; j++)
			{
				for (int k = 0; k < OUTER_N_COL; k++)
				{
					for (int l = 0; l < INNER_N_ROW_FIRST; l++)
					{
						float* pOut = out[i][j][l];
						for (int x = 0; x < INNER_N_COL_FIRST; x++)
						{							
							const __m256 a = _mm256_set1_ps(inA[i][k][l][x]);
							const float* pB = inB[k][j][x];
							for (int ms = 0; ms < INNER_N_COL_SECOND ; ms+=16)
							{
								_mm256_storeu_ps(pOut + ms, _mm256_fmadd_ps(a,
									_mm256_loadu_ps(pB + ms + 0), _mm256_loadu_ps(pOut + ms)));
								_mm256_storeu_ps(pOut + ms + 8, _mm256_fmadd_ps(a,
									_mm256_loadu_ps(pB + ms + 8), _mm256_loadu_ps(pOut + ms + 8)));
								//*(pOut+1) = _mm256_fmadd_ps(a, *(pB + ms), *(pOut + ms));
							}
						}
					}
				}
			}
		}
	}

	static void MultAvx(FltMxMxIn inA, FltMxMxIn inB, FltMxMxOut out)
	{
		//exit(0);//not implemented
		/*for (int i = 0; i < OUTER_N_ROW; i++)
		{
			for (int j = 0; j < OUTER_N_COL; j++)
			{
				for (int k = 0; k < INNER_N_ROW_FIRST; k++)
				{
					for (int l = 0; l < INNER_N_COL_SECOND; l++)
					{
						__m256* c = (__m256*)(out[i][j][l]);
						for (int x = 0; x < INNER_N_ROW_SECOND; x++)
						{
							__m256 a = _mm256_broadcast_ss(&inA[i][k][l][x]);
							__m256* b = (__m256*)(inB[k][j][x]);

							*(c + 0) = _mm256_add_ps(*(c + 0), _mm256_mul_ps(a, *(b + 0)));
							*(c + 1) = _mm256_add_ps(*(c + 1), _mm256_mul_ps(a, *(b + 1)));
						}
					}
				}
			}
		}*/
	}

	static void balance(float**** inA, float**** inB, float**** out, int i) {
		for (int j = 0; j < OUTER_N_COL; j++)
		{
			for (int k = 0; k < OUTER_N_COL; k++)
			{
				for (int l = 0; l < INNER_N_ROW_FIRST; l++)
				{
					for (int x = 0; x < INNER_N_COL_FIRST; x++)
					{
#pragma loop( no_vector )	
						for (int m = 0; m < INNER_N_COL_SECOND; m++)
						{
							out[i][j][l][m] += inA[i][k][l][x] * inB[k][j][x][m];
						}
					}
				}
			}
		}
	}


	static void multnovec(FltMxMxIn inA, FltMxMxIn inB, FltMxMxOut out)
	{
		for (int i = 0; i < OUTER_N_ROW; i++)
			balance(inA, inB, out, i);
			
	}

	static bool equal(FltMxMxIn inA, FltMxMxIn inB)
	{
		for (int i = 0; i < OUTER_N_ROW; i++)
		{
			for (int j = 0; j < OUTER_N_COL; j++)
			{
				for (int k = 0; k < INNER_N_ROW_FIRST; k++)
				{
					for (int l = 0; l < INNER_N_COL_SECOND; l++)
					{
						if (inA[i][j][k][l] != inB[i][j][k][l])
							return false;
					}
				}
			}
		}
		return true;
	}
};


float**** Create( const size_t nOutRow, const size_t nOutCol, const size_t nInRow, const size_t nInCol, const std::function<int()>& generator)
{
	float**** a = new float***[nOutRow];
	for (int i = 0; i < nOutRow; i++)
	{
		a[i] = new float**[nOutCol];
		for (int j = 0; j < nOutCol; j++)
		{
			a[i][j] = new float*[nInRow];
			for (int k = 0; k < nInRow; k++)
			{
				a[i][j][k] = new float[nInCol]; /*(float*) operator new[](sizeof(float)* nInCol, std:: align_val_t (std::size_t{32}));;*/
				for (int l = 0; l < nInCol; l++)
				{
					a[i][j][k][l]=generator();
				}
			}
		}
	}
	return a;
}

int main()
{
	srand(time(NULL));

	float**** a = Create(OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_FIRST, []() {return rand()%20; });
	float**** b = Create(OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_SECOND, INNER_N_COL_SECOND, []() {return rand() % 20;});
	float**** c = Create(OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND, []() {return 0; });
	float**** d = Create(OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND, []() {return 0; });
	float**** e = Create(OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND, []() {return 0; });
	float**** f = Create(OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND, []() {return 0; });

	//SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);


	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
	start = std::chrono::high_resolution_clock::now();

	MatrixProcessor::Mult(a, b, c);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time    vec: " << (end - start).count() << " ns" << std::endl;


	start = std::chrono::high_resolution_clock::now();

	MatrixProcessor::multnovec(a, b, d);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time no vec: " << (end - start).count() << " ns" << std::endl;


	start = std::chrono::high_resolution_clock::now();

	MatrixProcessor::MullIntrinNoUnroll(a, b, e);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time    sse: " << (end - start).count() << " ns" << std::endl;


	start = std::chrono::high_resolution_clock::now();

	MatrixProcessor::MultIntrinUnroll(a, b, f);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time    avx: " << (end - start).count() << " ns" << std::endl;


	if (!(MatrixProcessor::equal(c, d) && MatrixProcessor::equal(d, e) /*&& MatrixProcessor::equal(e, f)*/))
		std::cout << "Not equal. Not Xdd" << std::endl;

	/*MatrixProcessor::Print(a,OUTER_N_ROW,OUTER_N_COL,INNER_N_ROW_FIRST,INNER_N_COL_FIRST);

	std::cout << "-----------------------------" << std::endl;

	MatrixProcessor::Print(b, OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_SECOND, INNER_N_COL_SECOND);

	std::cout << "-----------------------------" << std::endl;

	MatrixProcessor::Print(c, OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND);

	std::cout << "-----------------------------" << std::endl;

	MatrixProcessor::Print(d, OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND);

	std::cout << "-----------------------------" << std::endl;

	MatrixProcessor::Print(e, OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND);

	std::cout << "-----------------------------" << std::endl;

	MatrixProcessor::Print(f, OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND);*/

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