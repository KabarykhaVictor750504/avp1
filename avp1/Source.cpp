#include <iostream>
#include <xmmintrin.h>
#include <immintrin.h>
#include <ctime>
#include <functional>
#include <chrono>
#include <Windows.h>
#include <new>

constexpr size_t OUTER_N_ROW = 1;
constexpr size_t OUTER_N_COL = 1;

constexpr size_t INNER_N_ROW_FIRST = 8;
constexpr size_t INNER_N_COL_FIRST = 5;
								   	 
constexpr size_t INNER_N_ROW_SECOND= 5;
constexpr size_t INNER_N_COL_SECOND= 16;

constexpr size_t INNER_N_COL_SECOND_MODULUS_16 = INNER_N_COL_SECOND % 16;
constexpr size_t INNER_N_COL_SECOND_ALIGNED_TO_16_DOWN = INNER_N_COL_SECOND - INNER_N_COL_SECOND_MODULUS_16;

constexpr size_t INNER_N_ROW_FIRST_MODULUS_4 = INNER_N_ROW_FIRST % 4;
constexpr size_t INNER_N_ROW_FIRST_ALIGNED_TO_4_DOWN = INNER_N_ROW_FIRST - INNER_N_ROW_FIRST_MODULUS_4;

constexpr size_t INNER_N_COL_FIRST_MODULUS_4 = INNER_N_COL_FIRST % 4;
constexpr size_t INNER_N_COL_FIRST_ALIGNED_TO_4_DOWN = INNER_N_COL_FIRST - INNER_N_COL_FIRST_MODULUS_4;

class MatrixProcessor
{
	using FltMxMxIn = const float const*const*const*const*const;
	using FltMxMxOut = float *const *const *const *const;
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

	static void MultFunctionForVector(FltMxMxIn inA, FltMxMxIn inB, FltMxMxOut out, int i) {
		for (int j = 0; j < OUTER_N_COL; ++j)
		{
			for (int k = 0; k < OUTER_N_COL; ++k)
			{
				for (int l = 0; l < INNER_N_ROW_FIRST; ++l)
				{
					float* const    pOut = out[i][j][l];
					for (int x = 0; x < INNER_N_ROW_SECOND; ++x)
					{
						const float const* pb = inB[k][j][x];
						const float tmp = inA[i][k][l][x];
						for (int m = 0; m < INNER_N_COL_SECOND; ++m) {
							pOut[m] += tmp * pb[m];
						}
					}
				}
			}
		}
	}

	static void Mult(FltMxMxIn inA, FltMxMxIn inB, FltMxMxOut out)
	{
		for (int i = 0; i < OUTER_N_ROW; ++i)
		{
			MultFunctionForVector(inA, inB, out, i);
		}

	}



	static void MultAvxNoUnroll(FltMxMxIn inA, FltMxMxIn inB, FltMxMxOut out)
	{
		for (int i = 0; i < OUTER_N_ROW; ++i)
		{
			for (int j = 0; j < OUTER_N_COL; j++)
			{
				for (int k = 0; k < OUTER_N_COL; ++k)
				{
					const float const* const*const pB = inB[k][j];
					float*const*const pOut = out[i][j];
					for (int l = 0; l < INNER_N_ROW_FIRST; l++)
					{
						for (int x = 0; x < INNER_N_COL_SECOND_ALIGNED_TO_16_DOWN; x += 16)
						{
							__m256 c00 = _mm256_setzero_ps();
							__m256 c01 = _mm256_setzero_ps();
							__m256 a1, a2, b1, b2;
							for (int m = 0; m < INNER_N_COL_FIRST; m++)
							{
								b1 = _mm256_loadu_ps(pB[m] + x);
								b2 = _mm256_loadu_ps(pB[m] + x + 8);

								a1 = _mm256_set1_ps(inA[i][k][l][m]);
								c00 = _mm256_fmadd_ps(a1, b1, c00);
								c01 = _mm256_fmadd_ps(a1, b2, c01);
								
							}
							_mm256_storeu_ps(pOut[l] + x, _mm256_add_ps(c00, _mm256_loadu_ps(pOut[l] + x)));
							_mm256_storeu_ps(pOut[l] + x + 8, _mm256_add_ps(c01, _mm256_loadu_ps(pOut[l] + x + 8)));

						}
						for (int xxxxx = INNER_N_COL_SECOND_ALIGNED_TO_16_DOWN; xxxxx < INNER_N_COL_SECOND_ALIGNED_TO_16_DOWN + INNER_N_COL_SECOND_MODULUS_16; ++xxxxx) {
							for (int mmmmm = 0; mmmmm < INNER_N_COL_FIRST; ++mmmmm) {
								pOut[l][xxxxx] += pB[mmmmm][xxxxx] * inA[i][k][l][mmmmm];
							}
						}
					}
				}
			}
		}
	}


	static void MullAvxUnroll(FltMxMxIn inA, FltMxMxIn inB, FltMxMxOut out)
	{
		for (int i = 0; i < OUTER_N_ROW; i++)
		{
			for (int j = 0; j < OUTER_N_COL; j++)
			{
				for (int k = 0; k < OUTER_N_COL; k++)
				{
					const float const *const *const  pB = inB[k][j];
					float*const*const pOut = out[i][j];
					for (int l = 0; l < INNER_N_ROW_FIRST_ALIGNED_TO_4_DOWN; l += 4) 
					{
						for (int x = 0; x < INNER_N_COL_SECOND_ALIGNED_TO_16_DOWN; x += 16) 
						{
							__m256 c00 = _mm256_setzero_ps();
							__m256 c10 = _mm256_setzero_ps();
							__m256 c01 = _mm256_setzero_ps();
							__m256 c11 = _mm256_setzero_ps();
							__m256 c20 = _mm256_setzero_ps();
							__m256 c30 = _mm256_setzero_ps();
							__m256 c21 = _mm256_setzero_ps();
							__m256 c31 = _mm256_setzero_ps();
							__m256 a1, b1, a2, b2;
							b1 = _mm256_loadu_ps(pB[0] + x);
							b2 = _mm256_loadu_ps(pB[0] + x + 8);

							a1 = _mm256_set1_ps(inA[i][k][l][0]);
							a2 = _mm256_set1_ps(inA[i][k][l + 1][0]);
							c00 = _mm256_fmadd_ps(a1, b1, c00);
							c01 = _mm256_fmadd_ps(a1, b2, c01);
							c10 = _mm256_fmadd_ps(a2, b1, c10);
							c11 = _mm256_fmadd_ps(a2, b2, c11);

							a1 = _mm256_set1_ps(inA[i][k][l + 2][0]);
							a2 = _mm256_set1_ps(inA[i][k][l + 3][0]);
							c20 = _mm256_fmadd_ps(a1, b1, c20);
							c21 = _mm256_fmadd_ps(a1, b2, c21);
							c30 = _mm256_fmadd_ps(a2, b1, c30);
							c31 = _mm256_fmadd_ps(a2, b2, c31);

							b1 = _mm256_loadu_ps(pB[1] + x);
							b2 = _mm256_loadu_ps(pB[1] + x + 8);

							a1 = _mm256_set1_ps(inA[i][k][l][1]);
							a2 = _mm256_set1_ps(inA[i][k][l + 1][1]);
							c00 = _mm256_fmadd_ps(a1, b1, c00);
							c01 = _mm256_fmadd_ps(a1, b2, c01);
							c10 = _mm256_fmadd_ps(a2, b1, c10);
							c11 = _mm256_fmadd_ps(a2, b2, c11);

							a1 = _mm256_set1_ps(inA[i][k][l + 2][1]);
							a2 = _mm256_set1_ps(inA[i][k][l + 3][1]);
							c20 = _mm256_fmadd_ps(a1, b1, c20);
							c21 = _mm256_fmadd_ps(a1, b2, c21);
							c30 = _mm256_fmadd_ps(a2, b1, c30);
							c31 = _mm256_fmadd_ps(a2, b2, c31);

							b1 = _mm256_loadu_ps(pB[2] + x);
							b2 = _mm256_loadu_ps(pB[2] + x + 8);

							a1 = _mm256_set1_ps(inA[i][k][l][2]);
							a2 = _mm256_set1_ps(inA[i][k][l + 1][2]);
							c00 = _mm256_fmadd_ps(a1, b1, c00);
							c01 = _mm256_fmadd_ps(a1, b2, c01);
							c10 = _mm256_fmadd_ps(a2, b1, c10);
							c11 = _mm256_fmadd_ps(a2, b2, c11);

							a1 = _mm256_set1_ps(inA[i][k][l + 2][2]);
							a2 = _mm256_set1_ps(inA[i][k][l + 3][2]);
							c20 = _mm256_fmadd_ps(a1, b1, c20);
							c21 = _mm256_fmadd_ps(a1, b2, c21);
							c30 = _mm256_fmadd_ps(a2, b1, c30);
							c31 = _mm256_fmadd_ps(a2, b2, c31);


							b1 = _mm256_loadu_ps(pB[3] + x);
							b2 = _mm256_loadu_ps(pB[3] + x + 8);

							a1 = _mm256_set1_ps(inA[i][k][l][3]);
							a2 = _mm256_set1_ps(inA[i][k][l + 1][3]);
							c00 = _mm256_fmadd_ps(a1, b1, c00);
							c01 = _mm256_fmadd_ps(a1, b2, c01);
							c10 = _mm256_fmadd_ps(a2, b1, c10);
							c11 = _mm256_fmadd_ps(a2, b2, c11);

							a1 = _mm256_set1_ps(inA[i][k][l + 2][3]);
							a2 = _mm256_set1_ps(inA[i][k][l + 3][3]);
							c20 = _mm256_fmadd_ps(a1, b1, c20);
							c21 = _mm256_fmadd_ps(a1, b2, c21);
							c30 = _mm256_fmadd_ps(a2, b1, c30);
							c31 = _mm256_fmadd_ps(a2, b2, c31);

							_mm256_storeu_ps(pOut[l] + x, _mm256_add_ps(c00, _mm256_loadu_ps(pOut[l] + x)));
							_mm256_storeu_ps(pOut[l] + x + 8, _mm256_add_ps(c01, _mm256_loadu_ps(pOut[l] + x + 8)));

							_mm256_storeu_ps(pOut[l + 1] + x, _mm256_add_ps(c10, _mm256_loadu_ps(pOut[l + 1] + x)));
							_mm256_storeu_ps(pOut[l + 1] + x + 8, _mm256_add_ps(c11, _mm256_loadu_ps(pOut[l + 1] + x + 8)));

							_mm256_storeu_ps(pOut[l + 2] + x, _mm256_add_ps(c20, _mm256_loadu_ps(pOut[l + 2] + x)));
							_mm256_storeu_ps(pOut[l + 2] + x + 8, _mm256_add_ps(c21, _mm256_loadu_ps(pOut[l + 2] + x + 8)));

							_mm256_storeu_ps(pOut[l + 3] + x, _mm256_add_ps(c30, _mm256_loadu_ps(pOut[l + 3] + x)));
							_mm256_storeu_ps(pOut[l + 3] + x + 8, _mm256_add_ps(c31, _mm256_loadu_ps(pOut[l + 3] + x + 8)));

						}
					
						/*for (int mmm = INNER_N_COL_FIRST_ALIGNED_TO_4_DOWN; mmm < INNER_N_COL_FIRST; ++mmm) 
						{
							for (int xx = 0; xx < INNER_N_COL_SECOND; ++xx)
							{
								pOut[l][xx] += pB[mmm][xx] * inA[i][k][l][mmm];
							}
						}*/
					}	
					for (int x = INNER_N_COL_SECOND_ALIGNED_TO_16_DOWN; x < INNER_N_COL_SECOND; ++x)
					{
						for (int l = 0; l < INNER_N_COL_FIRST; ++l)
						{
							const float * const* pA = inB[k][j];
							const float tmp = inB[i][k][l][x];
							for (int m = 0; m < INNER_N_COL_FIRST; ++m)
							{
								pOut[m][x] += tmp * pA[m][l];

							}
						}
					}
					for (int x = INNER_N_COL_FIRST_ALIGNED_TO_4_DOWN; x < INNER_N_ROW_SECOND; ++x)
					{
						for(int l=0;l<INNER_N_ROW_FIRST;++l)
						{
						const float const* pb = inB[k][j][x];
						const float tmp = inA[i][k][l][x];
						for (int m = 0; m < INNER_N_COL_SECOND; ++m)
						{
							pOut[l][m] += tmp * pb[m];

						}
						}
					}
					for (int lll = INNER_N_ROW_FIRST_ALIGNED_TO_4_DOWN; lll < INNER_N_ROW_FIRST; ++lll) {
						for (int xxxx = 0; xxxx < INNER_N_COL_FIRST; xxxx++)
						{
							for (int mmmm = 0; mmmm < INNER_N_COL_SECOND; mmmm++) {
								out[i][j][lll][mmmm] += inA[i][k][lll][xxxx] * inB[k][j][xxxx][mmmm];
							}
						}
					}
				}
			}
		}
	}

	static void MultNoVec(FltMxMxIn inA, FltMxMxIn inB, FltMxMxOut out)
	{
		for (int i = 0; i < OUTER_N_ROW; i++)
			for (int j = 0; j < OUTER_N_COL; j++)
			{
				for (int k = 0; k < OUTER_N_COL; k++)
				{
					for (int l = 0; l < INNER_N_ROW_FIRST; l++)
					{
						for (int x = 0; x < INNER_N_COL_FIRST; x++)
						{
#pragma loop( no_vector )
							for (int m = 0; m < INNER_N_COL_SECOND; m++) {
								out[i][j][l][m] += inA[i][k][l][x] * inB[k][j][x][m];
							}
						}
					}
				}
			}
			
	}

	static bool AreEqual(FltMxMxIn inA, FltMxMxIn inB,const float epsilon)
	{
		for (int i = 0; i < OUTER_N_ROW; i++)
		{
			for (int j = 0; j < OUTER_N_COL; j++)
			{
				for (int k = 0; k < INNER_N_ROW_FIRST; k++)
				{
					for (int l = 0; l < INNER_N_COL_SECOND; l++)
					{
						if (abs(inA[i][j][k][l] - inB[i][j][k][l])>epsilon)
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
				a[i][j][k] =new float[nInCol] /*(float*)operator new[](sizeof(float)* nInCol, std::align_val_t{ 32 })*/;
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
	MatrixProcessor::MultNoVec(a, b, c);
	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time no vec: " << std:: chrono:: duration_cast<std::chrono::microseconds> (end - start).count() << "mcs" << std::endl;


	start = std::chrono::high_resolution_clock::now();

	MatrixProcessor::Mult(a, b, d);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time    vec: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "mcs" << std::endl;


	start = std::chrono::high_resolution_clock::now();

	MatrixProcessor::MultAvxNoUnroll(a, b, e);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time    AVX: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "mcs" << std::endl;


	start = std::chrono::high_resolution_clock::now();

	MatrixProcessor::MullAvxUnroll(a, b, f);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time   avx2: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "mcs" << std::endl;

	constexpr float epsilon = 0.00001f;
	if (!(MatrixProcessor::AreEqual(c, d, epsilon) && MatrixProcessor::AreEqual(d, e, epsilon) && MatrixProcessor::AreEqual(e, f, epsilon)))
		std::cout << "Not equal. Not Xdd" << std::endl;

	MatrixProcessor::Print(a,OUTER_N_ROW,OUTER_N_COL,INNER_N_ROW_FIRST,INNER_N_COL_FIRST);

	std::cout << "-----------------------------" << std::endl;

	MatrixProcessor::Print(b, OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_SECOND, INNER_N_COL_SECOND);

	std::cout << "-----------------------------" << std::endl;

	MatrixProcessor::Print(c, OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND);

	std::cout << "-----------------------------" << std::endl;

	MatrixProcessor::Print(d, OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND);

	std::cout << "-----------------------------" << std::endl;

	MatrixProcessor::Print(e, OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND);

	std::cout << "-----------------------------" << std::endl;

	MatrixProcessor::Print(f, OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_SECOND);

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