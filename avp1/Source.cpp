#include <iostream>
#include <xmmintrin.h>
#include <immintrin.h>
#include <ctime>
#include <functional>
#include <chrono>
#include <Windows.h>
#include <new>

constexpr size_t OUTER_N_ROW = 2;
constexpr size_t OUTER_N_COL = 2;

constexpr size_t INNER_N_ROW_FIRST = 5;
constexpr size_t INNER_N_COL_FIRST = 5;


constexpr size_t INNER_N_ROW_SECOND = 5;
constexpr size_t INNER_N_COL_SECOND = 5;

constexpr size_t INNER_N_COL_SECOND_MODULUS_16 = INNER_N_COL_SECOND % 16;
constexpr size_t INNER_N_COL_SECOND_ALIGNED_TO_16_DOWN = INNER_N_COL_SECOND - INNER_N_COL_SECOND_MODULUS_16;

constexpr size_t INNER_N_ROW_FIRST_MODULUS_4 = INNER_N_ROW_FIRST % 4;
constexpr size_t INNER_N_ROW_FIRST_ALIGNED_TO_4_DOWN = INNER_N_ROW_FIRST - INNER_N_ROW_FIRST_MODULUS_4;

constexpr size_t INNER_N_COL_FIRST_MODULUS_4 = INNER_N_COL_FIRST % 4;
constexpr size_t INNER_N_COL_FIRST_ALIGNED_TO_4_DOWN = INNER_N_COL_FIRST - INNER_N_COL_FIRST_MODULUS_4;

class MatrixProcessor
{
	using FltMxMxIn = const float const* const* const* const* const;
	using FltMxMxOut = float* const* const* const* const;
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
					const float const* const* const pB = inB[k][j];
					float* const* const pOut = out[i][j];
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
					const float const* const* const  pB = inB[k][j];
					float* const* const pOut = out[i][j];
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

							for (int m = 0; m < INNER_N_COL_FIRST_ALIGNED_TO_4_DOWN; m += 4) {
								b1 = _mm256_loadu_ps(pB[m] + x);
								b2 = _mm256_loadu_ps(pB[m] + x + 8);

								a1 = _mm256_set1_ps(inA[i][k][l][m]);
								a2 = _mm256_set1_ps(inA[i][k][l + 1][m]);
								c00 = _mm256_fmadd_ps(a1, b1, c00);
								c01 = _mm256_fmadd_ps(a1, b2, c01);
								c10 = _mm256_fmadd_ps(a2, b1, c10);
								c11 = _mm256_fmadd_ps(a2, b2, c11);

								a1 = _mm256_set1_ps(inA[i][k][l + 2][m]);
								a2 = _mm256_set1_ps(inA[i][k][l + 3][m]);
								c20 = _mm256_fmadd_ps(a1, b1, c20);
								c21 = _mm256_fmadd_ps(a1, b2, c21);
								c30 = _mm256_fmadd_ps(a2, b1, c30);
								c31 = _mm256_fmadd_ps(a2, b2, c31);

								b1 = _mm256_loadu_ps(pB[m + 1] + x);
								b2 = _mm256_loadu_ps(pB[m + 1] + x + 8);

								a1 = _mm256_set1_ps(inA[i][k][l][m + 1]);
								a2 = _mm256_set1_ps(inA[i][k][l + 1][m + 1]);
								c00 = _mm256_fmadd_ps(a1, b1, c00);
								c01 = _mm256_fmadd_ps(a1, b2, c01);
								c10 = _mm256_fmadd_ps(a2, b1, c10);
								c11 = _mm256_fmadd_ps(a2, b2, c11);

								a1 = _mm256_set1_ps(inA[i][k][l + 2][m + 1]);
								a2 = _mm256_set1_ps(inA[i][k][l + 3][m + 1]);
								c20 = _mm256_fmadd_ps(a1, b1, c20);
								c21 = _mm256_fmadd_ps(a1, b2, c21);
								c30 = _mm256_fmadd_ps(a2, b1, c30);
								c31 = _mm256_fmadd_ps(a2, b2, c31);

								b1 = _mm256_loadu_ps(pB[m + 2] + x);
								b2 = _mm256_loadu_ps(pB[m + 2] + x + 8);

								a1 = _mm256_set1_ps(inA[i][k][l][m + 2]);
								a2 = _mm256_set1_ps(inA[i][k][l + 1][m + 2]);
								c00 = _mm256_fmadd_ps(a1, b1, c00);
								c01 = _mm256_fmadd_ps(a1, b2, c01);
								c10 = _mm256_fmadd_ps(a2, b1, c10);
								c11 = _mm256_fmadd_ps(a2, b2, c11);

								a1 = _mm256_set1_ps(inA[i][k][l + 2][m + 2]);
								a2 = _mm256_set1_ps(inA[i][k][l + 3][m + 2]);
								c20 = _mm256_fmadd_ps(a1, b1, c20);
								c21 = _mm256_fmadd_ps(a1, b2, c21);
								c30 = _mm256_fmadd_ps(a2, b1, c30);
								c31 = _mm256_fmadd_ps(a2, b2, c31);


								b1 = _mm256_loadu_ps(pB[m + 3] + x);
								b2 = _mm256_loadu_ps(pB[m + 3] + x + 8);

								a1 = _mm256_set1_ps(inA[i][k][l][m + 3]);
								a2 = _mm256_set1_ps(inA[i][k][l + 1][m + 3]);
								c00 = _mm256_fmadd_ps(a1, b1, c00);
								c01 = _mm256_fmadd_ps(a1, b2, c01);
								c10 = _mm256_fmadd_ps(a2, b1, c10);
								c11 = _mm256_fmadd_ps(a2, b2, c11);

								a1 = _mm256_set1_ps(inA[i][k][l + 2][m + 3]);
								a2 = _mm256_set1_ps(inA[i][k][l + 3][m + 3]);
								c20 = _mm256_fmadd_ps(a1, b1, c20);
								c21 = _mm256_fmadd_ps(a1, b2, c21);
								c30 = _mm256_fmadd_ps(a2, b1, c30);
								c31 = _mm256_fmadd_ps(a2, b2, c31);
							}
							_mm256_storeu_ps(pOut[l] + x, _mm256_add_ps(c00, _mm256_loadu_ps(pOut[l] + x)));
							_mm256_storeu_ps(pOut[l] + x + 8, _mm256_add_ps(c01, _mm256_loadu_ps(pOut[l] + x + 8)));

							_mm256_storeu_ps(pOut[l + 1] + x, _mm256_add_ps(c10, _mm256_loadu_ps(pOut[l + 1] + x)));
							_mm256_storeu_ps(pOut[l + 1] + x + 8, _mm256_add_ps(c11, _mm256_loadu_ps(pOut[l + 1] + x + 8)));

							_mm256_storeu_ps(pOut[l + 2] + x, _mm256_add_ps(c20, _mm256_loadu_ps(pOut[l + 2] + x)));
							_mm256_storeu_ps(pOut[l + 2] + x + 8, _mm256_add_ps(c21, _mm256_loadu_ps(pOut[l + 2] + x + 8)));

							_mm256_storeu_ps(pOut[l + 3] + x, _mm256_add_ps(c30, _mm256_loadu_ps(pOut[l + 3] + x)));
							_mm256_storeu_ps(pOut[l + 3] + x + 8, _mm256_add_ps(c31, _mm256_loadu_ps(pOut[l + 3] + x + 8)));

							for (int mm = INNER_N_COL_FIRST_ALIGNED_TO_4_DOWN; mm < INNER_N_COL_FIRST; ++mm)
							{
								for (int ll = l; ll < l + 4; ++ll)
								{
									const float* const* pB = inB[k][j];
									for (int xx = x; xx < x + 16; ++xx)
									{
										pOut[ll][xx] += inA[i][k][ll][mm] * pB[mm][xx];

									}
								}
							}
						}

						for (int xx = INNER_N_COL_SECOND_ALIGNED_TO_16_DOWN; xx < INNER_N_COL_SECOND; ++xx)
						{
							for (int ll = l; ll < l + 4; ++ll)
							{
								const float* const* pB = inB[k][j];
								for (int mm = 0; mm < INNER_N_COL_FIRST; ++mm)
								{
									pOut[ll][xx] += inA[i][k][ll][mm] * pB[mm][xx];

								}
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

	using FltMxIn = const float const* const;
	using FltMxOut = float* const;
	static void CacheL1Optimazation(const size_t nCol1stRow2nd
		, FltMxIn inA, const size_t  nRowA
		, FltMxIn inB, const size_t nColB, FltMxOut out)
	{
		for (int l = 0; l < nRowA; l++)
		{
			for (int x = 0; x < nColB; x++)
			{
				for (int m = 0; m < nCol1stRow2nd; m++)
				{
					out[l * nColB + x] += inA[l * nCol1stRow2nd + m] * inB[m * nColB + x];
				}
			}
		}
	}
	static void DefaultMul(const size_t nCol1stRow2nd
		, FltMxIn inA, const size_t  nRowA
		, FltMxIn inB, const size_t nColB, FltMxOut out)
	{
		//const size_t L3 = 32* 16*8/;
		for (int l = 0; l < nRowA; l++)
		{
			for (int x = 0; x < nColB; x++)
			{
				for (int m = 0; m < nCol1stRow2nd; m++)
				{
					out[l * nColB + x] += inA[l * nCol1stRow2nd + m] * inB[m * nColB + x];
				}
			}
		}
	}

	static bool Equals(FltMxIn inA, const size_t  nRowA
		, FltMxIn inB, const size_t nColB)
	{
		for (int i = 0; i < nRowA; i++) {
			for (int j = 0; j < nColB; j++) {
				if (inA[i * nColB + j] != inB[i * nColB + j])
					return false;
			}
		}
		return true;
	}

	//static void cacheOptimization(const size_t nCol1stRow2nd, const size_t nRow1st
	//	,float* a, const size_t nCol2nd, float* b, float* result) {
	//	//int blockSize = 64;
	//	
	//	const size_t L1SizeOfBlock = min(nCol2nd,32*16/64);
	//	const size_t L2SizeOfBlock = min(nCol1stRow2nd, 16 * 16 / (sizeof(float)) / L1SizeOfBlock);
	//	const size_t L3SizeOfBlock = min(nCol1stRow2nd, 4 * 16 * 16 / sizeof(float) / L1SizeOfBlock);

	//	size_t newL3CacheSize, newL2CacheSize, newL1CacheSize;
	//	//int i, j, k, I, J, K;

	//	for (int i = 0; i < nRow1st; i += L3SizeOfBlock)
	//	{
	//		newL3CacheSize = min(L3SizeOfBlock, nRow1st - i);
	//		for (int j = 0; j < nCol2nd; j += L2SizeOfBlock)
	//		{
	//			newL2CacheSize = min(L2SizeOfBlock, nCol2nd - j);
	//			for (int k = 0; k <nCol1stRow2nd ;k += L1SizeOfBlock)
	//			{
	//				newL1CacheSize = min(L1SizeOfBlock, nCol1stRow2nd - k);
	//				for (int iInner = i; iInner <i+ newL3CacheSize; iInner++)
	//				{
	//					for (int jInner = k; jInner <k+ newL1CacheSize; jInner++)
	//					{
	//						for (int kInner = j; kInner <j+ newL2CacheSize; kInner++)
	//						{
	//							result[iInner * nCol2nd + kInner] += a[iInner*nCol1stRow2nd+jInner] * b[jInner*nCol2nd+kInner];
	//						}
	//					}
	//				}
	//			}
	//		}
	//	}
	//}
	static void cacheOptimization(const
		size_t nCol1stRow2nd, const size_t nRow1st
		, float* a, const size_t nCol2nd, float* b, float* result)
	{
		const size_t L1SizeOfBlock = min(nCol2nd, 32 * 1024 / 64);
		const size_t L2SizeOfBlock = min(nCol1stRow2nd, 1024 * 512 / (sizeof(float)) / L1SizeOfBlock);
		const size_t L3SizeOfBlock = min(nCol1stRow2nd, 4 * 1024 * 1024 / sizeof(float) / L1SizeOfBlock);

		size_t  Aligned_L2, ModL2;


		size_t newL3CacheSize, newL2CacheSize, newL1CacheSize;

		for (int i = 0; i < nRow1st; i += L3SizeOfBlock)
		{
			newL3CacheSize = min(L3SizeOfBlock, nRow1st - i);
			for (int j = 0; j < nCol2nd; j += L2SizeOfBlock)
			{
				newL2CacheSize = min(L2SizeOfBlock, nCol2nd - j);
				for (int k = 0; k < nCol1stRow2nd; k += L1SizeOfBlock)
				{
					newL1CacheSize = min(L1SizeOfBlock, nCol1stRow2nd - k);
					Aligned_L2 = L2SizeOfBlock - L2SizeOfBlock % 16;
					ModL2 = L2SizeOfBlock % 16;

					for (int l = i; l < i+newL3CacheSize; l++)
					{
						for (int x = j; x <j+ Aligned_L2; x += 16)
						{
							__m256 c00 = _mm256_setzero_ps();
							__m256 c01 = _mm256_setzero_ps();
							__m256 a1, a2, b1, b2;
							for (int m = k; m < k+newL1CacheSize; m++)
							{
								b1 = _mm256_loadu_ps(b +m*nCol2nd+x);
								b2 = _mm256_loadu_ps(b + m * nCol2nd+x + 8);

								a1 = _mm256_set1_ps(a[l*nCol1stRow2nd+m]);
								c00 = _mm256_fmadd_ps(a1, b1, c00);
								c01 = _mm256_fmadd_ps(a1, b2, c01);

							}
							_mm256_storeu_ps(result+l*nCol2nd + x, _mm256_add_ps(c00, _mm256_loadu_ps(result+ l * nCol2nd + x)));
							_mm256_storeu_ps(result + l * nCol2nd + x + 8, _mm256_add_ps(c01, _mm256_loadu_ps(result + l * nCol2nd + x + 8)));

						}
						for (int xxxxx =j+ Aligned_L2; xxxxx <j+Aligned_L2 + ModL2; ++xxxxx) {
							for (int mmmmm = 0; mmmmm < newL1CacheSize; ++mmmmm) {
								result[l * nCol2nd + xxxxx] += b[mmmmm*nCol2nd+ xxxxx] * a[l*nCol1stRow2nd+ mmmmm];
							}
						}
					}
				}

			}
		}
	}
	

	


	

	static void Print(FltMxIn a,const size_t nrow, const size_t ncol)
	{
		for (size_t i = 0; i < nrow; i++)
		{
			for (size_t j = 0; j < ncol; j++)
			{
				std::cout << a[i*ncol+j]<< " ";
			}
			std::cout << std::endl;
		}
	}
	static constexpr size_t PADDING_SIZE = 32;

	static float* Create(const size_t nrow, const size_t ncol, const std::function<int()>& generator,bool& was_aligned, const size_t padding)
	{
		const size_t bufSz = nrow * ncol;
		size_t bufSzWithPadding = bufSz + padding;

		void* buf = new float[bufSzWithPadding];

		was_aligned = std::align(padding, bufSz, buf, bufSzWithPadding) ? true : false;

		float* ret = reinterpret_cast<float*> (buf);

		for (size_t i = 0; i < nrow; i++)
		{
			for (size_t j = 0; j < ncol; j++)
			{
				ret[i * ncol + j] = generator();
			}
		}
		return ret;
	}


	static float* Transform(float**** from_is, size_t nColOuter, size_t nRowOuter, size_t nColInner, size_t nRowInner)
	{
		bool was_aligned_mx1;
		float* to_it = Create(nRowInner * nRowOuter, nColInner * nColOuter, []() {return 0; }, was_aligned_mx1, MatrixProcessor::PADDING_SIZE);
		if (!was_aligned_mx1)
		{
			std::cout << std::endl << "Not Aligned" << std::endl;
		}
		for (size_t i = 0; i < nRowOuter; i++)
		{
			for (size_t k = 0; k < nColOuter; k++)
			{
			    for (size_t j = 0; j < nRowInner; j++) 
				{
					for (size_t l = 0; l < nColInner; l++)
					{
						to_it[(i * nColOuter )*nRowInner*nColInner+k*nColInner +j*(nColInner*nColOuter) +l] = from_is[i][k][j][l];
					}
				}
			}
		}
		return to_it;
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


	float* cd = MatrixProcessor::Transform(a, OUTER_N_COL, OUTER_N_ROW, INNER_N_COL_FIRST, INNER_N_ROW_FIRST);

	MatrixProcessor::Print(cd, INNER_N_ROW_FIRST *OUTER_N_ROW, OUTER_N_COL* INNER_N_COL_FIRST);

	MatrixProcessor::Print(a, OUTER_N_ROW, OUTER_N_COL, INNER_N_ROW_FIRST, INNER_N_COL_FIRST);

	std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

	bool was_aligned_mx1;
	float* mx = MatrixProcessor::Create(10, 10, []() { return rand()%10; },was_aligned_mx1,MatrixProcessor::PADDING_SIZE);
	if (!was_aligned_mx1)
	{
		std::cout << std::endl << "Not Aligned" << std::endl;
	}
	//MatrixProcessor::Print(mx, 10, 10);

	bool was_aligned_mx2;
	float* mx2 = MatrixProcessor::Create(10, 10, []() {return rand() % 10; }, was_aligned_mx2,MatrixProcessor::PADDING_SIZE);
	if (!was_aligned_mx2)
	{
		std::cout << std::endl << "Not Aligned" << std::endl;
	}
	//MatrixProcessor::Print(mx2,4, 10);

	bool was_aligned_mx3;
	float* mx3 = MatrixProcessor::Create(10, 10, []() {static int i = 0; return i; }, was_aligned_mx3, MatrixProcessor::PADDING_SIZE);
	if (!was_aligned_mx3)
	{
		std::cout << std::endl << "Not Aligned" << std::endl;
	}
	

	

	bool was_aligned_mx4;
	float* mx4 = MatrixProcessor::Create(10, 10, []() {static int i = 0; return i; }, was_aligned_mx4, MatrixProcessor::PADDING_SIZE);
	if (!was_aligned_mx4)
	{
		std::cout << std::endl << "Not Aligned" << std::endl;
	}

	start = std::chrono::high_resolution_clock::now();

	MatrixProcessor::cacheOptimization(10, 10, mx, 10, mx2, mx3);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time   cache: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "mcs" << std::endl;

	start = std::chrono::high_resolution_clock::now();

	MatrixProcessor::DefaultMul(10, mx, 10, mx2, 10, mx4);

	end = std::chrono::high_resolution_clock::now();
	std::cout << "Time default: " << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "mcs" << std::endl;

	//MatrixProcessor::Print(mx3, 4, 10);

	//MatrixProcessor::Print(mx4, 4, 10);

	std::cout << MatrixProcessor::Equals(mx4, 10, mx3, 10);

	//SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);

	
	/*std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

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
		std::cout << "Not equal. Not Xdd" << std::endl;*/

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