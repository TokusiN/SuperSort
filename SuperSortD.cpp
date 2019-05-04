#include <stdio.h>
#include <memory>
#include <immintrin.h>

// ソート本体
void SuperSortD(double* array, size_t num);


namespace {
	size_t g_bufsize = 0;
	double* g_buf1;
	double* g_buf2;
	void* AlignedMalloc(size_t size)
	{
		void* ptr = _mm_malloc(size, 32);
		return ptr;
	}

	void AlignedFree(void* ptr)
	{
		_mm_free(ptr);
	}
	void SuperSortDAligned(double* array, size_t num);
	void SuperSortD32(double* arr, double* dst = NULL);
	void SuperSortD48(double* arr, double* dst = NULL);
	void SuperSortD64(double* arr, double* dst = NULL);
} // namespace

#define PADDING_MAX INFINITY

void SuperSortD(double* arr, size_t num)
{
	bool isAligned = (((size_t)arr) & 16) == 0;
	size_t alignedsize;
	if (num < 32)
	{
		alignedsize = 32;
	}
	else
	{
		alignedsize = (num - 1 | 15) + 1;
	}
	if (alignedsize > g_bufsize)
	{
		if (g_bufsize)
		{
			AlignedFree(g_buf1);
			AlignedFree(g_buf2);
		}
		g_bufsize = alignedsize * 2;
		g_buf1 = (double*)AlignedMalloc(sizeof(double) * g_bufsize);
		g_buf2 = (double*)AlignedMalloc(sizeof(double) * g_bufsize);
	}
	if (num == alignedsize && isAligned)
	{
		if (alignedsize > 64)
		{
			SuperSortDAligned(arr, alignedsize / 16);
		}
		else if (alignedsize == 32)
		{
			SuperSortD32(arr);
		}
		else if (alignedsize == 48)
		{
			SuperSortD48(arr);
		}
		else
		{
			SuperSortD64(arr);
		}
	}
	else
	{
		double* buf = g_buf1;
		size_t i;
		for (i = num; i < alignedsize; i++)
		{
			buf[i] = PADDING_MAX;
		}
		memcpy(buf, arr, sizeof(double) * num);
		if (alignedsize > 64)
		{
			SuperSortDAligned(buf, alignedsize / 16);
		}
		else if (alignedsize == 32)
		{
			SuperSortD32(buf);
		}
		else if (alignedsize == 48)
		{
			SuperSortD48(buf);
		}
		else
		{
			SuperSortD64(buf);
		}
		memcpy(arr, buf, sizeof(double) * num);
	}
}

namespace {

	// 比較器
	auto Comparator = [](__m256d & lo, __m256d & hi) {
		__m256d t;
		t = _mm256_min_pd(lo, hi);
		hi = _mm256_max_pd(lo, hi);
		lo = t;
	};
	auto Swap01 = [](__m256d & lo, __m256d & hi) {
		__m256d t;
		t = _mm256_shuffle_pd(lo, hi, 0);
		hi = _mm256_shuffle_pd(lo, hi, 15);
		lo = t;
	};
	auto Swap02 = [](__m256d & lo, __m256d & hi) {
		__m256d t;
		t = _mm256_permute2f128_pd(lo, hi, 0x20);
		hi = _mm256_permute2f128_pd(lo, hi, 0x31);
		lo = t;
	};

#define Merge1616() {\
	m4 = _mm256_permute4x64_pd(m4, 0x1B);\
	m5 = _mm256_permute4x64_pd(m5, 0x1B);\
	m6 = _mm256_permute4x64_pd(m6, 0x1B);\
	m7 = _mm256_permute4x64_pd(m7, 0x1B);\
	Comparator(m0, m7);\
	Comparator(m1, m6);\
	Comparator(m2, m5);\
	Comparator(m3, m4);\
	Comparator(m0, m2);\
	Comparator(m1, m3);\
	Comparator(m4, m6);\
	Comparator(m5, m7);\
	Comparator(m0, m1);\
	Comparator(m2, m3);\
	Comparator(m4, m5);\
	Comparator(m6, m7);\
	Swap02(m0, m2);\
	Swap02(m1, m3);\
	Swap02(m4, m6);\
	Swap02(m5, m7);\
	Swap01(m0, m1);\
	Swap01(m2, m3);\
	Swap01(m4, m5);\
	Swap01(m6, m7);\
	Comparator(m0, m2);\
	Comparator(m1, m3);\
	Comparator(m4, m6);\
	Comparator(m5, m7);\
	Comparator(m0, m1);\
	Comparator(m2, m3);\
	Comparator(m4, m5);\
	Comparator(m6, m7);\
	Swap02(m0, m2);\
	Swap02(m1, m3);\
	Swap02(m4, m6);\
	Swap02(m5, m7);\
	Swap01(m0, m1);\
	Swap01(m2, m3);\
	Swap01(m4, m5);\
	Swap01(m6, m7);\
}
	void SuperSortD32(double* arr, double* dst)
	{
		__m256d m0, m1, m2, m3, m4, m5, m6, m7, ms, mt;

		if (!dst)
		{
			dst = arr;
		}
		// 規定のメモリにロード
		m0 = _mm256_load_pd((arr + 0));
		m1 = _mm256_load_pd((arr + 4));
		m2 = _mm256_load_pd((arr + 8));
		m3 = _mm256_load_pd((arr + 12));
		m4 = _mm256_load_pd((arr + 16));
		m5 = _mm256_load_pd((arr + 20));
		m6 = _mm256_load_pd((arr + 24));
		m7 = _mm256_load_pd((arr + 28));
		// 4並列でバッチャー奇偶マージソートを実行
		Comparator(m0, m1);
		Comparator(m2, m3);
		Comparator(m4, m5);
		Comparator(m6, m7);
		Comparator(m0, m2);
		Comparator(m1, m3);
		Comparator(m4, m6);
		Comparator(m5, m7);
		Comparator(m1, m2);
		Comparator(m5, m6);
		Comparator(m0, m4);
		Comparator(m1, m5);
		Comparator(m2, m6);
		Comparator(m3, m7);
		Comparator(m2, m4);
		Comparator(m3, m5);
		Comparator(m1, m2);
		Comparator(m3, m4);
		Comparator(m5, m6);
		// 0と1、2と3をスワップ
		m4 = _mm256_permute4x64_pd(m4, 0xB1);
		m5 = _mm256_permute4x64_pd(m5, 0xB1);
		m6 = _mm256_permute4x64_pd(m6, 0xB1);
		m7 = _mm256_permute4x64_pd(m7, 0xB1);

		Comparator(m0, m7);
		Comparator(m1, m6);
		Comparator(m2, m5);
		Comparator(m3, m4);
		// m0の0とm7の1、m0の2とm7の3、・・・をスワップ
		Swap01(m0, m7);
		Swap01(m1, m6);
		Swap01(m2, m5);
		Swap01(m3, m4);

		// バイトニック列をソート
		auto SortBitnic = [&]() {
			Comparator(m0, m4);
			Comparator(m1, m5);
			Comparator(m2, m6);
			Comparator(m3, m7);
			Comparator(m0, m2);
			Comparator(m1, m3);
			Comparator(m4, m6);
			Comparator(m5, m7);
			Comparator(m0, m1);
			Comparator(m2, m3);
			Comparator(m4, m5);
			Comparator(m6, m7);
		};
		SortBitnic();
		m4 = _mm256_permute4x64_pd(m4, 0x1B);
		m5 = _mm256_permute4x64_pd(m5, 0x1B);
		m6 = _mm256_permute4x64_pd(m6, 0x1B);
		m7 = _mm256_permute4x64_pd(m7, 0x1B);
		Comparator(m0, m7);
		Comparator(m1, m6);
		Comparator(m2, m5);
		Comparator(m3, m4);
		Swap01(m0, m4);
		Swap01(m1, m5);
		Swap01(m2, m6);
		Swap01(m3, m7);
		Comparator(m0, m4);
		Comparator(m1, m5);
		Comparator(m2, m6);
		Comparator(m3, m7);
		Swap02(m0, m7);
		Swap02(m1, m6);
		Swap02(m2, m5);
		Swap02(m3, m4);
		SortBitnic();
		// ソート完了
		Swap02(m0, m2);
		Swap02(m1, m3);
		Swap02(m4, m6);
		Swap02(m5, m7);
		Swap01(m0, m1);
		Swap01(m2, m3);
		Swap01(m4, m5);
		Swap01(m6, m7);
		// メモリの詰め替え完了
		// ストア
		_mm256_store_pd((dst + 0), m0);
		_mm256_store_pd((dst + 4), m4);
		_mm256_store_pd((dst + 8), m2);
		_mm256_store_pd((dst + 12), m6);
		_mm256_store_pd((dst + 16), m1);
		_mm256_store_pd((dst + 20), m5);
		_mm256_store_pd((dst + 24), m3);
		_mm256_store_pd((dst + 28), m7);

	}
	void SuperSortD48(double* arr, double* dst)
	{
		if (!dst)
		{
			dst = arr;
		}
		SuperSortD32(arr);
		SuperSortD32(arr + 16, dst+16);
		__m256d m0, m1, m2, m3, m4, m5, m6, m7;


		m0 = _mm256_load_pd(arr + 0);
		m1 = _mm256_load_pd(arr + 4);
		m2 = _mm256_load_pd(arr + 8);
		m3 = _mm256_load_pd(arr + 12);
		m4 = _mm256_load_pd(dst + 16 + 0);
		m5 = _mm256_load_pd(dst + 16 + 4);
		m6 = _mm256_load_pd(dst + 16 + 8);
		m7 = _mm256_load_pd(dst + 16 + 12);
		Merge1616();
		_mm256_store_pd(dst + 0, m0);
		_mm256_store_pd(dst + 4, m1);
		_mm256_store_pd(dst + 8, m2);
		_mm256_store_pd(dst + 12, m3);
		_mm256_store_pd(dst + 16 + 0, m4);
		_mm256_store_pd(dst + 16 + 4, m5);
		_mm256_store_pd(dst + 16 + 8, m6);
		_mm256_store_pd(dst + 16 + 12, m7);
	}

	void SuperSortD64(double* arr, double* dst)
	{
		if (!dst)
		{
			dst = arr;
		}
		SuperSortD32(arr);
		SuperSortD32(arr + 32);
		__m256d m0, m1, m2, m3, m4, m5, m6, m7;

		m0 = _mm256_load_pd(arr + 0);
		m1 = _mm256_load_pd(arr + 4);
		m2 = _mm256_load_pd(arr + 8);
		m3 = _mm256_load_pd(arr + 12);
		m4 = _mm256_load_pd(arr + 32 + 0);
		m5 = _mm256_load_pd(arr + 32 + 4);
		m6 = _mm256_load_pd(arr + 32 + 8);
		m7 = _mm256_load_pd(arr + 32 + 12);
		Merge1616();
		_mm256_store_pd(dst + 0, m0);
		_mm256_store_pd(dst + 4, m1);
		_mm256_store_pd(dst + 8, m2);
		_mm256_store_pd(dst + 12, m3);
		_mm256_store_pd(arr + 32 + 0, m4);
		_mm256_store_pd(arr + 32 + 4, m5);
		_mm256_store_pd(arr + 32 + 8, m6);
		_mm256_store_pd(arr + 32 + 12, m7);
		m0 = _mm256_load_pd(arr + 16 + 0);
		m1 = _mm256_load_pd(arr + 16 + 4);
		m2 = _mm256_load_pd(arr + 16 + 8);
		m3 = _mm256_load_pd(arr + 16 + 12);
		m4 = _mm256_load_pd(arr + 48 + 0);
		m5 = _mm256_load_pd(arr + 48 + 4);
		m6 = _mm256_load_pd(arr + 48 + 8);
		m7 = _mm256_load_pd(arr + 48 + 12);
		Merge1616();
		_mm256_store_pd(dst + 48 + 0, m4);
		_mm256_store_pd(dst + 48 + 4, m5);
		_mm256_store_pd(dst + 48 + 8, m6);
		_mm256_store_pd(dst + 48 + 12, m7);
		m4 = _mm256_load_pd(arr + 32 + 0);
		m5 = _mm256_load_pd(arr + 32 + 4);
		m6 = _mm256_load_pd(arr + 32 + 8);
		m7 = _mm256_load_pd(arr + 32 + 12);
		Merge1616();
		_mm256_store_pd(dst + 16 + 0, m0);
		_mm256_store_pd(dst + 16 + 4, m1);
		_mm256_store_pd(dst + 16 + 8, m2);
		_mm256_store_pd(dst + 16 + 12, m3);
		_mm256_store_pd(dst + 32 + 0, m4);
		_mm256_store_pd(dst + 32 + 4, m5);
		_mm256_store_pd(dst + 32 + 8, m6);
		_mm256_store_pd(dst + 32 + 12, m7);
	}

	void MergeD(double* src1, size_t size1, double* src2, size_t size2, double* dst)
	{
		size_t i, j;
		i = j = 1;
		__m256d m0, m1, m2, m3, m4, m5, m6, m7;

		m0 = _mm256_load_pd(src1 + 0);
		m1 = _mm256_load_pd(src1 + 4);
		m2 = _mm256_load_pd(src1 + 8);
		m3 = _mm256_load_pd(src1 + 12);
		m4 = _mm256_load_pd(src2 + 0);
		m5 = _mm256_load_pd(src2 + 4);
		m6 = _mm256_load_pd(src2 + 8);
		m7 = _mm256_load_pd(src2 + 12);
		Merge1616();
		_mm256_store_pd(dst + 0, m0);
		_mm256_store_pd(dst + 4, m1);
		_mm256_store_pd(dst + 8, m2);
		_mm256_store_pd(dst + 12, m3);
		src1 += 16;
		src2 += 16;
		dst += 16;
		while (1)
		{
			if (src1[0] > src2[0])
			{
				m0 = _mm256_load_pd(src2 + 0);
				m1 = _mm256_load_pd(src2 + 4);
				m2 = _mm256_load_pd(src2 + 8);
				m3 = _mm256_load_pd(src2 + 12);
				src2 += 16;
				j++;
				Merge1616();
				_mm256_store_pd(dst + 0, m0);
				_mm256_store_pd(dst + 4, m1);
				_mm256_store_pd(dst + 8, m2);
				_mm256_store_pd(dst + 12, m3);
				dst += 16;
				if (j == size2)
				{
					while (i < size1)
					{
						m0 = _mm256_load_pd(src1 + 0);
						m1 = _mm256_load_pd(src1 + 4);
						m2 = _mm256_load_pd(src1 + 8);
						m3 = _mm256_load_pd(src1 + 12);
						src1 += 16;
						i++;
						Merge1616();
						_mm256_store_pd(dst + 0, m0);
						_mm256_store_pd(dst + 4, m1);
						_mm256_store_pd(dst + 8, m2);
						_mm256_store_pd(dst + 12, m3);
						dst += 16;
					}
					break;
				}
			}
			else
			{
				m0 = _mm256_load_pd(src1 + 0);
				m1 = _mm256_load_pd(src1 + 4);
				m2 = _mm256_load_pd(src1 + 8);
				m3 = _mm256_load_pd(src1 + 12);
				src1 += 16;
				i++;
				Merge1616();
				_mm256_store_pd(dst + 0, m0);
				_mm256_store_pd(dst + 4, m1);
				_mm256_store_pd(dst + 8, m2);
				_mm256_store_pd(dst + 12, m3);
				dst += 16;
				if (i == size1)
				{
					while (j < size2)
					{
						m0 = _mm256_load_pd(src2 + 0);
						m1 = _mm256_load_pd(src2 + 4);
						m2 = _mm256_load_pd(src2 + 8);
						m3 = _mm256_load_pd(src2 + 12);
						src2 += 16;
						j++;
						Merge1616();
						_mm256_store_pd(dst + 0, m0);
						_mm256_store_pd(dst + 4, m1);
						_mm256_store_pd(dst + 8, m2);
						_mm256_store_pd(dst + 12, m3);
						dst += 16;
					}
					break;
				}
			}
		}
		_mm256_store_pd(dst + 0, m4);
		_mm256_store_pd(dst + 4, m5);
		_mm256_store_pd(dst + 8, m6);
		_mm256_store_pd(dst + 12, m7);
	}
	void SuperSortRecD(double* src, double* dst, double* org, size_t num)
	{
		if (num > 4)
		{
			SuperSortRecD(dst, src, org, num / 2);
			SuperSortRecD(dst + num / 2 * 16, src + num / 2 * 16, org + num / 2 * 16, num - num / 2);
			MergeD(src, num / 2, src + num / 2 * 16, num - num / 2, dst);
		}
		else
		{
			if (num == 4)
			{
				SuperSortD64(org, dst);
			}
			else if (num == 3)
			{
				SuperSortD48(org, dst);
			}
			else
			{
				SuperSortD32(org, dst);
			}
		}
	}

	void SuperSortDAligned(double * array, size_t num)
	{
		double* buf = g_buf2;
		SuperSortRecD(buf, array, array, num);
	}
}// namespace
