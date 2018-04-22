#include <stdio.h>
#include <algorithm>
#include <vector>
#include <memory.h>
#include <time.h>
#include <cmath>
#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include "SuperSort.h"

namespace {
	void SuperSortAligned(int* array, size_t num);
	void SuperSort64(int* array, int* dst = NULL);
	void SuperSort96(int* array, int* dst = NULL);
	void SuperSort128(int* array, int* dst = NULL);
} // namespace

void* AlignedMalloc(size_t size)
{
	void* ptr = _mm_malloc(size, 256);
	return ptr;
}

void AlignedFree(void* ptr)
{
	_mm_free(ptr);
}

void SuperSort(int* array, size_t num)
{
	bool isAligned = (((unsigned int)array) & 31) == 0;
	int alignedsize;
	if (num < 64)
	{
		alignedsize = 64;
	}
	else
	{
		alignedsize = (num - 1 | 31) + 1;
	}
	if (num == alignedsize && isAligned)
	{
		if (alignedsize > 128)
		{
			SuperSortAligned(array, alignedsize / 32);
		}
		if (alignedsize == 64)
		{
			SuperSort64(array);
		}
		else if (alignedsize == 96)
		{
			SuperSort96(array);
		}
		else
		{
			SuperSort128(array);
		}
	}
	else
	{
		int* buf = (int*)AlignedMalloc(sizeof(int) * alignedsize);
		size_t i;
		for (i = num; i < alignedsize; i++)
		{
			buf[i] = 0x7fffffff;
		}
		memcpy(buf, array, sizeof(int) * num);
		if (alignedsize > 128)
		{
			SuperSortAligned(buf, alignedsize / 32);
		}
		else if (alignedsize == 64)
		{
			SuperSort64(buf);
		}
		else if (alignedsize == 96)
		{
			SuperSort96(buf);
		}
		else
		{
			SuperSort128(buf);
		}
		memcpy(array, buf, sizeof(int) * num);
		AlignedFree(buf);
	}
}

namespace {

	// 比較器
	auto Comparator = [](__m256i& lo, __m256i& hi) {
		__m256i t;
		t = _mm256_min_epi32(lo, hi);
		hi = _mm256_max_epi32(lo, hi);
		lo = t;
	};

	// レジスタ内で8並列のバイトニックソートを行う
	#define LineBitonicSort() \
		Comparator(m0, m4);\
		Comparator(m1, m5);\
		Comparator(m2, m6);\
		Comparator(m3, m7);\
		Comparator(m0, m2);\
		Comparator(m1, m3);\
		Comparator(m4, m6);\
		Comparator(m5, m7);\
		Comparator(m0, m1);\
		Comparator(m2, m3);\
		Comparator(m4, m5);\
		Comparator(m6, m7);

	// xmmレジスタの0番目と2番目、1番目と3番目の要素をそれぞれソートする
	auto ComparatorLR2 = [](__m256i& m0, __m256i& m1) {
		__m256i mt, ms, mu;
		mt = _mm256_alignr_epi8(m0, m1, 8);
		ms = _mm256_blend_epi32(m1, m0, 0xcc);
		mu = _mm256_max_epi32(mt, ms);
		mt = _mm256_min_epi32(mt, ms);
		m0 = _mm256_unpackhi_epi64(mt, mu);
		m1 = _mm256_unpacklo_epi64(mt, mu);
	};

	// xmmレジスタの0番目と1番目、2番目と3番目の要素をそれぞれソートする
	auto ComparatorLR = [](__m256i& m) {
		__m256i ms, mt;
		ms = _mm256_slli_si256(m, 4);
		mt = _mm256_min_epi32(m, ms);
		m = _mm256_max_epi32(m, ms);
		mt = _mm256_srli_si256(mt, 4);
		m = _mm256_blend_epi32(m, mt, 0x55);
	};

	// loの上位とhiの下位をスワップする
	auto Swapupdn4 = [](__m256i& lo, __m256i& hi) {
		__m256i mt;
		mt = _mm256_permute2f128_si256(lo, hi, 0x20);
		hi = _mm256_permute2f128_si256(lo, hi, 0x31);
		lo = mt;
	};

	auto Unpack = [](__m256i& lo, __m256i& hi) {
		__m256i mt;
		mt = _mm256_unpacklo_epi32(lo, hi);
		hi = _mm256_unpackhi_epi32(lo, hi);
		lo = mt;
	};

	#define Merge3232() {\
		m0 = _mm256_permutevar8x32_epi32(m0, maskflip8);\
		Comparator(m0, m7);\
		m1 = _mm256_permutevar8x32_epi32(m1, maskflip8);\
		Comparator(m1, m6);\
		m2 = _mm256_permutevar8x32_epi32(m2, maskflip8);\
		Comparator(m2, m5);\
		m3 = _mm256_permutevar8x32_epi32(m3, maskflip8);\
		Comparator(m3, m4);\
		Comparator(m0, m2);\
		Comparator(m1, m3);\
		Comparator(m4, m6);\
		Comparator(m5, m7);\
		Comparator(m0, m1);\
		Comparator(m2, m3);\
		Comparator(m4, m5);\
		Comparator(m6, m7);\
		Swapupdn4(m0, m4);\
		Swapupdn4(m1, m5);\
		Swapupdn4(m2, m6);\
		Swapupdn4(m3, m7);\
		Unpack(m0, m2);\
		Unpack(m1, m3);\
		Unpack(m4, m6);\
		Unpack(m5, m7);\
		Unpack(m0, m1);\
		Unpack(m2, m3);\
		Unpack(m4, m5);\
		Unpack(m6, m7);\
		LineBitonicSort();\
		Swapupdn4(m0, m4);\
		Swapupdn4(m1, m5);\
		Swapupdn4(m2, m6);\
		Swapupdn4(m3, m7);\
		Unpack(m0, m2);\
		Unpack(m1, m3);\
		Unpack(m4, m6);\
		Unpack(m5, m7);\
		Unpack(m0, m1);\
		Unpack(m2, m3);\
		Unpack(m4, m5);\
		Unpack(m6, m7);\
	}

	void SuperSort64(int* array, int* dst)
	{
		__m256i m0, m1, m2, m3, m4, m5, m6, m7, ms, mt, mu;

		if (!dst)
		{
			dst = array;
		}
		// 規定のメモリにロード
		m0 = _mm256_load_si256((__m256i*)(array + 0));
		m1 = _mm256_load_si256((__m256i*)(array + 8));
		m2 = _mm256_load_si256((__m256i*)(array + 16));
		m3 = _mm256_load_si256((__m256i*)(array + 24));
		m4 = _mm256_load_si256((__m256i*)(array + 32));
		m5 = _mm256_load_si256((__m256i*)(array + 40));
		m6 = _mm256_load_si256((__m256i*)(array + 48));
		m7 = _mm256_load_si256((__m256i*)(array + 56));
		// 8並列でバッチャー奇偶マージソートを実行
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

		// 4並列でバイトニックソートを1段実行
		//	DebugPrint();
		mt = _mm256_alignr_epi8(m0, m0, 8);
		ms = _mm256_min_epi32(m7, mt);
		mt = _mm256_max_epi32(m7, mt);
		m0 = _mm256_unpacklo_epi32(ms, mt);
		m7 = _mm256_unpackhi_epi32(ms, mt);

		mt = _mm256_alignr_epi8(m1, m1, 8);
		ms = _mm256_min_epi32(m6, mt);
		mt = _mm256_max_epi32(m6, mt);
		m1 = _mm256_unpacklo_epi32(ms, mt);
		m6 = _mm256_unpackhi_epi32(ms, mt);

		mt = _mm256_alignr_epi8(m2, m2, 8);
		ms = _mm256_min_epi32(m5, mt);
		mt = _mm256_max_epi32(m5, mt);
		m2 = _mm256_unpacklo_epi32(ms, mt);
		m5 = _mm256_unpackhi_epi32(ms, mt);

		mt = _mm256_alignr_epi8(m3, m3, 8);
		ms = _mm256_min_epi32(m4, mt);
		mt = _mm256_max_epi32(m4, mt);
		m3 = _mm256_unpacklo_epi32(ms, mt);
		m4 = _mm256_unpackhi_epi32(ms, mt);

		LineBitonicSort();

		mt = _mm256_shuffle_epi32(m0, 0x1b);
		ms = _mm256_min_epi32(m7, mt);
		mt = _mm256_max_epi32(m7, mt);
		m0 = _mm256_unpacklo_epi32(ms, mt);
		m7 = _mm256_unpackhi_epi32(ms, mt);

		mt = _mm256_shuffle_epi32(m1, 0x1b);
		ms = _mm256_min_epi32(m6, mt);
		mt = _mm256_max_epi32(m6, mt);
		m1 = _mm256_unpacklo_epi32(ms, mt);
		m6 = _mm256_unpackhi_epi32(ms, mt);

		mt = _mm256_shuffle_epi32(m2, 0x1b);
		ms = _mm256_min_epi32(m5, mt);
		mt = _mm256_max_epi32(m5, mt);
		m2 = _mm256_unpacklo_epi32(ms, mt);
		m5 = _mm256_unpackhi_epi32(ms, mt);

		mt = _mm256_shuffle_epi32(m3, 0x1b);
		ms = _mm256_min_epi32(m4, mt);
		mt = _mm256_max_epi32(m4, mt);
		m3 = _mm256_unpacklo_epi32(ms, mt);
		m4 = _mm256_unpackhi_epi32(ms, mt);

		ComparatorLR2(m0, m1);
		ComparatorLR2(m2, m3);
		ComparatorLR2(m4, m5);
		ComparatorLR2(m6, m7);

		LineBitonicSort();

		__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
		m0 = _mm256_permutevar8x32_epi32(m0, maskflip8);
		Comparator(m0, m7);
		m1 = _mm256_permutevar8x32_epi32(m1, maskflip8);
		Comparator(m1, m6);
		m2 = _mm256_permutevar8x32_epi32(m2, maskflip8);
		Comparator(m2, m5);
		m3 = _mm256_permutevar8x32_epi32(m3, maskflip8);
		Comparator(m3, m4);

		ComparatorLR(m0);
		ComparatorLR(m1);
		ComparatorLR(m2);
		ComparatorLR(m3);
		ComparatorLR(m4);
		ComparatorLR(m5);
		ComparatorLR(m6);
		ComparatorLR(m7);

		ComparatorLR2(m0, m1);
		ComparatorLR2(m2, m3);
		ComparatorLR2(m4, m5);
		ComparatorLR2(m6, m7);

		Swapupdn4(m0, m7);
		Swapupdn4(m1, m6);
		Swapupdn4(m2, m5);
		Swapupdn4(m3, m4);
		LineBitonicSort();
		// ここでソート終了

		// メモリ配置を詰め替える
		Swapupdn4(m0, m4);
		Swapupdn4(m1, m5);
		Swapupdn4(m2, m6);
		Swapupdn4(m3, m7);

		Unpack(m0, m2);
		Unpack(m1, m3);
		Unpack(m4, m6);
		Unpack(m5, m7);
		Unpack(m0, m1);
		Unpack(m2, m3);
		Unpack(m4, m5);
		Unpack(m6, m7);
		// ストア
		_mm256_store_si256((__m256i*)(dst + 0), m0);
		_mm256_store_si256((__m256i*)(dst + 8), m2);
		_mm256_store_si256((__m256i*)(dst + 16), m1);
		_mm256_store_si256((__m256i*)(dst + 24), m3);
		_mm256_store_si256((__m256i*)(dst + 32), m4);
		_mm256_store_si256((__m256i*)(dst + 40), m6);
		_mm256_store_si256((__m256i*)(dst + 48), m5);
		_mm256_store_si256((__m256i*)(dst + 56), m7);
	}

	void SuperSort96(int* array, int* dst)
	{
		if (!dst)
		{
			dst = array;
		}
		SuperSort64(array);
		SuperSort64(array + 32);
		__m256i m0, m1, m2, m3, m4, m5, m6, m7;
		__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
		// 規定のメモリにロード
		m0 = _mm256_load_si256((__m256i*)(array + 0));
		m1 = _mm256_load_si256((__m256i*)(array + 8));
		m2 = _mm256_load_si256((__m256i*)(array + 16));
		m3 = _mm256_load_si256((__m256i*)(array + 24));
		m4 = _mm256_load_si256((__m256i*)(array + 32 + 0));
		m5 = _mm256_load_si256((__m256i*)(array + 32 + 8));
		m6 = _mm256_load_si256((__m256i*)(array + 32 + 16));
		m7 = _mm256_load_si256((__m256i*)(array + 32 + 24));
		Merge3232();
		_mm256_store_si256((__m256i*)(dst + 0), m0);
		_mm256_store_si256((__m256i*)(dst + 8), m1);
		_mm256_store_si256((__m256i*)(dst + 16), m2);
		_mm256_store_si256((__m256i*)(dst + 24), m3);
		_mm256_store_si256((__m256i*)(dst + 32 + 0), m4);
		_mm256_store_si256((__m256i*)(dst + 32 + 8), m5);
		_mm256_store_si256((__m256i*)(dst + 32 + 16), m6);
		_mm256_store_si256((__m256i*)(dst + 32 + 24), m7);
		if (array != dst)
		{
			memcpy(dst + 64, array + 64, 128);
		}
	}

	void SuperSort128(int* array, int* dst)
	{
		if (!dst)
		{
			dst = array;
		}
		SuperSort64(array);
		SuperSort64(array + 64);
		__m256i m0, m1, m2, m3, m4, m5, m6, m7;
		__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

		m0 = _mm256_load_si256((__m256i*)(array + 0));
		m1 = _mm256_load_si256((__m256i*)(array + 8));
		m2 = _mm256_load_si256((__m256i*)(array + 16));
		m3 = _mm256_load_si256((__m256i*)(array + 24));
		m4 = _mm256_load_si256((__m256i*)(array + 64 + 0));
		m5 = _mm256_load_si256((__m256i*)(array + 64 + 8));
		m6 = _mm256_load_si256((__m256i*)(array + 64 + 16));
		m7 = _mm256_load_si256((__m256i*)(array + 64 + 24));
		Merge3232();
		_mm256_store_si256((__m256i*)(dst + 0), m0);
		_mm256_store_si256((__m256i*)(dst + 8), m1);
		_mm256_store_si256((__m256i*)(dst + 16), m2);
		_mm256_store_si256((__m256i*)(dst + 24), m3);
		_mm256_store_si256((__m256i*)(array + 64 + 0), m4);
		_mm256_store_si256((__m256i*)(array + 64 + 8), m5);
		_mm256_store_si256((__m256i*)(array + 64 + 16), m6);
		_mm256_store_si256((__m256i*)(array + 64 + 24), m7);
		m0 = _mm256_load_si256((__m256i*)(array + 32 + 0));
		m1 = _mm256_load_si256((__m256i*)(array + 32 + 8));
		m2 = _mm256_load_si256((__m256i*)(array + 32 + 16));
		m3 = _mm256_load_si256((__m256i*)(array + 32 + 24));
		m4 = _mm256_load_si256((__m256i*)(array + 96 + 0));
		m5 = _mm256_load_si256((__m256i*)(array + 96 + 8));
		m6 = _mm256_load_si256((__m256i*)(array + 96 + 16));
		m7 = _mm256_load_si256((__m256i*)(array + 96 + 24));
		Merge3232();
		_mm256_store_si256((__m256i*)(dst + 96 + 0), m4);
		_mm256_store_si256((__m256i*)(dst + 96 + 8), m5);
		_mm256_store_si256((__m256i*)(dst + 96 + 16), m6);
		_mm256_store_si256((__m256i*)(dst + 96 + 24), m7);
		m4 = _mm256_load_si256((__m256i*)(array + 64 + 0));
		m5 = _mm256_load_si256((__m256i*)(array + 64 + 8));
		m6 = _mm256_load_si256((__m256i*)(array + 64 + 16));
		m7 = _mm256_load_si256((__m256i*)(array + 64 + 24));
		Merge3232();
		_mm256_store_si256((__m256i*)(dst + 32 + 0), m0);
		_mm256_store_si256((__m256i*)(dst + 32 + 8), m1);
		_mm256_store_si256((__m256i*)(dst + 32 + 16), m2);
		_mm256_store_si256((__m256i*)(dst + 32 + 24), m3);
		_mm256_store_si256((__m256i*)(dst + 64 + 0), m4);
		_mm256_store_si256((__m256i*)(dst + 64 + 8), m5);
		_mm256_store_si256((__m256i*)(dst + 64 + 16), m6);
		_mm256_store_si256((__m256i*)(dst + 64 + 24), m7);
	}

	void Merge(int* src1, size_t size1, int* src2, size_t size2, int* dst)
	{
		int i, j;
		i = j = 1;
		__m256i m0, m1, m2, m3, m4, m5, m6, m7;
		__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);

		m0 = _mm256_load_si256((__m256i*)(src1 + 0));
		m1 = _mm256_load_si256((__m256i*)(src1 + 8));
		m2 = _mm256_load_si256((__m256i*)(src1 + 16));
		m3 = _mm256_load_si256((__m256i*)(src1 + 24));
		m4 = _mm256_load_si256((__m256i*)(src2 + 0));
		m5 = _mm256_load_si256((__m256i*)(src2 + 8));
		m6 = _mm256_load_si256((__m256i*)(src2 + 16));
		m7 = _mm256_load_si256((__m256i*)(src2 + 24));
		Merge3232();
		_mm256_store_si256((__m256i*)(dst + 0), m0);
		_mm256_store_si256((__m256i*)(dst + 8), m1);
		_mm256_store_si256((__m256i*)(dst + 16), m2);
		_mm256_store_si256((__m256i*)(dst + 24), m3);
		src1 += 32;
		src2 += 32;
		dst += 32;
		while (i != size1 || j != size2)
		{
			if (i == size1)
			{
				m0 = _mm256_load_si256((__m256i*)(src2 + 0));
				m1 = _mm256_load_si256((__m256i*)(src2 + 8));
				m2 = _mm256_load_si256((__m256i*)(src2 + 16));
				m3 = _mm256_load_si256((__m256i*)(src2 + 24));
				src2 += 32;
				j++;
			}
			else if (j == size2)
			{
				m0 = _mm256_load_si256((__m256i*)(src1 + 0));
				m1 = _mm256_load_si256((__m256i*)(src1 + 8));
				m2 = _mm256_load_si256((__m256i*)(src1 + 16));
				m3 = _mm256_load_si256((__m256i*)(src1 + 24));
				src1 += 32;
				i++;
			}
			else if (src1[0] > src2[0])
			{
				m0 = _mm256_load_si256((__m256i*)(src2 + 0));
				m1 = _mm256_load_si256((__m256i*)(src2 + 8));
				m2 = _mm256_load_si256((__m256i*)(src2 + 16));
				m3 = _mm256_load_si256((__m256i*)(src2 + 24));
				src2 += 32;
				j++;
			}
			else
			{
				m0 = _mm256_load_si256((__m256i*)(src1 + 0));
				m1 = _mm256_load_si256((__m256i*)(src1 + 8));
				m2 = _mm256_load_si256((__m256i*)(src1 + 16));
				m3 = _mm256_load_si256((__m256i*)(src1 + 24));
				src1 += 32;
				i++;
			}
			Merge3232();
			_mm256_store_si256((__m256i*)(dst + 0), m0);
			_mm256_store_si256((__m256i*)(dst + 8), m1);
			_mm256_store_si256((__m256i*)(dst + 16), m2);
			_mm256_store_si256((__m256i*)(dst + 24), m3);
			dst += 32;
		}
		_mm256_store_si256((__m256i*)(dst + 0), m4);
		_mm256_store_si256((__m256i*)(dst + 8), m5);
		_mm256_store_si256((__m256i*)(dst + 16), m6);
		_mm256_store_si256((__m256i*)(dst + 24), m7);
	}

	void SuperSortRec(int* src, int* dst, int* org, size_t num)
	{
		if (num > 4)
		{
			SuperSortRec(dst, src, org, num / 2);
			SuperSortRec(dst + num / 2 * 32, src + num / 2 * 32, org + num / 2 * 32, num - num / 2);
			Merge(src, num / 2, src + num / 2 * 32, num - num / 2, dst);
		}
		else
		{
			if (num == 4)
			{
				SuperSort128(org, dst);
			}
			else if (num == 3)
			{
				SuperSort96(org, dst);
			}
			else
			{
				SuperSort64(org, dst);
			}
		}
	}

	void SuperSortAligned(int* array, size_t num)
	{
		int* buf = (int*)AlignedMalloc(sizeof(int) * num * 32);
		size_t i;
		SuperSortRec(buf, array, array, num);
		AlignedFree(buf);
	}
} // namespace
