/*
	Copyright 2018 Toshihiro Shirakawa

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.
*/
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <memory.h>
#include <immintrin.h>

#include "SuperQuickSort.h"

#ifdef SUPERQUICKSORT_UNSIGNED
typedef unsigned int T;
const T PADDING_MAX = 0xFFFFFFFFU;
#define _mm256_max_epi32 _mm256_max_epu32
#define _mm256_min_epi32 _mm256_min_epu32
#define m256i_i32 m256i_u32
#else
typedef int T;
const T PADDING_MAX = 0x7FFFFFFF;
#endif

namespace {
	int SuperQuickSortRec(T* array, size_t num);
	void SuperQuickSortRecAligned(T* array, size_t num);
	//void SuperQuickSortEnd(T* array, size_t num);
	void SuperSort64(T* array);

	// 32ワードをメモリからレジスタにロードする
	void Load32(T* p, __m256i& m0, __m256i& m1, __m256i& m2, __m256i& m3)
	{
		m0 = _mm256_load_si256((__m256i*)(p + 0));
		m1 = _mm256_load_si256((__m256i*)(p + 8));
		m2 = _mm256_load_si256((__m256i*)(p + 16));
		m3 = _mm256_load_si256((__m256i*)(p + 24));
	}

	// 32ワードをレジスタからメモリに格納する
	void Store32(T* p, __m256i m0, __m256i m1, __m256i m2, __m256i m3)
	{
		_mm256_store_si256((__m256i*)(p + 0), m0);
		_mm256_store_si256((__m256i*)(p + 8), m1);
		_mm256_store_si256((__m256i*)(p + 16), m2);
		_mm256_store_si256((__m256i*)(p + 24), m3);
	}

	// 比較器
	void Comparator(__m256i& lo, __m256i& hi)
	{
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
	void ComparatorLR2(__m256i& m0, __m256i& m1)
	{
		__m256i mt, ms, mu;
		mt = _mm256_alignr_epi8(m0, m1, 8);
		ms = _mm256_blend_epi32(m1, m0, 0xcc);
		mu = _mm256_max_epi32(mt, ms);
		mt = _mm256_min_epi32(mt, ms);
		m0 = _mm256_unpackhi_epi64(mt, mu);
		m1 = _mm256_unpacklo_epi64(mt, mu);
	};

	// xmmレジスタの0番目と1番目、2番目と3番目の要素をそれぞれソートする
	void ComparatorLR(__m256i& m)
	{
		__m256i ms, mt;
		ms = _mm256_slli_si256(m, 4);
		mt = _mm256_min_epi32(m, ms);
		m = _mm256_max_epi32(m, ms);
		mt = _mm256_srli_si256(mt, 4);
		m = _mm256_blend_epi32(m, mt, 0x55);
	};

	// loの上位とhiの下位をスワップする
	void Swapupdn4(__m256i& lo, __m256i& hi)
	{
		__m256i mt;
		mt = _mm256_permute2f128_si256(lo, hi, 0x20);
		hi = _mm256_permute2f128_si256(lo, hi, 0x31);
		lo = mt;
	};

	void Unpack(__m256i& lo, __m256i& hi)
	{
		__m256i mt;
		mt = _mm256_unpacklo_epi32(lo, hi);
		hi = _mm256_unpackhi_epi32(lo, hi);
		lo = mt;
	};

#if 0
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
#else

	// m0～m3レジスタ、m4～m7レジスタに格納されているソート済み列をマージする
	#define Merge3232() _Merge3232(m0, m1, m2, m3, m4, m5, m6, m7)
	void _Merge3232(__m256i& m0, __m256i& m1, __m256i& m2, __m256i& m3, __m256i& m4, __m256i& m5, __m256i& m6, __m256i& m7)
	{
		__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
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
#endif
	/*
#define BitonicMerge64() {\
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
	*/
	// 64要素のデータを受け取り、ソート済みの32要素の列2つにする
	void SuperSort3232(T* array)
	{
		__m256i m0, m1, m2, m3, m4, m5, m6, m7, ms, mt;

		// 規定のメモリにロード
		Load32(array, m0, m1, m2, m3);
		Load32(array + 32, m4, m5, m6, m7);
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
		//	DebugPrT();
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
		Store32(array, m0, m2, m1, m3);
		Store32(array + 32, m4, m6, m5, m7);
	}

	// 64要素のデータをソートする
	void SuperSort64(T* array)
	{
		__m256i m0, m1, m2, m3, m4, m5, m6, m7, ms, mt;

		// 規定のメモリにロード
		Load32(array, m0, m1, m2, m3);
		Load32(array + 32, m4, m5, m6, m7);
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
		//	DebugPrT();
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

		// 32要素のバイトニックマージを実行
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
		Store32(array, m0, m2, m1, m3);
		Store32(array + 32, m4, m6, m5, m7);
	}

#if 0
	// 32要素毎にソート済みの96要素のデータをソートする
	void Merge32x3(T* array)
	{
		__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
		__m256i m0, m1, m2, m3, m4, m5, m6, m7;
		Load32(array, m0, m1, m2, m3);
		Load32(array + 32, m4, m5, m6, m7);
		Merge3232();
		Store32(array, m0, m1, m2, m3);
		Load32(array+64, m0, m1, m2, m3);
		Merge3232();
		Store32(array + 64, m4, m5, m6, m7);
		Load32(array, m4, m5, m6, m7);
		Merge3232();
		Store32(array, m0, m1, m2, m3);
		Store32(array + 32, m4, m5, m6, m7);
	}

	// 32要素毎にソート済みの128要素のデータをソートする
	void Merge32x4(T* array)
	{
		__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
		__m256i m0, m1, m2, m3, m4, m5, m6, m7;
		Load32(array, m0, m1, m2, m3);
		Load32(array + 32, m4, m5, m6, m7);
		Merge3232();
		Store32(array, m0, m1, m2, m3);
		Store32(array + 32, m4, m5, m6, m7);
		Load32(array + 64, m0, m1, m2, m3);
		Load32(array + 96, m4, m5, m6, m7);
		Merge3232();
		Store32(array + 32, m0, m1, m2, m3);
		Store32(array + 96, m4, m5, m6, m7);
		Load32(array, m0, m1, m2, m3);
		Load32(array + 64, m4, m5, m6, m7);
		Merge3232();
		Store32(array, m0, m1, m2, m3);
		Load32(array + 32, m0, m1, m2, m3);
		Merge3232();
		Store32(array + 32, m0, m1, m2, m3);
		Store32(array + 64, m4, m5, m6, m7);
	}
#endif
	// m0～m7レジスタに格納されているデータをソートする
#define SuperSort64Reg() _SuperSort64Reg(m0, m1, m2, m3, m4, m5, m6, m7)
	void _SuperSort64Reg(__m256i& m0, __m256i& m1, __m256i& m2, __m256i& m3, __m256i& m4, __m256i& m5, __m256i& m6, __m256i& m7)
	{\
		__m256i ms, mt;\
		/* 8並列でバッチャー奇偶マージソートを実行 */\
		Comparator(m0, m1);\
		Comparator(m2, m3);\
		Comparator(m4, m5);\
		Comparator(m6, m7);\
		Comparator(m0, m2);\
		Comparator(m1, m3);\
		Comparator(m4, m6);\
		Comparator(m5, m7);\
		Comparator(m1, m2);\
		Comparator(m5, m6);\
		Comparator(m0, m4);\
		Comparator(m1, m5);\
		Comparator(m2, m6);\
		Comparator(m3, m7);\
		Comparator(m2, m4);\
		Comparator(m3, m5);\
		Comparator(m1, m2);\
		Comparator(m3, m4);\
		Comparator(m5, m6);\
\
		/* 4並列でバイトニックソートを1段実行 */\
		mt = _mm256_alignr_epi8(m0, m0, 8);\
		ms = _mm256_min_epi32(m7, mt);\
		mt = _mm256_max_epi32(m7, mt);\
		m0 = _mm256_unpacklo_epi32(ms, mt);\
		m7 = _mm256_unpackhi_epi32(ms, mt);\
\
		mt = _mm256_alignr_epi8(m1, m1, 8);\
		ms = _mm256_min_epi32(m6, mt);\
		mt = _mm256_max_epi32(m6, mt);\
		m1 = _mm256_unpacklo_epi32(ms, mt);\
		m6 = _mm256_unpackhi_epi32(ms, mt);\
\
		mt = _mm256_alignr_epi8(m2, m2, 8);\
		ms = _mm256_min_epi32(m5, mt);\
		mt = _mm256_max_epi32(m5, mt);\
		m2 = _mm256_unpacklo_epi32(ms, mt);\
		m5 = _mm256_unpackhi_epi32(ms, mt);\
\
		mt = _mm256_alignr_epi8(m3, m3, 8);\
		ms = _mm256_min_epi32(m4, mt);\
		mt = _mm256_max_epi32(m4, mt);\
		m3 = _mm256_unpacklo_epi32(ms, mt);\
		m4 = _mm256_unpackhi_epi32(ms, mt);\
\
		LineBitonicSort();\
\
		mt = _mm256_shuffle_epi32(m0, 0x1b);\
		ms = _mm256_min_epi32(m7, mt);\
		mt = _mm256_max_epi32(m7, mt);\
		m0 = _mm256_unpacklo_epi32(ms, mt);\
		m7 = _mm256_unpackhi_epi32(ms, mt);\
\
		mt = _mm256_shuffle_epi32(m1, 0x1b);\
		ms = _mm256_min_epi32(m6, mt);\
		mt = _mm256_max_epi32(m6, mt);\
		m1 = _mm256_unpacklo_epi32(ms, mt);\
		m6 = _mm256_unpackhi_epi32(ms, mt);\
\
		mt = _mm256_shuffle_epi32(m2, 0x1b);\
		ms = _mm256_min_epi32(m5, mt);\
		mt = _mm256_max_epi32(m5, mt);\
		m2 = _mm256_unpacklo_epi32(ms, mt);\
		m5 = _mm256_unpackhi_epi32(ms, mt);\
\
		mt = _mm256_shuffle_epi32(m3, 0x1b);\
		ms = _mm256_min_epi32(m4, mt);\
		mt = _mm256_max_epi32(m4, mt);\
		m3 = _mm256_unpacklo_epi32(ms, mt);\
		m4 = _mm256_unpackhi_epi32(ms, mt);\
\
		ComparatorLR2(m0, m1);\
		ComparatorLR2(m2, m3);\
		ComparatorLR2(m4, m5);\
		ComparatorLR2(m6, m7);\
\
		LineBitonicSort();\
\
		__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);\
		m0 = _mm256_permutevar8x32_epi32(m0, maskflip8);\
		Comparator(m0, m7);\
		m1 = _mm256_permutevar8x32_epi32(m1, maskflip8);\
		Comparator(m1, m6);\
		m2 = _mm256_permutevar8x32_epi32(m2, maskflip8);\
		Comparator(m2, m5);\
		m3 = _mm256_permutevar8x32_epi32(m3, maskflip8);\
		Comparator(m3, m4);\
\
		ComparatorLR(m0);\
		ComparatorLR(m1);\
		ComparatorLR(m2);\
		ComparatorLR(m3);\
		ComparatorLR(m4);\
		ComparatorLR(m5);\
		ComparatorLR(m6);\
		ComparatorLR(m7);\
\
		ComparatorLR2(m0, m1);\
		ComparatorLR2(m2, m3);\
		ComparatorLR2(m4, m5);\
		ComparatorLR2(m6, m7);\
\
		Swapupdn4(m0, m7);\
		Swapupdn4(m1, m6);\
		Swapupdn4(m2, m5);\
		Swapupdn4(m3, m4);\
		LineBitonicSort();\
		/* ここでソート終了 */\
\
		/* メモリ配置を詰め替える */\
		Swapupdn4(m0, m4);\
		Swapupdn4(m1, m5);\
		Swapupdn4(m2, m6);\
		Swapupdn4(m3, m7);\
\
		Unpack(m0, m2);\
		Unpack(m1, m3);\
		Unpack(m4, m6);\
		Unpack(m5, m7);\
		Unpack(m0, m1);\
		Unpack(m2, m3);\
		Unpack(m4, m5);\
		Unpack(m6, m7);\
		auto t = m1;\
		m1 = m2;\
		m2 = t;\
		t = m5;\
		m5 = m6;\
		m6 = t;\
	}\

	// 32要素毎にソートされた32要素でアライメントされたデータを受け取り、クイックソートを行う
	void SuperQuickSortRecAligned(T* array, size_t num)
	{
		T* alignedArray = array;
		size_t alignedSize = num;
		if (num <= 256)
		{
#if 0
			if (num <= 128)
			{
				if (num == 64)
				{
					__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
					__m256i m0, m1, m2, m3, m4, m5, m6, m7;
					Load32(array, m0, m1, m2, m3);
					Load32(array + 32, m4, m5, m6, m7);
					Merge3232();
					Store32(array, m0, m1, m2, m3);
					Store32(array + 32, m4, m5, m6, m7);
				}
				else if (num == 96)
				{
					Merge32x3(array);
				}
				else
				{
					Merge32x4(array);
				}
			}
			else
			{
				__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
				__m256i m0, m1, m2, m3, m4, m5, m6, m7;
				if (num == 192)
				{
					Load32(array + 128, m0, m1, m2, m3);
					Load32(array + 160, m4, m5, m6, m7);
					Merge3232();
					Store32(array + 128, m0, m1, m2, m3);
					Store32(array + 160, m4, m5, m6, m7);
				}
				else if (num == 224)
				{
					Merge32x3(array + 128);
				}
				else if (num == 256)
				{
					Merge32x4(array + 128);
				}
				if (num >= 224)
				{
					Load32(array + 64, m0, m1, m2, m3);
					Load32(array + 192, m4, m5, m6, m7);
					Merge3232();
					Store32(array + 64, m0, m1, m2, m3);
					Store32(array + 192, m4, m5, m6, m7);
				}
				if (num == 256)
				{
					Load32(array + 96, m0, m1, m2, m3);
					Load32(array + 224, m4, m5, m6, m7);
					Merge3232();
					Store32(array + 96, m0, m1, m2, m3);
					Store32(array + 224, m4, m5, m6, m7);
				}
				Load32(array, m0, m1, m2, m3);
				Load32(array + 128, m4, m5, m6, m7);
				Merge3232();
				Store32(array, m0, m1, m2, m3);
				Store32(array + 128, m4, m5, m6, m7);
				if (num >= 192)
				{
					Load32(array+32, m0, m1, m2, m3);
					Load32(array + 160, m4, m5, m6, m7);
					Merge3232();
					Store32(array + 32, m0, m1, m2, m3);
					Load32(array + 96, m0, m1, m2, m3);
					Merge3232();
					Store32(array + 96, m0, m1, m2, m3);
					if (num >= 224)
					{
						Load32(array + 192, m0, m1, m2, m3);
						Merge3232();
						Store32(array + 160, m0, m1, m2, m3);
						Store32(array + 192, m4, m5, m6, m7);
					}
					else
					{
						Store32(array + 160, m4, m5, m6, m7);
					}
				}
				Load32(array + 0, m0, m1, m2, m3);
				Load32(array + 128, m4, m5, m6, m7);
				Merge3232();
				Store32(array + 0, m0, m1, m2, m3);
				Load32(array + 64, m0, m1, m2, m3);
				Merge3232();
				Store32(array + 64, m0, m1, m2, m3);
				Load32(array + 96, m0, m1, m2, m3);
				Merge3232();
				Store32(array + 96, m0, m1, m2, m3);
				Store32(array + 128, m4, m5, m6, m7);
				Load32(array + 32, m0, m1, m2, m3);
				Load32(array + 64, m4, m5, m6, m7);
				Merge3232();
				Store32(array + 32, m0, m1, m2, m3);
				Store32(array + 64, m4, m5, m6, m7);
			}
#else
			int i, j;
			__m256i m0, m1, m2, m3, m4, m5, m6, m7;
			for (i = (int)num; i > 32; i-=32)
			{
				Load32(array, m4, m5, m6, m7);
				for (j = 32; j < i; j+=32)
				{
					Load32(array + j, m0, m1, m2, m3);
					Merge3232();
					Store32(array + j - 32, m0, m1, m2, m3);
				}
				Store32(array + j - 32, m4, m5, m6, m7);
			}
#endif
		}
		else
		{
			T pivot;
			__m256i m0, m1, m2, m3, m4, m5, m6, m7;
			if (alignedSize < 2048 * 3)
			{
				__m256i index = _mm256_setr_epi32(0, 32, 32 * 2, 32 * 3, 32 * 4, 32 * 5, 32 * 6, 32 * 7);
				__m256i padding = _mm256_setr_epi32(PADDING_MAX, PADDING_MAX, PADDING_MAX, PADDING_MAX, PADDING_MAX, PADDING_MAX, PADDING_MAX, PADDING_MAX);

				int * ofsArray = (int*)alignedArray + 15;
				m0 = _mm256_i32gather_epi32((int*)ofsArray + 256 * 0, index, 4);
				m1 = alignedSize < 256 * 2 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 1, index, 4);
				m2 = alignedSize < 256 * 3 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 2, index, 4);
				m3 = alignedSize < 256 * 4 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 3, index, 4);
				m4 = alignedSize < 256 * 5 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 4, index, 4);
				m5 = alignedSize < 256 * 6 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 5, index, 4);
				m6 = alignedSize < 256 * 7 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 6, index, 4);
				m7 = alignedSize < 256 * 8 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 7, index, 4);
				SuperSort64Reg();
				if (alignedSize < 256 * 5)
				{
					if (alignedSize < 256 * 2)
					{
						pivot = m0.m256i_i32[4];
					}
					else if (alignedSize < 256 * 3)
					{
						pivot = m1.m256i_i32[0];
					}
					else if (alignedSize < 256 * 4)
					{
						pivot = m1.m256i_i32[4];
					}
					else
					{
						pivot = m2.m256i_i32[0];
					}
				}
				else
				{
					if (alignedSize < 256 * 6)
					{
						pivot = m2.m256i_i32[4];
					}
					else if (alignedSize < 256 * 7)
					{
						pivot = m3.m256i_i32[0];
					}
					else if (alignedSize < 256 * 8)
					{
						pivot = m3.m256i_i32[4];
					}
					else
					{
						pivot = m4.m256i_i32[0];
					}
				}
			}
			else
			{
				size_t i, j;
				__m256i index = _mm256_setr_epi32(0, 32, 32 * 2, 32 * 3, 32 * 4, 32 * 5, 32 * 6, 32 * 7);
				size_t idx = 0;
				for (i = 0; i + 2048 <= alignedSize; i += 2048)
				{
					int* ofsArray = (int*)alignedArray + i + 15;
					m0 = _mm256_i32gather_epi32(ofsArray + 256 * 0, index, 4);
					m1 = _mm256_i32gather_epi32(ofsArray + 256 * 1, index, 4);
					m2 = _mm256_i32gather_epi32(ofsArray + 256 * 2, index, 4);
					m3 = _mm256_i32gather_epi32(ofsArray + 256 * 3, index, 4);
					m4 = _mm256_i32gather_epi32(ofsArray + 256 * 4, index, 4);
					m5 = _mm256_i32gather_epi32(ofsArray + 256 * 5, index, 4);
					m6 = _mm256_i32gather_epi32(ofsArray + 256 * 6, index, 4);
					m7 = _mm256_i32gather_epi32(ofsArray + 256 * 7, index, 4);
					SuperSort64Reg();
					T center = m4.m256i_i32[0];
					for (j = 0; j < 64; j++)
					{
						if (ofsArray[j * 32] == center)
						{
							ofsArray[j * 32] = alignedArray[idx];
							alignedArray[idx] = center;
							if ((T*)ofsArray - 15 + j * 32 != alignedArray)
							{
								SuperSort64((T*)ofsArray - 15 + j * 32);
							}
							idx++;
							break;
						}
					}
				}
				SuperQuickSort(alignedArray, idx);

				pivot = alignedArray[idx / 2];
				if (idx % 32)
				{
					SuperSort64(alignedArray + (idx & ~31));
				}
			}
			// ピボット選択終了
			T* l;
			T* r;
			__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
			l = alignedArray;
			r = alignedArray + alignedSize - 32;
			Load32(l, m0, m1, m2, m3);
			Load32(r, m4, m5, m6, m7);
			while (1)
			{
				Merge3232();
				if (m3.m256i_i32[7] <= pivot)
				{
					Store32(l, m0, m1, m2, m3);
					l += 32;
					if (l == r)
					{
						Store32(r, m4, m5, m6, m7);
						break;
					}
					Load32(l, m0, m1, m2, m3);
				}
				if (m4.m256i_i32[0] >= pivot)
				{
					Store32(r, m4, m5, m6, m7);
					r -= 32;
					if (l == r)
					{
						Store32(l, m0, m1, m2, m3);
						break;
					}
					Load32(r, m4, m5, m6, m7);
				}
			}
			if (r != array)
			{
				SuperQuickSortRecAligned(array, (r + 32) - array);
			}
			if (l != array + num - 32)
			{
				SuperQuickSortRecAligned(l, array - l + num);
			}
		}
	}

	// 32要素毎にソートされたデータを受け取り、クイックソートを行う
	int SuperQuickSortRec(T* array, size_t num)
	{
		T* alignedArray = (T*)(((size_t)array) + 31 & ~31);
		size_t alignedSize = (array + num - alignedArray) & ~31;
		int leftFraction = (int)(alignedArray - array);
		int rightFraction = (int)(num - alignedSize - leftFraction);
		if (alignedSize < 128 && (leftFraction == 0 || rightFraction == 0))
		{
			// 左右の端数を含む128ワード未満のソート
			// 再帰の末尾で呼びたくないのでサイズだけ記録しておく。
			if (leftFraction)
			{
				return (int)num << 16;
			}
			return (int)num;
		}
		else
		{
			T pivot;
			__m256i m0, m1, m2, m3, m4, m5, m6, m7;
			if (alignedSize < 256)
			{
				T a, b, c, t;
				a = alignedArray[15];
				b = alignedArray[15 + 32];
				c = alignedArray[15 + 64];
				if (a > b)
				{
					t = a;
					a = b;
					b = t;
				}
				pivot = b < c ? b : a < c ? c : a;
			}
			else if (alignedSize < 2048 * 3)
			{
				__m256i index = _mm256_setr_epi32(0, 32, 32 * 2, 32 * 3, 32 * 4, 32 * 5, 32 * 6, 32 * 7);
				__m256i padding = _mm256_setr_epi32(PADDING_MAX, PADDING_MAX, PADDING_MAX, PADDING_MAX, PADDING_MAX, PADDING_MAX, PADDING_MAX, PADDING_MAX);

				int * ofsArray = (int*)alignedArray + 15;
				m0 = _mm256_i32gather_epi32(ofsArray + 256 * 0, index, 4);
				m1 = alignedSize < 256 * 2 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 1, index, 4);
				m2 = alignedSize < 256 * 3 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 2, index, 4);
				m3 = alignedSize < 256 * 4 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 3, index, 4);
				m4 = alignedSize < 256 * 5 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 4, index, 4);
				m5 = alignedSize < 256 * 6 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 5, index, 4);
				m6 = alignedSize < 256 * 7 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 6, index, 4);
				m7 = alignedSize < 256 * 8 ? padding : _mm256_i32gather_epi32(ofsArray + 256 * 7, index, 4);
				SuperSort64Reg();
				if (alignedSize < 256 * 5)
				{
					if (alignedSize < 256 * 2)
					{
						pivot = m0.m256i_i32[4];
					}
					else if (alignedSize < 256 * 3)
					{
						pivot = m1.m256i_i32[0];
					}
					else if (alignedSize < 256 * 4)
					{
						pivot = m1.m256i_i32[4];
					}
					else
					{
						pivot = m2.m256i_i32[0];
					}
				}
				else
				{
					if (alignedSize < 256 * 6)
					{
						pivot = m2.m256i_i32[4];
					}
					else if (alignedSize < 256 * 7)
					{
						pivot = m3.m256i_i32[0];
					}
					else if (alignedSize < 256 * 8)
					{
						pivot = m3.m256i_i32[4];
					}
					else
					{
						pivot = m4.m256i_i32[0];
					}
				}
			}
			else
			{
				size_t i, j;
				__m256i index = _mm256_setr_epi32(0, 32, 32 * 2, 32 * 3, 32 * 4, 32 * 5, 32 * 6, 32 * 7);
				size_t idx = 0;
				for (i = 0; i + 2048 <= alignedSize; i += 2048)
				{
					int* ofsArray = (int*)alignedArray + i + 15;
					m0 = _mm256_i32gather_epi32(ofsArray + 256 * 0, index, 4);
					m1 = _mm256_i32gather_epi32(ofsArray + 256 * 1, index, 4);
					m2 = _mm256_i32gather_epi32(ofsArray + 256 * 2, index, 4);
					m3 = _mm256_i32gather_epi32(ofsArray + 256 * 3, index, 4);
					m4 = _mm256_i32gather_epi32(ofsArray + 256 * 4, index, 4);
					m5 = _mm256_i32gather_epi32(ofsArray + 256 * 5, index, 4);
					m6 = _mm256_i32gather_epi32(ofsArray + 256 * 6, index, 4);
					m7 = _mm256_i32gather_epi32(ofsArray + 256 * 7, index, 4);
					SuperSort64Reg();
					T center = m4.m256i_i32[0];
					for (j = 0; j < 64; j++)
					{
						if (ofsArray[j * 32] == center)
						{
							ofsArray[j * 32] = alignedArray[idx];
							alignedArray[idx] = center;
							if ((T*)ofsArray - 15 + j * 32 != alignedArray)
							{
								SuperSort64((T*)ofsArray - 15 + j * 32);
							}
							idx++;
							break;
						}
					}
				}
				SuperQuickSort(alignedArray, idx);

				pivot = alignedArray[idx / 2];
				if (idx % 32)
				{
					SuperSort64(alignedArray + (idx & ~31));
				}
			}
			// ピボット選択終了
			T* l;
			T* r;
			__m256i maskflip8 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
			l = alignedArray;
			r = alignedArray + alignedSize - 32;
			Load32(l, m0, m1, m2, m3);
			Load32(r, m4, m5, m6, m7);

			bool fracL = leftFraction;
			bool fracR = rightFraction;
			while (1)
			{
				Merge3232();
				if (m3.m256i_i32[7] <= pivot)
				{
					Store32(l, m0, m1, m2, m3);
					if (fracL && l == alignedArray)
					{
						fracL = false;
						for (int i = 0; i < leftFraction; i++)
						{
							T tmp = array[i];
							array[i] = alignedArray[i];
							alignedArray[i] = tmp;
						}
						Load32(l, m0, m1, m2, m3);
						SuperSort64Reg();
					}
					else
					{
						l += 32;
						if (l == r)
						{
							Store32(r, m4, m5, m6, m7);
							break;
						}
						Load32(l, m0, m1, m2, m3);
					}
				}
				if (m4.m256i_i32[0] >= pivot)
				{
					Store32(r, m4, m5, m6, m7);
					if (fracR && r == alignedArray + alignedSize - 32)
					{
						fracR = false;
						for (int i = 0; i < rightFraction; i++)
						{
							T tmp = alignedArray[i + alignedSize - rightFraction];
							alignedArray[i + alignedSize - rightFraction] = alignedArray[i + alignedSize];
							alignedArray[i + alignedSize] = tmp;
						}
						Load32(r, m4, m5, m6, m7);
						SuperSort64Reg();
					}
					else
					{
						r -= 32;
						if (l == r)
						{
							Store32(l, m0, m1, m2, m3);
							break;
						}
						Load32(r, m4, m5, m6, m7);
					}
				}
			}
			assert(!(fracL || fracR));
			int lfrac = 0, rfrac = 0;
			if (rightFraction)
			{
				rfrac = SuperQuickSortRec(l, array - l + num);
			}
			if (leftFraction)
			{
				lfrac = SuperQuickSortRec(array, (r + 32) - array);
			}
			else
			{
				if (r != array)
				{
					SuperQuickSortRecAligned(array, (r + 32) - array);
				}
			}
			if (!rightFraction)
			{
				if (l != array + num - 32)
				{
					SuperQuickSortRecAligned(l, array - l + num);
				}
			}
			return lfrac + rfrac;
		}

		return 0;
	}

	// 128要素未満のソート
	void SuperSortSmall(T* array, size_t num)
	{
		bool isAligned = (((size_t)array) & 31) == 0;
		size_t alignedsize;
		T stackArray[135];
		T* buf = (T*)(((size_t)stackArray) + 31 & ~31);

		if (num < 64)
		{
			// 64要素未満は64要素にパディングして処理
			alignedsize = 64;
		}
		else
		{
			// 32要素アライメントに調整
			alignedsize = (num - 1 | 31) + 1;
		}
		size_t i;
		for (i = num; i < alignedsize; i++)
		{
			buf[i] = PADDING_MAX;
		}
		memcpy(buf, array, sizeof(T) * num);
		if (alignedsize == 64)
		{
			SuperSort64(buf);
		}
		else if (alignedsize == 96)
		{
			__m256i m0, m1, m2, m3, m4, m5, m6, m7;
			SuperSort64(buf);
			SuperSort64(buf + 32);
			Load32(buf, m0, m1, m2, m3);
			Load32(buf +32, m4, m5, m6, m7);
			Merge3232();
			Store32(buf, m0, m1, m2, m3);
			Store32(buf + 32, m4, m5, m6, m7);

		}
		else
		{
			__m256i m0, m1, m2, m3, m4, m5, m6, m7;
			SuperSort64(buf);
			SuperSort64(buf + 64);
			Load32(buf, m0, m1, m2, m3);
			Load32(buf + 64, m4, m5, m6, m7);
			Merge3232();
			Store32(buf, m0, m1, m2, m3);
			Store32(buf + 64, m4, m5, m6, m7);
			Load32(buf +32, m0, m1, m2, m3);
			Load32(buf + 96, m4, m5, m6, m7);
			Merge3232();
			Store32(buf + 96, m4, m5, m6, m7);
			Load32(buf + 64, m4, m5, m6, m7);
			Merge3232();
			Store32(buf + 32, m0, m1, m2, m3);
			Store32(buf + 64, m4, m5, m6, m7);
		}
		memcpy(array, buf, sizeof(T) * num);
	}
} // namespace

// SuperQuickSort本体
void SuperQuickSort(T* array, size_t num)
{
	if (((size_t)array) & 3)
	{
		// 4バイトアライメント違反
		abort();
	}
	if (num <= 128)
	{
		// 128要素未満の時は専用のルーチンを使用
		SuperSortSmall(array, num);
	}
	else
	{
		T* alignedArray = (T*)(((size_t)array) + 31 & ~31);
		size_t alignedSize = (array + num - alignedArray) & ~63;
		size_t i;
		int leftFraction = (int)(alignedArray - array);
		int rightFraction = (int)(num - alignedSize - leftFraction);
		int frac;
		for (i = 0; i * 64 < alignedSize; i++)
		{
			SuperSort3232(alignedArray + i * 64);
		}
		if (rightFraction >= 32)
		{
			SuperSort3232(alignedArray + alignedSize - 32);
			rightFraction -= 32;
		}
		if (leftFraction || rightFraction)
		{
			frac = SuperQuickSortRec(array, num);
		}
		else
		{
			SuperQuickSortRecAligned(array, num);
		}
		if (leftFraction)
		{
			int n = (frac >> 16);
			SuperSortSmall(array, n);
			//SuperQuickSortEnd(array, n);
		}
		if (rightFraction)
		{
			int n = (frac & 65535);
			SuperSortSmall(array + num - n, n);
			//SuperQuickSortEnd(array + (num - n), n);
		}
	}
}
#if 0
namespace {
	// アライメントされていない部分の処理。不要なので削除
	void SuperQuickSortEnd(T* array, size_t num)
	{
		T* alignedArray = (T*)(((size_t)array) + 31 & ~31);
		size_t alignedSize = (array + num - alignedArray) & ~31;
		int leftFraction = alignedArray - array;
		int rightFraction = num - alignedSize - leftFraction;
		assert(1);
		// 左右両方アライメントが狂っている場合、ソートに失敗する。
		assert(leftFraction && rightFraction);
		// アライメントされた要素が64要素無いとソート出来ないので、外側のメモリを利用する
		if (alignedSize == 32)
		{
			if (leftFraction)
			{
				alignedSize = 64;
			}
			else
			{
				alignedSize = 64;
				alignedArray -= 32;
			}
		}
		// 小さいときは別アルゴリズムでソートしたい
		int i, j;
		__m256i m0, m1, m2, m3, m4, m5, m6, m7;
		if (leftFraction)
		{
			Load32(alignedArray + alignedSize - 32, m0, m1, m2, m3);
			for (i = alignedSize - 32; i > 0; i -= 32)
			{
				Load32(alignedArray + i - 32, m4, m5, m6, m7);
				Merge3232();
				Store32(alignedArray + i, m4, m5, m6, m7);
			}
			for (i = 0; i < leftFraction; i++)
			{
				T t = array[i];
				array[i] = alignedArray[39 - i];
				alignedArray[39 - i] = t;
			}
			m4 = _mm256_load_si256((__m256i*)(alignedArray + 32));
			SuperSort64Reg();
			_mm256_store_si256((__m256i*)(alignedArray), m0);
			for (i = 0; i < leftFraction; i++)
			{
				T t = array[i];
				array[i] = alignedArray[i];
				alignedArray[i] = t;
			}
			m0 = _mm256_load_si256((__m256i*)(alignedArray));
			SuperSort64Reg();
			//BitonicMerge64();
			Store32(alignedArray, m0, m1, m2, m3);
			Store32(alignedArray + 32, m4, m5, m6, m7);
		}
		else if (rightFraction)
		{
			Load32(alignedArray, m0, m1, m2, m3);
			Load32(alignedArray + 32, m4, m5, m6, m7);
			Merge3232();
			Store32(alignedArray, m0, m1, m2, m3);
		}
		if (rightFraction)
		{
			for (i = 64; i < alignedSize; i += 32)
			{
				Load32(alignedArray + i, m0, m1, m2, m3);
				Merge3232();
				Store32(alignedArray + i - 32, m0, m1, m2, m3);
			}
			for (i = 0; i < rightFraction; i++)
			{
				T t = alignedArray[alignedSize - 33 - i];
				alignedArray[alignedSize - 33 - i] = alignedArray[alignedSize + i];
				alignedArray[alignedSize + i] = t;
			}
			Load32(alignedArray + alignedSize - 64, m0, m1, m2, m3);
			SuperSort64Reg();
			Store32(alignedArray + alignedSize - 32, m4, m5, m6, m7);
			for (i = 0; i < rightFraction; i++)
			{
				T t = alignedArray[alignedSize - rightFraction + i];
				alignedArray[alignedSize - rightFraction + i] = alignedArray[alignedSize + i];
				alignedArray[alignedSize + i] = t;
			}
			Load32(alignedArray + alignedSize - 32, m4, m5, m6, m7);
			SuperSort64Reg();
			//BitonicMerge64();
			Store32(alignedArray + alignedSize - 32, m4, m5, m6, m7);
			Store32(alignedArray + alignedSize - 64, m0, m1, m2, m3);
		}
		for (i = alignedSize; i > 32; i -= 32)
		{
			Load32(alignedArray, m4, m5, m6, m7);
			for (j = 32; j < i; j += 32)
			{
				Load32(alignedArray + j, m0, m1, m2, m3);
				Merge3232();
				Store32(alignedArray + j - 32, m0, m1, m2, m3);
			}
			Store32(alignedArray + j - 32, m4, m5, m6, m7);
		}
	}
}// namespace
#endif
