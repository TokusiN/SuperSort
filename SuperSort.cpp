// ============================================================
//
//    SuperSort.cpp
//
// ============================================================
#include "SuperSort.h"
#include <immintrin.h>

void* AlignedMalloc(size_t size)
{
	void* ptr = _mm_malloc(size, 256);
	return ptr;
}

void AlignedFree(void* ptr)
{
	_mm_free(ptr);
}
