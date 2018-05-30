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
