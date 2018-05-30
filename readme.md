# SuperSort
std::sortの7倍速で動作する外部ソートです。Haswell以降のCPUで動作します。  
8ワードアライメントされた32要素の倍数のデータの場合元の配列と同じサイズ、  
そうでない場合は元の配列の2倍のサイズのワーキングメモリを必要とします。  

# SuperQuickSort
std::sortの5倍速で動作する内部ソートです。Haswell以降のCPUで動作します。  
4バイトアライメントされていないデータの場合abortします。

SuperSort, SuperQuickSort by Toshihiro Shirakawa is licensed under the Apache License, Version2.0
