
1. map
//创建
unordered_map<int, int> mp

//插入
//不存在k时自动创建
mp['k'] = v 

//查找
//判断某个key是否存在，所以存k-v的时候注意顺序
  if(mp.find(key) != mp.end() ) 