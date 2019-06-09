
1. map
//创建
unordered_map<int, int> mp

map对应的迭代器，也是返回对应的k-v pair, it->second就是v
unordered_map<int, int>::iterator it

//插入
//不存在k时自动创建
mp['k'] = v 

//查找
判断某个key是否存在，所以存k-v的时候注意顺序
返回迭代器
  if(mp.find(key) != mp.end() ) 


2. vector
//查找
不同于map，没有find成员方法，是依靠algorithm来实现find
vector<int>::iterator it = find(vec.begin(), vec.end(), 6);