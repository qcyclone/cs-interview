
## map
1. 创建
>* unordered_map<int, int> mp
>* map对应的迭代器，也是返回对应的k-v pair, it->second就是v
`unordered_map<int, int>::iterator it`
2. 插入
//不存在k时自动创建
mp['k'] = v 
3. 查找
判断某个key是否存在，所以存k-v的时候注意顺序
返回迭代器
  if(mp.find(key) != mp.end() ) 
4. 删除
>* `size_type erase( const key_type& key );`返回删除的个数


## vector
1. 查找
不同于map，vector没有find成员方法，是依靠algorithm来实现find
`vector<int>::iterator it = find(vec.begin(), vec.end(), 6);`

2. 声明string数组方式
>* `vector<string> ansRow(numRows);`

3. 声明二维数组
`vector< vector<int> > vv`

4. 截取vector
`vector<int> a(sequence.begin(), sequence.begin() + i );`
4. **如果不声明大小，不能用下标访问**

5. inserts value before pos
>*  `insert(v.begin(), tmp) `
实现头部插入，与push_back相反。
>* 或者可以`reverse(v.begin(), v.end())`
6. 反转 
reverse(v.begin(), v.end())
7. 截取
>* `vector<int> a(v.begin(), v.begin() + 4)`  
//截取前4个数

## string
1. 返回长度
>* string s; 
>* s.size(); s.length()
2. 截取子串
>* `s.substr( pos, count);`

## queue
1. 用front()获取头部，栈是top