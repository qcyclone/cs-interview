
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
>* `vector<int> tmp(10, 0) ` 声明大小并初始化为0

3. 声明二维数组
`vector< vector<int> > v(m, vector<int>(n))` 


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

## set
1. ？？大根堆 `multiset<int, greater<int>> s;`
2.      `s.erase(*iterMax); s.insert(*it);`

## priority_queue 可看成是大根堆，内部基于堆实现
0. 默认 队首 元素是最大的
1. priority_queue <int,vector<int>,greater<int> > q; //头部元素最小
2. 除此可用make_heap(), 来在vector基础上创建堆，这样建堆的复杂度是O(N)

## list 双向链表
1. list::splice实现list拼接的功能。将源list的内容部分或全部元素删除，拼插入到目的list。
`l.splice(l.begin(), l, it->second())` 将it插入到头部

## C++动态声明数组
1.  `int* a = new int[length]`  声明时不能用 [], 要用指针形式
    `delete[] a`
    
