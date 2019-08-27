
## map
1. 创建
>* unordered_map<int, int> mp
>* map对应的迭代器，也是返回对应的k-v pair, it->second就是v
`unordered_map<int, int>::iterator it`
2. 插入
* 每个节点存储了k-v
//不存在k时自动创建
mp['k'] = v 
3. 查找， 
* 是根据key进行排序的
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
>* `v.resize(n, 4)` //通过resize可根据输入声明全局数组，并初始化
>* `v.reserve(10)` // 只能传入1个参数，指定capacity
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
8. 释放内存
`vector<int>().swap(v)`
* swap()是交换函数，使vector离开其自身的作用域，从而强制释放vector所占的内存空间

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
2. 
## C++动态声明数组
1.  `int* a = new int[length]`  声明时不能用 [], 要用指针形式
    `delete[] a`
    
## 迭代器失效 
1. 慎用erase()， 返回删除元素下一个位置 
  * 关联容器： list、set、map遍历删除元素时
  * 序列式容器： vector、deque遍历删除元素也可使用这种方式
2. 对于序列式容器(如vector,deque)，删除当前的iterator会使后面所有元素的iterator都失效。这是因为vetor,deque使用了连续分配的内存，删除一个元素导致后面所有的元素会向前移动一个位置。还好erase方法可以返回下一个有效的iterator。
`std::list< int> List;`
`std::list< int>::iterator itList;`
`for( itList = List.begin(); itList != List.end(); )`
`{`
`      if( WillDelete( *itList) )`
`            itList = List.erase( itList);`
`       else`
`            itList++;`
`}`
