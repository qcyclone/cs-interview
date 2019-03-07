# cs-interview

## C++
### C++ STL 
c++中哈希表对应的容器  unordered_map<int, int> mp  查找复杂度 o(1)

map<int, int> mp  底层红黑树实现 插入查找删除复杂度 logn， 插入n个元素就是nlogn

if(mp.find(key) != mp.end() )

## 计网
客户端和服务器都可主动发起挥手动作，对应socket编程的close()，任何一方都可发起

连接动作connect()，对应三次握手，accept()从队列中取出连接

为什么三次握手，A还要发送一次确认？

* 防止已经失效的连接请求送到了B, A没有建立连接，B缺认为已建立连接，等待数据到来，浪费资源。
* 同步双方序列号，确认号
