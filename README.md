# cs-interview

# C++
### C++ STL 
1. unordered_map<int, int>mp c++中哈希表对应的容器    查找复杂度 o(1)

2. map<int, int> mp  底层红黑树实现 插入查找删除复杂度 logn， 插入n个元素就是nlogn

if(mp.find(key) != mp.end() ) 判断是否存在

---
1. 引用，在声明时要初始化，sizeof引用得到的是所指向的变量(对象)的大小
2. 指针，sizeof是指针本身的大小。指针和引用的自增(++)运算意义不一样；
---
1. 指针常量：int* const p,指针所保存的地址不可改变，但指向的变量的值能变
2. 常指针： const int* p, 指向常量的指针，指向的地址可以变，指向的是个常量，内容不可变

---
### define 和 const区别
1. define在预处理阶段被替换，不占内存
2. const在编译运行阶段被处理，占用内存

### extern作用
1. 置于变量或函数前，表示定义在别的文件中，提示编译器遇到此变量或函数在其他文件中寻找定义。

### volatile
1. 强制cpu每次从内存中访问变量，因为寄存器有可能（被别的程序）改变。
2. 内存中的值发生变化，寄存器还没变

### 封装、继承、多态
* 接口的多种不同实现方式为多态
1. 使用虚函数才会调用子类同函数，运行时多态（动态多态，通过虚函数和继承关系，在运行时确定），使用上层操作来执行下层具体操作
2. 父类指针指向子类对象（父类指针调用时，通过查找子类对象的虚函数表，找到指向哪个函数）
3. 编译时多态（静态多态）：重载函数，模板技术

# 计网
客户端和服务器都可主动发起挥手动作，对应socket编程的close()，任何一方都可发起

连接动作对应connect()，对应三次握手，accept()从队列中取出连接

---
为什么三次握手而不是两次，A还要发送一次确认？
1. 防止已经失效的连接请求送到了B, A没有建立连接，B缺认为已建立连接，等待数据到来，浪费资源。
2. 同步双方序列号，确认号

---
为什么要这有TIME_WAIT，为什么不直接给转成CLOSED状态?

1. TIME_WAIT确保有足够的时间让对端收到了ACK，如果被动关闭的那方没有收到Ack，就会触发被动端重发Fin，一来一去正好2个MSL，

2. 有足够的时间让这个连接不会跟后面的连接混在一起（你要知道，有些自做主张的路由器会缓存IP数据包，如果连接被重用了，那么这些延迟收到的包就有可能会跟新连接混在一起）
---
* TCP面向字节流，面向连接，都是同一台主机发送的，一直发送没关系
* UDP面向报文，无连接，应用层交给UDP多长报文，照发，不拆分也不合并。如果读取超过一个报文，多个发送方的报文可能混在一起

---
UDP没有流量控制和拥塞控制，所以在网络拥塞时不会使源主机发送速率降低（对实时通信很有用，比如QQ电话，视频会议等）

TCP流量控制: 针对单个tcp连接，使发送方速度小于接收方

* RTT(Round Trip Time)：一个连接的往返时间，即数据发送时刻到接收到确认的时刻的差值； 
* RTO(Retransmission Time Out)：重传超时时间，即从数据发送时刻算起，超过这个时间便执行重传。 

* 为了避免流量控制引发的死锁，TCP使用了持续计时器。每当发送者收到一个零窗口的应答后就启动该计时器。时间一到便主动发送报文询问接收者的窗口大小。若接收者仍然返回零窗口，则重置该计时器继续等待；若窗口不为0，则表示应答报文丢失了，此时重置发送窗口后开始发送，这样就避免了死锁的产生。

1. 滑动窗口(连续ARQ协议)：表示接收方有多大缓存可接收数据。发送方窗口内的序列号代表了已经被发送，但是还没有被确认的帧，或者可以被发送的帧。发送方接收的ACK中有接收窗口的大小，以控制发送方的速度。

TCP的拥塞控制: 针对整个网络，使每个tcp连接都能高速
* MSS，maximum segment size：单个TCP包所含最大字节数
* CWND（congestion window）:拥塞窗口，发送方的 Window 大小，表示当前可以发送多少个tcp包。取决于网络拥塞情况，动态变化
* RWND（receiver window):接收方的窗口大小：表示当前还能接收多少个TCP包
* 发送窗口的上限值 = Min{rwnd, cwnd}。rwnd会发给发送方

1. 慢开始： 双方在建立连接时，先慢慢发包，1，2，4，8。（慢慢拧开水龙头）
2. 拥塞避免：当cwnd超过ssthresh,进入拥塞避免阶段，cwnd加法线性增大
3. 快重传：收到3个连续重复确认立即重传，不必等待RTO（重传计时器）到时
4. 快恢复：是快重传的后续处理。快重传后，阈值减半，并且cwnd=ssthresh/2，乘法减小，（cwnd等于乘法减小后的阈值）而不是从1开始（慢开始）

---
udp如何实现可靠性传输？

1.  传输层无法保证数据的可靠传输，只能通过应用层来实现.
2. 通过包的分片、确认、重发
# 操作系统
### cache（类比于停车位）
* 直接映射，每组只有一个cacheline,只用大cache。内存中某块只能映射到特定组
* 组相联映射，s组，每组e个cacheline
* 全相联映射，适用小容量cache，只有一个组

CPU的程序计数器和各种寄存器都只有一份，当进程切换时，就需要保存进程上下文

---
### Linux IO模式
1. 同步 
* 阻塞：等待
* 非阻塞：轮询查看

2. 异步
* 非阻塞：全部交给cpu/内核处理，只等待完成信号
---

* 同步/异步 针对程序和内核的交互
* 阻塞/非阻塞 关心单个进程的执行状态
---
io多路复用

1. select

* 效率O(n)，有事件发生，不知道哪几个流，需要轮询找出数据
* 大量数据拷贝，需要用户空间向内核空间拷贝fd数组，内核变量fd数组
* 有最大连接数限制：1024

2. poll

* 没有最大连接限制，采用链表
* 其他缺点和select一样

3. epoll

* 复杂度O(1), 会把哪几个流发生了什么io时间通知，是事件驱动的
* 使用事件的就绪通知方式，通过epoll_ctl注册fd，一旦fd就绪，内核就用callback回调机制激活fd，epoll_wait收到通知

* 为什么会有ET？

1. 一旦有大量不需要读写的fd，每次epoll_wait都会返回，浪费资源
2. 比LT效率高，系统不会有大量不关心的就绪fd。

* epoll优点？
1. 没有最大并发限制
2. O(1),不是轮询方式。只有活跃可用的fd才会调用callback。只会管活跃的连接，和总连接数无关。即时上百万个总连接，同时活跃的连接数很少。所以效率和活跃的连接数有关。

当连接数少且都十分活跃是，select和poll可能会更高。

# 数据库
### B+树
1个索引节点就是1个磁盘页

m阶B+树，xx个关键字，xx个指针

1. 非叶子节点不保存关键字记录的指针，只进行数据索引，可保存更多的索引信息
（非叶节点不包含指向 数据记录 存放地址的指针，没有data域）
2. 叶子节点保存了父节点所有关键字记录的指针
3. 叶子节点间有指针相连，从小到大，方便进行范围查找

### 红黑树
1. 根节点是黑色的
2. 从跟到叶子的路径上黑色节点数相同
3. 保证最长路径不大于最短路径的2倍

为什么STL用红黑树？
1. 最坏情况下，AVL树有最多O(logN)次旋转，而红黑树最多三次

# Linux
### 文件
1. 一个文件占用1个inode，记录文件属性，记录内容所在的block编号，一个文件会占用多个block
2. 磁盘锁片：文件所在block过于分散，block只能被1个文件使用
3. 文件名记录在目录中，目录的x属性可修改文件名

### 硬链接
1. 有相同的inode仅文件名不同，文件由相同的inode和data block
2. 不能对目录进行创建
3. 会增加inode中的链接数（表示有多少个文件名指向该inode）

### mmap
https://www.cnblogs.com/huxiao-tee/p/4660352.html#_label1
* 位于堆和栈中间区域，将文件或其他对象映射到进程的地址空间。
* Memory-map. 可翻译成内存映射, 这是一种文件I/O操作, 它将文件(linux世界中, 设备也是文件)映射到内存中, 用户可以像操作内存一样操作文件.
* 进程就可以采用指针的方式读写操作这一段内存（，而系统会自动回写脏页面到对应的文件磁盘上），完成了对文件的操作而不必再调用read,write等系统调用函数。相反，内核空间对这段区域的修改也直接反映用户空间，从而可以实现不同进程间的文件共享。
1. **常规文件操作为了提高读写效率和保护磁盘，使用了页缓存机制**。 使读文件时需要先将文件页从磁盘拷贝到页缓存中，*由于页缓存处在内核空间*，不能被用户进程直接寻址，还需要将页缓存中数据页再次拷贝到内存对应的用户空间中。这样，通过了2数据拷贝过程，才能完成进程对文件内容的获取。写操作也是一样，待写入的buffer在内核空间不能直接访问，必须要先拷贝至内核空间对应的主存，再写回磁盘中（延迟写回），需要两次数据拷贝。
2. 使用mmap时，创建 新的虚拟内存区域(映射区域) 和 建立文件磁盘地址和虚拟内存区域映射 这两步，没有任何文件拷贝操作（真正的文件读取是当进程发起读或写操作时）。而访问数据时发现内存中并无数据而发起的缺页异常过程，可以通过已经建立好的映射关系，只用1数据拷贝，就从磁盘中将数据传入内存的用户空间中，供进程使用。

#### mmap总结
1. 对文件的读取操作跨过了 页缓存，减少了数据的拷贝次数，用**内存读写取代I/O读写**，提高了文件读取效率。
2. 实现了 **用户空间和内核空间** 的高效交互方式。两空间的各自修改操作可以直接反映在映射的区域内，从而被对方空间及时捕捉，提高交互效率。
3. 提供进程间共享内存及相互通信的方式。不管是父子进程还是无亲缘关系的进程，都可以将自身用户空间映射到同一个文件或匿名映射到同一片区域。从而通过各自对映射区域的改动，达到进程间通信和进程间共享的目的。
4. 可用于实现高效的大规模数据传输。内存空间不足，是制约大数据操作的一个方面，往往借助硬盘空间协助操作，补充内存的不足。但是进一步会造成大量的文件I/O操作，极大影响效率。这个问题可以通过mmap映射很好的解决。换句话说，但凡是需要用磁盘空间代替内存的时候，mmap都可以发挥其功效。

* page cache：页缓存是面向文件，面向内存的。通俗来说，它位于内存和文件之间缓冲区，文件IO操作实际上只和page cache交互。
* 如果页缓存命中，那么直接返回文件内容；
* 如果页缓存缺失，那么产生一个页缺失异常，创建一个页缓存页，同时通过inode找到文件该页的磁盘地址，读取相应的页填充该缓存页；重新进行第6步查找页缓存；
* 一个页缓存中的页如果被修改，那么会被标记成脏页。

####
mmap与文件操作的区别
linux还有一套文件操作的系统调用, 即open read write, mmap与他们区别在哪儿呢?
* 最大的区别就在于, 普通文件读写操作, 需要借助处于内核空间的页缓存作为中转, 将数据载入主存, 
而mmap却是逻辑映射到物理文件, 直接载入数据到主存.
前者总是向页缓存索取数据, 内核通过查找页缓存来判断能否立即返回数据, 在未缓存的情况下, 先从物理磁盘拷贝数据到页缓存, 再从页缓存拷贝数据到用户空间中, 写回时也是同样, 需要先将用户空间的文件缓存拷贝到内核空间, 再由内核写回到磁盘, 都需要经过两次拷贝.
后者则是在发现需要载入数据时, 通过事先建立的页表映射关系, 直接由内核将数据拷贝到用户空间, 一次拷贝.

mmap缺点？

1. 如果你要处理的文件大到很难为之分配一个连续的虚拟地址空间
2. 地址的偏移量不是物理页大小(page_size)的整数倍(此时你需要自己解决偏移量问题)

* MAP_SHARED: 在映射内容上发生变更，对所有共享同一个映射的其他进程都可见，对文件映射来讲，变更将会发生在底层的文件上。
* MAP_PRIVATE: 在映射内容上发生的变更，对其他进程不可见。内核使用了写时复制（copy-on-write）技术完成了这个任务。这意味着，每当一个进程试图修改一个分页的内容时，内核首先会为该进程创建一个新分页，

---
写时复制（copy-on-write）是一种可以推迟甚至避免复制数据的技术。内核此时并不是复制整个进程空间，而是让父进程和子进程共享同一个副本。只有在需要写入的时候，数据才会被复制，从而使父进程、子进程拥有各自的副本。也就是说，资源的复制只有在需要写入的时候才进行，**在此之前以只读方式共享**。这种优化可以避免复制大量根本就不会使用的数据。