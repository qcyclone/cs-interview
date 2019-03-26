# cs-interview
 
# C++
C和C++区别？
1. C面向过程，没有函数重载
2. C++面向对象，有封装、继承、多态等特性，有函数重载（函数名相同，参数不同）
---
### C++ 四种 强制类型 转换的区别?
* C风格是TYPE b = (TYPE)a
* 去const属性用const_cast。
* 基本类型转换用static_cast。
* 多态类之间的类型转换用daynamic_cast。
* 不同类型的指针类型转换用reinterpret_cast。

1. static_cast<T>(expression);
    * 基本数据类型转换；基类和子类之间转换，其中子类指针转换成父类指针是安全的

2. dynamic_cast<T>(expression);
    * 首先检查能否成功转换，如能成功转换则转换。如果转换失败, 对于指针返回nullptr，对于引用会抛出异常。bad_cast
    * 只能用于存在 虚函数的父子关系 的强制类型转换，安全的基类子类转换
3. reinterpret_cast<T>(expression);
    * 为数据的二进制形式重新解释，但是不改变其值。只是简单的从一个指针到别的指针的值的二进制拷贝。
4. const_cast<T>(expression);
    * 主要用来去const属性，当然也可以加上const属性。主要是用前者，后者很少用。
    * 对于未定义const版本的成员函数，通常使用const_cast来去除 const引用对象 的const，完成函数调用。
    * 常量指针（引用）被转化成非常量的指针（引用），并且仍然指向原来的对象
    * 另外一种使用方式，结合static_cast，可以在 非const版本的成员函数 内添加const，调用完const版本的成员函数后，再使用const_cast去除const限定。
    * 只能改变运算对象的底层const
---
### 隐式类型转换，explicit
1. 对于内置类型，低精度 变量 给高精度变量 赋值会发生隐式类型转换（小->大）
    * 相对应的是 显式类型转换（强制类型转换），高精度->低精度，会有精度的损失
2. 可以用 单个形参来调用 的构造函数定义了从 形参类型 到 该类类型 的一个隐式转换。
编译器会自动调用其构造函数生成临时对象。
3. explicit关键字只能用于类 的 构造函数 声明上。防止隐式转换
    通常将可以用一个实参进行调用的构造函数都声明为explicit。
---
### RTTI？
* Runtime Type Information，运行时类型信息，它提供了运行时确定对象类型的方法。
c++通过下面两个操作符提供RTTI。
1. typeid：返回指针或引用所指对象的实际类型。
2. dynamic_cast：将 基类类型 的指针或引用安全的转换为 派生类型 的指针或引用。
* 对于带虚函数的类，在运行时执行RTTI操作符，返回动态类型信息；
* 对于其他类型，在编译时执行RTTI，返回静态类型信息。

---
### 引用和指针的区别？ int &a = b;
1. 引用是别名，不是实体类型，不为引用单独分配内存空间。必须初始化，不能更换目标
    * 使用：引用型参数 
2. 指针是变量，存储的是一个地址。指针可以有多级，可以有const指针。
    * 自增运算意义不一样
---
### memcpy和strcpy区别
都是C库函数
1. strcpy提供了字符串的复制，只能复制字符串。不需要指定长度，'\0'
    * strcpy效率稍高，因为memcpy有count变量的变化，还有类型转换。
2. memcpy: 一般的内存复制。
---
### const成员函数
1. const成员函数只能被 const对象 调用。
2. const成员函数不能修改任何 成员数据 的值。
---
### C++里是怎么定义常量的？常量存放在内存的哪个位置？
1. 常量在C++里的定义就是一个top-level const加上对象类型，常量定义必须初始化。
2. 对于局部对象，常量存放在栈区，对于全局对象，常量存放在全局/静态存储区。对于字面值常量，常量存放在常量存储区。
---
申请内存
* 所申请的内存是由多个 内存块 构成的链表
    * 有meta区，数据区

malloc free 实现
1. malloc, 可用空间是分散的一块块的，用 ***空闲链表*** 标识。
    * 多分配4个字节，存储这个块有多大，释放的时候就知道了大小；指向下一块的指针
    * malloc分配一块连续的内存。寻找合适的block，first fit；best fit；next fit
    * 没有足够大的空间，相邻的合并，还没有，通过sbrk向内核申请堆
2. free，把内存块标记未被使用。is_avilible=1
---
构造函数 析构函数 虚函数
1. 构造函数不能是虚函数，构造函数调用之前找不到虚函数表。
    * 因为派生类不能继承基类的构造函数，将构造函数声明为虚函数没有意义
    * 从使用上来说，虚函数通过 基类指针 来调用派生类的成员，则在调用之前，对象必须存在。而构造函数调用之前没有对象。
2. 析构函数可以是虚函数，甚至是纯虚函数：是为了避免内存泄露
    * 不把 基类的析构函数 设置为虚函数，则在删除对象时，如果直接删除基类指针，系统就只能调用基类 析构函数 ，而不会调用派生类构造函数。这就会导致内存泄露。
为什么不把析构函数默认设置为虚函数？
* 原因是虚函数表的开销以及和C语言的类型的兼容性，虚函数表指针有额外的内存开销
3. 纯虚析构函数
想把Base做出抽象类（虚基类），不能直接构造对象；需要在其中定义一个纯虚函数。如果其中没有其他合适的函数，可以把析构函数定义为纯虚。

***注意：如果基类的析构函数设置为虚函数，所有派生类也默认为虚析构函数，即使没有带关键字Virtual。如果父类函数func()为虚函数，则子类中，函数func()是否加virtual关键字，都将是虚函数。为了提高程序的可读性，建议后代中虚函数都加上virtual关键字。***

3. 把所有的类的析构函数都设置为虚函数好吗？(虚函数的缺点)
* 运行时确定根据不同的对象调用不同的虚函数，要存储额外的信息用于查找。
* 系统为每一个对象存储了一个 ***虚函数表指针***，指向这个类的虚函数表。
* 使用虚函数后的类对象要比占的空间多，而且在查找具体使用哪一个虚函数时，还会有时间代价。即当一个类不打算作为基类时，不用将其中的函数设置为虚函数。

---
### 右值引用
* 左值：非临时对象
* 右值：指临时的对象，只在当前的语句中有效
* 右值引用解决了移动语义问题
---

### 浅拷贝 深拷贝
* 对于含有指针成员的类，直接拷贝可能会出现两个对象的 指针成员 指向同一个数据区。这时候一般先new个内存，然后复制内容 
* 对于堆上的内存需要深拷贝
---
 内存对齐？和机器字长有关
 * 32位机器最小存储单位是4字节
---
C++空类大小：1
*   类中初始化列表的顺序：与变量的声明顺序一样。一般变量只能在构造函数或初始化列表中初始化，否则为随机值

---
### C++ STL 
1. unordered_map<int, int>mp c++中哈希表对应的容器    查找复杂度 o(1)

2. map<int, int> mp  底层红黑树实现 插入查找删除复杂度 logn， 插入n个元素就是nlogn，有序的

if(mp.find(key) != mp.end() ) 判断是否存在

---
1. 引用，在声明时要初始化，sizeof引用得到的是所指向的变量(对象)的大小
2. 指针，sizeof是指针本身的大小。指针和引用的自增(++)运算意义不一样；

---
1. 指针常量：int* const p,指针所保存的地址不可改变，但指向的变量的值能变
2. 常量指针： const int *p, 指向常量的指针，指向的地址可以变，指向的是个常量，内容不可变
            int const *p;

---
### define 和 const区别
1. define在预处理阶段被替换，不占内存
2. const在编译运行阶段被处理，占用内存

---
### new 和malloc区别
1. malloc是C语言的库函数, 指定分配内存空间的大小，返回的是void*指针(也可建立对象，不会构造函数）
2. new是C++的关键字(运算符)，new建立对象，自动调用构造函数，析构函数。

* new底层先调用malloc申请一块适合的内存空间，然后使用该类型的构造函数进行函数的构造生成。
---
### static作用
1. C++的修饰符，控制变量的存储方式和可见性（生命周期和作用域）
    * 非静态全局变量的作用域是整个源程序，多个源文件中都有效
    * 静态全局变量：只在当前文件中有效
2. 一个数据为 整个类 服务而非某个对象
3. static函数：在内存中只有一份，普通函数在每个被调用中维持一份拷贝
    * 作用域不同，只被本源文件调用
---
### 迭代器失效
vector<int>::iterator it;
1. 关联式容器（map，set，multimap，multiset）删除当前的iterator，仅仅会导致当前iterator失效。这是因为对于这些关联式容器，底层使用红黑树来实现，插入、删除一个节点不会对其他节点产生影响。
2. 对于序列式容器（特别是支持随机存取的容器：vector、deque），删除当前iterator会使后面所有元素的iterator都失效，因为这些支持随机存取的容器使用连续分配的内存，删除一个元素会导致后面的所有元素都往前移动一个位置。

---
### 智能指针
定义一个类来封装资源的分配和释放，构造析构由编译器自动调用。
* auto_ptr问题：调用拷贝构造和赋值运算符重载函数时，将一块空间的权限完全交给别人。
1. shared_ptr，是引用计数的智能指针。可以有多个指针指向同一块内存，底层用引用计数的方式实现，即每增加一个指针，该内存的引用计数+1，减少一个指针，内存引用计数-1，当为0的时候，内存就会被析构函数销毁。
2. unique_ptr，对象里存放的内容在整个程序中只能出现一次
3. weak_ptr，指向一个shared_ptr管理的对象。不控制所指向对象的生存期。引用计数会带来循环引用产生无法删除的情况，导致内存泄漏。可使用weak_ptr，弱引用只引用但不增加或删减计数。如果一块内存被shared_ptr和weak_ptr同时引用，当所有的shared_ptr析构了之后，不管有没有weak_ptr引用该内存，内存会被释放。 
---
悬垂指针，野指针
1. 悬垂指针，指向的内存已经被释放了，但是指针还存在
    * 释放完空间后就是悬垂指针，需要将其再次指向NULL
2. 野指针，是指随机指向一块内存的指针，
    * 例如没有初始化;释放后没有置空;超过变量范围

---
### extern作用
1. 置于变量或函数前，表示定义在别的文件中，提示编译器遇到此变量或函数在其他文件中寻找定义。
extern “C” ？
* C++调用C函数需要extern C，因为C语言没有函数重载。
### volatile
1. 强制cpu每次从内存中访问变量，因为寄存器有可能（被别的程序）改变。
2. 内存中的值发生变化，寄存器还没变

### 封装、继承、多态
* 接口的多种不同实现方式为多态
1. 使用虚函数才会调用子类同函数，运行时多态（动态多态，通过虚函数和继承关系，在运行时确定），使用上层操作来执行下层具体操作
2. 父类指针指向子类对象（父类指针调用时，通过查找子类对象的虚函数表，找到指向哪个虚函数）
3. 编译时多态（静态多态）：重载函数，模板技术
如何实现多态？
* 子类若重写父类虚函数，虚函数表中，该函数的地址会被替换。
* 对于存在虚函数的类的对象，在VS中，对象的对象模型的头部存放指向虚函数表的指针，通过该机制实现多态。

# 计网
客户端和服务器都可主动发起挥手动作，对应socket编程的close()，任何一方都可发起

连接动作对应connect()，对应三次握手，accept()从队列中取出连接

---
为什么三次握手而不是两次，A还要发送一次确认？
1. 防止已经失效的连接请求送到了B, A没有建立连接，B缺认为已建立连接，等待数据到来，浪费资源。
1. 防止失效的连接请求到达服务器，让服务器错误打开连接。
2. 服务器端收到客户端的连接确认，同步双方序列号，确认号

---
为什么要这有TIME_WAIT，为什么不直接给转成CLOSED状态?

1. TIME_WAIT确保有足够的时间让对端收到了ACK，如果被动关闭的那方没有收到Ack，就会触发被动端重发Fin，一来一去正好2个MSL，
2. 有足够的时间让这个连接不会跟后面的连接混在一起，端口重用（你要知道，有些自做主张的路由器会缓存IP数据包，如果连接被重用了，那么这些延迟收到的包就有可能会跟新连接混在一起）
---
* TCP面向字节流，面向连接，都是同一台主机发送的，一直发送没关系
* UDP面向报文，无连接，应用层交给UDP多长报文，照发，不拆分也不合并。如果读取超过一个报文，多个发送方的报文可能混在一起

---
UDP没有流量控制和拥塞控制，所以在网络拥塞时不会使源主机发送速率降低（对实时通信很有用，比如QQ电话，视频会议等）

TCP流量控制: 针对单个tcp连接，使发送方速度小于接收方

* RTT(Round Trip Time)：一个连接的往返时间，即数据发送时刻到接收到确认的时刻的差值； 
* RTO(Retransmission Time Out)：重传超时时间，即从数据发送时刻算起，超过这个时间便执行重传。 

* 为了避免流量控制引发的死锁，TCP使用了持续计时器。每当发送者收到一个零窗口的应答后就启动该计时器。时间一到便主动发送报文询问接收者的窗口大小。若接收者仍然返回零窗口，则重置该计时器继续等待；若窗口不为0，则表示应答报文丢失了，此时重置发送窗口后开始发送，这样就避免了死锁的产生。

1. 滑动窗口(连续ARQ协议)：表示接收方有多大缓存可接收数据。发送方窗口内的序列号代表了已经被发送，但是还没有被确认的帧，或者可以被发送的帧。发送方接收的ACK中有 接收窗口 的大小，以控制发送方的速度。

### TCP的拥塞控制: 针对整个网络，使每个tcp连接都能高速
* MSS，maximum segment size：单个TCP包所含最大字节数
* CWND（congestion window）:拥塞窗口，发送方的 Window 大小，表示当前可以发送多少个tcp包。取决于网络拥塞情况，动态变化
* RWND（receiver window):接收方的窗口大小：表示当前还能接收多少个TCP包
* 发送窗口的上限值 = Min{rwnd, cwnd}。rwnd会发给发送方

1. 慢开始： 双方在建立连接时，先慢慢发包，1，2，4，8。（慢慢拧开水龙头）
2. 拥塞避免：当cwnd超过ssthresh,进入拥塞避免阶段，cwnd加法线性增大
3. 快重传：收到3个连续重复确认立即重传，不必等待RTO（重传计时器）到时
4. 快恢复：是快重传的后续处理。快重传后，阈值减半，并且cwnd=ssthresh/2，乘法减小，（cwnd等于乘法减小后的阈值）而不是从1开始（慢开始）

---
### udp如何实现可靠性传输？

1. 传输层无法保证数据的可靠传输，只能通过应用层来实现.
2. 通过包的分片、确认、重发
---
1. 短连接的优点是：管理起来简单，存在的连接都是有用的连接，不需要额外的控制手段，一边只有一次读写。WEB网站的http服务用短链接，因为长连接对于服务端会耗费资源，而像WEB网站频繁连接用短连接会更省资源。
2. 长连接：省去较多的TCP建立和关闭的操作，减少浪费，节约时间。长连接多用于操作频繁，点对点的通讯，而且连接数不能太多。数据库的连接用长连接。
    * 2小时内没动作，服务器就向客户发一个探测报文段
---
socket有哪几种
1. 流格式套接字（Stream Sockets）也叫“面向连接的套接字”，使用TCP协议
2. 数据报格式套接字（Datagram Sockets），也叫“无连接的套接字”，UDP

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

---
### 协程
1. 比线程更轻量级，一个线程有若干个协程
2. 协程不是被操作系统内核所管理，而完全是由程序所控制（也就是在用户态执行）。性能更高，不会由于切换消耗资源。
2. 由协程执行，在执行A的过程中，可以随时中断，去执行B，B也可能在执行过程中中断再去执行A，结果可能是：
# 数据库
### B+树
1个索引节点就是1个磁盘页

m阶B+树，xx个关键字，xx个指针

1. 非叶子节点不保存关键字记录的指针，只进行数据索引，可保存更多的索引信息
（非叶节点不包含指向 数据记录 存放地址的指针，没有data域）
2. 叶子节点保存了父节点所有关键字记录的指针
3. 叶子节点间有指针相连，从小到大，方便进行范围查找

### 红黑树
1. 根节点是黑色的，叶子是黑色的
2. 从跟到叶子的路径上黑色节点数相同
3. 保证最长路径不大于最短路径的2倍

为什么STL用红黑树？
1. 最坏情况下，AVL树有最多O(logN)次旋转，而红黑树最多三次

聚集索引，非聚集索引？
1. 聚集索引：表记录的存储顺序与索引的排列顺序一致。因为连续，查询快，更新慢，因为要维持顺序
    * 规定了物理存储顺序，因此一个表只能包含一个聚集索引。
    * 聚集索引的叶节点就是数据节点，而非聚集索引的叶节点仍然是索引节点，只不过其包含一个指向对应数据块的指针。
2. 非聚集索引：都采用B+树，但非聚集索引的叶子层并不与实际的数据页相重叠，而采用叶子包含一个指向 对应数据块的指针。
    * 添加记录不会引起数据顺序的重组
    * 索引中相邻，实际的存储不相邻

# Linux
## 编译原理
预处理器、编译器、汇编器、连接器
1. 
2. 
---
虚拟地址到物理地址的映射：
* MMU(内存管理单元)：每个程序有一个页表，存放虚拟页面到物理页面的映射，如果MMU接收到了程序发出的虚拟地址，在查找相对应的物理页面号时，没有找到，那么将会通过缺页中断来将需要的虚拟页面从磁盘中加载到物理内存的页面中。
---
1. fork: 创建虚拟地址空间，复制父进程的内容。cow但无物理空间，其指向父进程的物理空间
    * 不带参数，复制所有资源给子进程
    * 父子进程执行顺序不确定
2. vfork: 内核连子进程的虚拟地址空间也不创建，直接共享父进程的虚拟空间，没有了复制产生的开销。而且vfork()保证父进程在子进程调用execve()或exit()之前不会执行。vfork完就exec，（共享父进程的地址空间--线程）
    * 子进程先运行，可用来创建线程
3. clone: 可以更细粒度地控制与子进程共享的资源，因而参数也更复杂。
用来创建线程
    * 带参数，可选哪些资源复制给子进程
---
静态库、共享库
1. 静态库缺点，维护和更新时，需要重新链接。对于scanf或printf等基本命令，系统中的每个进程都会复制一份到其地址空间中，浪费内存。
2. 共享库（shared library）.so（共享目标）,由动态链接器 动态链接。
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
* Memory-map，内存映射, 是一种文件I/O操作, 它将文件(linux世界中, 设备也是文件)映射到内存中, 用户可以像操作内存一样操作文件.
* 进程可采用指针读写操作这一段内存（而系统会自动回写脏页面到对应的文件磁盘上），完成了对文件的操作而不必再调用read,write等系统调用函数。相反，内核空间对这段区域的修改也直接反映用户空间，从而实现不同进程间的文件共享。
1. **常规文件操作为了提高读写效率和保护磁盘，使用了页缓存机制**。 读文件时需先将文件页从磁盘拷贝到页缓存中，*由于页缓存处在内核空间*，不能被用户进程直接寻址，需要将页缓存中数据页再次拷贝到内存对应的用户空间中。通过2数据拷贝，才能完成进程对文件内容的获取。写操作也是一样，待写入的buffer在内核空间不能直接访问，必须要先拷贝至内核空间对应的主存，再写回磁盘中（延迟写回），需要两次数据拷贝。
2. 使用mmap时，创建 新的虚拟内存区域(映射区域) 和 建立文件磁盘地址和虚拟内存区域映射 这两步，没有任何文件拷贝操作（真正的文件读取是当进程发起读或写操作时）。而访问数据时发现内存中并无数据而发起的缺页异常过程，可以通过已经建立好的映射关系，只用1次数据拷贝，从磁盘中将数据传入内存的用户空间中，供进程使用。
  
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

---
进程间通信方式，共享内存是最快的一种IPC方式
* 管道需要复制4次 
* 共享内存复制2次

# 算法
1. 所有出栈顺序可能性：
回溯
2. 判断出栈顺序是否合法：https://www.cnblogs.com/bendantuohai/p/4680814.html
    * 模拟一遍
    * 若a[i+1] > a[i]，则a[i+1] 大于前面所有的数。如果按小到大出来的数，必定是进去就出来，那么比它们还大的数，就不可能比他们还早出来。
3. 两个有序数组求中位数，logn，
https://www.cnblogs.com/TenosDoIt/p/3554479.html。
https://blog.csdn.net/hk2291976/article/details/51107778
非递归
4. 蓄水池算法，第k+1个元素，概率k/(k+1),当前并不知道n有多大
5. BST->双向链表

# 项目
docker和vm区别
1. vm: 在硬件基础上，虚拟出操作系统
2. docker: 对操作系统复用。直接利用宿主机资源，启动docker等于启动进程
    * 用namespace做权限隔离控制，cgroups做资源分配，
    * 提供了 标准化的应用发布方式

Map-Reduce：
1. 分为map，reduce两个操作，map读、处理数据；reduce是对数据进行整合

MPI优点：
* 可移植性，易用性好。有很多异步通信功能



   

