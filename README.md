# cs-interview
 
# C++
C和C++区别？
1. C面向过程，
    * 没有函数重载

2. C++面向对象，有封装、继承、多态等特性，有函数重载（函数名相同，参数不同）
    * 有函数重载，因为函数符号为_funii
    * 对C中的struct进行扩展，唯一不同在于struct默认访问修饰符为public
    * C++有引用
---
### sizeof编译还是运行期？
* sizeof是运算符，不是函数。编译期确定
* sizeof 不能求得动态分配的内存大小

---
### C++ 四种 强制类型 转换的区别?
* C风格是TYPE b = (TYPE)a
* 去const属性用const_cast。
* 不同的数据类型转换用reinterpret_cast。
* 基本类型，没有多态的父子类间。转换用static_cast。
* 指针或引用，多态类之间的类型转换用daynamic_cast。

https://troywu0.gitbooks.io/spark/content/qiang_zhi_zhuan_huan.html
1. static_cast<T>()
    * 基本数据类型转换，只要不包含底层const；
    * 父类和子类之间转换，其中子类指针转换成父类指针是安全的（上行转换是安全的，下行不安全）
    * void*转换成其他类型指针

2. dynamic_cast<T>()  C++11才有，旧式风格没有
    * 只能转换指针或引用。上行转换时，和static一样，都是安全的。下行转换时会做类型检查，是安全的。在多态类型间转换主要使用这个，因为是RTTI的一部分。
    * 首先检查能否成功转换，如能成功转换则转换。
        1. 如果转换失败, 对于指针返回0 (NULL)，
        2. 对于引用会抛出异常。bad_cast。因为引用只能初始化时进行赋值，没有所谓空引用
    * 只能用于存在 虚函数的父子关系（因为RTTI，就是运行时类型识别） 的强制类型转换，安全的基类 -> 子类转换
    * 使用场景：当父类指针指向子类对象，执行某个派生类中的 非虚函数 操作，要转换。父类指针转换成子类的指针。
        * 尽可能使用虚函数，这样就不用cast转换

3. reinterpret_cast<T>();
    * 为数据的二进制形式重新解释，但是不改变其值。
    * 用在指针或引用类型之间的转换
    * `char b = *reinterpret_cast<char*>(&a);`
    * `char b = *(char*)&a`
4. const_cast<T>();
    * 只能改变运算对象的底层const，对指针或引用去改变底层conset
    * 如果对象本身是一个常量，cast后执行写，会产生未定义的后果
    * 使用场景：
        1. 对于未定义const版本的成员函数，通常使用const_cast来去除 const引用对象 的const，完成函数调用。
        2. 函数需要一个非const指针，只有const指针，要去掉其const
        3. 另外一种使用方式，结合static_cast，可以在 非const版本的成员函数 内添加const，调用完const版本的成员函数后，再使用const_cast去除const限定。

---
### 隐式类型转换，explicit（显式的）
1. 对于内置类型，低精你 度 变量 给高精度变量 赋值会发生隐式类型转换（小->大）
    * 相对应的是 显式类型转换（强制类型转换），高精度->低精度，会有精度的损失
2. 可以用 单个形参来调用 的构造函数定义了从 形参类型 到 该类类型 的一个隐式转换。
编译器会自动调用其构造函数生成临时对象。
3. explicit关键字只能用于类 的 构造函数 声明上。防止隐式转换
    通常将可以用一个实参进行调用的构造函数都声明为explicit。
    * Circle A = 1.23;  算隐式调用了构造函数
    * Circle A(10) 显式
    * Circle A = Circle(10) 显式
    * 形式上不直观。为了避免这种错误的发生
---
### RTTI？
* Runtime Type Identification，RTTI运行时类型识别，它提供了运行时确定对象类型的方法。
c++通过下面两个操作符提供RTTI。

虚函数表头部-1位置有指向type_info的指针
1. typeid：返回指针或引用所指对象的实际类型。
2. dynamic_cast：将 基类类型 的指针或引用安全的转换为 派生类型 的指针或引用。
* 对于带虚函数的类，在运行时执行RTTI操作符，返回动态类型信息；
* 对于其他类型，在编译时执行RTTI，返回静态类型信息。

---
### 引用和指针的区别？ int &a = b;
1. 引用是别名，。必须初始化，不能更换目标（指针常量）
    * 使用：引用型参数 
2. 指针是变量，存储的是一个地址。指针可以有多级，可以有const指针。
    * 自增运算意义不一样
3. ？不为引用单独分配内存空间对
    * sizeof（指针），是4个字节
    * sizeof（引用），返回指向区域的内存大小
---

### memcpy和strcpy区别
都是C库函数
1. strcpy提供了字符串的复制，只能复制字符串。不需要指定长度，'\0'
    * strcpy效率稍高，因为memcpy有count变量的变化，还有类型转换。
2. memcpy: 一般的内存复制。
---

### const成员函数
1. const成员函数可被 const对象，普通对象 调用。
2. const成员函数不能修改 数据成员，也不能调用非成员函数（间接修改）（不能对对象任何修改）
*  int GetCount() const;
---

### C++ 类内可以定义引用数据成员吗？
1. 可以，（常量，引用类型）必须通过构造函数 初始化列表 初始化。
2. ??普通成员变量只能被 构造函数 或初始化列表 初始化。
---
### C++里是怎么定义常量的？常量存放在内存的哪个位置？
1. 常量在C++里的定义就是一个top-level const加上对象类型，常量定义必须初始化。
2. 对于局部对象，常量存放在栈区，对于全局对象，常量存放在全局/静态存储区。对于字面值常量，常量存放在常量存储区。
---
申请内存
* 所申请的内存是由多个 内存块 构成的链表
    * 有meta区，数据区

### malloc free 实现
1. malloc, 可用空间是分散的一块块的，用 ***空闲链表*** 标识。
    * 多分配4个字节，存储这个块有多大，释放的时候就知道了大小；指向下一块的指针
    * malloc分配一块连续的内存。内存分配算法：寻找合适的block，first fit；best fit；next fit
    * 没有足够大的空间，相邻的合并，还没有，通过sbrk向内核申请堆
2. free，把内存块标记未被使用。is_avilible=1
3. brk, mmap?
    * 这两种方式分配的都是虚拟内存，没有分配物理内存。在第一次访问已分配的虚拟地址空间的时候，发生缺页中断，操作系统负责分配物理内存，然后建立虚拟内存和物理内存之间的映射关系；

### 构造函数 析构函数 虚函数
1. 构造函数不能是虚函数，构造函数调用之前找不到虚函数表。（会有编译错误）
    * ??因为子类不能继承构造函数，将构造函数声明为虚函数没有意义
    * 从使用上来说，虚函数通过 基类指针 来调用派生类的成员，则在调用之前，对象必须存在。而构造函数调用之前没有对象。
    * 编译器会加代码，自动先调用父类默认构造函数
2. 析构函数可以是虚函数，甚至是纯虚函数：是为了避免内存泄露
    * 若不是virtual，则在删除对象时，如果直接删除基类指针(该指针指向了派生类)，系统就只能调用基类 析构函数 ，而不会调用派生类析构函数。这就会导致内存泄露。
    * 写子类析构函数时，编译器自动在最后调用父类析构函数

#### 为什么不把析构函数默认设置为虚函数？
* 原因是虚函数表的开销以及和C语言的类型的兼容性，虚函数表指针有额外的内存开销
3. 纯虚析构函数
想把Base做出抽象类（虚基类），不能直接构造对象；需要在其中定义 1 个纯虚函数。如果其中没有其他合适的函数，可以把析构函数定义为纯虚。

***注意：如果基类的析构函数设置为虚函数，所有派生类也默认为虚析构函数，即使没有带关键字Virtual。如果父类函数func()为虚函数，则子类中，函数func()是否加virtual关键字，都将是虚函数。为了提高程序的可读性，建议后代中虚函数都加上virtual关键字。***

3. 把所有的类的析构函数都设置为虚函数好吗？(虚函数的缺点)
* 运行时确定根据不同的对象调用不同的虚函数，要存储额外的信息用于查找。
* 系统为每一个对象存储了一个 ***虚函数表指针***，指向这个类的虚函数表。
* 使用虚函数后的类对象 占的空间多，而且在查找具体使用哪一个虚函数时，还会有时间代价。即当一个类不打算作为基类时，不用将其中的函数设置为虚函数。
---
### 纯虚函数是什么
* 虚函数的声明以=0结束，便可将它声明为纯虚函数。包含纯虚函数的类不允许实例化，称为抽象类。 
* 对于纯虚函数，其派生类一定要对其定义
---
### 在构造函数、析构函数里 调用 虚函数会发生啥？
1. 简单说就是虚函数机制失效
2. 先调用p的构造函数，再是s的。析构正好相反

### 只能在堆上动态声明对象
1. 将析构函数声明为虚函数，通过另外的成员函数调用析构函数
2. 在编译阶段会检查析构函数的可达性，
3. 因为动态对象是由程序员手动控制，所以系统不做检查

### 只能在栈上声明
1. operator new、operator delete 声明为私有

### 成员函数和继承
1. 有2类成员函数，
    1. 希望派生类进行覆盖override, 虚函数
    2. 直接使用, 非虚函数
2. 当使用指针或引用调用虚函数时，该调用将被动态绑定。根据指针或引用所绑定的对象类型不同，可能会执行基类、派生类的版本。

### 访问权限
1. protected，在派生类中可被访问，相当于public。但是外部访问不了

### 继承方式，用在类派生列表
1. public，   基类成员在派生类中访问权限不变
2. private，  所有基类成员都变为private
3. protected, 基类的public，和protected 在派生类中变为protected, 私有仍是私有


### 右值引用
* 左值：非临时对象
* 右值：指临时的对象，只在当前的语句中有效
* 右值引用解决了移动语义问题
---

### C++中的锁
1. 互斥锁
2. 读写锁读写锁
3. 自旋锁：当线程获取不到资源时，不是进入睡眠状态，而是让当前的线程不停地执行空循环，直到锁被是否
    * 节省了线程从睡眠状态到被唤醒期间的消耗，在加锁时间短暂时极大提高了效率
    * 由于自旋锁不改变线程的状态，所以线程的运行会比较快。
4. 条件锁（条件变量）：
5. 同步锁：类似于信号量，解决不同进程间同步问题
    * 同步：协调各线程的执行顺序，使得有特定的执行顺序。可以叫时序
---
### 多线程



### 浅拷贝 深拷贝
* 对于含有指针成员的类，直接拷贝可能会出现两个对象的 指针成员 指向同一个数据区。这时候一般先new个内存，然后复制内容 
* 对于堆上的内存需要深拷贝
---
### 内存对齐？和机器字长有关，还有编译器默认对齐值
 * 32位机器最小存储单位是4字节
 * 保证当前偏移量是每个变量的整数倍
 * 为什么要对齐？
    1. 合理安排结构体成员顺序，减少使用的内存 
    2. 提升数据读取效率，每次从内存中为8字节整数倍的地址开始读入8字节的数据
* 对齐单位：min(#pragma pack(4),最小的成员变量)
* 结构体的整体大小是 对齐单位的整倍数
```#pragma pack(4)
struct {
    char a;
    short b;
    char c;
} myStruct;
cout << sizeof myStruct << endl;  // 6
```

---
### C++空类大小：1
*   类中初始化列表的初始化顺序：与变量的声明顺序一样。一般变量只能在构造函数或初始化列表中初始化，否则为随机值

---
### C++ STL 
1. unordered_map<int, int>mp c++中哈希表对应的容器,查找复杂度 o(1);不是线程安全的，可用读写锁解决
    * 底层是hashtable，有一个大vector，以便动态扩容；vector上挂链表解决冲突
    * 当冲突的链表长度超过8，可转换红黑树解决。
    * 先根据key算出hash值，找到对应的桶，在桶里遍历
    * rehash：如果元素个数/size 大于阈值，二分找到下一个新(约2倍)的质数，扩容，并rehash。为了减少rehash时间，可实际用到某个key时再rehash。

如何解决冲突？
    1. 链表法
    2. 开放地址（线性探测，二次探测，二次哈希）
2. map<int, int> mp  底层红黑树实现 插入查找删除复杂度 logn， 插入n个元素就是nlogn，有序的

if(mp.find(key) != mp.end() ) 判断是否存在
---
### 红黑树
* 本质是BST，保证了其最坏复杂度logn
1. 若任意结点的左子树不空，则左子树上 **所有** 结点的值均小于它的根结点的值；
2. 
3. 任意结点的左、右子树也分别为二叉查找树。
4. 没有键值相等的结点（no duplicate nodes）。

1. 每个结点要么是红的，要么是黑的。  
2. 根结点是黑的。  
3. 每个叶结点（叶结点即指树尾端NIL指针或NULL结点）是黑的。  
4. 如果一个结点是红的，那么它的俩个儿子都是黑的。  
5. 对于任一结点而言，其到叶结点树尾端NIL指针的每一条路径都包含相同数目的黑结点。

红黑树与AVL树比较？
1. AVL是严格平衡树，因此在增加或者删除节点的时候，旋转的次数比红黑树（最多3次）要多； （logn）
2. 红黑树是弱平衡的，用非严格的平衡来换取*增删*节点时候旋转次数的降低；

---
1. 引用，在声明时要初始化，sizeof引用得到的是所指向的变量(对象)的大小
2. 指针，sizeof是指针本身的大小(地址,64位是8个字节)。指针和引用的自增(++)运算意义不一样；

---
1. 顶层const - 指针常量：int* const p, 指针所保存的地址不可改变，但指向的变量的值能变
2. 底层const - 常量指针：const int *p / int const *p, 指向常量的指针，指向的地址可以变，指向的是个常量，内容不可变

* 顶层const：指针本身是一个常量
    * 一般情况下，顶层 const 可以表示任意对象是一个常量
    * const int a = 10: 这是顶层const
* 底层const:指针所指的对象是一个常量
    * 底层 const 则与指针和？？引用等复合类型的基本类型部分有关


* const int* const p
* 底层  ----   顶层
---

### 函数指针，数组指针
1. int (*pf)(const int&, const int&)； 要有小括号，否则是声明一个函数
2. int (*p)[n]
    * int a[3][4];
    * int (*p)[4]; //该语句是定义一个数组指针，指向含4个元素的一维数组。

---
### define 和 const区别
1. define在预处理阶段被替换，不占内存
2. const在编译运行阶段被处理，有数据类型，有安全检查。占用内存
---
### define和内联函数
* 加号优先级高于三目运算符：define MAX(a, b) ((a) > (b) ? (a) : (b))
* 内联函数会 安全可靠，适用于短小函数，代码量大的话会复制过多的代码到代码区。
    1. 编译期间替换成函数代码

---
### 拷贝构造函数
1. 第一个参数是自身的引用，其余参数有默认值也可以

### 运算符重载
1. 拷贝构造函数：用于创建对象时调用的
    * String c=a; String c(a); 
2. 赋值运算符重载：String& operator=(const String& other)
    * c = a;
* 有4个步骤，
* https://troywu0.gitbooks.io/spark/content/kao_bei_gou_zao_han_shu_yu_fu_zhi_yun_suan_fu_zhon.html

---
### new 和malloc区别
1. malloc是C语言的库函数, 指定分配内存空间的大小，返回的是void*指针(也可建立对象，不会调用构造函数），用的时候要强制转换。
2. new可以用于自定义类型，类的动态内存分配，初始化。
2. new是C++的 操作符，new建立对象，自动调用构造函数，消亡之前调用析构函数。
3. 而malloc free是库函数不是操作符，不在编译器控制权限之内，无法把构造函数和析构函数任务强加给malloc和free。
* new底层先调用malloc申请一块适合的内存空间，然后使用该类型的构造函数进行函数的构造生成。
---

### delete[] p; delete p; 区别
* 对简单类型没有区别，对复杂类型，[]会析构所有对象；不加[]只析构1个。
    * 针对复杂类型，p的前4个字节会记录数组大小
1. delete[]时，数组中的元素按逆序的顺序进行销毁；
2. new将内存分配 和 对象构造组合在一起
3. delete将对象析构和 内存释放组合在一起
4. allocator释放内存，不初始化


#### 操作符、库函数区别？
* 运算符是语言自身的特性，它有固定的语义，而且编译器也知道意味着什么。就像 +-*/ 一样，由编译器解释语义，生成相应的代码。
* 库函数是依赖于库的，没有库就没有它，也就是一定程度上独立于语言的。

---
### static作用
1. C++的修饰符，控制变量的存储方式和可见性（生命周期和作用域）
    * 非静态全局变量的作用域是整个源程序，多个源文件中都有效
    * 静态全局变量：只在当前文件中有效
2. 一个数据为 整个类 服务而非某个对象
3. static函数：在内存中只有一份，普通函数在每个被调用中维持一份拷贝
    * 作用域不同，只被本源文件调用
---
### static局部变量和普通局部变量有什么区别 ？
1. 把局部变量改变为静态变量后 改变了它的存储方式即改变了它的生存期。
    * static局部变量只被初始化一次，下一次依据上一次结果值
2. 把全局变量改变为静态变量后 改变了它的作用域，限制了它的使用范围。  
    * 静态全局变量作用域在当前源文件，而非静态整个程序

### static函数与普通函数有什么区别？
* static函数与普通函数作用域不同,仅在本文件。只在当前源文件中使用的函数应该说明为内部函数(static修饰的函数)
* static函数在内存中只有一份，普通函数在每个被调用中维持一份拷贝

### 迭代器失效
vector<int>::iterator it;
1. 关联式容器（map，set，multimap，multiset）删除当前的iterator，仅仅会导致当前iterator失效。这是因为对于这些关联式容器，底层使用红黑树来实现，插入、删除一个节点不会对其他节点产生影响。
2. 对于序列式容器（特别是支持随机存取的容器：vector、deque），删除当前iterator会使后面所有元素的iterator都失效，因为这些支持随机存取的容器使用连续分配的内存，删除一个元素会导致后面的所有元素都往前移动一个位置。

---
### 智能指针
定义一个类来封装资源的分配和释放，构造析构由编译器自动调用。
* auto_ptr问题：调用拷贝构造和赋值运算符重载函数时，将一块空间的权限完全交给别人。
1. shared_ptr，是引用计数的智能指针。可以有多个指针指向同一块内存，底层用引用计数的方式实现，即每增加一个指针，该内存的引用计数+1，减少一个指针，内存引用计数-1，当为0的时候，内存就会被析构函数销毁。
    * 它的引用计数本身是安全且无锁的，但对象的读写则不是；shared_ptr 对象本身的线程安全级别，不是它管理的对象的线程安全级别。
    * 循环依赖问题：http://senlinzhan.github.io/2015/04/24/%E6%B7%B1%E5%85%A5shared-ptr/
    * 只有在拷贝构造，赋值时候才++？？

2. unique_ptr，对象里存放的内容在整个程序中只能出现一次
3. weak_ptr，指向一个shared_ptr管理的对象。不控制所指向对象的生存期。引用计数会带来循环引用产生无法删除的情况，导致内存泄漏。可使用weak_ptr，弱引用只引用但不增加或删减计数。如果一块内存被shared_ptr和weak_ptr同时引用，当所有的shared_ptr析构了之后，不管有没有weak_ptr引用该内存，内存会被释放。 
---
### 悬垂指针，野指针
1. 悬垂指针，2个指针指向同一区域，指向的内存已经被释放了，但是另一个指针还存在
    * 释放完空间后就是悬垂指针，需要将其再次指向NULL
2. 野指针：随机指向一块内存的指针，
    * 例如没有初始化;
    * 释放后没有置空NULL;
    * 超过变量范围

---
### extern作用
1. 置于变量或函数前，表示定义在别的文件中，提示编译器遇到此变量或函数在其他文件中寻找定义。

extern “C” ？
* C++调用C函数需要extern C，因为C语言没有函数重载。
### volatile
1. 强制cpu每次从内存中访问变量，因为寄存器有可能（被别的程序）改变。
2. 内存中的值发生变化，寄存器还没变

### 封装、继承、多态多态
* 接口的多种不同实现方式为多态
1. 使用虚函数才会调用子类同函数，运行时多态（动态多态，通过虚函数和继承关系，在运行时确定），使用上层操作来执行下层具体操作
2. 父类指针指向子类对象（父类指针调用时，通过查找子类对象的虚函数表，找到指向哪个虚函数）
3. 编译时多态（静态多态）：重载函数，模板技术
如何实现多态？
* 子类若重写父类虚函数，虚函数表中，该函数的地址会被替换。
* 对于存在虚函数的类的对象，在VS中，对象的对象模型的头部存放指向虚函数表的指针，通过该机制实现多态。
https://jocent.me/2017/08/07/virtual-table.html

### 多态的三种形态：
1. 通过基类指针调用基类和子类的同名虚函数时，会调用对象的实际类型中的虚函数。
2. 通过基类引用调用基类和子类的同名虚函数时，会调用对象的实际类型中的虚函数。
3. 基类或子类的 成员函数 中调用基类和子类的同名虚函数，会调用对象的实际类型中的虚函数。

### 多重继承、虚继承？
* class C：public A， publicB{};这样就是一个多重继承。
* 运行时多态
    1. 父类的指针指向子类的对象，调用子类中的函数。在运行时决定调用哪一个函数。
    2. 与静态多态不同的，静态多态就是函数的重载和模板。
* 虚继承？https://troywu0.gitbooks.io/spark/content/duo_zhong_ji_cheng_yu_xu_ji_cheng.html
    1. 为了解决多重继承中的菱形继承问题
    2. 用法：虚继承的类就叫做虚基类。 A就是是虚基类，B和C虚继承A，D中只有一个A对象
    3. 最后注意，虚基类总是先于非虚基类构造的，与继承的顺序是没有关系的。
* 虚函数表，继承几个父类就有几个虚函数表
https://blog.csdn.net/haoel/article/details/1948051/

### 重载 覆盖 隐藏
1. 重载overload: 在同一作用域中，仅有函数参数不同。
    * 在编译时确定，根据函数符号_ffii，以参数作为后缀
2. 覆盖override
    * 在虚函数的继承时候发生的，而且发生在不同的范围内，基类和子类。
    * 函数前必须有virtual。而重载有没有virtual无所谓。
    * 而且函数名字和参数必须相同。如果参数不同，就是隐藏，不是重载哦。
3. 隐藏overwrite
    * 隐藏发生在继承时候。如果基类的函数不是virtual，子类又重新定义了该函数，无论参数相同不相同都是隐藏。
    * 如果是virtual 函数， 参数相同就是覆盖，参数不同就是隐藏。
* 覆盖的条件比较苛刻，只有virtual，并且函数名，参数都相同时，才发生。



# 计网

### 网络编程
* 客户端
    1. 创建socket，绑定ip和端口
* 客户端和服务器都可主动发起挥手动作，对应socket编程的close()，任何一方都可发起
* 连接动作对应connect()，对应三次握手，accept()从队列中取出连接。第二次握手完成后connect()返回。
* int listen(int sockfd, int backlog); 如果不accept，也可建立连接，最多连接数就是backlog,即accept队列大小 
---
### 为什么三次握手而不是两次，A还要发送一次确认？
1. 防止已经失效的连接请求送到了B, A没有建立连接，B却认为已建立连接，让服务器错误打开连接。等待数据到来，浪费资源。
2. 服务器端收到客户端的连接确认，同步双方序列号，确认号。tcp是可靠的连接，确保双方同步。

---
### 为什么要这有TIME_WAIT，为什么不直接给转成CLOSED状态? 
1. TIME_WAIT确保有足够的时间让对端收到ACK，如果被动关闭方没有收到Ack，就会触发被动端重发Fin，一来一去正好2个MSL。（实现TCP全双工连接的终止可靠性）
2. ？？有足够的时间让这个连接中延迟的数据不会跟后面的连接混在一起，（当存在端口重用时）。
http://www.voidcn.com/article/p-aogsoptp-hw.html

---
### 如何避免过多的Time_wait?
1. 缩短2MSL的时间 
2. 使用SO_REUSEADDR允许连接重用 
3. 设计协议避免TIME_WAIT产生的问题

---
### 字节流、报文
1. TCP面向字节流、无边界
    * 有缓冲区，可一次接收
    * TCP面向字节流，面向连接，都是同一台主机发送的，一直发送没关系
2. UDP面向报文、有边界
    * 只能逐次接收
    * UDP面向报文，无连接，应用层交给UDP多长报文，照发，不拆分也不合并。如果读取超过一个报文，多个发送方的报文可能混在一起

---
UDP没有流量控制和拥塞控制，所以在网络拥塞时不会使源主机发送速率降低（对实时通信很有用，比如QQ电话，视频会议等）

### TCP可靠传输
* 在IP的不可靠上使用 校验、序号、确认、重传
    * 序号: 本报文段所发送数据第一个字节的序号
    * 确认: 期望收到对方下一个报文段数据 的第一个字节的序号
    * 重传: 2种事件导致TCP对报文段进行重传： 超时、冗余ACK(效率更高，因为RTO往往太长；快重传)
* RTT(Round Trip Time)：一个连接的往返时间，即数据 发送 时刻到接收到 确认 的时刻的差值； 
* RTO(Retransmission Time Out)：重传超时时间，即从数据发送时刻算起，超过这个时间便执行重传。
* 为了避免流量控制引发的死锁，TCP使用了持续计时器。每当发送者收到一个零窗口的应答后就启动该计时器。时间一到便主动发送报文询问接收者的窗口大小。若接收者仍然返回零窗口，则重置该计时器继续等待；若窗口不为0，则表示应答报文丢失了，此时重置发送窗口后开始发送，这样就避免了死锁的产生。

### 发送窗口的实际大小 = Min{rwnd, cwnd}。RWND = 流量控制，CWND = 拥塞控制

### TCP流量控制: 针对单个tcp连接，使发送方速度小于接收方
1. 滑动窗口 (连续ARQ协议) 用于流量控制：表示接收方 有多大缓存可接收数据。发送方窗口内的序列号代表已发送，但还没有被确认的帧，或者可以被发送的帧。发送方接收的ACK中有 接收窗口RWND 的大小，以控制发送方的速度。CWND是发送方根据网络拥塞情况调整的。

* MSS，maximum segment size：一个最大报文段长度
* CWND（congestion window）:拥塞窗口，发送方的 Window 大小，表示当前可以发送多少个tcp包。取决于网络拥塞情况，动态变化
* RWND（receiver window): 接收端窗口：表示当前还能接收多少个TCP包。接收方用于限制发送方。

### TCP的拥塞控制: 针对整个网络，使每个tcp连接都能高速 (根据网络情况，来维护CWND)
1. 慢开始 （乘法增大）： 双方在建立连接时，先慢慢发包，CWND = 1，2，4，8。即1个MSS，经过1个RTT
后，收到确认后再double（慢慢拧开水龙头）
2. 拥塞避免（加法增大）：当cwnd达到ssthresh, 进入拥塞避免阶段，每经过1个RTT，cwnd加法线性增大。
            当出现超时，阈值设为当前cwnd一半，cwnd = 1,重新进入慢开始。
            cwnd < ssthresh，根据大小关系来选择 慢开始 or 拥塞避免

* 3、4 是对 1、2的改进
3. 快重传：接收方收到 *失序的报文段* 后立即发出重复确认，发送方收到3个连续重复确认立即重传，不必等待RTO（重传计时器）到时
4. 快恢复：是快重传的后续处理。快重传后，阈值减半，并且cwnd=ssthresh/2，乘法减小，（cwnd等于乘法减小后的阈值）而不是从1开始（慢开始）。然后开始执行拥塞避免
    * 快恢复是对慢开始的改进，cwnd不从1开始。实际上，两种方案都使用，如果因为 超时 -> 慢开始；快重传-> 快恢复
---

### udp如何实现可靠性传输？
1. 传输层无法保证数据的可靠传输，只能通过应用层来实现.
2. 通过包的分片、确认、重发
---

1. 短连接的优点是：管理起来简单，存在的连接都是有用的连接，不需要额外的控制手段，一边只有一次读写。WEB网站的http服务用短链接，因为长连接对于 服务端 会耗费资源，而像WEB网站频繁连接用短连接会更省资源。
2. 长连接：省去较多的TCP建立和关闭的操作，减少浪费，节约时间。??长连接多用于操作频繁，点对点的通讯，而且连接数不能太多。数据库的连接用长连接。
    * 2小时内没动作，服务器就向客户发一个探测报文段
---
全连接、半连接？
* 一个tcp连接管理使用两个队列，一个是半链接队列（用来保存处于SYN_SENT和SYN_RECV状态的请求），
* 一个是全连接队列（accpetd队列）（用来保存处于established状态，但是应用层没有调用accept取走的请求）。 

---
socket有哪几种
1. 流格式套接字（Stream Sockets）也叫“面向连接的套接字”，使用TCP协议
2. 数据报格式套接字（Datagram Sockets），也叫“无连接的套接字”，UDP

---
请求Web过程？
* HTTP 连接是基于 TCP 连接之上的，HTTP 属于应用层，TCP 属于传输层。因此发送 HTTP请求（HTTP无状态也无连接的，没有HTTP连接这种说法）之前需要先建立 TCP 连接，那么就涉及到 TCP 的三次握手四次挥手等过程。

---
### http和https区别
* http:超文本传输协议,明文方式发生内容，不提供任何方式的数据加密;连接是无状态的;80port
* https = http + SSL/TLS : 在HTTP的基础上加入SSL协议，身份认证,保证数据传输的安全;443port。
    * TLS是传输层加密协议，前身是SSL协议。Transport Layer Security，即安全传输层协议

### HTTP报文结构
* 有请求报文，相应报文两种
* 请求报文包括
    1. 请求行：请求方式，URL，HTTP版本
    2. 请求头：Host: 接收请求的地址。User-Agent；Connection：与连接相关的属性。Accept-Charset、Accept-Encoding、Accept-Language、Cookie
    3. 请求体：
* 对应有
    1. 响应行：http版本，状态码。
    2. 

### HTTP状态码
1. 3xx:301是永久重定向，302临时重定向。通过响应头中的Location指定重定向的地址。
2. 502: Bad Gateway  服务器挂掉了；
3. 503: Service Unavailable。临时的服务器维护、过载，请求数太多，当前无法处理请求，从而拒绝某些用户的访问

### cookie和session
* 用于连接时保持状态
1. cookie 存在client端；不安全；单个cookie保存数据不超过4k
2. session 存在server端；在一定时间内保存在服务器，当访问增多，会占用服务器的性能，考虑减轻服务器压力，用cookie

### 各层协议
0. socket不属于任何一层，只是TCP/UDP的具体实现，提供了接口

1. 物理层：通过媒介传输比特,确定机械及电气规范,传输单位为bit
    * IEE802.3 CLOCK RJ45
    * hub(集线器)：把收到的每一个字节都复制到其他端口上去。
    * 中继器

2. 数据链路层：CSMA/CD协议，MAC VLAN PPP
    * 将比特组装成帧和点到点的传递,传输单位为帧
    * 交换机：解决hub集线器对任何数据报文都转发这个低效率的问题。维护一个转发表，根据mac地址寻址
    * 网桥

3. 网络层：ARP（地址解析协议，IP->MAC）,RAPR，IP，ICMP（ping,互联网控制报文协议）
    * ICMP，用于传递控制消息。控制消息是指网络通不通、主机是否可达、路由是否可用等网络本身的消息。
    * ARP, Address Resolution Protocol: 通过解析网络层地址来寻找数据链路层地址
    * 负责数据包从源到宿的传递和网际互连，传输单位为包,
    * 路由器：路由选择 和 分组转发
        1. 路由选择：根据 路由选择协议 构造出路由表，根据从相邻路由器得到的网络拓扑变化，动态改变所选择的路由，根据路由选择算法，维护路由表。
        2. 分组转发：根据转发将用户IP数据报从合适的端口转发出去。转发表是根据路由表得来的，一般统称为路由表。
        * 可以解析ip地址，根据ip地址寻址。用于连接不同的子网。
        * 连接不同子网提供路由、转发。
        * 发送数据报时，若在源和目标主机在不同子网，需要间接交付给路由器，按照转发表（路由表）将数据报转发给下一个路由器。 *隔离了广播域*
4. 传输层：TCP, UDP
    * 提供端到端的可靠报文传递和错误恢复，传输单位为报文
    * 网关：有协议转换功能设施，能在不同协议间移动数据

5. 应用层：HTTP，SMTP，POP3, FTP，DNS

### 有IP地址为什么还要有mac地址?
1. 网络中设备很多，根据mac地址怎么找呢？mac地址太多了，不可能存在一个表中。
1. mac地址不能表示子网的概念，（出厂前设定好的）
2. ip地址和地域相关，同一个子网中，ip前缀相同

### 路由器路由表工作原理
1. 

### 子网内怎么查找mac地址？不同子网怎么找mac地址？（也就是ARP地址解析过程）
1. 子网内，采用广播形式，广播目标IP地址。每台服务器判断自己是否为目的地址，如果是回答这个ARP请求，把自己的mac地址填入其中。并将其缓存在本地ARP表中。
    * 主机A在准备构造2层帧头时，首先根据目的IP去查找ARP表，如果找到对应项，则直接得到目的MAC地址，如果没有查到才执行上面所说的ARP广播请求。这样做是为了最大限度地减少广播。
2. 如果不在同一子网，就通过ARP询问默认网关对应的MAC地址。
3. 主机A首先比较目的IP地址与自己的IP地址是否在同一子网中，如果在同一子网，则向本网发送ARP广播，获得目标IP所对应的MAC地址；如果不在同一子网，就通过ARP询问默认网关（路由器）对应的MAC地址。

### http1.0 1.1 2.0
1. HTTP1.0需要使用keep-alive参数来告知服务器要建立一个长连接，而HTTP1.1默认支持长连接；
2. HTTP2.0使用了多路复用的技术，做到同一个连接并发处理多个请求
---
# 操作系统
### CAS Compare And Swap，比较并替换
* 3个基本操作数：内存地址V，旧的预期值A，要修改的新值B。
* 当且仅当预期值A 和 当前内存值V相同时，将内存值修改为B并返回true，否则什么都不做，并返回false。
* 在处理并发问题时，使用的一种方式。
 
* 缺点
    1. CPU开销大。多个线程反复尝试更新某变量，却一直失败
    2. 不能保证代码块的原子性；CAS机制所保证的只是一个变量的原子性操作
    3. ABA问题

### 进程线程区别？
1. 哪些资源是线程私有的？
    * 一组寄存器，栈
2. 线程共享？
    * 代码段、数据段、堆空间
---
### 堆、栈空间区别？
1. 为什么栈分配效率高?
    * 栈：在底层对栈提供支持，有专用寄存器存放栈的地址，入栈、出栈都有专门的指令。
    * 堆：堆则是库函数提供支持，有一定的分配算法。若空间不足，还会向系统请求资源。

---
### 虚拟内存
1. 核心思想，是把虚拟内存存在磁盘上，将主存作为磁盘的缓存
1. 内核虚拟内存，对所有进程都是一样的。里面有页缓存，可时间进程间通信。
---
为什么线程切换开销小？
1. 进程切换时要切页表，伴随着页调度，因为进程的数据段代码段要换出去，以便把将要执行的进程的内容换进来。
2. 线程只需要保存线程的上下文（相关寄存器状态和栈的信息）就好了，动作很小。
http://ourcoders.com/thread/show/4327/
* 切换的开销主要就是看是否需要中断进入内核和是否需要重新建立映射(建立页表)。
---
### 内存碎片问题（原因：小块内存频繁分配释放）
* Linux内存管理算法Buddy和Slab
1. 伙伴系统
    * 伙伴系统（buddy system）是以页为单位管理和分配内存。用于解决外部碎片。
    * 11个free list，最大可申请1024个连续的页框，对应4MB连续内存
    * 用页表可以把不连续的物理地址在虚拟地址上连续起来，但是内核态就没有办法获取大块连续的内存。例如GPU需要大块物理地址连续的内存
2. slab(是一种内存分配机制)
    * 以Byte字节为单位，用于解决内部碎片
    * 针对一些频繁分配释放的对象，如进程描述符。slab分配器是基于对象进行管理的，相同类型的对象归为一类。
    * 每次当要申请这样一个对象，slab分配器就从一个slab列表中分配一个这样大小的单元出去，而当要释放时，将其重新保存到该列表中，而不是直接返回给伙伴系统，
---
### 内存分配算法：
* First Fit: 每次从链首查找
* Next Fit、 
* Best Fit

### 为什么要清理内存碎片？分配空间物理是连续的吗？
* 使用malloc分配的内存空间在虚拟地址空间上是连续的，但是转换到物理内存空间上有可能是不连续的，因为有 可能相邻的两个字节是在不同的物理分页上；
* （内存碎片是虚拟内存空间的概念。程序员看到的只有虚拟内存空间，物理内存空间是透明的）

---
### 段和页区别？
1. 页的物理单位，长度固定
2. 段是程序的逻辑单位，长度不定

### 进程
* task_struct（PCB进程控制块，即进程描述符。在系统中是一个结构体）
1. CPU的程序计数器和各种寄存器都只有一份，当进程切换时，就需要保存进程上下文

---
* 系统向该 session 发出SIGHUP信号，session 将SIGHUP信号发给所有子进程。前台任务退出。
1. 守护进程：持续运行在后台的进程。一般用来提供服务，如httpd
2. 

### 进程调度算法
1. 先来先服务
2. 短作业优先
3. 时间片轮转
4. 多级反馈队列，
    * 每个队列有不同优先级，优先级越高时间片越短
    * 第一个队列时间片用完，放到第二个队列中
    * 仅当第一个队列空闲时，才调度第二个队列
5. 抢占式优先级调度算法
6. 高响应比优先调度： R = (w + s)/s, w是已等待时间，s表示服务期望时间

### 僵尸进程
1. 孤儿进程：父进程退出后，通常不会通知子进程，会将所有子进程过继给init（1）进程。可使用进程组；或利用通信机制，主动发送信号，通知子进程退出。
2. 僵尸进程：
    * 终止了但还未 被父进程回收 的进程。
    * 进程通过exit()结束进程后，会变成一个僵尸进程，僵尸进程几乎释放了其他所有的资源，但PCB没有释放，其中PCB中的这个字段就是记录退出的退出码。
    1. 子进程退出，而父进程并没有调用wait（是waitpid的简化版本）或waitpid获取子进程的状态信息，那么子进程的进程描述符（PCB）仍然保存在系统中。这种进程称之为僵死进程。
    2. 任何一个子进程(init除外)在exit()之后，并非马上就消失掉，而是留下一个称为僵尸进程(Zombie)的数据结构，等待父进程处理。这是每个子进程在结束时都要经过的阶段。
3. 怎么处理僵尸进程
* 僵尸进程很难被杀死，一般杀死僵尸爸爸。进程变成孤儿进程，这样进程就会自动交由 init 进程（pid 为 1 的进程）来处理，一般 init 进程都包含对僵尸进程进行处理的逻辑。
* 连续 fork 两次，然后 kill 第二个进程，使得要使用的父进程为init
* signal(SIGCHLD, SIG_IGN),通知内核对子进程的结束不关心，由内核回收


### 怎么创建守护进程
1. 在后台运行，
2. 守护进程必须与其运行前的环境隔离开来。这些环境包括未关闭的文件描述符，控制终端，会话和进程组，工作目录以及文件创建掩模等。这些环境通常是守护进程从执行它的父进程（特别是shell）中继承下来的。 
3. 父进程执行fork并exit退出
4. 子进程中setsid创建新的会话

---
### OS缺页置换算法，LRU如何实现
1. LRU，当未命中并且缓存已满时，需要按一定策略置换。
    * 插入和删除是O(1)的，用双向链表
    * 快速存储，查找O(1)，用HashMap

2. redis中如何实现LRU？


### cache（类比于停车位）
* 高速缓存分S组，每组E个cacheline，每行有B个数据块（block）。
* 根据每个组的行数E，被分为几类。
1. 1: 直接映射，每组只有一个cacheline, 适用大cache。内存中某块只能映射到特定组。可能N个块对应1个组。
    * 多个人用同一车位，停车难，找车方便。
2. e: 组相联映射，s组，每组e个cacheline。 E路组相联。
3. n: 全相联映射，适用小容量cache，遍历的开销小。只有一个组，主存中的一个地址可被映射进任意cacheline，需要遍历每一个cacheline来寻找是否被cache。
    * 可随便停，但是找车慢。
---
### buffer（缓冲区）与cache区别
1. 进行流量削锋，减少短期内突发io
2. 先存到buffer，满了再写到硬盘，减少io次数

---
什么是互斥与同步？
1. 同步是指用于实现控制多个进程按照一定的规则或顺序访问某些系统资源的机制。
2. 互斥是同步机制中的一种特殊情况。对资源的互斥访问（临界资源）
    * 访问临界资源的代码是 临界区

---
### 同步问题：
1. 生产者-消费者问题：用互斥锁，和2个信号量解决。先判断信号量，再上锁。

---



### Linux 5种IO模式
1. BIO 阻塞式IO(blocking I/O）
2. NIO 非阻塞式 I/O 模型(non-blocking I/O）
3. I/O 复用模型(I/O multiplexing）
    * I/O多路复用就通过一种机制，可以监视多个描述符，一旦某个描述符就绪（一般是读就绪或者写就绪），通知程序进行读写操作。但select，poll，epoll本质上都是同步I/O，因为需要在读写事件就绪后自己负责进行读写，也就是说这个读写过程是阻塞的，而异步I/O则无需自己负责进行读写，异步I/O的实现会负责把数据从内核拷贝到用户空间。
    * 采用单线程的方式，轮询去监视多个socket，真正有socket读写事件时，才会使用IO资源，大大减少了资源占用（而不是多个socket使用多个线程监听），适合；连接数比较多情况。
    * 另外多路复用IO为 非阻塞IO模型 效率高 因为：NOI不断地询问socket状态是通过用户线程去进行的，而在多路复用IO中，轮询每个socket状态是内核在进行的，这个效率要比用户线程要高的多。

4. RIO 信号驱动式 I/O 模型（signal-driven I/O)

5. AIO 异步 I/O 模型（即AIO，全称asynchronous I/O）
    * 应用程序告知内核启动某个操作，并让内核在整个操作（包括将数据从内核拷贝到应用程序的缓冲区）完成后通知应用程序。
    * https://zhuanlan.zhihu.com/p/36344554
https://zhuanlan.zhihu.com/p/43933717
---

1. 同步 ：执行完才返回结果给调用者
* 阻塞：等待
* 非阻塞：轮询查看
2. 异步：立即f返回结果给调用者
* 非阻塞：全部交给cpu/内核处理，只等待完成信号
---

* 阻塞是指调用方一直在等待而且别的事情什么都不做；
* 非阻塞是指调用方先去忙别的事情。
* 同步处理：同步处理是指被调用方得到最终结果之后才返回给调用方；
* 异步处理：是指被 调用方先返回应答，然后再计算调用结果，计算完最终结果后再通知并返回给调用方。

* 同步/异步： 针对程序和内核的交互
* 阻塞/非阻塞： 关心单个进程的执行状态，调用者

### IO分2个步骤
1. 发起IO请求
2. 执行IO操作
### 区分同步IO还是异步？
* 执行IO操作 是否被阻塞
* 同步IO：需要发起者进行内核态到用户态的数据拷贝过程，所以这里必须由个阻塞。
* 异步IO：的执行者是内核线程，内核线程将数据从内核态拷贝到用户态，所以这里没有阻塞
### 区分阻塞IO还是非阻塞
* 发起IO请求是否被阻塞

---

### io多路复用
* 本质都是同步IO，需要在IO时间就绪后进程自己负责读写，这个过程是阻塞的。
* 而异步IO，是将读写过程完全交给内核。

1. select
    * 效率O(n)，有事件发生，不知道哪几个流，需要轮询找出数据
    * 大量数据拷贝，需要用户空间向内核空间拷贝fd数组，内核遍历fd数组
    * 单进程监听的fd有连接数限制：1024
2. poll
    * 没有最大连接限制，采用链表
    * 其他缺点和select一样
3. epoll
    * epoll也是同步IO的一种
    * 复杂度O(1), 会把哪几个流发生了什么io事件通知，是事件驱动的
    * 使用事件的就绪通知方式，通过epoll_ctl注册fd，一旦fd就绪，内核就用callback回调机制激活fd，epoll_wait收到通知

* epoll优点？
    1. 没有最大并发限制
    2. O(1),不是轮询方式。只有活跃可用的fd才会调用callback。只会管活跃的连接，和总连接数无关。
    3. 即时上百万个总连接，同时活跃的连接数很少。所以效率和活跃的连接数有关。
* 当连接数少且都十分活跃，select和poll可能会更高。
    4. epoll_create() 创建epoll对象；epoll_ctl(), 向epoll对象注册事件
### 什么是EL、LT？
1. ET边缘触发：只有数据来才会有通知
2. 

### 为什么会有ET（边缘触发）？
1. 如果采用LT，一旦有大量不需要读写的fd，每次epoll_wait都会返回，浪费资源。
2. 所以ET是只有资源来的时候触发，就要一次性读完。while循环。
3. 比LT效率高，LT是只要可读就会读。系统不会有大量不关心的就绪fd。
---

### 协程（Coroutines）
1. 比线程更轻量级，一个线程有若干个协程
2. 协程不被 内核管理调度，而完全由 *程序* 控制（也就是在用户态执行）。性能更高，不会由于切换消耗资源。
    * 线程是由内核管理、调度，切换涉及到系统调用，开销比协程大。
3. 由协程执行，在执行A的过程中，可以随时中断，去执行B，B也可能在执行过程中中断再去执行A，结果可能是：
---

### 死锁的四个必要条件
1. 互斥：一个资源每次只能被一个进程持有，资源必须是**临界资源**。比如打印机
2. 不可抢占：已经分配给一个进程的资源不能强制性地被抢占，它只能被占有它的进程显式地释放。
3. 占有和等待：已经得到了某个资源的进程可以再请求新的资源。
4. 环路等待：有两个或者两个以上的进程组成一条环路，该环路中的每个进程都在等待下一个进程所占有的资源。

死锁预防，破坏4个条件？
1. 规定所有进程在开始执行前一次性获取全部资源。
2. 规定在进程在请求资源失败时，释放它获得的所有资源。
3. 破坏环路等待：给资源统一编号，进程只能按编号顺序来请求资源。

死锁避免的基本思想？
* 相比于预防没有那么严格，因为即使死锁的必要条件存在,也不一定发生死锁
1. 系统对进程发出每一个 资源申请 进行动态检查,并根据检查结果决定是否分配资源。如果分配后系统可能发生死锁,则不予分配,否则予以分配。这是一种保证系统不进入死锁状态的**动态策略**。

死锁检测与恢复？
* 检测方法：超时机制和检测是否存在环路。
* 检测死锁之后，可用 进程回退 或者事务回滚等机制，释放获取的资源，之后再重新执行。

---
#### 计算机硬件有两种存储数据的方式：
* 假设低地址到高地址，从左到右。计算机是从低地址按顺序读取字节的。
* 一般操作系统是小端(x86)、通信协议是大端
1. 大端字节序(big endian)：低地址存储的是高位，符合人类的从左向右读取
    * 网络传输，文件存储
2. 小端字节序(little endian)：计算都是从数据的低位开始的，
    * 计算机内部都是小端字节序。

#### 怎么判断机器的字节序
1. 强制类型转换
```
int a = 0x1234;
//char b = (char)a;  这样也可以
char b = *(char*)&a; //相当于取a的低地址部分
if(b == 0x12)  //相同则是大端
```
2. union
```
union node{
    int a;
    char b;
}num;
num.a = 0x1234;
if(num.b = 0x12)  //
```


---
# Linux命令
1. 查看进程 ps -ef/ps aux 
    * ps -ef 能把启动的命令显示全
2. 查看cpu负载均衡
    * top，w，uptime
3. awk 对数据分析并生成报告
    1. 统计词频，并排序
    * awk '{for(i=1;i<=NF;i++) num[$i]++} END{for(k in num) print k, num[k]}' words.txt | sort -nr -k 2
    2. 转置文件
    * awk '{for(i=1;i<=NF;i++) if(NR==1){res[i]=$i}else{res[i]=res[i]" "$i} } \
END{for(i=1;i<=NF;i++) print res[i]}' file.txt

4. sed 编辑，并不改变文件内容。逐行处理
    * sed 's/test/mytest/g' example
5. grep 查找
    1. 有效电话号码，注意括号要转义
        * grep -P "^(\(\d{3}\)\s|\d{3}-)\d{3}-\d{4}$" file.txt
6. 查看端口号
    1. netstat -aon
    2. lsof -i:端口号
7. rsync -azv
8. 查找包含xxx的文件名
    1. find ./ -name "*ist*"
    2. grep -r "*ist*" ./
9. git查询提交记录：git log
10. rebase变基： 

# 数据库
### B树
1. 除了存储指向子节点的索引，还有data域
2. m阶B树，m=3，非叶子节点有2个关键字，3个指针

### B+树?
* https://swsmile.info/2019/06/10/%E3%80%90Database%E3%80%91%E6%95%B0%E6%8D%AE%E5%BA%93%E7%B4%A2%E5%BC%95%E4%B8%BA%E4%BB%80%E4%B9%88%E4%BD%BF%E7%94%A8B-%E6%A0%91/
* 1个索引节点就是1个磁盘块=页的大小=等于节点大小
* 1个扇区为512字节，主存和磁盘之间交换是以n个扇区（页）为单位，通常是8个扇区，4KB
* 预读的长度一般为页（page）的整倍数。页是计算机管理存储器的逻辑块，硬件及操作系统往往将主存和磁盘存储区分割为连续的大小相等的块，每个存储块称为一页（在许多操作系统中，页得大小通常为 4KB），主存和磁盘以页为单位交换数据。
* m阶B+树，xx个关键字，xx个指针

1. 非叶子节点不保存关键字记录的指针，只进行数据索引，可保存更多的索引信息
（非叶节点不包含指向 数据记录 存放地址的指针，没有data域）
2. 叶子节点保存了父节点所有关键字记录的指针
3. 叶子节点间有指针相连，从小到大，方便进行范围查找

https://github.com/julycoding/The-Art-Of-Programming-By-July/blob/master/ebook/zh/03.02.md
### 红黑树?
1. 根节点是黑色的，叶子是黑色的
2. 从跟到叶子的路径上黑色节点数相同
3. 保证最长路径不大于最短路径的2倍

### 为什么STL用红黑树？
1. 最坏情况下，AVL树有最多O(logN)次旋转，而红黑树最多三次

### 唯一索引 主键索引 聚集索引
1. 唯一索引：记录中该字段不能有重复
2. 主键索引：一列或多列的组合，称为主键。为表定义主键，将自动创建主键索引，
3. 聚集索引：记录的物理存储顺序与 索引的顺序 相同。


### 聚集索引，非聚集索引？
1. 聚集索引：表记录的 存储顺序 与 索引的排列顺序 一致。因为物理存储连续，范围查询快；更新慢，因为要维持顺序
    * 规定了物理存储顺序，因此一个表只能包含一个聚集索引。
    * 聚集索引的叶节点就是数据节点，而非聚集索引的叶节点仍然是索引节点，再去聚集索引中查找，只不过其包含一个指向对应数据块的指针。
2. 非聚集索引：都采用B+树，但非聚集索引的叶子层并不与实际的数据页相重叠，而采用叶子包含一个指向 对应数据块的指针。
    * 添加记录不会引起数据顺序的重组
    * 索引中相邻，实际的存储不相邻
    * 非聚集索引的查找：现在索引表中查找主键id，再根据主键id的聚集索引找到相应行。

* 聚簇索引能提高 多行（范围查找？） 检索的速度，而非聚簇索引对于 单行 的检索很快。
* 注意一个表只能有一个聚集索引，但是可以由多个非聚集索引。

### 不同引擎区别?
* https://blog.csdn.net/voidccc/article/details/40077329
1. innodb存储引擎：（数据存储方式是：聚集索引，数据文件本身就是主索引）
    1. 在存储实现上，非叶节点和叶节点结构不一样，因为叶节点直接存储了数据，而不是指针
    * 支持事务、有表锁和行锁、外键。 行级锁大幅度提高多用户并发操作的性能。大量Insert和Update是性能更高。
    1. 是聚集索引（叶子节点存储数据，决定物理存储顺序）。数据文件是和索引绑在一起的，必须要有主键，通过主键索引效率很高。
    2. 存储是只有1个表空间数据文件，一个日志文件
    
2. myISAM存储引擎：(非聚集的，先查索引，通过索引再查表)
    * 不支持事务。只有表级锁。强调性能，每次查询具有原子性。多线程并发访问时，要加表锁，并发性能不高。
    1. 大量select读性能更高。保存表的行数，执行count(*)时直接返回行数。
    2. 是非聚集索引（叶子节点存储，数据地址）。数据文件是分离的，索引保存了数据文件的指针。
    3. 3个文件：1个索引文件、1个数据文件、1个表存储定义文件

### 索引的作用？
1. 避免全表查询，降低CPU,IO资源使用：
    1. CPU：不全表查询则减少了搜索次数、缓存读取次数
    2. IO：内存有限，全表查询则可能多次从磁盘读取，而索引则最多需4次左右
2. innodb中行级锁实现的基础
3. 过多索引会带来维护开销，写数据时需要更新索引树，此处会导致节点分裂、合并，产生开销

### 什么样的字段不适合创建索引
1. 对于那些在查询中很少使用列不应该创建索引
2. 只有很少数据值的列 （比如性别）
3. 当增加索引时，会提高检索性能，但是会降低修改性能。因此，当修改性能远远大于检索性能时，不应该创建索引。
---
### 数据库事务
1. 逻辑上的一组操作，要么完全执行，要么完全不执行。
2. 并发访问时，事务可以在应用程序间提供一个隔离方法；
3. 另一方面，为数据库操作序列提供了从失败恢复正常的方法。

事务的4个特性 ACID?
1. atomicity   原子性: 事务中的操作不可拆分
2. consistency 一致性: 指事务的执行不能破坏数据库的一致性，一致性也称为完整性。一个事务在执行后，数据库必须从一个一致性状态转变为另一个一致性状态。
3. isolation   隔离性: 事务的隔离型是指并发的事务相互隔离，不能互相干扰。
4. durability  持久性: 指事务一旦提交，对数据的状态变更应该被永久保存。

### 事务隔离级别
* 对于同时运行的多个事务,当这些事务访问数据库中相同的数据时,如果没有采取必要的隔离机制。
* 3种并发问题
1. 脏读：      T1读取了T2已更新但没提交的字段，T2回滚，T1读到的就是临时无效的。
2. 不可重复读：事务T1需要读2次；T1读--T2修改--T1读，T1执行相同sql读的的不一致
3. 幻读 ：     事务T1需要读2次；T1读，T2插入或删除，T1读。T1读取搜索若干行。

* 4种隔离级别
1. 读未提交（Read Uncommited): 允许其他事务查到未提交的内容
2. 读已提交（Read Committed): 读到的是提交后的数据，同一事务里，2次相同的select结果不同。
3. 可重复读（Repeatable Read）： 在同一个事务里，SELECT的结果是事务开始时间点的状态，同样的SELECT操作读到的结果会是一致的。但是，会有幻读现象。
（MySQL的默认隔离级别）
4. 串行化  （Serializable）：所有事务只能串行执行

### 实现事务的技术
1. 日志文件(redo log 和 undo log),重做日志
    * redo log: 重做日志记录数据被修改后的信息; redo log是用来恢复数据的 用于保障，已提交事务的持久化特性
    * undo log: 回滚日志，用于记录数据被修改前的信息; undo log是用来回滚数据的用于保障 未提交事务的原子性
2. 锁
3. MVCC(MultiVersion Concurrency Control) 叫做多版本并发控制。

### 数据库中的锁
1. 共享锁（读锁，S锁），select时加
2. 排它锁（独占锁，写锁，X锁），insert, update, delete时加
3. 更新锁（U锁）

### 一致性Hash （Consistent Hashing）
* memcached: 分布式缓存系统。其服务器本身不提供分布式cache的一致性，而是由客户端来提供。
    1. 求出memcached服务器的哈希值，映射到2^32的圆上。 
    2. 同样求出存储数据key的哈希值，并映射到圆上
    3. 从数据映射的位置开始顺时针查找，将数据保存到找到的第一台服务器上。
* 考虑分布式系统中节点会动态的增加、失效，如何保证对外提供良好服务。
* 如果不采用算法保证一致性，有可能所有数据都失效，即hash值改变
#### 分布不均匀，可以用虚拟节点

---
# SQL语句
* left join 左连接查询，主要返回左表信息。再加上on的字段
* join 就是只返回存在关联关系的结果

1. 从一张用户信息表中统计出年龄最大的10个人（limit+order by）
    select person
    from xxx
    order by age desc
    limit 10
2. 查找第二高的薪水
    select
    (select distinct salary
    from employee
    order by salary desc
    limit 1 offset 1)
    as SecondHighestSalary
3. 查找薪水小于1000的员工
    select e.name, b.bonus
    from employee e left join bonus b
    on e.empid=b.empid
    where bonus < 1000 or bonus is null



# Linux
## 编译原理

### 系统调用是什么
1. 操作系统中的状态分为 内核态 和 用户态。
2. 大多数系统交互式操作需求在内核态执行。如设备IO操作或者进程间通信。
3. 用户态为了执行底层功能，如io，需要系统调用

用户态 --> 内核态方式
1. （软中断）系统调用
2. （软中断）异常, 在执行用户态程序时，发生异常。如缺页异常
3. （硬件中断）中断，外围设备按完成用户请求的操作后，会向cpu发送中断信号，cpu会转而执行中断信号的处理程序。比如硬盘读写完成

---
### 编译过程：预处理器、编译器、汇编器、链接器
* 每个符号对应1个函数（函数重载的原理，函数符号会以 函数名+参数，参数作为后缀。对编译器是唯一的）、全局变量、静态变量
* bss段。为了提高空间效率，不需要占用实际磁盘空间。仅仅是占位符。
    * Block Storage Start（块存储开始）, 可简记为Better Save Space
1. 编译器：生成汇编代码
2. 汇编器：生成可重定向目标程序

---
### 虚拟地址空间，主存作为硬盘的缓存
虚拟地址到物理地址的映射：
* MMU(内存管理单元)：每个程序有一个页表，存放虚拟页面到物理页面的映射，如果MMU接收到了程序发出的虚拟地址，在查找相对应的物理页面号时，没有找到，那么将会通过缺页中断来将需要的虚拟页面从磁盘中加载到物理内存的页面中。
---
### 程序和进程区别？
1. 程序是一堆代码和数据，可作为目标文件存在于磁盘上，或作为段存在于地址空间中。
2. 进程是 执行中程序的一个具体实例。 程序总是运行在某个进程的上下文中。fork函数创建新的子进程并运行相同的程序，这个程序可被execve替换。

---
### 创建进程
1. fork: 创建虚拟地址空间，复制父进程的内容。cow但无物理空间，其指向父进程的物理空间
    * 父子进程执行顺序不确定
    * 成功调用fork() 会克隆一个进程。
        * 调用一次，返回两次。  
        * 在子进程中，成功的fork()调用会返回0。在父进程中fork()返回子进程的pid。如果出现错误，fork( )返回一个负值。
    * fork完用execve()载入二进制映像，替换当前进程的映像。派生（fork）了新的进程，而子进程会执行一个新的二进制可执行文件的映像。这种**派生加执行**的方式是很常见的。用新的内存镜像取代原来的内存镜像，当地址空间很大时，复制的操作会很费，而且又是做无用功，所以就产生了vfork()。
    * https://www.cnblogs.com/snake-hand/p/3161450.html

2. vfork: 共享父进程的虚拟空间，没有了复制产生的开销。
    * 用 vfork创建子进程后，父进程会被阻塞直到子进程调用exec或exit。
    * 子进程先运行，可用来创建线程
    * 当创建子进程的目的仅仅是为了调用exec（）执行另一个程序时，子进程不会对父进程的地址空间有任何引用。因此，此时对地址空间的复制是多余的，通过vfork可以减少不必要的开销。你吧从 
    
3. clone: 可以更细粒度地控制与子进程共享的资源，因而参数也更复杂。
用来创建线程
    * 带参数，可选哪些资源复制给子进程
---
静态库、共享库
1. 静态库缺点，维护和更新时，需要重新链接。对于scanf或printf等基本命令，系统中的每个进程都会复制一份到其地址空间中，浪费内存。
2. 共享库（shared library）.so（共享目标）,由动态链接器 动态链接。

### 文件 
* inode（索引节点）: 储存文件元信息的区域。
1. 一个文件占用1个inode，记录文件属性，如权限，所有者，链接数量；记录内容所在的block编号，一个文件会占用多个block
2. 磁盘碎片：文件所在block过于分散，block只能被1个文件使用
3. 文件名存储在 目录文件 中，目录的x属性可修改文件名

### 权限对目录的作用
1. r，能否ls
2. w，能够增删改
3. x，能否cd

1. 不能对目录创建硬链接；
2. 不能对不同的文件系统创建硬链接；
3. 不能对不存在的文件创建硬链接；

### 硬链接，软连接
* inode是一种数据结构，保存了一个文件系统对象的元信息数据，但不包括 数据内容和文件名。
    * 比如有文件的字节数，拥有者，相关权限。链接数，即有多少文件名指向这个inode；文件数据block的位置

1. 硬链接有相同的inode仅文件名不同，文件有相同的inode和data block
2. 不能对目录进行创建
3. 硬链接会增加inode中的链接数（表示有多少个文件名指向该inode）
* 软连接类似快捷方式，存放路径

* 若1个inode号对应多个文件名，则为硬链接，即硬链接就是同一个文件使用了不同的别名,使用ln创建。
* 若文件中存放的是另一个文件的路径，是软连接。软连接是一个普通文件，有自己独立的inode。ln -s

### mmap
### 使用了内存映射的 文件读 / 写 操作，只需一次拷贝
https://www.jianshu.com/p/719fc4758813
https://www.cnblogs.com/huxiao-tee/p/4660352.html#_label1
* 位于堆和栈中间区域，将 磁盘区域 映射到进程的 虚拟地址 空间。
* Memory-map，内存映射, 是一种文件I/O操作, 它将文件(linux世界中, 设备也是文件)映射到内存中, 用户可以像操作内存一样操作文件。
* 进程可采用指针读写操作这一段内存（而系统会自动回写脏页面到对应的文件磁盘上），完成了对文件的操作而不必再调用read,write等系统调用函数。相反，内核空间对这段区域的修改也直接反映用户空间，从而实现不同进程间的文件共享。
1. **常规文件操作为了提高 读写效率 和 保护磁盘，使用了页缓存机制**。 读文件时先将文件页从 磁盘 拷贝到 页缓存 中，*由于页缓存处在内核空间*，需将 页缓存中 数据页再次拷贝到内存对应的 用户空间 中。通过2数据拷贝，完成进程对文件的获取。写操作也是一样，待写入的buffer在内核空间不能直接访问，必须要先拷贝至内核空间对应的？？主存，再写回磁盘中（延迟写回），需要两次数据拷贝。
2. 使用mmap时，创建 新的虚拟内存区域(映射区域) 和 建立文件磁盘地址和虚拟内存区域映射 这两步，没有任何文件拷贝操作（真正的文件读取是当进程发起读或写操作时）。而访问数据时发现内存中并无数据而发起的缺页异常过程，可以通过已经建立好的映射关系，只用1次数据拷贝，从磁盘中将数据传入内存的用户空间中，供进程使用。
  
### mmap总结  
1. 对文件的读取操作跨过了 页缓存，减少了数据的拷贝次数，用**内存读写取代I/O读写**，提高了文件读取效率。
2. 实现了 **用户空间和内核空间** 的高效交互方式。两空间的各自修改操作可以直接反映在映射的区域内，从而被对方空间及时捕捉，提高交互效率。
3. 提供进程间共享内存及通信的方式。不管是父子进程还是无亲缘关系的进程，都可以将自身用户空间映射到同一个文件或匿名映射到同一片区域。从而通过各自对映射区域的改动，达到进程间通信和进程间共享的目的。
4. 可用于实现高效的大规模数据传输。内存空间不足，是制约大数据操作的一个方面，往往借助硬盘空间协助操作，补充内存的不足。但是进一步会造成大量的文件I/O操作，极大影响效率。可以通过mmap映射很好的解决。凡是需要用磁盘空间代替内存的时候，mmap都可以发挥其功效。

### 位于内核空间的页缓存
* page cache：页缓存是面向文件，面向内存的。通俗来说，它位于内存和文件之间缓冲区，文件IO操作实际上只和page cache交互。
* 如果页缓存命中，那么直接返回文件内容；
* 如果页缓存缺失，那么产生一个页缺失异常，创建一个页缓存页，同时通过inode找到文件该页的磁盘地址，读取相应的页填充该缓存页；重新进行第6步查找页缓存；
* 一个页缓存中的页如果被修改，那么会被标记成脏页。

###
mmap与文件操作的区别
linux还有一套文件操作的系统调用, 即open read write, mmap与他们区别在哪儿呢?
* 普通文件读写操作, 需要借助处于 内核空间的页缓存 作为中转, 将数据载入主存;
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
### IPC（进程间通信有7种）
1. 匿名管道：实质是一个内核缓冲区，内容是无格式字节流。只能用于父子进程，单向的，1个读，1个写
2. 命名管道：以有名管道的 文件形式 存在于文件系统中，即使不存在亲缘关系的进程，只要可以访问该路径，就能相互通信。
3. 信号（signal）
4. 信号量（semaphore）:用于进程间或线程间的同步手段。
    1. P, 如果信号量的值大于0，就-1。如果为0，就挂起进程
    2. V, 如果有其他进程因等待该信号量而挂起，则将其唤醒；如果没有，就+1
5. 共享内存：（用信号量解决同步问题，PV操作）
6. 消息队列
7. socket： 用于不同机器间的通信


* https://www.jianshu.com/p/c1015f5ffa74
进程间通信方式，共享内存是最快的一种IPC方式
* 管道需要复制4次 
* 共享内存复制2次

### 线程间的同步与互斥（通信）
1. 锁，条件变量（锁保护的就是条件变量），信号量
2. 信号
3. 共享内存

### 为什么条件变量要加锁？
* 防止唤醒丢失。wait外部
* wait内部： 判断“条件”是否成立，在 lock, unlock 之间，受到保护。pthread_cond_wait 可能会阻塞。假如真的阻塞，mutex 这锁就一直不能被释放了。因此在 pthread_cond_wait 的实现内部，在阻塞之前，必须先将锁释放。在唤醒的时候，再重新获取锁。
* wait内部会释放锁；其他线程拿到才能拿到锁，去唤醒

### 管程
* 把同步、互斥复杂的进行封装
* 对信号量PV操作进行封装。以方便、简洁方式用来实现进程的同步、互斥
* 提供的接口：insert_item(); remove_item()

# 算法
1. 字符串替换、删除字符串。memcpy。
用两个指针分别指向new、old的头部或尾部
    * 这类题原地操作会发生覆盖情况
    * 压缩，删除-->从前向后
    * 扩充      -->从后向前


1. 所有出栈顺序可能性：
回溯
2. 判断出栈顺序是否合法：https://www.cnblogs.com/bendantuohai/p/4680814.html
    * 模拟一遍
    * 若a[i+1] > a[i]，则a[i+1] 大于前面所有的数。如果按小到大出来的数，必定是进去就出来，那么比它们还大的数，就不可能比他们还早出来。
3. 两个有序数组求中位数，logn，
https://www.cnblogs.com/TenosDoIt/p/3554479.html。
https://blog.csdn.net/hk2291976/article/details/51107778
非递归
4. 蓄水池算法，使每个元素都有k/n概率被选中。
第k+1个元素，概率k/(k+1),当前并不知道n有多大
5. 5*rand5() + rand5()：取值范围在[0, 24]，取每一个值的概率达到完美的相等；
 if x < 21 return x%7
6. 排序：
    * 归并排序，空间复杂度logn，可以递归或非递归，因为其在递归过程中不需要记录返回信息。
    * 快速排序，需要使用递归栈，平均复杂度logn
    * 堆排序，升序排列，建立大根堆；空间复杂度O(1);？？建堆的时间复杂度O(n)
7. 25匹马中选出跑得最快的3匹，每次只有5匹马同时跑，最少要比赛几次？
    * 最少比7次，
    * 第6次，A1>B1>C1>D1>E1。能选出第一名
    * 第7次，A2, A3, B1, B2, C1
8. 区间DP，两端取石子
    * 取数字是从外向里，我们枚举状态从里向外
    * dp[i][j] = max((s[i][j] - dp[i][j-1]), s[i][j] - dp[i+1][j]));
    * 表示P能取得的最大值，是从上一步Q能取得的最大值转移过来的
    * 最优子结构

# 项目
### docker和vm区别
* 容器只是虚拟了操作系统，而不像vm虚拟出计算机硬件。
1. vm: 虚拟出硬件，需要构建完整操作系统。
    * 并利用Hypervisor虚拟化CPU、内存、IO设备等
2. docker: 对操作系统复用。直接利用宿主机资源，启动docker等于启动进程
    * 用namespace做资源隔离; cgroups做资源限制，限制资源使用的最大值
    * 利用Linux内核特性实现隔离，运行容器就等同于启动进程
    * 利用宿主机资源，和宿主机共用1个内核

3. 容器解决了本地依赖问题，把应用程序和其他相关内容打包。包括代码、依赖。

* docker是一种CS架构模型，通过 客户端 连接到 守护进程

#### 怎么进入docker？
1. docker attach
2. docker exec -it 775c7c9ee1e1 /bin/bash  

#### 有几种namespace？
1. IPC
2. PID
3. network
4. mount: 挂载点、文件系统
5. user: 用户、用户组
6. UTS：主机名，域名

Map-Reduce：
1. 分为map，reduce两个操作，map读、处理数据；reduce是对数据进行整合

MPI优点：
* 可移植性，易用性好。有很多异步通信功能
---
### TKE
1. 资源、pod挂掉了，考虑迁移？
    * 看资源是否有状态，无状态资源可直接创建新的，这样花费时间少
    * 如果有状态，需要把旧pod迁移过来，这样花费时间多，但可保留状态
2. TKE架构图，各个组件
3. 如何调度？调度器策略？抢占式调度？
4. 调度时资源如何计算的，最小最多资源？
5. 解决bug：监控ectd指标失效。问了其他负责人，有个token模块被改掉了，把错误信息官网查。发现认证有问题，单独搞了个对ectd的证书认证。调试时候编辑deplyment，sleep，手动进入pod里面调试，挂载证书验证，测试使用证书是否成功。通过hostPath形式对证书挂载，修改代码。该了容忍，和新亲和性，调度其到master节点，etcd是部署在master节点上的。
6. 控制器多线程访问时，多个同时删除怎么处理？
* 加锁，多某个资源加锁
7. etcd
    * 高可用、强一致性的服务发现存储仓库
    * 使用了raft算法 

### PBS
0. PBS一种常见的集群作业调度系统
    * openPBS,收费
    * PBS pro，商业版
    * Torque, 开源的
1. PBS架构，如何工作的
* pbs_sched, 管理节点，负责调度。TORQUE集成一个简单的FIFO调度策略
* pbs_server, 管理节点，负责与mom通讯，对资源管理
* pbs_mom，作业执行器。当从服务器接收1个作业拷贝就将其放入执行队列
2. 调度器策略，
3. 如何监控进程
4. 亲和性

### 高性能计算
* 分支预测器能提高效率的原因->在于 指令的流水线化，是一种增加指令吞吐量的方法。即单位时间内提高同时执行指令的个数。-》流水线需要确定执行的序列和先后顺序，这样才会使流水线中充满指令。
1. GPU区别
    * GPU: 小cache，小control(没有分支预测)。 众多ALU。擅长大规模并发通用计算。核非常多。
        * throughput oriented。流处理器数量多，实现大量线程并行，使得同时走一条指令的数据变多，从而提高数据的吞吐量。
    * CPU：擅长逻辑控制，分支预测，cache，串行运算。
        * latency oriented。主频高，使得处理每条指令所需时间短、从而降低具有复杂跳转分支程序执行所需的时间。
2. 什么样的程序适合在GPU上运行？
    1. 计算密集型，即大部分时间花在寄存器运算上。
    2. 易于并行的程序，gpu是一种SIMD架构，有众多的核，可在同一时间做同样事情。

### golang
1. 垃圾回收
    1. 采用有向图方式管理内存。实习监控对象是否可达，如不可达，将其回收。
    2. 引用计数
2. gorouting调度器
3. slice和数组
4. 
# Special计划

### 海量数据处理
### 消息队列
1. 解耦、异步、削锋
2. 增加缓冲，提高系统响应速度
缺点：
1. 可用性降低，系统复杂性增加
### 连接池
### 线程池
* 同时维护多个线程，避免处理短时任务创建与销毁线程的开销
### rpc
### protobuf
* 是一种轻便高效的结构化数据存储格式，一种序列化传输协议。用于序列化
    * 可类比 xml, json, thrift
    * 于其他相比，性能更高
        1. 信息的表示紧凑，体积小
        2. 封解包的过程，Varint 是一种紧凑的表示数字的方法。它用一个或多个字节来表示一个数字，值越小的数字使用越少的字节数。这能减少用来表示数字的字节数。
1. 编写.proto文件，编译。定义了需要处理的结构化数据
2. 编写writer和reader
### 秒杀系统
1. redis做缓存。
2. 引入队列，然后将所有写DB操作在单队列中排队，完全串行处理。当达到库存阀值的时候就不在消费队列，并关闭购买功能。这就解决了超卖问题。

### 负载均衡
### 正则表达式
1. IP地址
* (2(5[0-5]|[0-4]\d)) 表示200——255
* [0-1]?\d{1,2} 表示0~199
* (\.(2(5[0-5]|[0-4]\d))|[0-1]?\d{1,2}){3}
* 
* ？表示前面内容重复出现0或1次
*
### 设计模式
1. 单例模式
    * 保证一个类仅有一个实例，并提供一个该实例的全局访问点
    * 利用静态局部变量，返回引用。
    * 局部静态变量不仅只会初始化一次，而且还是线程安全的。
2. 工厂模式
    * 用工厂方法代替new操作的一种模式。
    * 主要作用是解耦合



### 聊人生
1. 性格缺点
    1. 实习发日报，总结不细致。周报，写的不详细。把一天中的问题，解决思路，写到日报里
    2. 对于不喜欢的工作，没太多热情。-》先做不喜欢的，才能做喜欢的事情。