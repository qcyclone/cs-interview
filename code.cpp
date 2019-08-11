0. 二分查找
//左闭右开
int binery_search(int a[], int len, int key){
    int st = 0, ed = len;
    while(st < ed){
        mid = st + (ed - st) / 2;
        if (a[mid] < key)
            st = mid + 1;
        else
            ed = mid;
    }
    return st;
}

1. 快速排序
//简洁的方法
void quick_sort(int a[], int left, int right){
    if(left > right)
        return;
    int tmp = a[left];
    int i = left, j = right;
    while(i < j){
        //顺序很重要，要先从右边开始找
        while(a[j] >= tmp && i < j)
            j--;
        //要有等号，很关键
        while(a[i] <= tmp && i < j)
            i++;
        if(i < j){
            swap(a[i], a[j]);
        }
    }
   //注意要和a[left]换，不能是tmp 
    swap(a[left], a[i]);
    //使用了闭区间
    quick_sort(a, left, i - 1); 
    quick_sort(a, i + 1, right);
}

int main(){
    int a[] = {0, 1 ,4,3};
    //左闭右闭区间
    quick_sort(a, 0, 3);
    for(int i =0 ;i<4;i++){
        cout<<a[i]<<" ";
    }
    cout<<endl;
    return 0;
}

//填坑法
void quick_sort(int arr[], int left, int right)
{
    if(left >= right)
        return;
    int i = left, j = right, target = arr[left];
    while (i < j)
    {
        while (i < j && arr[j] >= target)
            j--;
        if (i < j)
            arr[i++] = arr[j];
        while (i < j && arr[i] <= target)
            i++;
        if (i < j)
            arr[j--] = arr[i];
    }
    arr[i] = target;
    quick_sort(arr, left, i - 1);   //分割完后，都不包括分割点
    quick_sort(arr, i + 1, right);
}

1. 归并排序
//将有二个有序数列a[first...mid]和a[mid + 1...last]合并。
void mergeArray(int a[], int first, int mid, int last, int temp[]){
    int i = first, j = mid + 1;
    int m = mid, n = last;
    int k = 0;
    while( i <= m && j <= n){
        if(a[i] <= a[j])
            temp[k++] = a[i++];
        else
            temp[k++] = a[j++];
    }
    while( i <= m)
        temp[k++] = a[i++];
    while( j <= n)
        temp[k++] = a[j++];
    for(int i = 0; i < k; i++){
        a[first + i] = temp[i];
    }
}
void mergeSort(int a[], int first, int last, int temp[]){
    if(first < last){
        int mid =  first + (last - first) / 2;
        //递归左边
        mergeSort(a, first, mid, temp);
        //递归右面
        mergeSort(a, mid + 1, last, temp);
        //递归结束后，合并过程
        mergeArray(a, first, mid, last, temp);
    }
}


//BST左子树 所有 节点都比根小，右子树 所有 节点都比根大
1.判断是否为BST
bool isBST(node *root, int x, int y){ //判断是否为BST, 返回条件一个true，一个false
    if(root == null)//
        return true;
    if(root.val < x || root.val > y)//如果中间有不符合条件的，直接返回,不用再判断后面
        //我不需要判断符合的，只需要判断不符合条件的
        return false;
    return isBST(root.l, x, root.val) &&  isBST(root.r, root.val, y);
}

int pre;
bool isBST(node *root){ //递归中序遍历，没有临时数组
    if(root == null)
        return true;
    if(! isBST(root->left)) return false;
    if(root->val > m)
        m= root->val;
    else
        return false;
    if(! isBST(root->right))    return false;
    return  ture; //如果上面判断都没生效，最后执行这句
}

long long pre = -1e10;
bool isValidBST(TreeNode* root) {
    if(root==NULL)  return true;
    bool l_res = isValidBST(root->left);
    if(l_res==false)   //看左边是否符合条件
        return false;
    // if(!isValidBST(root->left))
    //     return false;
    if(root->val <= pre)  return false;//看跟是否符合条件
    pre=root->val;
    return isValidBST(root->right);//看右侧是否符合条件
}
};

2. 非递归中序遍历
vector<int> inorderTraversal(TreeNode* root) { //非递归
    vector<int> ans;
    //if(root == NULL)    return ans;
    stack<TreeNode*> s;
    while(root!= NULL || !s.empty()){  // 左根右，先一直往左，再访问根和右
        while(root!=NULL){
            s.push(root);
            root=root->left;
        }
        root = s.top();
        ans.push_back(root->val);
        s.pop();
        root = root->right;
    }
    return ans;
}

3.递归中序遍历
void inOrder(node* root){ //递归
    if(root == null)
        return;
    inOrder(root.left);
    res.push_back(root->val);
    inOrder(root.right);
}
4.非递归先序遍历
void preOrder(node* root) {//非递归
    stack<int> s;
    s.push(root);
    while(!s.empty()){
        node = s.pop();
        if(node==null) continue;
        ans.push_back(node.val);
        s.push(node.right);
        s.push(node.left);
    }
}

5. 二叉树中序遍历的下一个节点
1) 有右子树，下一个就是 右子树 中最左节点
2) 如没右，如果cur是左子树，那么就是它父节点
                 右子树，向上找，直到找到一个左分叉，答案就是它的父节点

node* getNext(node* root){
    if(root==null)  return null;
    node* ans = null;
    if(root->r != null){
        node* pRight = root->r;
        while(pRight->l != null)
            pRight=pRight->l;
        ans = pRight;
    }eles if(root->p != null){ //没右节点，找父节点
        node* pCur = root;
        node* pParent = root->p;
        while(pParent != null && pCur == pParent->r){//如果当前是右子树
            pCur = pParent;
            pParent=pParent->p;
        }
        ans = pParent;//退出的条件就是当前是左分叉
    }
    return ans;
}
6.快速幂
double power(double base, int exponse){
    if(exponse==0)  return 1;
    if(exponse==1) return  base;
    double result=power(base, exponse >> 1);
    result*=result;
    if(exponse%2==1)
        result*=base;
    return  result;
}


7.BST每个节点都加上其后所有节点
遍历顺序为 右根左
int sum=0;
void addBiger(node* root){
    if(root == null)
        return;
    addBiger(root->r);
    root.val += sum;
    sum += root.val;
    addBiger(root->l)
}

8. 二叉树的最近公共祖先
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
    //有可能左边找不到，所以会有空的情况
    if(root==NULL || root==p || root==q)
        return root;
    TreeNode* left = lowestCommonAncestor( root->left,p, q);
    TreeNode* right = lowestCommonAncestor( root->right,p, q);
    if(left!=NULL && right != NULL)
        return root;
    if(left==NULL)
        return right;
    return left;
}
BST的最近公共祖先
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        //只向一边去找，所以肯定会找到
        if(root->val == p->val || root->val == q->val)
            return root;
        if(p->val < root->val && q->val > root->val)
            return root;
        if(q->val < root->val && p->val > root->val)
            return root;
        if(p->val < root->val && q->val < root->val)
            return lowestCommonAncestor(root->left, p, q);
        return lowestCommonAncestor(root->right,p, q);
}

9. BST中第K大的数字
//需要用到全局变量，因为左下角递归到时并不是第一个k==0
class Solution {
    int time = 0;
public:
    int kthSmallest(TreeNode* root, int k) {
        if(root == NULL) return -1;
        int ans;
        ans = kthSmallest(root->left, k);
        if(time == k)//已经找到了答案
            return ans;
        if(++time == k)//当前是答案
            return root->val;
        return kthSmallest(root->right, k);
    }
};
//没用到全局变量，k是引用，也相当于是全局变量吧
class Solution {
public:
    int kthSmallest(TreeNode* root, int& k) {
        if(root==NULL)  return -1;
        int ans = -1;
        ans = kthSmallest(root->left, k);
        if(ans != -1)
            return ans;
        if(k == 1)//找到了答案就要返回，否则会向右找，找不到答案(-1)
            return ans = root->val;
        k--;   
        return kthSmallest(root->right, k);
    }
};
//没用到全局变量，二分的思路, 复杂度nlogn
class Solution {
public:
    int getNodeNum(TreeNode* root){
        if(root == NULL)  return 0;
        return getNodeNum(root->left) + getNodeNum(root->right) + 1;
    }
    int kthSmallest(TreeNode* root, int k) {
        if(root == NULL)
            return -1;
        int leftNums = getNodeNum(root->left);
        if(leftNums == k-1)
            return root->val;
        else if(leftNums > k-1)
            return kthSmallest(root->left, k);
        return kthSmallest(root->right, k-leftNums-1);
    }
};

10. 对称二叉树 -> 判断两个树是否镜像
//递归，时间 空间 o(N)
//判断两个树, 一棵树只能向一个方向遍历，不能同时兼顾两侧
//所以这里要用两个递归过程
public:
bool isSymmetric(TreeNode* root) {
    return isMirror(root, root);
}
bool isMirror(TreeNode* t1, TreeNode* t2){
    if(t1==NULL && t2==NULL)
        return true;
    if(t1==NULL || t2==NULL)
        return false;
    return t1->val == t2->val &&
           isMirror(t1->left, t2->right) &&
           isMirror(t1->right, t2->left);
}

//也是同时处理两棵树
//非递归，时间 空间 o(N)
//有点类似于层次遍历
public boolean isSymmetric(TreeNode root) {
    Queue<TreeNode> q = new LinkedList<>();
    q.add(root);
    q.add(root);
    while (!q.isEmpty()) {
        TreeNode t1 = q.poll();
        TreeNode t2 = q.poll();
        if (t1 == null && t2 == null) continue;
        if (t1 == null || t2 == null) return false;
        if (t1.val != t2.val) return false;
        q.add(t1.left);
        q.add(t2.right);
        q.add(t1.right);
        q.add(t2.left);
    }
return true;
}
10.1 输出一棵二叉树的镜像，剑指

class Solution {
public:
    void Mirror(TreeNode *pRoot) {
        if(pRoot == nullptr)
            return;
        TreeNode* tmp = pRoot->left;
        pRoot->left = pRoot->right;
        pRoot->right = tmp;
        Mirror(pRoot->right);
        Mirror(pRoot->left);
    }
};

11. 树的最小/最大 深度

非递归-->层次遍历
public static int minDeep(BTNode node) {
        //如果为空
        if(node == null){
            return 0;
        }
        int minDeep = 0, width;
        ArrayDeque<BTNode> queue = new ArrayDeque<>();
        //根节点入队
        queue.add(node);
        while (!queue.isEmpty()) {
            width = queue.size();
            //若每一层的宽度大于maxWidth，则重新赋值
            minDeep += 1;
            //注意这里循环的次数是width,出队的仅仅是每一层的元素
            for (int i = 0; i < width; i++) {
                BTNode nodeTemp = queue.poll();
                //左右均为空表明是叶子结点
                //如果是最小，就多了这一句。及时返回
                if(nodeTemp.rightChild == null && nodeTemp.leftChild == null){
                    return minDeep;
                }
                if (nodeTemp.leftChild != null) {
                    queue.add(nodeTemp.leftChild);
                }
                if(nodeTemp.rightChild != null) {
                    queue.add(nodeTemp.rightChild);
                }
            }
        }
        return minDeep;
    }
public:
int minDepth(TreeNode* root) {
    if(root==NULL)  return 0;
    int l = minDepth(root->left);
    int r = minDepth(root->right);
    if(l == 0||r == 0) //如果有一侧没有子树，则高度为另一侧
        return max(l,r) + 1;
    return min(l,r) + 1;
}
class Solution {
public:l
    int maxDepth(TreeNode* root) {
        if(root==NULL)
            return  0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
16. !!树的直径
//实际就是求最大深度时，对于每个节点对左右子树的深度加和并求最大值
class Solution {
public:
    int ans = 0;
    int depth(TreeNode* node){
        if(node ==  NULL)
            return 0;
        int l = depth(node->left);
        int r = depth(node->right);
        ans = max(ans, l + r);
        //在最后返回的时候，才加上当前节点的高度
        //这里算直径是边的长度，没有加上 根节点的1 正好
        return max(l, r) + 1;
    }
    int diameterOfBinaryTree(TreeNode* root) {
        depth(root);
        return ans;
    }
};

12. 二叉树展开为单链表
//顺序为 右左根，想象成一个栈的顺序
class Solution {  
public:
    TreeNode* pre=NULL;
    void flatten(TreeNode* root) {
        if(root==NULL)
            return ;
        flatten(root->right);
        flatten(root->left);
        root->right = pre;
        root->left = NULL;
        pre = root;
    }
};

13. 有序链表 --> BST，快慢指针

    TreeNode* sortedListToBST(ListNode* head) {
        return my(head,nullptr);
    }
    TreeNode* my(ListNode* head,ListNode* end){
       if(head == end) {  
           return nullptr;
       }
        ListNode* one = head;
        ListNode* two = head;
        while(two!= end && two->next != end){  //都要判断快指针
            one = one->next;
            two = two->next->next;
        }
        TreeNode* root = new TreeNode(one->val);
        root->left = my(head, one);
        root->right = my(one->next, end);
        return root;
    } 


14. BST->双向链表
class Solution {
public:
    TreeNode* pre = NULL;
    TreeNode* head = NULL;//需要返回的头结点
    TreeNode* Convert(TreeNode* pRootOfTree)
    {
        visit(pRootOfTree);
        return head;
    }
    void visit(TreeNode* node){
        if(node==NULL)
            return;
       visit(node->left);
       create(node);
       visit(node->right);
    }
    void create(TreeNode* node){
         node->left = pre;
        if(pre == NULL)
            head = node;
       else
           pre->right = node;
        pre = node;
    }
};

15. 合并二叉树
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
        if(t1==NULL && t2==NULL)
            return NULL;
        TreeNode* node = new TreeNode(0);
        if(t1 != NULL)
            node->val += t1->val; 
        if(t2 != NULL)
            node->val += t2->val;
        //注意，如果为空了，就没有left指针
        node->left = mergeTrees(t1==NULL?NULL:t1->left, t2==NULL?NULL:t2->left);
        node->right = mergeTrees(t1==NULL?NULL:t1->right, t2==NULL?NULL:t2->right);
        return node;
    }
};


17. min栈
每次入栈和栈头部元素比较。
void push(int x) {
    if(a.empty()){
        a.push(x);
        b.push(x);
    }else{
        a.push(x);
        b.push(min(x,b.top()));
    }
}

18. 队列的最大值，滑动窗口
越后面新来的元素，如果较大，肯定会覆盖前面的元素
每次从队列头取最大值，递减队列，并且存储下标，方便知道哪个元素离开窗口
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& num, int k) {
        vector<int> ans;
        deque<int> win;
        int len = num.size();
        for(int i = 0;i < len;i++){
            //注意要判断为是否为空, 超出了窗口范围
            if(!win.empty() && i - win.front() >= k)
                win.pop_front();
            while(!win.empty() && num[i] > num[win.back()])
                win.pop_back();
            win.push_back(i);
            if(i >= k - 1)
                ans.push_back(num[win.front()]);
        }
        return ans;
    }
};

19. 模拟约瑟夫环
class Solution {
public:
    int LastRemaining_Solution(int n, int m){
        if(n<1 || m<1)    return -1;
        vector<int> vis(n, 0);
        int st = 0;
        int ans = -1;
        for(int i = 0;i < n;i++){
            int cur = 0 ;
            while(cur < m){
                if(vis[st] == 0){
                    cur++;
                    if(cur == m){
                        ans = st;
                        vis[st] = 1;
                    }
                }
                st = (st + 1) % n;
            }
        }
        return ans;
    }
};

20. 乘积数组
**vector不指定长度，不能够随机访问
class Solution {
public:
    vector<int> multiply(const vector<int>& A) {
        int len = A.size();
        vector<int> l(len);
        vector<int> r(len);
        vector<int> ans(len);
        if(len < 1)
            return ans;
        l[0] = A[0];
        for(int i = 1;i < len;i++){
            l[i] = l[i-1] * A[i];
        }
        r[len-1] = A[len-1];
        for(int i=len-2;i>=0;i--){
            r[i] = r[i+1] * A[i];
        }
        ans[0] = r[1];
        ans[len-1] = l[len-2];
        for(int i = 1;i < len-1;i++){
            ans[i] = l[i-1]*r[i+1];
        }
        return ans;
    }
};

21.旋转数组找最小值
int findMin(vector<int>& nums) {
    int l = 0;
    int r = nums.size() - 1; //右闭区间，因为下面要获取nums[r]
    int mid;
    while(l < r){
        mid = (l+r)/2;
        if(nums[mid] < nums[r])
            r= mid;
        else
            l = mid+1;
    }
    return nums[l];
}

22.旋转数组找最小值,有重复元素
int findMin(vector<int>& nums) {
    int l = 0;
    int r = nums.size() - 1; //右闭区间，因为下面要获取nums[r]
    int mid;
    while(l < r){
        mid = (l+r)/2;
        if(nums[mid] < nums[r])
            r = mid;
        else if(nums[mid] > nums[r])
            l = mid+1;
        else
            r--; 
    }
    return nums[l];
}
23.搜索旋转数组
int search(vector<int>& nums, int target) {
    if(nums.size() == 0 )
        return -1;
    int l = 0;
    int r = nums.size() - 1; //右闭区间，因为下面要获取nums[r]
    int mid;
    while(l <= r){   //等于
        mid = (l+r)/2;
        if(nums[mid] == target)
            return mid;
        if(nums[mid] < nums[r]){
            if(target <= nums[r] && nums[mid] < target)//只有一个等号
                l = mid + 1;
            else
                r = mid;   //-1也可以，可能是因为循环内有返回
        }else{
            if(target > nums[r] && nums[mid] > target)
                r = mid;
            else
                l = mid + 1;  
        }     
    }
    return -1;
}

24.搜索旋转数组，有重复值
bool search(vector<int>& nums, int target) {
    if(nums.size() == 0 )
     return false;
    int l = 0;
    int r = nums.size() - 1; //右闭区间，因为下面要获取nums[r]
    int mid;
    while(l <= r){
        mid = (l+r)/2;
        if(nums[mid] == target)
            return true;
        if(nums[mid] < nums[r]){
            if(target <= nums[r] && nums[mid] < target)
                l = mid + 1;
            else
                r = mid;   //-1也可以，可能是因为循环内有返回
        }else if(nums[mid] > nums[r]){
            if(target > nums[r] && nums[mid] > target)
                r = mid;
            else
                l = mid + 1;  
        }else
            r--;
    }
    return false;
}

25. 分割链表
头部各创建一个临时节点，注意这样实际是在原来链表空间上操作，会破坏原来链表。
头指针不变，要作为返回值。再创建一个指针用于遍历；创建链表时不能为空。
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode* l = new ListNode(0);
        ListNode* r = new ListNode(0);
        ListNode* p = l;
        ListNode* q = r;
        ListNode* tmp =  head;
        while(tmp != NULL){
            if(tmp -> val < x){
                p->next = tmp; 
                p = p -> next;
            }else{
                q -> next = tmp; 
                q = q -> next;
            }
            tmp = tmp->next;
        }
        q->next = NULL; //注意要断开，否则会有死循环
        p->next = r->next;
        return l->next;
    }
};
25.2 创建额外节点，不破坏原链表
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        ListNode* l = new ListNode(0);  //创建一个临时节点，方便
        ListNode* p = l;
        ListNode* tmp =  head;
        while(tmp != NULL){
           if(tmp->val < x){
               p -> next = new ListNode(tmp->val);
               p = p -> next;
           }
           tmp = tmp -> next;
        }
        tmp = head;
        while(tmp != NULL){
           if(tmp->val >= x){
               p -> next = new ListNode(tmp->val);
               p = p -> next;
           }
          tmp = tmp -> next;
        }
        return l->next;
    }
}; 

26. !!反转链表
掉转指针
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        ListNode* pre = NULL;
        ListNode* cur = head;
        ListNode* tmp = NULL;
        while(cur != nullptr){
            tmp = cur->next; //查找下一个节点
            cur->next = pre; //反转操作

            pre = cur;  //继续向后遍历
            cur = tmp;
        }
        return pre;
    }
};

26.2 递归法
class Solution {
public:
    ListNode* ans = NULL;
    ListNode* fun(ListNode* cur){
        if(cur == nullptr)
            return nullptr;
        if(cur -> next == nullptr) 
            ans = cur;
        ListNode* tmp = nullptr;
        if(cur->next != nullptr){
            tmp = fun(cur -> next);
            tmp -> next = cur;
        }
        return cur;
    }
    ListNode* reverseList(ListNode* head) {
        ListNode* tmp = fun(head);
        if(tmp!= nullptr)
            tmp->next = nullptr;
        return ans;
    }
};
26.3 !!简洁递归法，
ListNode* reverseList(ListNode* head) {
    //第一个判断防止输入就是null, 最后判断条件判断尾结点
    if(head == nullptr || head->next==nullptr)  
        return head;
    ListNode *p = reverseList(head -> next);
    head -> next -> next = head;
    //很重要，否则会死循环；后面的节点被覆盖，只允许覆盖空节点
    head -> next = nullptr; 
    return p;
}

27. 堆排序，堆是种特殊的完全二叉树，是一种选择排序。
最坏，最好，平均时间复杂度均为O(nlogn)，它也是不稳定排序。
void adjustHeap(int *arr, int i, int len){
    int tmp = arr[i];
    for(int k = 2*i+1; k < len; k = k*2+1){
        if(k+1<len && arr[k] < arr[k+1])
            k++;
        if(tmp < arr[k]){//每次要用tmp比较
            arr[i] = arr[k];
            i = k; //记录前一次到达的节点i
        }else{
            break;
        }
    }
    arr[i] = tmp;
}
void sort(int *arr,int len){
    //建堆 复杂度 O(N)
    for(int i = len/2-1; i>=0; i--){
        adjustHeap(arr, i, len);
    }
    for(int i = len - 1;i > 0; i--){
        swap(arr[0], arr[i]);
        adjustHeap(arr, 0, i); //每次范围减小
    }
}
//大根堆,从小到大排序
int main(){
    int arr[] = {8,7,3,6,2,4,9,1,5};
    sort(arr, 9);
    for(int i=0;i<9;i++)
        cout<<arr[i]<<" ";
    return 0;
}

28. lc1. 两数之和
map的用法，k-v不要反
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res;
        unordered_map<int, int> mp;
        int len = nums.size();
        for(int i = 0;i < len; i++){
            mp[ nums[i] ] = i; //k-v
        }
        for(int i = 0;i < len; i++){
            auto index = mp.find(target - nums[i]);
            if(index != mp.end() && index->second != i){
                //注意判断 两个数不能相同
                res.push_back(i);
                res.push_back(index->second);
                break;
            }
        }
        return res;
    }
};

29. lc2. 两数相加
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        //声明一个头结点，然后cur指向头结点，每次cur移动作为索引
        //最后返回 head -> next
        ListNode *head = new ListNode(0);
        ListNode *cur = head;
        
        int carry = 0;
        while(l1 != nullptr || l2 != nullptr){
            int a,b;
            a = 0;
            b = 0;
            if(l1 != nullptr){
                a = l1->val;
                l1 = l1->next;
            }
            if(l2 != nullptr){
                b = l2->val;
                l2 = l2->next;
            }
            ListNode *tmp = new ListNode((carry + a + b) % 10);
            if(carry + a + b >= 10)
                carry = 1;
            else
                carry = 0;
            
            cur -> next = tmp;
            cur = cur->next;
        }
        if(carry == 1)
           cur -> next = new ListNode(1);
        return head->next;
    }
};

30 无重复字符的最长子串
//用滑动窗口，双指针
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int ans = 0;
        int len = s.size();
        int a = 0;
        int b = 0;
        set<char> m;
       	while(a <= b && b < len) {
       		if(m.find(s[b]) != m.end() ){
                m.erase(s[a]);
       			a++;//搞清楚++的位置，先删除，再++
       			
       		}else{
                m.insert(s[b]);
       			b++;//先插入原来的值，++，再计算ans
       			ans = max(ans, b - a);
       		}
       	}
       	return ans;
    }
};

31.寻找两个有序数组的中位数？？？(背)
这里相当于对K二分，每次调整k的大小，使其为1
区别与传统二分，其是对数组大小二分，调整其大小，使其范围为1

!!! size_type v.size() 无符号，如果空的v, 
num1.size() - 1会返回一个极大的数，溢出

class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2){
        int m = nums1.size();
        int n = nums2.size();
        int l = (m + n + 1) / 2;
        int r = (m+n+2)/2;
        return (getKth(nums1,0,nums2,0,l)+getKth(nums1,0,nums2,0,r))/2.0;
    }

    int getKth(vector<int>& nums1, int start1, 
               vector<int>& nums2, int start2, int k){
        int lena = nums1.size();
        int lenb = nums2.size();
        if(start1 > lena - 1){
            return nums2[start2 + k - 1];
        }
        if(start2 > nums2.size() - 1){
            return nums1[start1+k-1];
        }
        if(k==1)
            return min(nums1[start1],nums2[start2]);
        int nums1Mid = start1 + k/2 - 1 < nums1.size()?nums1[start1+k/2-1]:INT_MAX;
        int nums2Mid = start2 + k/2 - 1 < nums2.size()?nums2[start2+k/2-1]:INT_MAX;
    //哪个数组的中间值小，就在哪里找。当越界后，标记为最大，保证不会在这里面找        
        if(nums1Mid <nums2Mid)
        //每次把范围大小折半，min(m,n)?
            return getKth(nums1,start1+k/2,nums2,start2,k - k/2);
        else
            return getKth(nums1,start1,nums2,start2+k/2, k - k/2);
    }                            
};


//O(N) 的做法
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int lena = nums1.size(), lenb = nums2.size();
        int i = 0, j = 0;
        int cnt = 1;
        int ans1 = (lena + lenb + 1)/2;
        int ans2 = (lena + lenb + 2)/2;
        int x, y;
        int flag = 0;
        double ans;
        while(i < lena && j < lenb){
            if(nums1[i] < nums2[j]){
                if(cnt == ans1){
                    x = nums1[i];            
                    flag = 1;
                }
                i++;
            }else{
                if(cnt == ans1){
                    x = nums2[j]; 
                    flag = 1;
                }
                j++;
            }  
            cnt++;
            if(flag)
                break;
        }
        if(flag == 0){
            if(i == lena){
                while(cnt++ < ans1){
                    j++;
                }
                x = nums2[j++];
            }else{
                while(cnt++ < ans1){
                    i++;
                }
                x = nums1[i++];
            }
        }
        if(( lena + lenb )%2==0){
            if(i < lena && j < lenb){
                y = nums1[i] < nums2[j] ? nums1[i] :nums2[j];
            }else{
                y = i == lena? nums2[j] : nums1[i];
            }
            ans = (x + y)/2.0;
        }else{
            ans = x;
        }
        return ans;
    }
};

32.最长回文子串
//主要是stl用法，n2复杂度
class Solution {
public:
    int ans = 0;
    int start;
    string longestPalindrome(string s) {
        int len = s.size();
        for(int i = 0; i< len; i++){
            isPalindrome(s, i, i);  //一种巧妙处理奇偶
            isPalindrome(s, i, i + 1); 
        }
        return s.substr(start, ans);
    }
    void isPalindrome(string s, int a, int b){
        int len = s.size();
        while(a >= 0 && b <= len - 1){
            if(s[a] == s[b]){
                if(b-a+1 > ans){
                    ans = b - a + 1;
                    start = a;
                }
            }else{
                return;
            }
            a--;
            b++;
        }
        return;
    }
};

33. Z字形变换
class Solution {
public:
    string convert(string s, int numRows) {
        //string数组的声明方式
        vector<string> ansRow(numRows);
        int cnt = 0;
        int len = s.size();
        while(cnt < len){
            //注意加条件 cnt < len, 有cnt++的地方要判断
            for(int i = 0 ;i < numRows && cnt < len; i++){
                ansRow[i] += s[cnt++];
            }
            for(int i = numRows - 2; i >= 1 && cnt < len; i--){
                ansRow[i] += s[cnt++];
            }
        }
        string ans;
        //按行存储，最后拼接
        for(int i = 0; i< numRows; i++){
            ans += ansRow[i];
        }
        return ans;
    }
};

34. 整数反转
//-17%10 = -7 所以这里不用考虑正负号
注意判断溢出  -2^31, 2^31-1
class Solution {
public:
    int reverse(int x) {
        long long ans = 0;
        while(x){
            ans = ans * 10 + x % 10;
            x/=10;
            if( ans < INT_MIN || ans > INT_MAX)
                return 0;
        }
        return ans;
    }
};

35. 合并两个有序链表
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        //首先有个实体节点，用于最后的返回。再搞个索引
        //因为并不知道头结点应该是L1还是L2的，相当于搞个公共的
        ListNode* ans = new ListNode(-1); 
        ListNode* p = ans;
        while(l1 != nullptr && l2 != nullptr){
            if(l1->val < l2-> val){
                p  -> next = l1;
                l1 = l1 -> next;
            }else{
                p -> next = l2;
                l2 = l2 -> next;
            }
            p = p -> next;
        }
        if(l1 == nullptr)
            p -> next = l2;
        else
            p -> next = l1;
        return ans -> next;
    }
};
//递归做法
class Solution {
public:
    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if(l1 == nullptr)
            return l2;
        if(l2 == nullptr)
            return l1;
        if(l1->val < l2->val){
            l1->next = mergeTwoLists(l1->next, l2);
            return l1;
        } else {
            l2 -> next = mergeTwoLists(l1, l2 -> next);
            return l2;
        }
    }
};

36. 另一个树的子树
    bool HasSubtree(TreeNode* pRoot1, TreeNode* pRoot2)
    {
        bool ans = false;
        //注意判断空指针
        if(pRoot1 == nullptr || pRoot2 == nullptr)
            return ans;
        if(pRoot1->val == pRoot2->val){
            ans = isAhasB(pRoot1, pRoot2);
        }
        if(!ans){
            ans = HasSubtree(pRoot1->left, pRoot2);
        }
        if(!ans)
            ans = HasSubtree(pRoot1->right, pRoot2);
        return ans;
    }
    bool isAhasB(TreeNode* a, TreeNode* b){
        if(b == nullptr)
            return true;
        if(a == nullptr)
            return false;
        if(a->val != b->val)
            return false;
        return isAhasB(a->left, b->left) && isAhasB(a->right,b->right);
    }

37. (lc54. 螺旋矩阵) !!顺时针打印矩阵
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        //m, n的获取方法
        int m = matrix.size();
        int n = 0;
        if(m > 0)
            n = matrix[0].size();
        //二维vector的声明方式，并初始化为0
        vector<vector<int>> vis(m, vector<int>(n, 0));
        vector<int> ans;
        int i = 0;
        int j = -1;//
        int cnt = 0;
        while(cnt < m * n){
            //要先试探，再走
            //否则先走，当前就可能是不满足条件了，退出循环，会越界
            //因为是连续循环，要使得每个状态都满足条件
            while(j + 1 < n && !vis[i][j + 1]){
                vis[i][j + 1] = 1;
                cnt++;
                ans.push_back(matrix[i][++j]);
            }
            while(i + 1 < m && !vis[i + 1][j]){
                vis[i + 1][j] = 1;
                cnt++;
                ans.push_back(matrix[++i][j]);
            }
            while(j - 1 >= 0 && !vis[i][j - 1]){
                vis[i][j - 1] = 1;
                cnt ++;
                ans.push_back(matrix[i][--j]);
            }
            while(i - 1 >= 0 && !vis[i - 1][j]){
                vis[i - 1][j] = 1;
                cnt++;
                ans.push_back(matrix[--i][j]);
            }
        }
        return ans;
    }

38. 栈的压入、弹出序列     
bool validateStackSequences(vector<int>& pushed, vector<int>& popped) {
    stack<int> s;
    int i = 0;
    int j = 0;
    int len = pushed.size();
    //考虑好循环条件，有一个就可以
    while(i < len || j < len){
        //如果下一个要弹出的数字是栈顶数字，直接弹出
        if(!s.empty() && s.top() == popped[j]){
            s.pop();
            j++;
        }else{
        //不是栈顶，
            if(i >= len)
                break;
            s.push(pushed[i]);
            i++;
        }
    } 
    if(i==len && j == len && s.empty())
        return true;
    return false;
}

39. !! 按行层次遍历二叉树
vector<vector<int> > Print(TreeNode* pRoot) {
    vector<vector<int>> ans;//
    queue<TreeNode*> q;
    if(pRoot == nullptr)
        return ans;
    q.push(pRoot);
    while(!q.empty()){
//关键，每次获取len
        int len = q.size();
        vector<int> v;//
        for(int i = 0; i < len; i++){
            TreeNode* tmp = q.front();
            q.pop();
            v.push_back( tmp -> val );//
            if(tmp -> left != nullptr)
                q.push(tmp->left);
            if(tmp->right != nullptr)
                q.push(tmp->right);
        }
        ans.push_back(v);//
    }
    return ans;
}

40. Z字形打印二叉树
只需在39基础上
if(lineNum % 2 == 0)
    reverse(v.begin(), v.end());


41. BST的后序遍历序列
class Solution {
public:
    bool VerifySquenceOfBST(vector<int> sequence) {
        int len = sequence.size();
        ///递归要有退出条件
        if(len == 0)
            return false;
        // if(len <= 1)
        //     return true;
        int root = sequence[len - 1];
        int i = 0 ;
        //加判断，别越界
        while(sequence[i] < root && i < len - 1){
            i++;
        }
        int j = i;
        while(sequence[j] > root && j < len - 1){
            j++;
        }
        if(j != len - 1)
            return false;
        //vector 截取
        //左闭右开区间好处就是，下标连续的
        vector<int> a(sequence.begin(), sequence.begin() + i );
        vector<int> b(sequence.begin() + i, sequence.end() - 1);
        bool ansA = false;
        if(i <= 1)
            ansA = true;
        else
            ansA = VerifySquenceOfBST(a);
        bool ansB = false;
        if(len - 1 - i <= 1)
            ansB = true;
        else
            ansB = VerifySquenceOfBST(b);
        return ansA && ansB;
    }
};

42. 最大子序列和
int maxSubArray(vector<int>& nums) {
    int len = nums.size();
    
    int sum = 0 ;
    //考虑特殊情况，所有数都负
    //注意ans赋值，有可能所有数都是负的
    int ans = nums[0];
    
    for(int i = 0; i < len; i++){
        if(sum < 0){
            sum = 0;
        }
        sum += nums[i];
        ans = max(sum, ans);
    }
    return ans;
}

43. TOP k
class Solution {
public:
    int findKthLargest(vector<int>& nums, int k) {
        int len = nums.size();
        
        int st = 0, ed = len - 1;
        int index = 0;
        //闭区间，注意边界条件
        while(st <= ed){
            index = partition(nums, st, ed);
            if(index == len - k)
                break;
            if(index  < len - k){
               st = index + 1;
            }else{
                ed = index - 1;
            }
        }
        //最后返回index
        return nums[index];
    } 
    
    int partition(vector<int>& nums, int st, int ed){
        if(st >= ed)
            return st;
        int priov = nums[st];
        int i = st, j = ed;
        while(i < j){
            while(i < j && nums[j] >= priov)
                j--;
            while(i < j && nums[i] <= priov)
                i++;
            if(i < j)
                swap(nums[i], nums[j]);
        }
        //要和nums数组交换，不能是swap(nums[i], priov)
        swap(nums[i], nums[st]);
        return i;
    }
};

44. 最小的K个数，复杂度NlogK,基于堆，红黑树
适用于海量数据，不用都读入内存
class Solution {
public:
    vector<int> GetLeastNumbers_Solution(vector<int> input, int k) {
        vector<int> ans;
        int len = input.size();
        if(len < k){
            return ans;
        }
        //从大到小排，根节点最大的写法
        multiset<int, greater<int>> s;
        multiset<int, greater<int>>::iterator iterMax;
        
        vector<int>:: iterator it = input.begin();
        for(; it != input.end(); it++){
            
            if(int(s.size()) < k) {
                s.insert(*it);
            }else{
                //s.begin()就是根节点，最大
                iterMax = s.begin();
                if( *it < *iterMax) {
                    s.erase(*iterMax);
                    s.insert(*it);
                }
            }
        }
        iterMax = s.begin();
        for(; iterMax != s.end(); iterMax++){
            ans.push_back(*iterMax);
        }
        reverse(ans.begin(), ans.end());
        return ans;
    }
};

45. 排序链表
    ListNode* sortList(ListNode* head) {
        if(head == nullptr || head->next == nullptr)
            return head;
        ListNode* slow = head;
        //想象只有两个节点的情况，slow指针循环完应该指向第一个节点。
        //所以这里fast要快
        ListNode* fast = head->next;
        while(fast != nullptr && fast->next != nullptr){
            slow = slow -> next;
            fast = fast -> next -> next;
        }
        //为了保证断开链，先搞后半段链表
        ListNode* right = sortList(slow->next);
        slow -> next = nullptr;
        ListNode* left = sortList(head); 
        //调用合并两个有序链表
        return merge(left, right);        
    }
    
    ListNode* merge(ListNode *l1, ListNode *l2){
        ListNode* head = new ListNode(0);
        ListNode* p = head;
        while(l1 != nullptr && l2 != nullptr){
            if(l1->val < l2->val){
                p->next = l1;
                l1 = l1 -> next;
            }else{
                p->next = l2;
                l2 = l2 -> next;
            }
            p = p->next;
        }
        if(l1 != nullptr){
            p -> next = l1;
        }else{
            p -> next = l2;
        }
        return head->next;
    }

46.1 删除链表所有重复元素  中等
ListNode* deleteDuplicates(ListNode* head) {
    if(head == nullptr || head->next == nullptr)
        return head;
    ListNode* ans = new ListNode(0);
    ans->next = head;
    ListNode* slow = ans;
    ListNode* fast = head;
    int tmp = -1;
    while(fast != nullptr && fast->next != nullptr){
        if(fast->next->val == fast -> val){
            tmp = fast->val;
            while(fast != nullptr && fast->val == tmp){
                fast = fast->next;
            }
            slow->next = fast;
        }else{
            slow = fast;
            fast = fast->next;
        }
    }
    return ans->next;
}
46.2

struct ListNode {
    int val;
    ListNode *next;
    ListNode(int x) : val(x), next(NULL) {}
};

class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if(head == nullptr || head->next == nullptr)
            return head;
        ListNode* p = head;
        while(head != nullptr && head->next != nullptr){
            if(head->val == head->next->val){
                head->next = head->next->next;
            }else
                head = head->next;
        }
        return p;
    }
};

47. LRU 

/******
//定义三个private数据，LRU尺寸，LRU pair<key,value>, LRU map<key,iterator of pair>

//利用splice操作erase,makepair 等完成LRUcache

//put()
1. 如果m中有对应的key,value,那么移除l中的key,value
2. 如果m中没有，而size又==cap，那么移除l最后一个key，value，并移除m中对应的key，iterator
3. 在l的开头插入key，value，然后m[key]=l.begin();

****/
class LRUCache {
private:
    int cap;//LRU size
    list<pair<int,int>> l;//pair<key,value>
    unordered_map<int,list<pair<int,int>>::iterator> m;
    //unordered_map<key,key&value's pair iterator>
};
public:
    LRUCache(int capacity){
        cap = capacity;
    }
    int get(int key){
        auto it = m.find(key);
        if(it == m.end()) 
            return -1;
        pair<int, int> kv = *map[key];
        l.erase(map[key]);
        l.push_front(kv);
        map[key] = l.begin();
      //  l.splice(l.begin(), l, it->second);//插入到list 头部
        return kv->second;
    }
    void put(int key,int value){
        auto it=m.find(key);
        if(it != m.end()){
            l.erase(it->second);
            l.push_front(make_pair(key,value));
            m[key] = l.begin();
        } else {
            if(l.size()==cap){
                //当删除节点时，要能取得最后一个k，所以链表中要存储k-v
                int k = l.rbegin()->first;
                l.pop_back();
                m.erase(k);//map可以根据key值和迭代器值移除，查找
            }
            l.push_front(make_pair(key,value));
            m[key] = l.begin();
        }
    }

48. 实现memcpy
https://zhuanlan.zhihu.com/p/70873246
覆盖有两种情况，
其中1种一定要从后向前复制
另一种一定要从前向后
void* memcpy(void *dst, const void *src, int len){
    if(dst==nullptr || src == nullptr)
        return null;
    char *psrc = (char*)src;
    char *pdst = (char*)dst;
    if(psrc < pdst && psrc + size > pdst){
        psrc = psrc + len -1;
        pdst = pdst + len - 1;
        while(len--){
            *pdst-- = *psrc--;
        }
    }else{
        while(len--){
            *pdst++ = *psrc++;
        }
    }
    return dst;
}

49. 单例模式
利用static局部变量
1. 懒汉式
//用到了才初始化，延迟初始化
class Singleton{
public:
    //返回静态的引用
    static Singleton& getInstance(){
        //局部静态变量
        static Singleton m_instance;
        return m_instance;
    }  
private:
    Singleton();
    Singleton(const Single& other);
}
2. 饿汉式，程序运行时立即初始化。
初始化了一直没有被使用，拿不到资源，导致饥饿
class Singleton{
public:
    static Singleton& getInstance(){
        return instance;
    }
private:
	static Singleton instance;
	Singleton();
	~Singleton();
	Singleton(const Singleton&);
	Singleton& operator=(const Singleton&);    
}
//初始化
Singleton Singleton::instance;

50. 验证UTF8

class Solution {
public:
    bool validUtf8(vector<int>& data) {
        int n = data.size();
        for(int i = 0; i < n; i++){
            int ans = 0; 
            //找前缀1个数
            for(int j = 7;j >= 0; j--){
                //注意这里是data移动。如果是1移动要判断是否等于128，而不是，这样比较麻烦
                if((data[i] >> j) & 1)
                    ans++;
                else
                    break;
            }
            if(ans == 0)
                continue;
            if(ans == 1|| ans > 4 || i + ans > n)
                return false;
            for(int j = 1; j < ans; j++){
                if((data[i + j] >>6) !=2)
                    return false;
            }
            //ans个前缀1，只向后找了ans-1
            i += ans - 1 ;
        }
        return true;
    }
};