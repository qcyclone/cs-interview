//BST左子树所有节点都比根小，右子树所有节点都比根大
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

11. 树的最小/最大 深度
public:
int minDepth(TreeNode* root) {
    if(root==NULL)  return 0;
    int l = minDepth(root->left) + 1;
    int r = minDepth(root->right) + 1;
    if(l==1 || r==1) //如果有一侧没有子树，则高度为另一侧
        return max(l,r);
    return min(l,r);
}
class Solution {
public:l
    int maxDepth(TreeNode* root) {
        if(root==NULL)
            return  0;
        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
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

16. 树的直径
//实际就是求最大深度时，对于每个节点对左右子树的深度加和并求最大值
class Solution {
public:
    int ans =0;
    int depth(TreeNode* node){
        if(node==NULL)
            return 0;
        int l = depth(node->left);
        int r = depth(node->right);
        ans = max(ans, l + r);
        return max(l, r) + 1;
    }
    int diameterOfBinaryTree(TreeNode* root) {
        depth(root);
        return ans;
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
