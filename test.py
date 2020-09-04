# from numpy import *
# import time
# class Solution(object):
#     def longestPalindrome(self, s):
#         """
#         :type str: str
#         :rtype: str
#         """
#         if len(s)<2:
#             return s
#         else:
#             le=0
#             result=''
#             for i in range(1,len(s)-1):
#                 l,r=i,i
#                 if s[i]==s[i+1]:
#                     l,r = self.hui(s, i, i+1)
#                 else:
#                     l,r=self.hui(s,i,i)
#                 if r-l+1>le:
#                     le=r-l+1
#                     result=s[l:r+1]
#             return result
#
#
#     def hui(self,st,i,j):
#         left=i
#         right=j
#         if st[left] == st[right]:
#             while (st[left] == st[right]):
#                 left -= 1
#                 right += 1
#                 if left ==0 or right ==len(st)-1:
#                     break
#             left += 1
#             right -= 1
#         return left,right
#
#
#
# h=Solution()
# s="aaa"
# n=2
# f=h.longestPalindrome(s)
# print(f)
# class Solution:
#     def lengthOfLongestSubstring(self, s):
#         """
#         :type s: str
#         :rtype: int
#         """
#         start = maxLength = 0
#         usedChar = {}
#         for index, char in enumerate(s):
#             if char in usedChar and start <= usedChar[char]:
#                 start = usedChar[char] + 1
#             else:
#                 maxLength = max(maxLength, index - start + 1)
#             usedChar[char] = index
#         return maxLength



def dfs(res, num, fg, i, cnt):
    if fg[0] == 0:  # 无法回到起点，起点被遍历两遍
        return 0
    if sum(fg) == 2 and fg[0] == 1 and num[i][0] != 0.0:  # 只剩两个城市 ,起点还未到达, 且可以返回起点
        res.append(cnt + num[i][0])

    for kk in range(n):
        if num[i][kk] != 0.0 and fg[kk] > 0 and i!=kk:  # 有路且去向城市未访问过
            fg[i] -= 1
            dfs(res, num, fg, kk, cnt + num[i][kk])  #
            fg[i] += 1
#
# num=[[0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 4.0, 4.0], [5.0, 4.0, 0.0, 2.0], [5.0, 2.0, 2.0, 0.0]]
num=[[1]]
n=len(num)
fg = [1] * n  # 改点是否访问标志
fg[0] = 2  # 起点需访问两遍
res = []
dfs(res, num, fg, 0, 0)
print(res)
print(".1f"%min(res))
#
# if len(res) == 0:  # 考虑没有路径的情况
#     print(-1)
# else:
#     print(min(res))
#
# class Solution(object):
#
#     def exist(self, board, word):
#         """
#         :type board: List[List[str]]
#         :type word: str
#         :rtype: bool
#         """
#         m=len(board)
#         n=len(board[0])
#         mark=[[0 for i in range(n)] for j in range(m)]
#         start=[]
#         for i in range(m):
#             for j in range(n):
#                 if board[i][j]==word[0]:
#                     start.append([i,j])
#         for i in start:
#             begin_i=i[0]
#             begin_j=i[1]
#             mark = [[0 for i in range(n)] for j in range(m)]
#             mark[begin_i][begin_j]=1
#             if self.back(begin_i,begin_j,board,word[1:],mark)==True:
#                 return True
#             else:
#                 mark[begin_i][begin_j] = 0
#         return  False
#
#     def back(self,s,t,board,word,mark):
#         directs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
#         if len(word)==0:
#             return True
#         else:
#             for i in directs:
#                 bi=s+i[0]
#                 bj=t+i[1]
#                 if 0<=bi<=len(board)-1 and 0<=bj<=len(board[0])-1:
#                     if board[bi][bj]==word[0] and mark[bi][bj]==0:
#                         mark[bi][bj]=1
#                         if self.back(bi,bj,board,word[1:],mark)==True:
#                             return True
#                         else:
#                             mark[bi][bj] = 0
#             return False

# class Solution(object):
#     def maxProfit(self, prices):
#         """
#         :type prices: List[int]
#         :rtype: int
#         """
#         n=len(prices)
#         if n==0:
#             return 0
#         elif n<4:
#             return max(prices[1]-prices[0],0,prices[2]-prices[0],prices[2]-prices[1])
#         else:
#             res=[]
#             s=prices[1]-prices[0]
#             for i in range(1,len(prices)-1):
#                 k=prices[i+1]-prices[i]
#                 if k*s>=0:
#                     s += k
#                 else:
#                     res.append(s)
#                     s=k
#             res.append(s)
#             print(res)
#             money=0
#             t=2
#             while max(res)>0 and t>0:
#                 money+=max(res)
#                 res[res.index(max(res))]=0
#                 t-=1
#             return money
# class Solution(object):
#     def numIslands(self, grid):
#         """
#         :type grid: List[List[str]]
#         :rtype: int
#         """
#         num=1
#         n=len(grid)
#         m=len(grid[0])
#         t=[]
#         res=[[-1,0],[0,-1]]
#         du=[[0 for i in range(m+1)] for j in range(n+1)]
#         for i in range(1,n+1):
#             for j in range(1,m+1):
#                 if grid[i-1][j-1]=='1':
#                     if du[i-1][j]==0 and du[i][j-1]==0:
#                         du[i][j]=num
#                         num+=1
#                     else:
#                         if min(du[i-1][j],du[i][j-1])>0:
#                             p=max(du[i - 1][j], du[i][j - 1])
#                             b=min(du[i - 1][j], du[i][j - 1])
#                             for ii in range(1,i+1):
#                                 for jj in range(1,m+1):
#                                     if du[ii][jj]==p:
#                                         du[ii][jj] =b
#                             t.append([du[i-1][j],du[i][j-1]])
#
#                         du[i][j]=max(du[i-1][j],du[i][j-1])
#         x=0
#         for i in du:
#             for j in i:
#                 if j>x:
#                     x=j
#         return x

# class Solution(object):
#     def partition(self, s):
#         """
#         :type s: str
#         :rtype: List[List[str]]
#         """
#
#         res = []
#         tem = []
#         self.dfs(res, tem, s)
#         return res
#
#     def dfs(self, res, tem, s):
#         if len(s)==0:
#             res.append(tem)
#         else:
#             for i in range(len(s)):
#                 if s[:i+1] == s[:i+1][::-1]:
#                     self.dfs(res, tem+[s[:i+1]],s[i+1:])
# class Solution(object):
#     def lengthOfLIS(self, nums):
#         """
#         :type nums: List[int]
#         :rtype: int
#         """
#         if len(nums)>1:
#             dp = [0 for i in range(len(nums))]
#             dp[0] = 1
#             l = 0
#             for i in range(1, len(nums)):
#                 z=1
#                 for j in range(i):
#                     if nums[j] < nums[i]:
#                         z =max(dp[j]+1,z)
#                 dp[i] =z
#                 l = max(l, dp[i])
#             return l
#         else:
#             if len(nums)==1:
#                 return 1
#             else:
#                 return 0


# s=[10]
# j=Solution()
# w=j.lengthOfLIS(s)
# print(w)
