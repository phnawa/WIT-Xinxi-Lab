#include<iostream>
#include<iomanip>
#include<queue>
#define inf 1000//无穷大 
#define n 11//城市数 
using namespace std;
int C[n][n];//代价矩阵 
int S[n];//解空间 
int bestX[n],bestC=inf;//最佳路径与最小代价 
//结点定义
struct Node
{
	int x[n];//路径
	int v[n];//访问标识
	int k;//结点所在层
	int cc;//当前路径的代价
};
queue<Node>Q;//定义结点队列

//对S初始化 
void init()
{
	for(int i=0;i<n;i++)
		S[i]=i;
}
//读入C 
void Read()
{
	int tmp;
	FILE *fp;//定义文件指针
	fp=fopen("d:/cost.txt","rt");//以读入方式打开d:\cost.txt 
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			fscanf(fp,"%d",&tmp);
			C[i][j]=tmp;
		}
	}
	fclose(fp);
} 
//显示代价矩阵 
void Display()
{
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
		{
			if(i==j)
				cout<<setw(4)<<"inf";
			else
				cout<<setw(4)<<C[i][j];
		}
		cout<<endl;
	}
	cout<<endl;
}
//输出最佳路径与最小代价
void Output()
{	cout<<"最佳路径 ";
	for(int j=0;j<n;j++)
		cout<<bestX[j]<<" ";
	cout<<bestX[0]<<endl;
	cout<<"最小代价 "<<bestC<<endl;
}
//广搜
void BFS()
{
	int i,j;
	Node node;
	//创建第0层结点
	for(i=0;i<n;i++)
	{	node.x[i]=-1;//路径均置空值
		node.v[i]=-1;//访问数组均置空值
	}
	node.k=0;
	node.cc=0;
	Q.push(node);//入队
	//创建第1层结点
	for(i=0;i<n;i++)
	{	node=Q.front();
		node.x[0]=S[i];
		node.v[S[i]]=1;
		node.k++;
		node.cc=0;
		Q.push(node);
	}
	Q.pop();//结点1出队
	while(!Q.empty())
	{
		node=Q.front();
		if(node.k==n)//如果结点层到达叶子结点层
		{
			node.cc+=C[node.x[n-1]][node.x[0]];//更新当前路径的总代价
			if(node.cc<bestC)
			{
				bestC=node.cc;//更新最小代价
				for(int i=0;i<n;i++)
					bestX[i]=node.x[i];//更新最佳路径 
			}
		}
		else//如果结点层未到达叶子结点层
		{
			for(i=0;i<n;i++)//扩展该结点下的分枝结点
			{
				node=Q.front();
				j=node.k;
				if(node.x[i]==S[i]||node.v[S[i]]==1)
					continue;
				else
				{	node.x[j]=S[i];
					node.v[S[i]]=1;
				}
				node.cc+=C[node.x[j-1]][node.x[j]];//更新当前路径代价
				node.k=j+1;//更新结点层
				Q.push(node);//扩展结点入队
			}
		}
		Q.pop();//队头结点出队
	}
}
//主函数
int main()
{
	Read();//读入代价矩阵C
	Display();//显示代价矩阵C
	init();//对S初始化
	BFS();//广搜
	Output();//输出最佳路径与最小代价 
	return 0;
}
