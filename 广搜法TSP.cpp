#include<iostream>
#include<iomanip>
#include<queue>
#define inf 1000//����� 
#define n 11//������ 
using namespace std;
int C[n][n];//���۾��� 
int S[n];//��ռ� 
int bestX[n],bestC=inf;//���·������С���� 
//��㶨��
struct Node
{
	int x[n];//·��
	int v[n];//���ʱ�ʶ
	int k;//������ڲ�
	int cc;//��ǰ·���Ĵ���
};
queue<Node>Q;//���������

//��S��ʼ�� 
void init()
{
	for(int i=0;i<n;i++)
		S[i]=i;
}
//����C 
void Read()
{
	int tmp;
	FILE *fp;//�����ļ�ָ��
	fp=fopen("d:/cost.txt","rt");//�Զ��뷽ʽ��d:\cost.txt 
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
//��ʾ���۾��� 
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
//������·������С����
void Output()
{	cout<<"���·�� ";
	for(int j=0;j<n;j++)
		cout<<bestX[j]<<" ";
	cout<<bestX[0]<<endl;
	cout<<"��С���� "<<bestC<<endl;
}
//����
void BFS()
{
	int i,j;
	Node node;
	//������0����
	for(i=0;i<n;i++)
	{	node.x[i]=-1;//·�����ÿ�ֵ
		node.v[i]=-1;//����������ÿ�ֵ
	}
	node.k=0;
	node.cc=0;
	Q.push(node);//���
	//������1����
	for(i=0;i<n;i++)
	{	node=Q.front();
		node.x[0]=S[i];
		node.v[S[i]]=1;
		node.k++;
		node.cc=0;
		Q.push(node);
	}
	Q.pop();//���1����
	while(!Q.empty())
	{
		node=Q.front();
		if(node.k==n)//������㵽��Ҷ�ӽ���
		{
			node.cc+=C[node.x[n-1]][node.x[0]];//���µ�ǰ·�����ܴ���
			if(node.cc<bestC)
			{
				bestC=node.cc;//������С����
				for(int i=0;i<n;i++)
					bestX[i]=node.x[i];//�������·�� 
			}
		}
		else//�������δ����Ҷ�ӽ���
		{
			for(i=0;i<n;i++)//��չ�ý���µķ�֦���
			{
				node=Q.front();
				j=node.k;
				if(node.x[i]==S[i]||node.v[S[i]]==1)
					continue;
				else
				{	node.x[j]=S[i];
					node.v[S[i]]=1;
				}
				node.cc+=C[node.x[j-1]][node.x[j]];//���µ�ǰ·������
				node.k=j+1;//���½���
				Q.push(node);//��չ������
			}
		}
		Q.pop();//��ͷ������
	}
}
//������
int main()
{
	Read();//������۾���C
	Display();//��ʾ���۾���C
	init();//��S��ʼ��
	BFS();//����
	Output();//������·������С���� 
	return 0;
}
