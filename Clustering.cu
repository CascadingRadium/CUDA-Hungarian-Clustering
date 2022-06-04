#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cfloat>
#include <cuda.h>
#include <cuda_runtime.h>
#define watch(x) cout << boolalpha << (#x) << " is " << (x) <<'\n'
#define watcharr(x) for(auto i:x)cout<<i<<' ';cout<<'\n';
#define SIZE 32
using namespace std;
typedef vector<vector<float>> matrix;
__global__ void step1a(double* Dr, double* d_arr, int n)
{
	int thread=threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<n)
	{
		double m=DBL_MAX;
		for(int j=0;j<n;j++)
		{
			m=min(m,d_arr[(thread*n)+j]);
		}
		Dr[thread]=m;
	}	
}
__global__ void step1b(double* Dr, double* Dc, double* d_arr, int n)
{
	int thread=threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<n)
	{
		double m=DBL_MAX;
		for(int i=0;i<n;i++)
			m=min(m,d_arr[(i*n)+thread]-Dr[i]);
		Dc[thread]=m;
	}	
}
__global__ void step2(int* Ar, bool* Vr, int* matchcount, int n)
{
	int thread=threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<n)
	{
		if(Ar[thread]!=-1)
		{
			Vr[thread]=1;
			atomicAdd(matchcount, 1);
		}
	}
}
__global__ void dualstep(double theta, bool* Vr, bool* Vc, int* Pc, double* Dr, double* Dc, double* slack,int n, bool* active)
{
	int thread=threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<n)
	{
		double val=theta/2.0;
		if(Vr[thread]==false)
			Dr[thread]=Dr[thread]+val;
		else
			Dr[thread]=Dr[thread]-val;
		if(Vc[thread]==false)
			Dc[thread]=Dc[thread]+val;
		else
			Dc[thread]=Dc[thread]-val;
		if(slack[thread]>0)
		{
			slack[thread]=slack[thread]-theta;
			if(slack[thread]==0)
				active[Pc[thread]]=true;
		}
	}
}
__global__ void revstep(int* frontier, int frontiersize, int* Pr, int* Pc, int* Sr, int* Sc, bool* augment)
{
	int thread=threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<frontiersize)
	{
		int j=frontier[thread];
		int rcur=-1;
		int ccur=j;
		while(ccur!=-1)
		{
			Sc[ccur]=rcur;
			rcur=Pc[ccur];
			Sr[rcur]=ccur;
			ccur=Pr[rcur];
		}
		augment[rcur]=true;
	}
}
__global__ void augstep(int* frontier, int frontiersize, int* Sr, int* Sc, int* Ar, int* Ac)
{
	int thread=threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<frontiersize)
	{
		int i=frontier[thread];
		int rcur=i;
		int ccur=-1;
		while(rcur!=-1)
		{
			ccur=Sr[rcur];
			Ar[rcur]=ccur;
			Ac[ccur]=rcur;
			rcur=Sc[ccur];
		}
	}
}
__global__ void step3a(int* frontier, int frontiersize, double* d_arr, double* Dr, double* Dc, int* Ac, bool* Vr, bool* Vc, double* slack, int n, int* Pr, int* Pc, bool* active, bool* reverse, bool* visited,bool* visited2)
{
	int thread=threadIdx.x+(blockIdx.x*blockDim.x);
	if(thread<n)
	{
		if(Vc[thread]==false)
		{
			for(int x = 0; x < frontiersize; x++)
			{
				int i=frontier[x];
				double g=d_arr[(i*n)+thread]-Dr[i]-Dc[thread];
				if(slack[thread] > g)
				{
					slack[thread]=g;
					Pc[thread]=i;
				}
				int i_new=Ac[thread];
				if(slack[thread]==0&&!visited2[i_new])
				{
					if(i_new!=-1)
					{
						Pr[i_new]=thread;
						Vr[i_new]=false;
						Vc[thread]=true;
						active[i_new]=true;
					}
					else
						reverse[thread]=true;
				}
				visited2[i]=true;
			}
		}
	}
}
__global__ void step0( bool* active, bool* visited,bool* visited2,bool* reverse,bool* augment, double* slack, int* Ar,int* Ac, bool* Vr, bool* Vc, int* Pr, int* Pc, int* Sr,int* Sc, int n)
{
	int i=threadIdx.x+(blockIdx.x*blockDim.x);
	if(i<n)
	{
		active[i]=false;
		visited[i]=false;
		visited2[i]=false;
		reverse[i]=false;
		augment[i]=false;
		slack[i]=DBL_MAX;
		Ar[i]=-1;
		Ac[i]=-1;
		Vr[i]=-1;
		Vc[i]=-1;
		Pr[i]=-1;
		Pc[i]=-1;
		Sr[i]=-1;
		Sc[i]=-1;
	}
}
__global__ void failure(double* slack,int* Pr, int* Pc, int* Sr,int* Sc,bool* visited,bool* visited2,bool* reverse,bool* augment, int n)
{
	int i=threadIdx.x+(blockIdx.x*blockDim.x);
	if(i<n)
	{
		slack[i]=DBL_MAX;
		Pr[i]=-1;
		Pc[i]=-1;
		Sr[i]=-1;
		Sc[i]=-1;
		visited[i]=false;
		visited2[i]=false;
		reverse[i]=false;
		augment[i]=false;
	}
}
__global__ void dualdonecheck(bool* visited,bool* visited2,bool* reverse,bool* augment, int n)
{
	int i=threadIdx.x+(blockIdx.x*blockDim.x);
	if(i<n)
	{
		visited[i]=false;
		visited2[i]=false;
		reverse[i]=false;
		augment[i]=false;
	}
}
__global__ void outcheck(bool* Vr, bool* Vc,int n)
{
	int i=threadIdx.x+(blockIdx.x*blockDim.x);
	if(i<n)
	{
		Vr[i]=false;
		Vc[i]=false;
	}
}
float roundoff(float value, unsigned char prec)
{
	float pow_10 = pow(10.0f, (float)prec);
	return round(value * pow_10) / pow_10;
}
void callcudahungarian(vector<vector<float>> &h_mat,vector<int>&traversal)
{
	int n=h_mat.size();
	int size=n*n;
	double* d_arr;
	cudaMallocManaged((void **)&d_arr, sizeof(double)*size);
	for(int i=0;i<n;i++)
	{
		for(int j=0;j<n;j++)
			d_arr[i*n+j]=h_mat[i][j];
	}
	double* Dr;
	double* Dc;
	double* slack;
	int* Ar;
	int* Ac;
	int* Pr;
	int* Pc;
	int* Sr;
	int* Sc;
	int* frontier;
	int* matchcount;
	bool* Vr;
	bool* Vc;
	bool* active;
	bool* visited;
	bool* visited2;
	bool* reverse;
	bool* augment;
	cudaMallocManaged((void **)&active, sizeof(bool)*n);
	cudaMallocManaged((void **)&visited, sizeof(bool)*n);
	cudaMallocManaged((void **)&visited2, sizeof(bool)*n);
	cudaMallocManaged((void **)&reverse, sizeof(bool)*n);
	cudaMallocManaged((void **)&augment, sizeof(bool)*n);
	cudaMalloc((void **)&Dr, sizeof(double)*n);
	cudaMallocManaged((void **)&slack, sizeof(double)*n);
	cudaMalloc((void **)&Dc, sizeof(double)*n);
	cudaMalloc((void **)&Vr, sizeof(bool)*n);
	cudaMalloc((void **)&Vc, sizeof(bool)*n);
	cudaMallocManaged((void **)&Ar, sizeof(int)*n);
	cudaMalloc((void **)&Ac, sizeof(int)*n);
	cudaMalloc((void **)&Pr, sizeof(int)*n);
	cudaMalloc((void **)&Pc, sizeof(int)*n);
	cudaMalloc((void **)&Sr, sizeof(int)*n);
	cudaMalloc((void **)&Sc, sizeof(int)*n);
	cudaMallocManaged((void **)&frontier, sizeof(int)*n);
	cudaMallocManaged(&matchcount, sizeof(int));
	*matchcount=0;
	int frontiersize=0;
	step0<<<(n/SIZE)+1,SIZE>>> (active, visited, visited2, reverse, augment, slack, Ar, Ac, Vr, Vc, Pr, Pc, Sr, Sc, n);
	step1a<<<(n/SIZE)+1,SIZE>>> (Dr,d_arr,n);
	step1b<<<(n/SIZE)+1,SIZE>>> (Dr,Dc,d_arr,n);
	bool dualdone=false;
	while(true)
	{
		if(!dualdone)
		{
			*matchcount=0;
			outcheck<<<(n/SIZE)+1,SIZE>>> (Vr, Vc, n);
			step2<<<(n/SIZE)+1,SIZE>>> (Ar,Vr,matchcount,n);
			cudaDeviceSynchronize();
			if(*matchcount==n)
			{
				for(int i=0;i<n;i++)
					traversal.push_back(Ar[i]);
				break;
			}
			failure<<<(n/SIZE)+1,SIZE>>> (slack, Pr, Pc, Sr, Sc, visited, visited2, reverse, augment, n);
		}
		else
			dualdonecheck<<<(n/SIZE)+1,SIZE>>> (visited, visited2, reverse, augment, n);
		dualdone=false;
		frontiersize=0;
		int activecount=0;
		for(int i=0;i<n;i++)
		{
			if(Ar[i]==-1||active[i])
			{
				activecount+=1;
				active[i]=true;
				frontier[frontiersize]=i;
				frontiersize+=1;
			}
			else
				active[i]=false;
		}
		int newactivecount=0;
		while(true)
		{
			newactivecount=0;
			step3a<<<(n/SIZE)+1,SIZE>>> (frontier,frontiersize,d_arr,Dr,Dc,Ac,Vr,Vc,slack,n,Pr,Pc,active,reverse,visited,visited2);
			cudaDeviceSynchronize();
			for(int i=0;i<frontiersize;i++)
				visited[frontier[i]]=true;
			for(int i=0;i<n;i++)
				visited2[i]=false;
			frontiersize=0;
			for(int i=0;i<n;i++)
			{
				if(active[i]==1&&!visited[i])
				{
					frontier[frontiersize]=i;
					newactivecount+=1;
					frontiersize+=1;
				}
			}
			if(newactivecount==0)
				break;
		}
		for(int i=0;i<n;i++)
			active[i]=false;
		frontiersize=0;	
		for(int i=0;i<n;i++)
		{
			if(reverse[i])
			{
				frontier[frontiersize]=i;
				frontiersize++;
			}
		}
		if(frontiersize==0)
		{
			dualdone=true;
			double theta=DBL_MAX;
			for(int i=0;i<n;i++)
			{
				if(slack[i]>0)
					theta=min(theta,slack[i]);
			}		
			dualstep<<<(n/SIZE)+1,SIZE>>>(theta, Vr, Vc, Pc, Dr, Dc, slack, n, active);
		}
		else
		{
			dualdone=false;
			revstep<<<(frontiersize/32)+1,32>>>(frontier,frontiersize , Pr, Pc, Sr, Sc,augment);
			cudaDeviceSynchronize();
			frontiersize=0;
			for(int i = 0;i<n;i++)
			{
				if(augment[i])
				{
					frontier[frontiersize]=i;
					frontiersize++;
				}
			}
			augstep<<<(frontiersize/32)+1,32>>>(frontier,frontiersize, Sr, Sc, Ar, Ac);
			cudaDeviceSynchronize();
		}		
	}
	cudaFree(active);
	cudaFree(visited);
	cudaFree(visited2);
	cudaFree(reverse);
	cudaFree(augment);
	cudaFree(Dr);
	cudaFree(slack);
	cudaFree(Dc);
	cudaFree(Vr);
	cudaFree(Vc);
	cudaFree(Ar);
	cudaFree(Ac);
	cudaFree(Pr);
	cudaFree(Pc);
	cudaFree(Sr);
	cudaFree(Sc);
	cudaFree(frontier);
	cudaFree(matchcount);
}

struct HASH
{
	size_t operator()(const pair<int,int>&x)const
	{
		return hash<long long>()(((long long)x.first)^(((long long)x.second)<<32));
	}
};
void dfs(int v,vector<bool> &used,unordered_map<pair<int,int>,float,HASH>& comp, vector<vector<pair<int,float>>>& adjlist) 
{
	used[v] = true;
	for (int i = 0; i < adjlist[v].size(); ++i)
	{
		auto to = adjlist[v][i].first;
		if (!used[to])
		{
			comp[{v,to}]=adjlist[v][i].second;
			dfs(to,used,comp,adjlist);
		}
	}
}
vector<unordered_map<pair<int,int>,float,HASH>> findconnectedcomponents(vector<vector<pair<int,float>>> &adjlist, int NoOfVertex)
{
	vector<bool> used(NoOfVertex,false);
	vector<unordered_map<pair<int,int>,float,HASH>> ListOfG;
	unordered_map<pair<int,int>,float,HASH> comp; 
	for (int i = 0; i < NoOfVertex ; ++i)
	{
		if (!used[i]) 
		{
			comp.clear();
			dfs(i,used,comp,adjlist);
			ListOfG.push_back(comp);
		}
	}
	return ListOfG;
}
vector<vector<pair<int,float>>> GetAdjList(unordered_map<pair<int,int>,float,HASH>& hmap, int NoOfVertex)
{
	vector<vector<pair<int,float>>> adjlist(NoOfVertex);
	for(auto i:hmap)
	{
		adjlist[i.first.first].push_back({i.first.second,i.second});
		adjlist[i.first.second].push_back({i.first.first,i.second});
	}
	return adjlist;
}
vector<unordered_map<pair<int,int>,float,HASH>> GetArrayOfG(unordered_map<pair<int,int>,float,HASH>& newhmap,int NoOfVertex)
{
	auto adjlist=GetAdjList(newhmap,NoOfVertex);
	auto connectedComponents=findconnectedcomponents(adjlist, NoOfVertex);
	return connectedComponents;
}

unordered_set<int> GetVertexSetFromHMap(unordered_map<pair<int,int>,float,HASH> Graph)
{
	unordered_set<int> Vset;
	for(auto i:Graph)
	{
		Vset.insert(i.first.first);
		Vset.insert(i.first.second);
	}
	return Vset;
}
pair<float,pair<int,int>> FindInterGraphDistance(unordered_map<pair<int,int>,float,HASH> G1, unordered_map<pair<int,int>,float,HASH> G2 ,matrix& OrigDistMat)
{
	auto G1VSet=GetVertexSetFromHMap(G1);
	auto G2VSet=GetVertexSetFromHMap(G2);
	float interGDist=FLT_MAX;
	pair<int,int> retpair;
	for(auto i: G1VSet)
	{
		for(auto j : G2VSet)
		{
			if(OrigDistMat[i][j]<interGDist)
			{
				interGDist=OrigDistMat[i][j];
				retpair={i,j};
			}
		}
	}
	return{interGDist,retpair};
}
vector<int> GetVlist(unordered_map<pair<int,int>,float,HASH> &G)
{
	unordered_set<int> vset;
	for(auto i: G)
	{
		vset.insert(i.first.first);
		vset.insert(i.first.second);
	}
	vector<int> Vset(vset.begin(),vset.end());
	return Vset;
}
bool CheckForCompleteness(unordered_map<pair<int,int>,float,HASH> &G, matrix &OrigDistMat, int point, float distance, int PARAM)
{
	auto vset= GetVlist(G);
	int count=0;
	for(auto i:vset)
	{
		if(OrigDistMat[i][point]<distance)
		{
			count++;
		}
	}
	if(count>PARAM)
		return true;
	return false;
}

void GetNewMat(vector<unordered_map<pair<int,int>,float,HASH>> &ListOfG, unordered_map<int,unordered_map<pair<int,int>,float,HASH>> &IndexToGraphEncoder, matrix &distmat, vector<vector<pair<int,int>>>& paircurmat, matrix& OrigDistMat, int PARAM)
{
	distmat.clear();
	paircurmat.clear();
	distmat.resize(ListOfG.size(),vector<float>(ListOfG.size()));
	paircurmat.resize(ListOfG.size(),vector<pair<int,int>>(ListOfG.size()));
	for(int i=0;i<ListOfG.size();i++)
	{
		IndexToGraphEncoder[i]=ListOfG[i];
	}
	for(int i=0;i<ListOfG.size();i++)
	{
		for(int j=i;j<ListOfG.size();j++)
		{
			if(i==j)
				distmat[i][j]=FLT_MAX;
			else
			{
				pair<float,pair<int,int>> res=FindInterGraphDistance(ListOfG[i],ListOfG[j], OrigDistMat);
				bool isCiComplete= false;
				bool isCjComplete= false;
				isCiComplete=CheckForCompleteness(ListOfG[i],OrigDistMat,res.second.first, res.first, PARAM);
				isCjComplete=CheckForCompleteness(ListOfG[j],OrigDistMat,res.second.second, res.first, PARAM);
				if(isCiComplete||isCjComplete)
				{
					distmat[i][j]=FLT_MAX;
					distmat[j][i]=FLT_MAX;
				}
				else
				{
					distmat[i][j]=res.first;
					distmat[j][i]=res.first;
					paircurmat[i][j]=res.second;
					paircurmat[j][i]=res.second;
				}

			}
		}
		int c=0;
		for(int k=0;k<distmat.size();k++)
		{
			if(distmat[i][k]==FLT_MAX)
				c++;
		}
		if(c==distmat.size())
			distmat[i][i]=FLT_MIN;
	}
}
float getEuclideanDistance(vector<float>& a, vector<float> &b)
{
	float sumsq = 0.0;
	for(int i = 0; i < a.size(); ++i)
	{
		sumsq += pow(a[i] - b[i], 2);
	}
	return sqrt(sumsq);
}
vector<unordered_map<pair<int,int>,float,HASH>> GetNextIter(vector<int>& traversal, matrix &curmat, vector<vector<pair<int,int>>> paircurmat, unordered_map<int,unordered_map<pair<int,int>,float,HASH>> &IndexToGraphEncoder, matrix &OrigDistMat)
{
	int V=traversal.size();
	unordered_map<pair<int,int>,float,HASH> newhmap;
	for(int i=0;i<V;i++)
	{
		int j=traversal[i];
		int matrixval=curmat[i][j];
		pair<int,int> joiningvertex=paircurmat[i][j];
		auto I = IndexToGraphEncoder[i];
		auto J = IndexToGraphEncoder[j];
		I.insert(J.begin(),J.end());
		I[joiningvertex]=matrixval;
		newhmap.insert(I.begin(),I.end());
	}
	vector<unordered_map<pair<int,int>,float,HASH>> ArrOfG=GetArrayOfG(newhmap,OrigDistMat.size());
	return ArrOfG;
}
void tokenize(string &str, char delim, vector<string> &out)
{
	size_t start;
	size_t end = 0;
	while ((start = str.find_first_not_of(delim, end)) != string::npos)
	{
		end = str.find(delim, start);
		out.push_back(str.substr(start, end - start));
	}
}
pair<matrix, vector<vector<float>>> DMatCalc(int noOfLine, int SKIP_COL, int SKIP_ROW, ifstream& IN_FILE)
{
	vector<vector<float>> arguments;
	string line;
	vector<string> tokens;
	if(SKIP_ROW!=0)
	{
		for(int i=0;i<SKIP_ROW;i++)
			getline(IN_FILE , line);
	}
	for(int i=0;i<noOfLine-SKIP_ROW;i++)
	{
		tokens.clear();
		IN_FILE>>line;
		tokenize(line,',',tokens);
		vector<float> point;
		for(int p = 0; p < tokens.size() - SKIP_COL; ++p)
		{
			point.push_back(stof(tokens[p]));
		}
		arguments.push_back(point);
	}
	size_t size=noOfLine-SKIP_ROW;
	vector<vector<float>> distmat{size,vector<float>(size)};
	for(int i=0;i<noOfLine-SKIP_ROW;i++)
	{
		for(int j=i;j<noOfLine-SKIP_ROW;j++)
		{
			if(i==j)
				distmat[i][j]=FLT_MAX;
			else
			{
				float dist=getEuclideanDistance(arguments[i], arguments[j]);
				distmat[i][j]=dist;
				distmat[j][i]=dist;
			}
		}
	}
	return {distmat, arguments};
}
int main(int argc, char *argv[])
{
	if(argc==1)
	{
		cout<<"Input File (.csv) as command line argument 1"<<'\n';
		exit(0);
	}
	if(argc==2)
	{
		cout<<"Parameter Value as command line argument 2" <<'\n';
		exit(0);
	}
	if(argc==3)
	{
		cout<<"Number of columns to skip from right as command line argument 3" <<'\n';
		exit(0);
	}
	if(argc==4)
	{
		cout<<"Number of rows to skip from top as command line argument 4" <<'\n';
		exit(0);
	}

	ifstream IN_FILE(argv[1]);
	int PARAM=atoi(argv[2]);
	int SKIP_COL=atoi(argv[3]);
	int SKIP_ROW=atoi(argv[4]);

	string line;
	size_t lines_count=0;
	while (getline(IN_FILE , line))
		++lines_count;

	IN_FILE.clear();
	IN_FILE.seekg(0);

	ofstream cout("output.csv");
	auto inipair=DMatCalc(lines_count,SKIP_COL,SKIP_ROW,IN_FILE);
	auto distmat=inipair.first;
	auto arguments=inipair.second;
	matrix OrigDistMat=distmat;

	unordered_map<int,unordered_map<pair<int,int>,float,HASH>> IndexToGraphEncoder;
	vector<vector<pair<int,int>>> paircurmat(distmat.size(),vector<pair<int,int>>(distmat.size()));
	int prevsize=0;
	int cursize=distmat.size();

	for(int i=0;i<cursize;i++)
	{
		for(int j=0;j<cursize;j++)
			paircurmat[i][j]={i,j};
	}
	while(prevsize!=cursize)
	{
		prevsize=cursize;
		vector<int> traversal;
		callcudahungarian(distmat,traversal);
		auto CycleVector=GetNextIter(traversal,distmat,paircurmat,IndexToGraphEncoder,OrigDistMat);
		GetNewMat(CycleVector,IndexToGraphEncoder,distmat,paircurmat, OrigDistMat, PARAM);
		cursize=distmat.size();
	}
	vector<vector<int>> clusters;
	for(int i=0;i<distmat.size();i++)
	{
		auto vlist=GetVlist(IndexToGraphEncoder[i]);
		clusters.push_back(vlist);
	}

	IN_FILE.clear();
	IN_FILE.seekg(0);
	if(SKIP_ROW==1)
	{
		getline(IN_FILE , line);
		vector<string> tokens;
		tokenize(line,',',tokens);
		line="";
		for(int i=0;i<tokens.size()-SKIP_COL;i++)
			line+=tokens[i]+',';
		line+="label\n";
		cout<<line;
	}
	for(int i=0;i<clusters.size();i++)
	{
		for(int j=0;j<clusters[i].size();j++)
		{
			for(int k = 0; k < arguments[clusters[i][j]].size(); ++k)
				cout << arguments[clusters[i][j]][k] << ",";
			cout << i << '\n';
		}
	}
	return 0;
}

