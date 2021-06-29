function [A1,L1]=smote(A,L)
%输入：A和L
%输出：拓展后的A1和L1
%% 选出正样本
A_posi=A;
A_posi(L==-1,:)=[];
num_posi=size(A_posi,1);
num_nega=size(A,1)-num_posi;
expend_num=num_nega-num_posi;
m=10;
k=floor(expend_num/num_posi);

%% 计算距离矩阵
dis=zeros(num_posi);
for i=1:num_posi
    for j=1:i
        dis(i,j)=norm(A_posi(i,:)-A_posi(j,:),2);
        dis(j,i)=dis(i,j);
    end
end
%% 循环
A_add=zeros(k*num_posi,300);
n=1;
for i=1:num_posi
    for j=1:k
        [~,c]=sort(dis(i,:));     
        d=c(randperm(m,1));
        alpha=rand;
        A_add(n,:)=alpha*A_posi(d,:)+(1-alpha)*A_posi(i,:);
        n=n+1;
    end
end
%% 人工数据混合
A1=zeros(num_nega+num_posi+num_posi*k,300);
L1=zeros(1,num_nega+num_posi+num_posi*k);
orin=randperm(num_nega+num_posi+num_posi*k,num_nega+num_posi);
L1(orin)=L;
A1(orin,:)=A;
A1(L1==0,:)=A_add;
L1(L1==0)=1;
end