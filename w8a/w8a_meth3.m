clear all
%%GT-SARAH
%% 数据加载
load('w8a_smote.mat');%载入A：A(95598x300):95598个数据
load('L_w8a_smote.mat');%载入L：A(1x95598):95598个结果
A=A1;
L=L1;%如果使用原始数据，不需要添加这两行，同时修改load的名称
A=double(A);
L=double(L);
L(L==0)=-1;
L(L==1)=1;
%% 参数设置
agent_num=10;% agent个数
%Maxgen=1600;% 迭代次数
% C=doubly_stochastic(agent_num);% 生成随机的邻接矩阵（行和列和都是1）
% C_store=C
load('data/C_meth1_smote_800.mat');%载入C_store
C=C_store
%% 数据预处理
%根据智能体个数裁剪数据，每个智能体十分之一的数据,同时变量初始化
for i=1:agent_num
    L_cut(:,i)=L((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num));
    A_cut(:,:,i)=A((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num),:); 
   
end
lamuda1=0.5*10^(-5);
lamuda2=0.5*10^(-5);
alpha=0.1;
q=23;
S=80;
xcell=cell(q+1,S+1);
ycell=cell(q+1,S+1);
vcell=cell(q+1,S+1);
tic;
x_0_1=zeros(300,agent_num);
for i=1:agent_num
    x_i(:,i)=zeros(300,1);
    y_i(:,i)=zeros(300,1);
    v_i(:,i)=zeros(300,1);
end
    xcell{2,1}=x_i;
    ycell{2,1}=y_i;
    vcell{1,1}=v_i;
    x_k_store{1,1}=x_i;
    x_k_store{2,1}=x_i;
    agent_m=floor(size(A,1)/agent_num);
for s=1:S
    s
    for i=1:agent_num
        mid=L_cut(:,i).*A_cut(:,:,i); 
        gradient=-mid.*exp(mid*xcell{2,s}(:,i))./(1+exp(mid*xcell{2,s}(:,i))).^2;
        gradient=sum(gradient,1)/agent_m+2*lamuda2*xcell{2,s}(:,i)';
        vcell{2,s}(:,i)=gradient';
        ysum=zeros(300,1);
        xsum=zeros(300,1);
        for j=1:agent_num
            ysum=ysum+C(i,j)*ycell{2,s}(:,j);
            xsum=xsum+C(i,j)*xcell{2,s}(:,j);
        end
        ycell{3,s}(:,i)=ysum+vcell{2,s}(:,i)-vcell{1,s}(:,i);
        xcell{3,s}(:,i)=xsum-alpha*ycell{3,s}(:,i);
    end
  
        for t=3:q
            for i=1:agent_num
                tau_i=randperm(agent_m,1);
                mid=L_cut(:,i).*A_cut(:,:,i); 
                gradient_x=-mid(tau_i,:)*exp(mid(tau_i,:)*xcell{t,s}(:,i))/(1+exp(mid(tau_i,:)*xcell{t,s}(:,i)))^2;
                gradient_x=sum(gradient_x,1)+2*lamuda2*xcell{t,s}(:,i)';
                gradient_x0=-mid(tau_i,:)*exp(mid(tau_i,:)*xcell{t-1,s}(:,i))/(1+exp(mid(tau_i,:)*xcell{t-1,s}(:,i)))^2;
                gradient_x0=sum(gradient_x0,1)+2*lamuda2*xcell{t-1,s}(:,i)';
                vcell{t,s}(:,i)=gradient_x'-gradient_x0'+vcell{t-1,s}(:,i);
                ysum_t=zeros(300,1);
                xsum_t=zeros(300,1);
                for j=1:agent_num
                    ysum_t=ysum_t+C(i,j)*ycell{t,s}(:,j);
                    xsum_t=xsum_t+C(i,j)*xcell{t,s}(:,j);
                end
                ycell{t+1,s}(:,i)=ysum_t+vcell{t,s}(:,i)-vcell{t-1,s}(:,i);
                xcell{t+1,s}(:,i)=xsum_t-alpha*ycell{t+1,s}(:,i);
                
               
            end
        end
        for i=1:agent_num
             xcell{2,s+1}(:,i)=xcell{q+1,s}(:,i);
             ycell{2,s+1}(:,i)=ycell{q+1,s}(:,i);
             vcell{1,s+1}(:,i)=vcell{q,s}(:,i);
             x_k_store{s}=xcell{2,s+1};
        end
    
end

for k=1:S
    gradient_ite_sum=zeros(size(A,2),1);
    for i=1:agent_num
     mid=L_cut(:,i).*A_cut(:,:,i); 
    gradient_ite=-mid.*exp(mid*x_k_store{k}(:,i))./(1+exp(mid*x_k_store{k}(:,i))).^2;
    gradient_ite=sum(gradient_ite,1)/floor(size(A,1)/agent_num)+2*lamuda2*x_k_store{k}(:,i)';
    gradient_ite_sum=gradient_ite_sum+gradient_ite';
    end
   gradient_sto{k}=gradient_ite_sum;
end
    

T=toc
