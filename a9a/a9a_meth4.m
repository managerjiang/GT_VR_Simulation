clear all
%%D-GET
%% 数据加载
load('a9a_smote.mat');%载入A：A(48243x123):48243个数据
load('L_a9a_smote.mat');%载入L：A(1x48243):48243个结果
A=A1;
L=L1;%如果使用原始数据，不需要添加这两行，同时修改load的名称
A=double(A);
L=double(L);
L(L==0)=-1;
L(L==1)=1;
%% 参数设置
agent_num=10;% agent个数
Maxgen=1600;% 迭代次数
% C=doubly_stochastic(agent_num);
% C=doubly_stochastic(agent_num);% 生成随机的邻接矩阵（行和列和都是1）
% C_store=C
load('data/C_meth1_smote_800.mat');%载入C_store
% C=C_store
%% 数据预处理
%根据智能体个数裁剪数据，每个智能体十分之一的数据,同时变量初始化
for i=1:agent_num
    L_cut(:,i)=L((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num));
    A_cut(:,:,i)=A((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num),:); 
   
end
lamuda1=0.5*10^(-5);
lamuda2=0.5*10^(-5);

x_k_last=zeros(123,agent_num);% x阵 
tau_ini=x_k_last;
y_k_last=zeros(123,agent_num);% y阵
v_k_last=zeros(123,agent_num);% v阵
lamuda1=0.5*10^(-5);
lamuda2=0.5*10^(-5);
%% 数据预处理
%根据智能体个数裁剪数据，每个智能体十分之一的数据,同时变量初始化
for i=1:agent_num
    L_cut(:,i)=L((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num));
    A_cut(:,:,i)=A((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num),:); 
    mid=L_cut(:,i).*A_cut(:,:,i); 
    gradient=-mid.*exp(mid*x_k_last(:,i))./(1+exp(mid*x_k_last(:,i))).^2;
    gradient=sum(gradient,1)/floor(size(A,1)/agent_num)+2*lamuda2*x_k_last(:,i)';
    y_k_last(:,i)=gradient';
    v_k_last(:,i)=gradient';
end
agent_m=floor(size(A,1)/agent_num);
eta=0.1;%%步长
T=8;
tic;
b0=30;
for k=1:Maxgen
    k   
    % 开始循环算法
    for i=1:agent_num       
        %-----更新x_i(k+1)--------
        xk=zeros(123,1);
        for j=1:agent_num
            x_k_j_last=x_k_last(:,j);
            xk=xk+C(i,j)*x_k_j_last;
        end
        x_k_new(:,i)=xk-eta*y_k_last(:,i);
        %%-----更新v---
        if mod(k+1,T)==0 
            mid=L_cut(:,i).*A_cut(:,:,i); 
            gradient=-mid.*exp(mid*x_k_new(:,i))./(1+exp(mid*x_k_new(:,i))).^2;
            gradient=sum(gradient,1)/floor(size(A,1)/agent_num)+2*lamuda2*x_k_new(:,i)';
            v_k_new(:,i)=gradient';
        else
            gradient_tau_sum=zeros(size(A,2),1);
            mid=L_cut(:,i).*A_cut(:,:,i); 
            s0_i=randperm(agent_m,b0);
            for j=1:b0
                gradient_0=-mid(s0_i(j),:)*exp(mid(s0_i(j),:)*x_k_last(:,i))/(1+exp(mid(s0_i(j),:)*x_k_last(:,i)))^2;
                gradient_0=sum(gradient_0,1)+2*lamuda2*x_k_last(:,i)';
                gradient_1=-mid(s0_i(j),:)*exp(mid(s0_i(j),:)*x_k_new(:,i))/(1+exp(mid(s0_i(j),:)*x_k_new(:,i)))^2;
                gradient_1=sum(gradient_1,1)+2*lamuda2*x_k_new(:,i)';
                gradient_tau_sum=gradient_tau_sum+gradient_1'-gradient_0';
            end
            v_k_new(:,i)=gradient_tau_sum/b0+v_k_last(:,i);% v阵
        end
   
    end
     %---------更新y_i(k+1)----------
    for i=1:agent_num
        ymid=zeros(size(A,2),1);%v(123x1)
        for j=1:agent_num
           ymid=ymid+C(i,j)*y_k_last(:,j);
        end
        y_k_new(:,i)=ymid+v_k_new(:,i)-v_k_last(:,i);
    end
    gradient_ite_sum=zeros(size(A,2),1);
    for i=1:agent_num
     mid=L_cut(:,i).*A_cut(:,:,i); 
    gradient_ite=-mid.*exp(mid*x_k_new(:,i))./(1+exp(mid*x_k_new(:,i))).^2;
    gradient_ite=sum(gradient_ite,1)/floor(size(A,1)/agent_num)+2*lamuda2*x_k_new(:,i)';
    gradient_ite_sum=gradient_ite_sum+gradient_ite';
    end
    gradient_sto{k}=gradient_ite_sum;
    x_k_store{k}=x_k_new;
    y_k_last=y_k_new;
    v_k_last=v_k_new;
    x_k_last=x_k_new;
     
end
T=toc