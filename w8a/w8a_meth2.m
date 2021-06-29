clear all
%%GT-SAGA
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
Maxgen=1600;% 迭代次数
% C=doubly_stochastic(agent_num);% 生成随机的邻接矩阵（行和列和都是1）
% C_store=C
load('data/C_meth1_smote_800.mat');%载入C_store
C=C_store
x_k_last=zeros(300,agent_num);% x阵

y_k_last=zeros(300,agent_num);% y阵
v_k_last=zeros(300,agent_num);% v阵
lamuda1=0.5*10^(-5);
lamuda2=0.5*10^(-5);
agent_m=floor(size(A,1)/agent_num);
%% 数据预处理
%根据智能体个数裁剪数据，每个智能体十分之一的数据,同时变量初始化
for i=1:agent_num
    L_cut(:,i)=L((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num));
    A_cut(:,:,i)=A((i-1)*floor(size(A,1)/agent_num)+1:i*floor(size(A,1)/agent_num),:); 
   
end
for i=1:agent_num
    for j=1:agent_m
        tau_k_last(:,i,j)=x_k_last(:,i);
    end
end
eta=0.1;%%步长
T=8;
tic;
%% 算法主体
for k=1:Maxgen
    k   
    % 开始循环算法
       %-----更新v_i(k+1)---------
    for i=1:agent_num  
         mid=L_cut(:,i).*A_cut(:,:,i); 
         gradient_tau_sum=zeros(size(A,2),1);
         for j=1:agent_m
            gradient=-mid(j,:)*exp(mid(j,:)*tau_k_last(:,i,j))/(1+exp(mid(j,:)*tau_k_last(:,i,j)))^2;
            gradient=sum(gradient,1)+2*lamuda2*tau_k_last(:,i,j)';
            gradient_tau_sum=gradient_tau_sum+gradient';
         end
             gradient_tau_sum=gradient_tau_sum/agent_m;
%         gradient=-mid.*exp(mid.*tau_k_last(:,i))./(1+exp(mid.*tau_k_last(:,i))).^2;
%         gradient=sum(gradient,1)/floor(size(A,1)/agent_num)+2*lamuda2.*tau_k_last(:,i)';
%         gradient_tau_sum=gradient';
        
        xk=x_k_last(:,i);
        s_i=randperm(agent_m,1);
        mid=L_cut(:,i).*A_cut(:,:,i); 
        gradient_x=-mid(s_i,:)*exp(mid(s_i,:)*xk)/(1+exp(mid(s_i,:)*xk))^2;
        gradient_x=sum(gradient_x,1)+2*lamuda2*xk';
        gradient_x_i_k=gradient_x';
        gradient_tau=-mid(s_i,:)*exp(mid(s_i,:)*tau_k_last(:,i,s_i))/(1+exp(mid(s_i,:)*tau_k_last(:,i,s_i)))^2;
        gradient_tau=sum(gradient_tau,1)+2*lamuda2*tau_k_last(:,i,s_i)';
        gradient_tau_i_k=gradient_tau';
        v_k_i_new=gradient_x_i_k-gradient_tau_i_k+gradient_tau_sum;
        v_k_new(:,i)=v_k_i_new; 
    end

        
     %---------更新y_i(k+1)----------
    for i=1:agent_num
        ymid=zeros(300,1);%v(300x1)
        for j=1:agent_num
           ymid=ymid+C(i,j)*(y_k_last(:,j)+v_k_new(:,j)-v_k_last(:,j));
        end
        y_k_new(:,i)=ymid;
    end
        
    tau2_i=randperm(agent_m,1);
    for i=1:agent_num
       % x_k_i_last=x_k_last(:,i);
        %-----更新x_i(k+1)--------
        xk=zeros(300,1);
        for j=1:agent_num
            x_k_j_last=x_k_last(:,j);
            y_k_j_new=y_k_new(:,j);
            xk=xk+C(i,j)*(x_k_j_last-eta*y_k_j_new);
        end
        x_k_new(:,i)=xk;
        if(j==tau2_i)
            tau_k_last(:,i,j)=xk;
        end
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
