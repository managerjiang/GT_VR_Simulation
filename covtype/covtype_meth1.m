clear all
%% 数据加载
load('cov.mat');%载入A：A(581012x54):581012个数据
load('L_cov.mat');%载入L：A(1x581012):581012个结果
A=double(A);
L=double(L);
L(L==1)=-1;
L(L==2)=1;
A(435756:581009,:)=[];
L(435756:581009)=[];
%% 参数设置
agent_num=10;% agent个数
Maxgen=1600;% 迭代次数
% C=doubly_stochastic(agent_num);
% C=doubly_stochastic(agent_num);% 生成随机的邻接矩阵（行和列和都是1）
% C_store=C

 load('data/C_meth1_smote_800.mat');%载入C_store
 C=C_store
x_k_last=zeros(54,agent_num);% x阵 
tau_ini=x_k_last;
y_k_last=zeros(54,agent_num);% y阵
v_k_last=zeros(54,agent_num);% v阵
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
    gradient_tau_sum=gradient';
end
eta=0.1;%%步长
T=8;
tic;
%% 算法主体
for k=1:Maxgen
    k   
    % 开始循环算法
    for i=1:agent_num       
        x_k_i_last=x_k_last(:,i);
        %-----更新x_i(k+1)--------
        xk=zeros(54,1);
        for j=1:agent_num
            x_k_j_last=x_k_last(:,j);
            y_k_j_last=y_k_last(:,j);
            xk=xk+C(i,j)*(x_k_j_last-eta*y_k_j_last);
        end
        x_k_new(:,i)=xk;
        s_i=randperm(floor(size(A,1)/agent_num),1);
        l_i=randperm(agent_num,1);
        if k==1
            tau_k_i_new=tau_ini(:,i);
        end
%         if mod(l_i,2)==0
        if (rand(1)>0.7)
%             tau_k_i_new=x_k_last(:,i);%%给tau重新赋值
%         else
            tau_k_i_new=xk;
        
            %%----更新个体和梯度-----
            mid=L_cut(:,i).*A_cut(:,:,i); 
            gradient=-mid.*exp(mid*tau_k_i_new)./(1+exp(mid*tau_k_i_new)).^2;
            gradient=sum(gradient,1)/floor(size(A,1)/agent_num)+2*lamuda2*tau_k_i_new';
            gradient_tau_sum=gradient';
        end
        %-----更新v_i(k+1)---------
       
         mid=L_cut(:,i).*A_cut(:,:,i); 
        gradient_x=-mid(s_i,:)*exp(mid(s_i,:)*xk)/(1+exp(mid(s_i,:)*xk))^2;
        gradient_x=sum(gradient_x,1)+2*lamuda2*xk';
        gradient_x_i_k=gradient_x';
        gradient_tau=-mid(s_i,:)*exp(mid(s_i,:)*tau_k_i_new)/(1+exp(mid(s_i,:)*tau_k_i_new))^2;
        gradient_tau=sum(gradient_tau,1)+2*lamuda2*tau_k_i_new';
        gradient_tau_i_k=gradient_tau';
        v_k_i_new=gradient_x_i_k-gradient_tau_i_k+gradient_tau_sum;
        v_k_new(:,i)=v_k_i_new;
    end
     %---------更新y_i(k+1)----------
    for i=1:agent_num
        ymid=zeros(54,1);%v(123x1)
        for j=1:agent_num
           ymid=ymid+C(i,j)*(y_k_last(:,j)+v_k_new(:,j)-v_k_last(:,j));
        end
        y_k_new(:,i)=ymid;
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
