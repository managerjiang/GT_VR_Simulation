close all;
%% meth1
% ==============需要读取.mat文件================
  load('data/G_meth1_smote_800.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth1_smote_800.mat');%加载保存的迭代解信息gradient()
  load('data/C_meth1_smote_800.mat');%加载保存的邻接矩阵信息C_store
  load('w8a.mat');
  load('L_w8a.mat');
  L=double(L);
  L(L==0)=-1;
  C=C_store;
% 参数设置
Maxgen00=size(x_k_store,2);
Maxgen=size(x_k_store,2)/20;%迭代次数
agent_num=size(C);%智能体个数

% 训练集
for k=1:Maxgen
  x_k=x_k_store{k*20};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  fi=sum(1./(1+exp(L'.*A*x_k_1)),1);
  obj_m1(k)=fi/size(A,1)+lamuda2*norm(x_k_1,2)^2; %%函数值
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m1(k)=XLX;%%一致性
  % 梯度值
   %取所有智能体中第一个
   g_k=gradient_sto{k*20};
   zz_m1(k)=norm(g_k);
end
% 测试集
  load('w8a_test.mat');
  load('L_w8a_test.mat');
  L=double(L);
  L(L==0)=-1;
for k=1:Maxgen
     % 测试集正确率
  x_k=x_k_store{k*20};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  result=A*x_k_1;
  result(result>=0)=1;
  result(result<0)=-1;
  yy_m1(k)=sum((result==L'))/size(L,2);
  
end

%% meth2
% ==============需要读取.mat文件================
  load('data/G_meth2_smote_800.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth2_smote_800.mat');%加载保存的迭代解信息gradient()
  %load('data/C_meth2_smote_800.mat');%加载保存的邻接矩阵信息C_store
  load('w8a.mat');
  load('L_w8a.mat');
  L=double(L);
  L(L==0)=-1;
  C=C_store;
% 参数设置
Maxgen=size(x_k_store,2)/20;%迭代次数
agent_num=size(C);%智能体个数

% 训练集
for k=1:Maxgen
  x_k=x_k_store{k*20};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  fi=sum(1./(1+exp(L'.*A*x_k_1)),1);
  obj_m2(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)+lamuda2*norm(x_k_1,2)^2; 
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m2(k)=XLX;
  % 梯度值
   %取所有智能体中第一个
   g_k=gradient_sto{k*20};
   zz_m2(k)=norm(g_k);
end
% 测试集
  load('w8a_test.mat');
  load('L_w8a_test.mat');
  L=double(L);
  L(L==0)=-1;
for k=1:Maxgen
     % 测试集正确率
  x_k=x_k_store{k*20};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  result=A*x_k_1;
  result(result>=0)=1;
  result(result<0)=-1;

  yy_m2(k)=sum((result==L'))/size(L,2);
  
end

%% meth3--------------------------------------------------------------%%%%%%%%%%%%
% ==============需要读取.mat文件================
  load('data/G_meth4_smote_800.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth4_smote_800.mat');%加载保存的迭代解信息gradient()
%  load('data/C_meth3_smote_800.mat');%加载保存的邻接矩阵信息C_store
  load('w8a.mat');
  load('L_w8a.mat');
  L=double(L);
  L(L==0)=-1;
  C=C_store;
% 参数设置
Maxgen=size(x_k_store,1);%迭代次数
agent_num=size(C);%智能体个数
Maxgen4=Maxgen;
% 训练集
for k=1:Maxgen
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  fi=sum(1./(1+exp(L'.*A*x_k_1)),1);
  obj_m4(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)+lamuda2*norm(x_k_1,2)^2; 
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m4(k)=XLX;
  % 梯度值
   %取所有智能体中第一个
   g_k=gradient_sto{k};
   zz_m4(k)=norm(g_k);
end
% 测试集
  load('w8a_test.mat');
  load('L_w8a_test.mat');
  L=double(L);
  L(L==0)=-1;
for k=1:Maxgen
     % 测试集正确率
  x_k=x_k_store{k};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  result=A*x_k_1;
  result(result>=0)=1;
  result(result<0)=-1;

  yy_m4(k)=sum((result==L'))/size(L,2);
  
end

%% meth4
% ==============需要读取.mat文件================
  load('data/G_meth5_smote_800.mat');%加载保存的迭代梯度信息x_k_store{}
  load('data/X_meth5_smote_800.mat');%加载保存的迭代解信息gradient()
%  load('data/C_meth3_smote_800.mat');%加载保存的邻接矩阵信息C_store
  load('w8a.mat');
  load('L_w8a.mat');
  L=double(L);
  L(L==0)=-1;
  C=C_store;
% 参数设置
Maxgen=size(x_k_store,2)/20;%迭代次数
agent_num=size(C);%智能体个数

% 训练集
for k=1:Maxgen
  x_k=x_k_store{k*20};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  % 目标函数
  fi=sum(1./(1+exp(L'.*A*x_k_1)),1);
  obj_m5(k)=fi/size(A,1)+lamuda1*norm(x_k_1,1)+lamuda2*norm(x_k_1,2)^2; 
  %  XLX
  XLX=0;
  for i=1:agent_num
    temp_X=zeros(size(A,2),1);
    for j=1:agent_num
        temp_X=temp_X+C(i,j)*(x_k(:,i)-x_k(:,j));
    end
    XLX=XLX+x_k(:,i)'*temp_X;
  end
  w_m5(k)=XLX;
  % 梯度值
   %取所有智能体中第一个
   g_k=gradient_sto{k*20};
   zz_m5(k)=norm(g_k);
end
% 测试集
  load('w8a_test.mat');
  load('L_w8a_test.mat');
  L=double(L);
  L(L==0)=-1;
 
for k=1:Maxgen
     % 测试集正确率
  x_k=x_k_store{k*20};
  x_k_1=x_k(:,1);%取所有智能体中第一个
  result=A*x_k_1;
  result(result>=0)=1;
  result(result<0)=-1;

  yy_m5(k)=sum((result==L'))/size(L,2);
  
end


%% 绘图  
figure(1);

subplot(1,2,1)
plot(1:20:Maxgen00,w_m1,'r-.','linewidth',1),hold on;
plot(1:20:Maxgen00,w_m2,'b--','linewidth',1),hold on;
plot(1:20:Maxgen00,w_m4,'k-.','linewidth',1),hold on;
plot(1:20:Maxgen00,w_m5,'m-','linewidth',1),hold on;
 legend('GT-VR','GT-SAGA','GT-SARAH','D-GET');
ylabel('$$\|D(\bar{x})\|$$','Interpreter','latex')
xlabel('iterations');
 subplot(1,2,2)
 plot(1:20:Maxgen00,obj_m1,'r-.','linewidth',1),hold on;
 plot(1:20:Maxgen00,obj_m2,'b--','linewidth',1),hold on;
   plot(1:20:Maxgen00,obj_m4,'k-.','linewidth',1),hold on;
 plot(1:20:Maxgen00,obj_m5,'m-','linewidth',1),hold on;
 legend('GT-VR','GT-SAGA','GT-SARAH','D-GET');
 ylabel('$$f(x)$$','Interpreter','latex')
 xlabel('iterations')

