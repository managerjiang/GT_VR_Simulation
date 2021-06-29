clear all
%% ���ݼ���
load('a9a_smote.mat');%����A��A(48243x123):48243������
load('L_a9a_smote.mat');%����L��A(1x48243):48243�����
A=A1;
L=L1;%���ʹ��ԭʼ���ݣ�����Ҫ��������У�ͬʱ�޸�load������
A=double(A);
L=double(L);
L(L==0)=-1;
L(L==1)=1;
%% ��������
agent_num=10;% agent����
Maxgen=1600;% ��������
% C=doubly_stochastic(agent_num);
% C=doubly_stochastic(agent_num);% ����������ڽӾ����к��кͶ���1��
% C_store=C

 load('data/C_meth1_smote_800.mat');%����C_store
% C=C_store
x_k_last=zeros(123,agent_num);% x�� 
tau_ini=x_k_last;
y_k_last=zeros(123,agent_num);% y��
v_k_last=zeros(123,agent_num);% v��
lamuda1=0.5*10^(-5);
lamuda2=0.5*10^(-5);
%% ����Ԥ����
%��������������ü����ݣ�ÿ��������ʮ��֮һ������,ͬʱ������ʼ��
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
eta=0.1;%%����
T=8;
tic;
%% �㷨����
for k=1:Maxgen
    k   
    % ��ʼѭ���㷨
    for i=1:agent_num       
        x_k_i_last=x_k_last(:,i);
        %-----����x_i(k+1)--------
        xk=zeros(123,1);
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
%             tau_k_i_new=x_k_last(:,i);%%��tau���¸�ֵ
%         else
            tau_k_i_new=xk;
        
            %%----���¸�����ݶ�-----
            mid=L_cut(:,i).*A_cut(:,:,i); 
            gradient=-mid.*exp(mid*tau_k_i_new)./(1+exp(mid*tau_k_i_new)).^2;
            gradient=sum(gradient,1)/floor(size(A,1)/agent_num)+2*lamuda2*tau_k_i_new';
            gradient_tau_sum=gradient';
        end
        %-----����v_i(k+1)---------
       
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
     %---------����y_i(k+1)----------
    for i=1:agent_num
        ymid=zeros(123,1);%v(123x1)
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
