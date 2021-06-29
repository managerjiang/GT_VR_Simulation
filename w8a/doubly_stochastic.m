function A=doubly_stochastic(n)
%产生 n 阶双随机矩阵 A
A(1,1)=rand;
for i=2:n-1
    d=1-sum(A(1,1:i-1));
    A(1,i)=d*rand;
end
for i=2:n-1
    d=1-sum(A(1:i-1,1));
    A(i,1)=d*rand;
end
for i=2:n-1
    for j=2:n-1
        d1=1-sum(A(i,1:j-1));
        d2=1-sum(A(1:i-1,j));
        d=min([d1 d2]);
        A(i,j)=d*rand;
    end
end
for i=1:n-1
    A(n,i)=1-sum(A(1:n-1,i));
end
for i=1:n
    A(i,n)=1-sum(A(i,1:n-1));
end
if A(n,n)<0
    A=1/n*ones(n);
end
%对称化A
A=(A'+A)/2;