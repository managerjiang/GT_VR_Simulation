文件说明：
1.数据集
a9a.mat 是训练集数据（生成A矩阵[32561x123]）
L.mat 是训练集标签（生成A矩阵[1x32561]）
a9a_test.mat 是测试集数据（生成A矩阵[16281x123]）
L_test.mat 是测试集标签（生成A矩阵[1x16281]）
a9a_smote.mat是smote之后的训练集数据（生成A矩阵[48243x123]）
L_smote.mat是smote之后的训练集标签（生成A矩阵[1x48243]）
2.主程序
a9a_meth1.m 是GT-VR算法
a9a_meth2.m 是GT-SAGA算法
a9a_meth3.m 是GT-SARAH算法
a9a_meth4.m 是D-GET算法

3.功能程序
drawthem.m绘图程序（执行完上面的主程序，直接运行，也可以手动输入数据）
smote.m：另一种新的数据处理方法，对原始数据正负样本差距过大进行纠正。具体原理：smote法，在7800个正样本里面，依次for循环，每次循环找距离该点最近的m个点，随机选其中一个连线，再在连线上随机找1个点作为插值点，重复k次。（由于负样本大约是32500-7800个，因此k取3，m取10，拓充后的数据约为48200个，正负样本比例约为0.96：1）