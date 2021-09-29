#用户数据
user_watch.npy 存储了用户的watch记录，64维
user_search.npy存储了用户的search记录，64维
user_feat.npy存储了用户的如年龄，工作等特征，32维
user_labels.npy存储了标签，用户观看3952个电影中的哪一个

#数据分割
distribute_data.py
user vecotr一共64+64+32=160维，我们划分为64维和64+32=96维

#SPlitNN网络结构
splitnn_net.py
是splitNN的网络结构

#训练过程
训练过程在split.py中

