## Tensorflow运作方式入门  
 
### 1.目的    
用含有2个隐含层的神经网络实现mnist数据集手写数字识别。    
同时可以用Tensorboard可视化学习。    

### 2.code

此部分用到了2个python文件：mnist包下的mnist.py和fully_connected_feed.py    

### 3.知识点  
#### 3.1. tf.size(input,name=None,out_type=tf.int32)  
作用：返回一个tensor的size,即input的元素数量  
参数：  
name:操作的名字  
out_type：输出的类型(int32或int64)，默认为int32  
eg：  
 't' is [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]]]  
size(t) ==> 12  


#### 3.2.tf.sparse_to_dense(sparse_indices, output_shape, sparse_values,default_value)  
输出一个onehot标签的矩阵  
参数说明：  
第一个参数sparse_indices：稀疏矩阵中那些个别元素对应的索引值。  
第二个参数output_shape：输出的稀疏矩阵的shape  
第三个参数sparse_values：个别元素的值。  
第四个参数default_value：未指定元素的默认值，一般如果是稀疏矩阵的话就是0了  

#### 3.3. tf.concat(concat_dim, values, name='concat')：  
连接两个矩阵  
参数说明：  
name:操作的名字  
concat_dim：必须是一个数，表明在哪一维上连接  
     如果concat_dim是0，那么在某一个shape的第一个维度上连，对应到实际，就是叠放到列上  
values：就是两个或者一组待连接的tensor  

#### 3.4. 利用Tensorboard可视化学习
首先启动Tensorboard：python tensorflow/tensorboard/tensorboard.py --logdir=path/to/log-directory    

然后在浏览器中输入 localhost:6006 来查看 TensorBoard。 
