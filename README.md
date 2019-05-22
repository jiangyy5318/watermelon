# watermelon

### 训练数据准备
- 如data中, 每个文件夹代表一个西瓜, 则样本集中有三个西瓜, 分别是fold15, fold16, fold17;
- 每个文件夹中存有若干个wav文件, 文件是拍击西瓜的录音, 要求为.wav格式, 通道数为1, 如果不符合要求使用ffpmeg转化;
- score.txt按照csv个数存储了每个西瓜的成熟度;
- train.txt记录每个录音以及对应西瓜的成熟度, 由脚本generate_dataset.sh生成, 执行如下

```
cd $(projects)
cd data
sh generate_dataset.sh > train.txt
```

### 如何训练
当准备好数据之后, 就可以直接进行训练了, 训练的代码在src/tf_ffm_sound.py中实现, 主要使用因子分解机和快速傅里叶变换实现, 具体执行:

```
python3 src/tf_ffm_sound.py
```
PS:会有若干需要安装的库, 看到error就开始装吧!
### 线上代码
- 鉴于之前申请域名导致项目周期无限期延长并且间接导致代码丢失, 这里只提供线上预测的调用, 不提供后台;
- 注意送入预测代码的wav文件需要进行检测, 如果不是wav或者不是单通道, 请自行处理, 预测代码只接受正确的文件;

```
python3 src/predict.py
```
