## 情绪分类（图片）

在testImages_artphoto上进行图像分类。训练测试划分大概是7:3。

最初就是一个简单的图像分类问题。但是随便拿了个模型一跑，嗯。。。精度低的离谱。
最初好像top1才刚到20%。训练集一共五百多张图片，还是八分类，图片还抽象的离谱。所以就开始不断地尝试提点了。。。


####项目地址： https://github.com/mycfhs/mycf_emo_recog

## 文件目录

----mycf_emo_recog

--------haarshare(dir)

--------testImage_artphoto(dir)

--------两个数据集的txt和一堆py文件

###下载地址（百度网盘）：
haarshare：
链接：https://pan.baidu.com/s/1qNbRf8ShCIZc8d-ttA129Q 

提取码：vc7j

testImage_artphoto：
链接：https://pan.baidu.com/s/1pmNznSOOeY6-FfmIZM2C0Q 

提取码：3ixg

### main.py

新手村的程序（）。这个程序主要是用来测试不同的dataloader的。

### train_test_func.py

训练和测试用的函数。

### find_model.py

用这个程序试了一部分模型，结果差强人意。至少找模型的任务是完成了，虽然普遍精度都不高，但有几个相对高一点点，矮子里的高个子？
靠着这个最后把模型锁定在了vgg19和resnet152了。哦对了，考虑到数据集的抽象，还把Vit作为了备选模型。

### dataloader.py

自己手撸的读数据模型，优点是便于加功能，缺点是，在有些模型上精度持续为0，
可能是哪里写的有问题吧，没找到。。。欢迎指正~


### divideDataset.py 和 datasetLarger,py

后来尝试用torch的从文件夹读取数据的那个函数，这俩就是一个把训练验证分开，一个做数据增广。

### autogluon,py

尝试的autogluon程序。算是最终结果吧。到了50%。

没思路了，五十就五十吧，开摆了。
