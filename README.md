## 可视化图像搜索Workshop
这个Workshop是实现了端到端的一个可视化图像搜索应用程序。它主要使用使用Amazon SageMaker和Amazon Elasticsearch等组件。架构如下：

![arch](./pics/arch.png)

这个workshop面向AWS中国区域部署，在Global部署请参考：

[aws-samples/amazon-sagemaker-visual-search](https://github.com/aws-samples/amazon-sagemaker-visual-search)

## 如何工作?

我们将使用feidegger (zalandoresearch数据集)的Fashion Images作为参考图像或者自定义数据集比如地形图，使用卷积神经网络生成2048个特征向量，并存储到Amazon Elasticsearch KNN索引中

![diagram](./pics/ref.png)

当我们提出一个新的查询图像时，它正在计算来自Amazon SageMaker托管模型的相关特征向量，并查询Amazon Elasticsearch KNN索引来找到类似的图像

![diagram](./pics/query.png)

## 实验步骤
1、复制项目代码，如果在中国无法直接git clone，可以浏览器进入下载项目zip包。
```
git clone https://github.com/comdaze/sagemaker-visual-image-search.git
```
2、采用cfn/initial_template_cn.yaml文件在Cloudformation服务中运行，创建初始化资源。

3、在Notebook实例的的jupterlab中打开[visual-image-search-gluoncv-cn.ipynb](./visual-image-search-gluoncv-cn.ipynb)，按照这个notebook继续执行。


## License

This library is licensed under the MIT-0 License. See the LICENSE file.
