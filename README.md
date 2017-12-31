## 作业1

作业实现步骤：
    1、fork https://github.com/AI100-CSDN/quiz_slim 代码
    2、修改train_image_classifier.py 40行配置 clone_on_cpu 为 True
    3、下载数据集ai100-quiz-w7到本地目录
    4、下载checkpoint inception_v4.ckpt 到本地目录
    5、执行脚本
    python train_eval_image_classifier.py --dataset_name=quiz --dataset_dir=H:\000---Study\3_Python-ML\CSDN\HomeWork\Week_07\ai100-quiz-w7 --checkpoint_path=H:\000---Study\3_Python-ML\CSDN\HomeWork\Week_07\inception_v4\inception_v4.ckpt --model_name=inception_v4 --checkpoint_exclude_scopes=InceptionV4/Logits,InceptionV4/AuxLogits/Aux_logits --optimizer=rmsprop --train_dir=H:\000---Study\3_Python-ML\CSDN\HomeWork\Week_07\inception_v4\train_dir\ckpt --learning_rate=0.001 --dataset_split_name=validation --eval_dir=H:\000---Study\3_Python-ML\CSDN\HomeWork\Week_07\inception_v4\train_dir\eval --max_num_batches=128
    6、期间可以临时查看预测效果，执行脚本
    python eval_image_classifier.py --checkpoint_path=H:\000---Study\3_Python-ML\CSDN\HomeWork\Week_07\inception_v4\train_dir\ckpt --eval_dir=H:\000---Study\3_Python-ML\CSDN\HomeWork\Week_07\inception_v4\train_dir\eval  --dataset_name=quiz  --dataset_split_name=validation  --dataset_dir=H:\000---Study\3_Python-ML\CSDN\HomeWork\Week_07\ai100-quiz-w7  --model_name=inception_v4
    7、训练一段时间之后，修改train_image_classifier.py 40行配置 clone_on_cpu 为 False，并将GitHub项目关联至tinymind模型quiz-w1-1-lwl
    8、上传本地目录train_dir/ckpt下最新checkpoint3文件到tinymind新建数据集 inceptionv4checkpoint
    9、将模型dataset_dir 设置为 /data/ai100/quiz-w7  checkpoint_path设置为 inceptionv4checkpoint
    10、新建运行

PS：由于tinymind点数不够，又将checkpoint下载下来，最终在本地勉强跑出结果，已上传截图到码云附件中，名称为Inception_V4_作业结果截图.png，
结果为：

 eval/Recall_5[0.880859375]
 eval/Accuracy[0.713378906]


## 作业2

densenet网络结构实现参考了论文中对于ImageNet数据集的实现方式及Inception_V3中的实现方法。

    结构组成如下：
    1、4个blocks：dense_1、dense_2、dense_3、dense_final
    2、3个transition layers:trans_1、trans_2、trans_3
    3、进入dense_1之前，对输入数据进行了reshape，然后进行了 7×7 conv, stride 2 和 3×3 max pool, stride 2
    4、在dense_final之后，用一个全局平均池化Global_avg_pooling将输入tensor由[batch_size, 7, 7, 1440]变为[batch_size, 1, 1, 1440]
    5、全链接层：使用tf.layers.dense，将将输入tensor由[batch_size, 1, 1, 1440]变为[batch_size, 1, 1, 1000]
    6、对全链接层的数据加入dropout操作，防止过拟合
    7、Logits层，对dropout层的输出Tensor，执行分类操作

    执行脚本：
    python train_eval_image_classifier.py --dataset_name=quiz --dataset_dir=H:\\000---Study\\3_Python-ML\\CSDN\\HomeWork\Week_07\\ai100-quiz-w7 --model_name=densenet --train_dir=H:\\000---Study\\3_Python-ML\\CSDN\HomeWork\\Week_07\\desenet\\train_dir\\ckpt --learning_rate=0.001 --dataset_split_name=validation --eval_dir=H:\\000---Study\\3_Python-ML\\CSDN\HomeWork\\Week_07\\desenet\\train_dir\\eval --max_num_batches=128

    临时查看预测结果：
    python eval_image_classifier.py --checkpoint_path=H:\000---Study\3_Python-ML\CSDN\HomeWork\Week_07\desenet\train_dir\ckpt --eval_dir=H:\000---Study\3_Python-ML\CSDN\HomeWork\Week_07\desenet\train_dir\eval  --dataset_name=quiz  --dataset_split_name=validation  --dataset_dir=H:\000---Study\3_Python-ML\CSDN\HomeWork\Week_07\ai100-quiz-w7  --model_name=densenet --batch_size=32 --max_num_batches=128

PS: densenet.py实现过程中，有一些关于其他实现方法的注释、以及每一步tensor维度的变化，似懂非懂，希望助教老师帮我分析一下总结的对不对，谢谢。


