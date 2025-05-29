# CV-mid
本实验分为两个部分。任务一对应Resnet18，任务二对应mmdetection。
## Resnet18
1. 下载caltech-101数据集，并存放在`caltech-101/caltech-101/101_ObjectCategories`目录下。
2. 开始训练直接运行`python main.py`。\
   如果需要调整参数组，在`main.py`的 line 152-154 处直接调整参数选项。
3. 输出的曲线在`/plots`文件夹中，超参数搜索结果在`grid_search.csv`文件中。


## mmdetection
1. 下载VOC数据集（包括VOC2007与VOC2012），并存放至`/mmdetection/data/VOCdevkit`中。
2. 运行`python voc2coco.py`将VOC数据集划分成训练集、验证集与测试集，并转化为coco格式。
3. 将`voc_mask_rcnn_r50_fpn.py`存放至`/mmdetection/configs/mask_rcnn`中。
   将`voc_sparse_rcnn_r50_fpn.py`存放至`/mmdetection/configs/sparse_rcnn`中。
4. 首先运行`cd mmdetection`切换至mmdetection目录下，开始训练Mask R-CNN时输入以下命令：
   
   ```
   python tools/train.py \
   configs/mask_rcnn/voc_mask_rcnn_r50_fpn.py \
   --work-dir work_dirs/voc_mask_rcnn_r50_fpn
   ```

   运行会持续一段时间，并在`work_dirs/voc_mask_rcnn_r50_fpn`这个文件夹记录模型。

   
   类似地，对于Sparse R-CNN，我们输出以下命令：
    ```
   python tools/train.py \
   configs/sparse_rcnn/voc_sparse_rcnn_r50_fpn.py \
   --work-dir work_dirs/voc_sparse_rcnn_r50_fpn
   ```
   运行会持续一段时间，并在`work_dirs/voc_sparse_rcnn_r50_fpn`这个文件夹记录模型。



5. 训练结束后，要测试Mask R-CNN模型效果，执行以下命令：
   
   
   ```
   python tools/test.py \
   configs/Mask_RCNN/voc_mask_rcnn_r50_fpn.py \
   work_dirs/voc_mask_rcnn_r50_fpn/latest.pth \ 
   --eval mAP
   ```

   (latest.pth也可以用epoch_12.pth代替)
   类似地，要测试Sparse R-CNN模型效果，执行以下命令：
   ```
   python tools/test.py \
   configs/Sparse_RCNN/voc_sparse_rcnn_r50_fpn.py \
   work_dirs/voc_sparse_rcnn_r50_fpn/latest.pth \
   --eval mAP
   ```
   (latest.pth也可以用epoch_12.pth代替)

6. 若要在图像上进行测试验证，可以在填写`xxx_rcnn_test.py`中的路径后，直接执行
   ```
   python mask_rcnn_test.py
   ```
   或
   ```
   python sparse_rcnn_test.py
   ```
   结果会在外部保存下来。
