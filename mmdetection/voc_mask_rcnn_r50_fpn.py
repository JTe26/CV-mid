_base_ = [
    '/root/mmdetection/configs/_base_/models/mask-rcnn_r50_fpn.py',
    '/root/mmdetection/configs/_base_/datasets/coco_instance.py',
    '/root/mmdetection/configs/schedule_1x.py',
    '/root/mmdetection/configs/_base_/default_runtime.py'
]

class_names = (
    'aeroplane','bicycle','bird','boat','bottle',
    'bus','car','cat','chair','cow',
    'diningtable','dog','horse','motorbike',
    'person','pottedplant','sheep','sofa',
    'train','tvmonitor'
)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(class_names)),
        mask_head=dict(num_classes=len(class_names))
    )
)

# JSON files are in combined_coco directory
data_root = '/root/autodl-tmp/data/combined_coco/'

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=class_names),
        ann_file='voc0712_train.json',
        data_prefix=dict(img='JPEGImages/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    )
)

val_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=class_names),
        ann_file='voc0712_val.json',
        data_prefix=dict(img='JPEGImages/'),
        test_mode=True,
    )
)

# Separate test dataloader for test JSON
test_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        metainfo=dict(classes=class_names),
        ann_file='voc0712_test.json',
        data_prefix=dict(img='JPEGImages/'),
        test_mode=True,
    )
)

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'voc0712_val.json',
    metric=['bbox', 'segm']
)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'voc0712_test.json',
    metric=['bbox', 'segm']
)
