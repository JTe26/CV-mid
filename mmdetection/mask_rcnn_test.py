from mmdet.apis import init_detector, inference_detector
from mmdet.registry import VISUALIZERS
import mmcv
import os
import matplotlib.pyplot as plt

config_file     = ''
pretrained_ckpt = ''
finetuned_ckpt  = ''  #这三行补充config训练的路径
device = 'cuda:0'

model_pre  = init_detector(config_file, pretrained_ckpt, device=device)
model_post = init_detector(config_file, finetuned_ckpt,  device=device)

out_dir = 'demo/results'
os.makedirs(out_dir, exist_ok=True)

imgs_pre_vis, imgs_post_vis = [], []
img_paths = ['demo/000100.jpg'] # 这里填写要测试的图像
    img = mmcv.imread(img_path)
    res_pre  = inference_detector(model_pre,  img)
    res_post = inference_detector(model_post, img)

    # 前模型可视化
    viz_pre = VISUALIZERS.build(model_pre.cfg.visualizer)
    viz_pre.dataset_meta = model_pre.dataset_meta
    viz_pre.add_datasample(
        'pre', image=img, data_sample=res_pre,
        draw_gt=False, pred_score_thr=0.15, show=False
    )
    imgs_pre_vis.append(viz_pre.get_image())

    # 后模型可视化
    viz_post = VISUALIZERS.build(model_post.cfg.visualizer)
    viz_post.dataset_meta = model_post.dataset_meta
    viz_post.add_datasample(
        'post', image=img, data_sample=res_post,
        draw_gt=False, pred_score_thr=0.5, show=False
    )
    imgs_post_vis.append(viz_post.get_image())

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(mmcv.bgr2rgb(imgs_pre_vis[0]))
axes[0].set_title('img_1 — Before')
axes[0].axis('off')

axes[1].imshow(mmcv.bgr2rgb(imgs_post_vis[0]))
axes[1].set_title('img_1 — After')
axes[1].axis('off')

plt.tight_layout()

# 保存对比图
output_path = 'demo/results/Mask_RCNN/img_1_comparison_in.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')  # dpi 可以根据需求调整
plt.show()

print(f'对比图已保存到：{output_path}')
