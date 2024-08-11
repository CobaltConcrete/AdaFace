import torch
from mmcv import Config
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import cv2
import numpy as np

# Step 1: Set up the environment
# Make sure you have installed the necessary libraries:
# pip install mmcv-full mmdet opencv-python

# Step 2: Load the configuration
config_dict = {
    'data_pipeline': [
        dict(typename='LoadImageFromFile'),
        dict(
            typename='MultiScaleFlipAug',
            img_scale=(1100, 1650),
            flip=False,
            transforms=[
                dict(typename='Resize', keep_ratio=True),
                dict(typename='RandomFlip', flip_ratio=0.0),
                dict(typename='Normalize', mean=[123.675, 116.28, 103.53], std=[1, 1, 1], to_rgb=True),
                dict(typename='Pad', size_divisor=32, pad_val=0),
                dict(typename='ImageToTensor', keys=['img']),
                dict(typename='Collect', keys=['img'])
            ])
    ],
    'model': dict(
        typename='SingleStageDetector',
        backbone=dict(
            typename='ResNet',
            depth=50,
            num_stages=4,
            out_indices=(0, 1, 2, 3),
            norm_cfg=dict(typename='GN', num_groups=32, requires_grad=True),
            norm_eval=False,
            dcn=dict(typename='DCN', deformable_groups=1, fallback_on_stride=False),
            stage_with_dcn=(False, False, True, True),
            style='pytorch'),
        neck=[
            dict(
                typename='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                start_level=0,
                add_extra_convs='on_input',
                num_outs=6,
                norm_cfg=dict(typename='GN', num_groups=32, requires_grad=True),
                upsample_cfg=dict(mode='bilinear')),
            dict(
                typename='Inception',
                in_channel=256,
                num_levels=6,
                norm_cfg=dict(typename='GN', num_groups=32, requires_grad=True),
                share=True)
        ],
        head=dict(
            typename='IoUAwareRetinaHead',
            num_classes=1,
            num_anchors=3 * 1,  # scales_per_octave * len(ratios)
            in_channels=256,
            stacked_convs=4,
            feat_channels=256,
            norm_cfg=dict(typename='GN', num_groups=32, requires_grad=True),
            use_sigmoid=True)),
    'infer_engine': dict(
        typename='InferEngine',
        model=None,
        meshgrid=dict(
            typename='BBoxAnchorMeshGrid',
            strides=[4, 8, 16, 32, 64, 128],
            base_anchor=dict(
                typename='BBoxBaseAnchor',
                octave_base_scale=2**(4 / 3),
                scales_per_octave=3,
                ratios=[1.3],
                base_sizes=[4, 8, 16, 32, 64, 128])),
        converter=dict(
            typename='IoUBBoxAnchorConverter',
            num_classes=1,
            bbox_coder=dict(
                typename='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            nms_pre=10000,
            use_sigmoid=True),
        num_classes=1,
        test_cfg=dict(
            min_bbox_size=0,
            score_thr=0.4,
            nms=dict(
                typename='nms',
                iou_thr=0.45),
            max_per_img=300),
        use_sigmoid=True),
    'weights': dict(filepath='pretrained/tinaface_r50_fpn_gn_dcn.pth'),
    'class_names': ('face', )
}

cfg = Config(config_dict)

# Step 3: Initialize the model
model = init_detector(cfg, cfg.weights['filepath'], device='cuda:0')

# Step 4: Inference
image_path = 'Projects/AdaFace/face_alignment/test_images/RDJ1.jpeg'
result = inference_detector(model, image_path)

# Extract bounding boxes and corresponding face images
def extract_faces(image_path, result, score_thr=0.4):
    image = cv2.imread(image_path)
    bboxes = result[0]
    faces = []

    for bbox in bboxes:
        if bbox[4] >= score_thr:
            x1, y1, x2, y2, score = bbox
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            face = image[y1:y2, x1:x2]
            faces.append((face, (x1, y1, x2, y2)))

    return faces

faces = extract_faces(image_path, result)

# Step 5: Visualization
show_result_pyplot(model, image_path, result, score_thr=0.4, class_names=cfg.class_names)

# Print bounding boxes and display extracted faces
for i, (face, bbox) in enumerate(faces):
    print(f"Face {i+1}: Bounding box {bbox}")
    cv2.imshow(f"Face {i+1}", face)
    cv2.waitKey(0)

cv2.destroyAllWindows()
