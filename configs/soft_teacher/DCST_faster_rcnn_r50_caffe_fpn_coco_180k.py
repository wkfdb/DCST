_base_ = "base.py"


model = dict(
    roi_head=dict(
        bbox_head = dict(
            type='Shared2FCBBoxHead2',
            loss_cls=dict(
                type='mySoftCrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0)
        )
    ),
)



semi_wrapper = dict(
    train_cfg=dict(
        rpn_pseudo_threshold=0.7,
        cls_pseudo_threshold=0.7,
    )
)


data = dict(
    samples_per_gpu=10,
    workers_per_gpu=2,
    train=dict(
        sup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}.json",
            img_prefix="data/coco/train2017/",
        ),
        unsup=dict(
            type="CocoDataset",
            ann_file="data/coco/annotations/semi_supervised/instances_train2017.${fold}@${percent}-unlabeled.json",
            img_prefix="data/coco/train2017/",
        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 4],
        )
    ),
)

fold = 1
percent = 1

work_dir = "work_dirs/${cfg_name}/${percent}/${fold}"
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
