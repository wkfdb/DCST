_base_="base.py"


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


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(

        sup=dict(

            ann_file="data/coco/annotations/instances_train2017.json",
            img_prefix="data/coco/train2017/",

        ),
        unsup=dict(

            ann_file="data/coco/annotations/instances_unlabeled2017.json",
            img_prefix="data/coco/unlabeled2017/",

        ),
    ),
    sampler=dict(
        train=dict(
            sample_ratio=[1, 1],
        )
    ),
)

semi_wrapper = dict(
    train_cfg=dict(
        unsup_weight=2.0,
    )
)

# semi_wrapper = dict(
#     train_cfg=dict(
#     rpn_pseudo_threshold=0.9,
#     cls_pseudo_threshold=0.9
#     )
# )

lr_config = dict(step=[120000 * 4, 160000 * 4])
runner = dict(_delete_=True, type="IterBasedRunner", max_iters=180000 * 4)

work_dir = "work_dirs/coco_full/"

