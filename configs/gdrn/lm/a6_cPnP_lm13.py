

_base_ = ["../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output/gdrn/lm/a6_cPnP_lm13"#改变一下输出路径以同时跑两个
INPUT = dict(
    DZI_PAD_SCALE=1.5,
    COLOR_AUG_PROB=0.0,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        "Sometimes(0.4, CoarseDropout( p=0.1, size_percent=0.05) ),"
        # "Sometimes(0.5, Affine(scale=(1.0, 1.2))),"
        "Sometimes(0.5, GaussianBlur(np.random.rand())),"
        "Sometimes(0.5, Add((-20, 20), per_channel=0.3)),"
        "Sometimes(0.4, Invert(0.20, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.7, 1.4), per_channel=0.8)),"
        "Sometimes(0.5, Multiply((0.7, 1.4))),"
        "Sometimes(0.5, ContrastNormalization((0.5, 2.0), per_channel=0.3))"
        "], random_order=False)"
    ),
)

SOLVER = dict(
    IMS_PER_BATCH=24,  #barchsize 24?
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,# 修改之前是0.72
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    # OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),#
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),#修改一下学习率
    
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
)

DATASETS = dict(
    TRAIN=("lm_13_train", "lm_imgn_13_train_1k_per_obj"),
    TEST=("lm_13_test",),
    DET_FILES_TEST=("datasets/BOP_DATASETS/lm/test/test_bboxes/bbox_faster_all.json",),#get box by fastercnn in the file  by wenhua
)

MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    CDPN=dict(
        BACKBONE=dict( FREEZE=True) ,#同时不训练
        ROT_HEAD=dict(

            # ENABLED=False,#去掉旋转tou
            FREEZE=True,  #同时不训练
            ENABLED=True,#去掉旋转tou
            # FREEZE=False,  #同时不训练
            # FREEZE=False,
            ROT_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
            XYZ_LW=1.0,
            REGION_CLASS_AWARE=False,
            # NUM_REGIONS=64,  #这里是不是要改
            NUM_REGIONS=0,  #这里是不是要改

        ),
        PNP_NET=dict(
            # R_ONLY=False,  #这里改为true会怎样？
            FREEZE=False,
            # FREEZE=True,
            R_ONLY=True,
            CENTER_TRANS=True,
            # REGION_ATTENTION=True,
            # WITH_2D_COORD=True,
            REGION_ATTENTION=False,
            WITH_2D_COORD=False,
            ROT_TYPE="allo_rot6d",
            # ROT_TYPE="allo_quat",
            
            TRANS_TYPE="centroid_z",
            PM_NORM_BY_EXTENT=True,
            PM_R_ONLY=True,
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,
           
        ),
        TRANS_HEAD=dict(FREEZE=True),
        # TRANS_HEAD=dict(FREEZE=False),

    ),
)

TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="est")  # gt | est可见可以选择bbox，先看一看可视化吧
