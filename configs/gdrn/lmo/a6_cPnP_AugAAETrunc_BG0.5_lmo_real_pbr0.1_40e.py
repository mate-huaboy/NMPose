_base_ = ["../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output2/gdrn/lmo/a6_cPnP_AugAAETrunc_BG0.5_lmo_real_pbr0.1_40e"
INPUT = dict(
    DZI_PAD_SCALE=1.5,
      DZI_SCALE_RATIO=0.15,
 DZI_SHIFT_RATIO=0.1,
    TRUNCATE_FG=True,
    CHANGE_BG_PROB=0.5,
    COLOR_AUG_PROB=0.8,
    COLOR_AUG_TYPE="code",
    COLOR_AUG_CODE=(
        "Sequential(["
        # Sometimes(0.5, PerspectiveTransform(0.05)),
        # Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
        # Sometimes(0.5, Affine(scale=(1.0, 1.2))),
        "Sometimes(0.5, CoarseDropout( p=0.2, size_percent=0.05) ),"
        "Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),"
        "Sometimes(0.5, Add((-25, 25), per_channel=0.3)),"
        "Sometimes(0.3, Invert(0.2, per_channel=True)),"
        "Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),"
        "Sometimes(0.5, Multiply((0.6, 1.4))),"
        "Sometimes(0.5, LinearContrast((0.5, 2.2), per_channel=0.3))"
        "], random_order = False)"
        # aae
    ),
)

SOLVER = dict(
    IMS_PER_BATCH=24,
    TOTAL_EPOCHS=60,#原来是40
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.72,
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
)

DATASETS = dict(
    TRAIN=("lmo_train",),
    TRAIN2=("lmo_pbr_train",),
    TRAIN2_RATIO=0.1,
    TEST=("lmo_test",),

    # TRAIN=("train",),
    # TRAIN2=("train_pbr",),
    # TRAIN2_RATIO=0.1,
    # TEST=("test",),

    # AP	AP50	AR	inf.time
    # 60.657	89.625	66.2	0.024449
    DET_FILES_TEST=(
        "datasets/BOP_DATASETS/lmo/test/test_bboxes/faster_R50_FPN_AugCosyAAE_HalfAnchor_lmo_pbr_lmo_fuse_real_all_8e_test_480x640.json",
    ),
)

MODEL = dict(
    LOAD_DETS_TEST=True,
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    CDPN=dict(
          WEIGHTS='',
        BACKBONE=dict(
            PRETRAINED='torchvision://resnet34',
            ARCH='resnet',
            NUM_LAYERS=34,
            INPUT_CHANNEL=5,#修改
            INPUT_RES=256,
            OUTPUT_RES=64,
            ENABLED=True,
            FREEZE=False),#修改
        ROT_HEAD=dict(
            ENABLED=True,
            FREEZE=False,
            ROT_CLASS_AWARE=False,
            MASK_CLASS_AWARE=False,
             XYZ_LOSS_TYPE='Cos_smi',#
            XYZ_LW=1.0,
            REGION_CLASS_AWARE=False,
             REGION_LOSS_TYPE='R_cos',
              REGION_ATTENTION=True,
            # NUM_REGIONS=64,
            NUM_REGIONS=3,

        ),
        PNP_NET=dict(
            FREEZE=True,
            ENABLE=False,
            R_ONLY=True,
            # REGION_ATTENTION=True,改
             NUM_LAYERS=4,
            REGION_ATTENTION=True,
             PM_LOSS_TYPE='R_normal_loss',#修改
            WITH_2D_COORD=False,#去掉2d对应
            ROT_TYPE="allo_rot6d",
            TRANS_TYPE="centroid_z",
            PM_NORM_BY_EXTENT=True,
            PM_LOSS_SYM=True,  #注意这个参数
            PM_R_ONLY=True,
            CENTROID_LOSS_TYPE="L1",
            CENTROID_LW=1.0,
            Z_LOSS_TYPE="L1",
            Z_LW=1.0,  
            TRUE_NORMAL=False,
            CENTER_TRANS=False
        ),
        TRANS_HEAD=dict(ENABLED=True,
        FREEZE=False
        ),
    ),
)

TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="est")  # gt | est
