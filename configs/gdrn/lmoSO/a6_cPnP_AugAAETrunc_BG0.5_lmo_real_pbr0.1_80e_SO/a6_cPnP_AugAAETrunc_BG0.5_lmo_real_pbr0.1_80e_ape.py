_base_ = ["../../../_base_/gdrn_base.py"]

OUTPUT_DIR = "output3/gdrn/lmoSO/a6_cPnP_AugAAETrunc_lmo_real_pbr0.1_80e_SO/ape"#修改
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
    TOTAL_EPOCHS=80,#修改
    LR_SCHEDULER_NAME="flat_and_anneal",
    ANNEAL_METHOD="cosine",  # "cosine"
    ANNEAL_POINT=0.64,
    # REL_STEPS=(0.3125, 0.625, 0.9375),
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),#修改
    WEIGHT_DECAY=0.0,
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
)

DATASETS = dict(
    TRAIN=("lm_real_ape_all",),
    TRAIN2=("lmo_pbr_ape_train",),
    TRAIN2_RATIO=0.1,
    TEST=("lmo_test",),
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
        NAME='GDRN',
        TASK='rot',
        USE_MTL=False,
           WEIGHTS='',
        BACKBONE=dict(
            PRETRAINED='torchvision://resnet34',
            ARCH='resnet',
            NUM_LAYERS=34,
            INPUT_CHANNEL=3,
            INPUT_RES=256,
            OUTPUT_RES=64,
            ENABLED=True,
            FREEZE=True),#修改
        ROT_HEAD=dict(
            ROT_CONCAT=False,
            XYZ_BIN=64,
            NUM_LAYERS=3,
            NUM_FILTERS=256,
            CONV_KERNEL_SIZE=3,
            NORM='BN',
            NUM_GN_GROUPS=32,
            OUT_CONV_KERNEL_SIZE=1,
            NUM_CLASSES=13,
            ROT_CLASS_AWARE=False,
            XYZ_LOSS_TYPE='Cos_smi',#
            XYZ_LOSS_MASK_GT='visib',
            XYZ_LW=1.0,
            MASK_CLASS_AWARE=False,
            MASK_LOSS_TYPE='L1',
            MASK_LOSS_GT='trunc',
            MASK_LW=1.0,
            MASK_THR_TEST=0.5,
            REGION_ATTENTION=True,
            NUM_REGIONS=2,
            REGION_CLASS_AWARE=False,
            REGION_LOSS_TYPE='CE',
            REGION_LOSS_MASK_GT='visib',
            REGION_LW=1.0,
            FREEZE=True,
            ENABLED=True),#修改
        PNP_NET=dict(
            R_ONLY=True,
            LR_MULT=1.0,
            PNP_HEAD_CFG=dict(
                type='ConvPnPNet', norm='GN', num_gn_groups=32, drop_prob=0.0),
            WITH_2D_COORD=False,#修改
            REGION_ATTENTION=False,
            MASK_ATTENTION='none',
            TRANS_WITH_BOX_INFO='none',
            ROT_TYPE='allo_rot6d',
            TRANS_TYPE='centroid_z',
            Z_TYPE='REL',
            NUM_PM_POINTS=3000,
            # PM_LOSS_TYPE='normal_loss',#修改
            PM_LOSS_TYPE='L1',#修改
            PM_SMOOTH_L1_BETA=1.0,
            PM_LOSS_SYM=False,  #注意这个参数
            PM_NORM_BY_EXTENT=True,
            PM_R_ONLY=True,
            PM_DISENTANGLE_T=False,
            PM_DISENTANGLE_Z=False,
            PM_T_USE_POINTS=False,
            PM_LW=1.0,
            ROT_LOSS_TYPE='angular',
            ROT_LW=0.0,
            CENTROID_LOSS_TYPE='L1',
            CENTROID_LW=1.0,
            Z_LOSS_TYPE='L1',
            Z_LW=1.0,
            TRANS_LOSS_TYPE='L1',
            TRANS_LOSS_DISENTANGLE=True,
            TRANS_LW=0.0,
            BIND_LOSS_TYPE='L1',
            BIND_LW=0.0,
            FREEZE=False,
            ENABLE=True,#修改
            CENTER_TRANS=True),#可以修改
        TRANS_HEAD=dict(
            ENABLED=True,
            FREEZE=True,#修改
            LR_MULT=1.0,
            NUM_LAYERS=3,
            NUM_FILTERS=256,
            NORM='BN',
            NUM_GN_GROUPS=32,
            CONV_KERNEL_SIZE=3,
            OUT_CHANNEL=3,
            TRANS_TYPE='centroid_z',
            Z_TYPE='REL',
            CENTROID_LOSS_TYPE='L1',
            CENTROID_LW=0.0,
            Z_LOSS_TYPE='L1',
            Z_LW=0.0,
            TRANS_LOSS_TYPE='L1',
            TRANS_LW=0.0)),
)

VAL = dict(
    DATASET_NAME="lmo",
    SCRIPT_PATH="lib/pysixd/scripts/eval_pose_results_more.py",
    TARGETS_FILENAME="test_targets_all.json",
    ERROR_TYPES="ad,rete,re,te,proj",
    RENDERER_TYPE="egl",  # cpp, python, egl
    SPLIT="test",
    SPLIT_TYPE="bb8",
    N_TOP=1,  # SISO: 1, VIVO: -1 (for LINEMOD, 1/-1 are the same)
    EVAL_CACHED=False,  # if the predicted poses have been saved
    SCORE_ONLY=False,  # if the errors have been calculated
    EVAL_PRINT_ONLY=False,  # if the scores/recalls have been saved
    EVAL_PRECISION=False,  # use precision or recall
    USE_BOP=True,  # whether to use bop toolkit
)

TEST = dict(EVAL_PERIOD=0, VIS=False, TEST_BBOX_TYPE="est")  # gt | est
