OUTPUT_ROOT = 'output'
OUTPUT_DIR = 'output7/gdrn/lm/a6_cPnP_lm13'#可以修改
EXP_NAME = ''
DEBUG = False
SEED = -1
CUDNN_BENCHMARK = True
VIS_PERIOD = 0
INPUT = dict(
    FORMAT='BGR',
    MIN_SIZE_TRAIN=(480, ),
    MAX_SIZE_TRAIN=640,
    MIN_SIZE_TRAIN_SAMPLING='choice',
    MIN_SIZE_TEST=480,
    MAX_SIZE_TEST=640,
    WITH_DEPTH=False,
    AUG_DEPTH=False,
    COLOR_AUG_PROB=0.0,
    COLOR_AUG_TYPE='code',
    COLOR_AUG_CODE=
    'Sequential([Sometimes(0.4, CoarseDropout( p=0.1, size_percent=0.05) ),Sometimes(0.5, GaussianBlur(np.random.rand())),Sometimes(0.5, Add((-20, 20), per_channel=0.3)),Sometimes(0.4, Invert(0.20, per_channel=True)),Sometimes(0.5, Multiply((0.7, 1.4), per_channel=0.8)),Sometimes(0.5, Multiply((0.7, 1.4))),Sometimes(0.5, ContrastNormalization((0.5, 2.0), per_channel=0.3))], random_order=False)',
    COLOR_AUG_SYN_ONLY=False,
    BG_TYPE='VOC_table',
    BG_IMGS_ROOT='datasets/VOCdevkit/VOC2012/',
    NUM_BG_IMGS=10000,
    CHANGE_BG_PROB=0.5,
    TRUNCATE_FG=False,
    BG_KEEP_ASPECT_RATIO=True,
    DZI_TYPE='uniform',
    DZI_PAD_SCALE=1.5,
    DZI_SCALE_RATIO=0.25,
    DZI_SHIFT_RATIO=0.25,
    SMOOTH_XYZ=False)
DATASETS = dict(
    TRAIN=('lm_13_train', 'lm_imgn_13_train_1k_per_obj'),
    TRAIN2=(),
    TRAIN2_RATIO=0.0,
    PROPOSAL_FILES_TRAIN=(),
    PRECOMPUTED_PROPOSAL_TOPK_TRAIN=2000,
    TEST=('lm_13_test', ),
    PROPOSAL_FILES_TEST=(),
    PRECOMPUTED_PROPOSAL_TOPK_TEST=1000,
    DET_FILES_TEST=(
        'datasets/BOP_DATASETS/lm/test/test_bboxes/bbox_faster_all.json', ),
    DET_TOPK_PER_OBJ=1,
    DET_THR=0.0,
    SYM_OBJS=['bowl', 'cup', 'eggbox', 'glue'])
DATALOADER = dict(
    NUM_WORKERS=4,
    ASPECT_RATIO_GROUPING=False,
    SAMPLER_TRAIN='TrainingSampler',
    REPEAT_THRESHOLD=0.0,
    FILTER_EMPTY_ANNOTATIONS=True,
    FILTER_EMPTY_DETS=True,
    FILTER_VISIB_THR=0.0)
SOLVER = dict(
    IMS_PER_BATCH=24,
    TOTAL_EPOCHS=160, #可以修改
    OPTIMIZER_CFG=dict(type='Ranger', lr=0.0001, weight_decay=0),#可以修改
    GAMMA=0.1,
    BIAS_LR_FACTOR=1.0,
    LR_SCHEDULER_NAME='flat_and_anneal',
    WARMUP_METHOD='linear',
    WARMUP_FACTOR=0.001,
    WARMUP_ITERS=1000,
    ANNEAL_METHOD='cosine',
    ANNEAL_POINT=0.72,  #修改
    POLY_POWER=0.9,
    REL_STEPS=(0.5, 0.75),
    CHECKPOINT_PERIOD=5,
    CHECKPOINT_BY_EPOCH=True,
    MAX_TO_KEEP=5,
    AMP=dict(ENABLED=False),
    WEIGHT_DECAY=0,
    OPTIMIZER_NAME='Ranger',
    BASE_LR=0.0001,
    MOMENTUM=0.9)
TRAIN = dict(PRINT_FREQ=500, VERBOSE=False, VIS=False, VIS_IMG=True)
VAL = dict(
    DATASET_NAME='lm',
    SCRIPT_PATH='lib/pysixd/scripts/eval_pose_results_more.py',
    RESULTS_PATH='',
    TARGETS_FILENAME='lm_test_targets_bb8.json',
    ERROR_TYPES='ad,rete,re,te,proj',#可以修改
    RENDERER_TYPE='cpp',
    SPLIT='test',
    SPLIT_TYPE='bb8',
    N_TOP=1,
    EVAL_CACHED=False,
    SCORE_ONLY=False,
    EVAL_PRINT_ONLY=False,
    EVAL_PRECISION=False,
    USE_BOP=False)
TEST = dict(
    EVAL_PERIOD=0,
    VIS=False,
    TEST_BBOX_TYPE='est',
    PRECISE_BN=dict(ENABLED=False, NUM_ITER=200),
    AMP_TEST=False,
    USE_PNP=False,
    PNP_TYPE='ransac_pnp')
MODEL = dict(
    DEVICE='cuda',
    WEIGHTS='output2/gdrn/lm/a6_cPnP_lm13/model_final.pth',
    PIXEL_MEAN=[0.0, 0.0, 0.0],
    PIXEL_STD=[255.0, 255.0, 255.0],
    LOAD_DETS_TEST=True,
    CDPN=dict(
        NAME='GDRN',
        TASK='rot',
        USE_MTL=False,
        BACKBONE=dict(
            PRETRAINED='torchvision://resnet34',
            ARCH='resnet',
            NUM_LAYERS=34,
            INPUT_CHANNEL=3,
            INPUT_RES=256,
            OUTPUT_RES=64,
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
            XYZ_LOSS_TYPE='Cos_smi',
            XYZ_LOSS_MASK_GT='visib',
            XYZ_LW=1.0,
            MASK_CLASS_AWARE=False,
            MASK_LOSS_TYPE='L1',
            MASK_LOSS_GT='trunc',
            MASK_LW=1.0,
            MASK_THR_TEST=0.5,
            REGION_ATTENTION=False,
            NUM_REGIONS=0,
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
            CENTER_TRANS=False),#可以修改
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
    KEYPOINT_ON=False,
    LOAD_PROPOSALS=False)
EXP_ID = 'a6_cPnP_lm13_test'
RESUME = False
