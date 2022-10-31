_base_ = ["config_base.py"]
OUTPUT_DIR = "output30/gdrn/lm/a6_cPnP_lm13"#改变一下输出路径以同时跑两个
INPUT = dict(
  DZI_SCALE_RATIO=0.15,
 DZI_SHIFT_RATIO=0.1,
)
SOLVER = dict(
    TOTAL_EPOCHS=160,
    ANNEAL_POINT=0.72,# 修改之前是0.72
    OPTIMIZER_CFG=dict(_delete_=True, type="Ranger", lr=1e-4, weight_decay=0),#修改一下学习率
)

MODEL = dict(
  CDPN=dict(
     WEIGHTS='',
        BACKBONE=dict( FREEZE=False,
         ENABLED=True,
         INPUT_RES=256,
            OUTPUT_RES=64) ,#同时不训练
        ROT_HEAD=dict(
            ENABLED=True,#去掉旋转tou
            FREEZE=False,  #同时不训练
            XYZ_LOSS_TYPE='Cos_smi',
            REGION_ATTENTION=True,
            NUM_REGIONS=3,
            REGION_LOSS_TYPE='R_cos',#"CE"
        ),
        PNP_NET=dict(     
            FREEZE=False,
            ENABLE=True,
            R_ONLY=False,
            CENTER_TRANS=False,
            WITH_2D_COORD=False,#加上这个2d对应
            ROT_TYPE="allo_rot6d",
            TRUE_NORMAL=True,
            PM_LOSS_TYPE='Rot_cos_loss',#"Rot_cos_loss"|L1,
            NUM_LAYERS=5,
            MASK_ATTENTION='none',
            REGION_ATTENTION=True
        ),
        TRANS_HEAD=dict(  ENABLED=False,
            FREEZE=False,),
    ),
)
TEST = dict(TEST_BBOX_TYPE="est")  # gt | est可见可以选择bbox，先看一看可视化吧
