from __future__ import division, print_function

import mmcv
import numpy as np
from tqdm import tqdm
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../.."))
sys.path.insert(0, PROJ_ROOT)



from lib.utils.mask_utils import mask2bbox_xyxy
from lib.pysixd import misc
from lib.vis_utils.image import grid_show
# from lib.meshrenderer.meshrenderer_phong_normals import Renderer
from verify_idea.render.diffrenderNormal import DiffRenderer_Normal_Wrapper
import torch
import ref

DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))

LM_OCC_OBJECTS = ["ape", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher"]
lm_model_root = "BOP_DATASETS/lm/models/"
lmo_model_root = "BOP_DATASETS/lmo/models/"
SPLITS_LM_PBR = dict(
    # lm_pbr_13_train=dict(
    #     name="lm_pbr_13_train",
    #     objs=LM_13_OBJECTS,  # selected objects
    #     dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_pbr"),
    #     models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
    #     xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_pbr/xyz_crop"),
    #     nxyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_pbr/nxyz_crop"),

    #     scale_to_meter=0.001,
    #     with_masks=True,  # (load masks but may not use it)
    #     with_depth=True,  # (load depth path here, but may not use it)
    #     height=480,
    #     width=640,
    #     cache_dir=osp.join(PROJ_ROOT, ".cache"),
    #     use_cache=True,
    #     num_to_load=-1,
    #     filter_invalid=True,
    #     ref_key="lm_full",
    # ),
    lmo_pbr_train=dict(
        name="lmo_pbr_train",
        objs=LM_OCC_OBJECTS,  # selected objects
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_pbr"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
        xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_pbr/xyz_crop"),
        nxyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_pbr/nxyz_crop"),

        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_invalid=True,
        ref_key="lmo_full",
    ),
)

# single obj splits
# for obj in ref.lm_full.objects:
#     for split in ["train"]:
#         name = "lm_pbr_{}_{}".format(obj, split)
#         if split in ["train"]:
#             filter_invalid = True
#         elif split in ["test"]:
#             filter_invalid = False
#         else:
#             raise ValueError("{}".format(split))
#         if name not in SPLITS_LM_PBR:
#             SPLITS_LM_PBR[name] = dict(
#                 name=name,
#                 objs=[obj],  # only this obj
#                 dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_pbr"),
#                 models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
#                 xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_pbr/xyz_crop"),
#                 nxyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_pbr/nxyz_crop"),

#                 scale_to_meter=0.001,
#                 with_masks=True,  # (load masks but may not use it)
#                 with_depth=True,  # (load depth path here, but may not use it)
#                 height=480,
#                 width=640,
#                 cache_dir=osp.join(PROJ_ROOT, ".cache"),
#                 use_cache=True,
#                 num_to_load=-1,
#                 filter_invalid=filter_invalid,
#                 ref_key="lm_full",
#             )

# lmo single objs
#  ref.lmo_full.objects定义了lmo数据集中的类，指出了需要渲染的类别
for obj in ref.lmo_full.objects:
    for split in ["train"]:
        name = "lmo_pbr_{}_{}".format(obj, split)
        if split in ["train"]:
            filter_invalid = True
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_LM_PBR:
            SPLITS_LM_PBR[name] = dict(
                name=name,
                objs=[obj],  # only this obj
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_pbr"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
                xyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_pbr/xyz_crop"),
                nxyz_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/train_pbr/nxyz_crop"),

                scale_to_meter=0.001,
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                height=480,
                width=640,
                cache_dir=osp.join(PROJ_ROOT, ".cache"),
                use_cache=True,
                num_to_load=-1,
                filter_invalid=filter_invalid,
                ref_key="lmo_full",
            )

cur_dir = osp.abspath(osp.dirname(__file__))
PROJ_ROOT = osp.join(cur_dir, "../..")
sys.path.insert(0, PROJ_ROOT)
# from lib.meshrenderer.meshrenderer_phong import Renderer


idx2class = {
    1: "ape",
    2: "benchvise",
    3: "bowl",
    4: "camera",
    5: "can",
    6: "cat",
    7: "cup",
    8: "driller",
    9: "duck",
    10: "eggbox",
    11: "glue",
    12: "holepuncher",
    13: "iron",
    14: "lamp",
    15: "phone",
}

class2idx = {_name: _id for _id, _name in idx2class.items()}#gen a  dictionary

# classes = idx2class.values()
# classes = sorted(classes)

# # DEPTH_FACTOR = 1000.
# IM_H = 480
# IM_W = 640
near = 0.01
far = 6.5
def normalize_to_01(img):
    if img.max() != img.min():
        return (img - img.min()) / (img.max() - img.min())
    else:
        return img


def get_emb_show(bbox_emb):
    show_emb = bbox_emb.copy()
    show_emb = normalize_to_01(show_emb)
    return show_emb

class XyzGen(object):
    def __init__(self, split="train", scene="all"):
     
        self.objs = SPLITS_LM_PBR["lmo_pbr_train"]["objs"]  # selected objects

        self.dataset_root = SPLITS_LM_PBR["lmo_pbr_train"].get("dataset_root", osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/train_pbr"))
        self.xyz_root = SPLITS_LM_PBR["lmo_pbr_train"].get("xyz_root", osp.join(self.dataset_root, "xyz_crop"))
        self.nxyz_root = SPLITS_LM_PBR["lmo_pbr_train"].get("nxyz_root", osp.join(self.dataset_root, "nxyz_crop"))
        self.models_root =SPLITS_LM_PBR["lmo_pbr_train"]["models_root"]  # BOP_DATASETS/lm/models
        # self.models_root ="datasets/BOP_DATASETS/lm/models"  #修改一下

        self.height = SPLITS_LM_PBR["lmo_pbr_train"]["height"]  # 480
        self.width = SPLITS_LM_PBR["lmo_pbr_train"]["width"]  # 640

        ##################################################

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.lm_full.id2obj.items() if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]

        self.scenes = [f"{i:06d}" for i in range(50)]
        self.cam =None
       
        ##################################################
        if self.cam is None:
            self.cam = np.array(
                [[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])

        # NOTE: careful! Only the selected objects
        self.cat_ids = [cat_id for cat_id, obj_name in ref.lm_full.id2obj.items(
        ) if obj_name in self.objs]
        # map selected objs to [0, num_objs-1]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.label2cat = {label: cat for cat, label in self.cat2label.items()}
        self.renderer = None
        # self.obj2label = OrderedDict((obj, obj_id)
        #                              for obj_id, obj in enumerate(self.objs))
        ##########################################################

    def get_renderer(self):
        if self.renderer is None:
            # self.renderer = Renderer(
            #     model_paths, vertex_tmp_store_folder=osp.join(PROJ_ROOT, ".cache"), vertex_scale=0.001
            # )
            model_paths = [osp.join(self.models_root, f"obj_{_id:06d}.ply") for _id in self.cat_ids]
            # self.renderer = Renderer(
            #     model_paths, vertex_tmp_store_folder=osp.join(
            #         PROJ_ROOT, ".cache")
            # )
            self.renderer=DiffRenderer_Normal_Wrapper(
                    model_paths,device="cuda"
            )
        return self.renderer

    def main(self):

        for scene in tqdm(self.scenes):
            scene_id = int(scene)
            if scene_id>2:
                break
            scene_root = osp.join(self.dataset_root, scene)

            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            gt_info_dict = mmcv.load(osp.join(scene_root, "scene_gt_info.json"))
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))

            for str_im_id in tqdm(gt_dict, postfix=f"{scene_id}"):
                # str_im_id='58'
                int_im_id = int(str_im_id)

                scene_im_id = f"{scene_id}/{int_im_id}"

                K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
                depth_factor = 1000.0 / cam_dict[str_im_id]["depth_scale"]  # 10000
                insts = []
                render_obj_id=[]
              
                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    
                    obj_id = anno["obj_id"]
                    if obj_id not in self.cat_ids:
                        continue
                    id=self.cat_ids.index(obj_id)
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0#3*1

                    xyz_path = osp.join(self.xyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-xyz.pkl")
                    nxyz_path = osp.join(self.nxyz_root, f"{scene_id:06d}/{int_im_id:06d}_{anno_i:06d}-nxyz.pkl")
                    render_obj_id=id

                # t=np.array([0,0,4])  #  #固定平移
                    T=np.eye(4)
                    T[:3,:3]=R
                    T[:3,3]=t
                    T=T[None]
                    
                    T=torch.tensor(T,device='cuda:0',dtype=torch.float32)
                    K=torch.tensor(K,device='cuda:0',dtype=torch.float32)
                    K=K.view(1,3,3)
                    nomal_img = self.get_renderer()([render_obj_id],T,K,(self.width,self.height),near,far)#如果固定t则如何呢
                    nomal_img=nomal_img.cpu().numpy()
                    nomal_img=nomal_img[0]
                    # nomal_img=nomal_img[:,:,0]
                    mask1 = (nomal_img != np.array([0, 0, 0])).astype("uint8")

                    if mask1.sum() <2000:
                            # cv2.imwrite("nxyz_crop.png",nxyz_crop*255)
                            # cv2.imwrite("nxyz.png",nomal_img*255)
                            continue
                    else:
                        x1, y1, x2, y2 = mask2bbox_xyxy(mask1)
                        nxyz_crop = nomal_img[y1: y2 + 1, x1: x2 + 1, :]
                        nxyz_info = {
                            "nxyz_crop": nxyz_crop,  # save disk space w/o performance drop
                            "nxyxy": [x1, y1, x2, y2],
                        }
                        if VIS:
                            print(
                                f"xyz_crop min {nxyz_crop.min()} max {nxyz_crop.max()}")
                            import cv2
                            cv2.imwrite("nxyz_crop.png",nxyz_crop*255)
                            cv2.imwrite("nxyz.png",nomal_img*255)
                            # show_ims = [
                            #     bgr_gl[:, :, [2, 1, 0]],
                            #     # get_emb_show(xyz_np),
                            #     get_emb_show(nomal_img),
                            #     get_emb_show(nxyz_crop),

                            # ]
                            # show_titles = ["bgr_gl", "nxyz", "nxyz_crop"]
                            # grid_show(show_ims, show_titles, row=1, col=3)
                            break #跳过查看下一个类
                    # break #跳过查看下一个类
                    if not args.no_save:  # save file
                        mmcv.mkdir_or_exist(osp.dirname(nxyz_path))
                        mmcv.dump(nxyz_info, nxyz_path)
        # if self.renderer is not None:
        #     self.renderer.close()



if __name__ == "__main__":
    import argparse
    import time

    import setproctitle

    parser = argparse.ArgumentParser(description="gen lm train_pbr xyz")
    parser.add_argument("--split", type=str, default="train", help="split")
    parser.add_argument("--scene", type=str, default="all", help="scene id")
    parser.add_argument("--vis", default=False,
                        action="store_true", help="vis")
    parser.add_argument("--no-save", default=False,
                        action="store_true", help="do not save results")
    args = parser.parse_args()

    # height = IM_H
    # width = IM_W

    VIS = args.vis

    T_begin = time.perf_counter()
    setproctitle.setproctitle(
        f"gen_xyz_lm_train_pbr_{args.split}_{args.scene}")
    xyz_gen = XyzGen(args.split, args.scene)
    xyz_gen.main()
    T_end = time.perf_counter() - T_begin
    print("split", args.split, "scene", args.scene, "total time: ", T_end)