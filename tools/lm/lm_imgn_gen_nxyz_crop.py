from __future__ import division, print_function
import os
os.environ["PYOPENGL_PLATFORM"] = "egl"
import os.path as osp
import sys
cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../.."))
sys.path.insert(0, PROJ_ROOT)


# import hashlib
from lib.utils.mask_utils import mask2bbox_xyxy
from lib.pysixd import misc
from lib.vis_utils.image import grid_show
from lib.meshrenderer.meshrenderer_phong_normals import Renderer
# import logging

import ref

import mmcv
import numpy as np
from tqdm import tqdm




DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))

LM_13_OBJECTS = [
    "ape",
    "benchvise",
    "camera",
    "can",
    "cat",
    "driller",
    "duck",
    "eggbox",
    "glue",
    "holepuncher",
    "iron",
    "lamp",
    "phone",
]  # no bowl, cup ---why no bowl and cup by wenhua
################################################################################

# SPLITS_LM_IMGN_13的获取是脚本，无论如何都会运行

SPLITS_LM_IMGN_13 = dict(  # 这里面有几个字典阿，关键字是那一个呢
    lm_imgn_13_train_1k_per_obj=dict(
        name="lm_imgn_13_train_1k_per_obj",  # BB8 training set
        # dataset_root is datasets
        dataset_root=osp.join(DATASETS_ROOT, "lm_imgn/"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
        objs=LM_13_OBJECTS,  # selected objects
        ann_files=[  # !!!!!!!!!!!!!!1by wenhua 获取的文件引入
            osp.join(DATASETS_ROOT, "lm_imgn/image_set/{}_{}.txt".format("train", _obj)) for _obj in LM_13_OBJECTS
        ],
        image_prefixes=[osp.join(DATASETS_ROOT, "lm_imgn/imgn")
                        for _obj in LM_13_OBJECTS],
        nxyz_prefixes=[osp.join(DATASETS_ROOT, "lm_imgn/nxyz_crop_imgn/")
                      for _obj in LM_13_OBJECTS],
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        depth_factor=1000.0,
        cam=ref.lm_full.camera_matrix,
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        n_per_obj=1000,  # 1000 per class
        filter_scene=True,
        filter_invalid=False,
        ref_key="lm_full",
    )
)

# single obj splits
for obj in ref.lm_full.objects:
    for split in ["train"]:
        name = "lm_imgn_13_{}_{}_1k".format(obj, split)
        ann_files = [
            osp.join(DATASETS_ROOT, "lm_imgn/image_set/{}_{}.txt".format(split, obj))]
        if split in ["train"]:
            filter_invalid = True
        elif split in ["test"]:
            filter_invalid = False
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_LM_IMGN_13:
            SPLITS_LM_IMGN_13[name] = dict(
                name=name,
                dataset_root=osp.join(DATASETS_ROOT, "lm_imgn/"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
                objs=[obj],  # only this obj
                ann_files=ann_files,
                image_prefixes=[osp.join(DATASETS_ROOT, "lm_imgn/imgn/")],
                nxyz_prefixes=[
                    osp.join(DATASETS_ROOT, "lm_imgn/nxyz_crop_imgn/")],
                scale_to_meter=0.001,
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                depth_factor=1000.0,
                cam=ref.lm_full.camera_matrix,
                height=480,
                width=640,
                cache_dir=osp.join(PROJ_ROOT, ".cache"),
                use_cache=True,
                n_per_obj=1000,
                filter_invalid=False,
                filter_scene=True,
                ref_key="lm_full",
            )


# os.environ["PYOPENGL_PLATFORM"] = "egl"


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
IM_H = 480
IM_W = 640
near = 0.01
far = 6.5

# data_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/train"))

# cls_indexes = sorted(idx2class.keys())
# cls_names = [idx2class[cls_idx] for cls_idx in cls_indexes]
# lm_model_dir = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/models"))
# model_paths = [osp.join(lm_model_dir, f"obj_{obj_id:06d}.ply") for obj_id in cls_indexes]
# texture_paths = None

# scenes = [i for i in range(1, 15 + 1)]   #场景
# # xyz_root = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/train_pbr/xyz_crop"))
# xyz_root = osp.normpath(osp.join(PROJ_ROOT, "datasets/BOP_DATASETS/lm/test/nxyz_crop"))

# K = np.array([[572.4114, 0, 325.2611], [0, 573.57043, 242.04899], [0, 0, 1]])  #这里可能要改


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
        # self.name = data_cfg["name"]
        # self.data_cfg = data_cfg

        self.objs = SPLITS_LM_IMGN_13["lm_imgn_13_train_1k_per_obj"]["objs"]  # selected objects

        # idx files with image ids
        self.ann_files = SPLITS_LM_IMGN_13["lm_imgn_13_train_1k_per_obj"]["ann_files"]
        self.image_prefixes = SPLITS_LM_IMGN_13["lm_imgn_13_train_1k_per_obj"]["image_prefixes"]
        self.nxyz_prefixes = SPLITS_LM_IMGN_13["lm_imgn_13_train_1k_per_obj"]["nxyz_prefixes"]

        # self.dataset_root = data_cfg["dataset_root"]  # lm_imgn
        self.models_root = SPLITS_LM_IMGN_13["lm_imgn_13_train_1k_per_obj"]["models_root"]  # BOP_DATASETS/lm/models
        # self.scale_to_meter = data_cfg["scale_to_meter"]  # 0.001===zhuyi

        # True (load masks but may not use it)
        # self.with_masks = data_cfg["with_masks"]
        # True (load depth path here, but may not use it)
        # self.with_depth = data_cfg["with_depth"]
        # self.depth_factor = data_cfg["depth_factor"]  # 1000.0

        self.cam = SPLITS_LM_IMGN_13["lm_imgn_13_train_1k_per_obj"]["cam"]  #
        # self.height = data_cfg["height"]  # 480
        # self.width = data_cfg["width"]  # 640

        # self.cache_dir = data_cfg["cache_dir"]  # .cache
        # self.use_cache = data_cfg["use_cache"]  # True
        # # sample uniformly to get n items
        # self.n_per_obj = data_cfg.get("n_per_obj", 1000)
        # self.filter_invalid = data_cfg["filter_invalid"]
        # self.filter_scene = data_cfg.get("filter_scene", False)
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
            model_paths = [osp.join(self.models_root, f"obj_{_id:06d}.ply") for _id in ref.lm_full.id2obj]
            self.renderer = Renderer(
                model_paths, vertex_tmp_store_folder=osp.join(
                    PROJ_ROOT, ".cache")
            )
        return self.renderer

    def main(self):
        for ann_file, scene_root, xyz_root in zip(self.ann_files, self.image_prefixes, self.nxyz_prefixes):
            # ann_file ='/home/lyn/Desktop/GDR-Net-main/datasets/lm_imgn/image_set/train_ape.txt'
            # scene_root='/home/lyn/Desktop/GDR-Net-main/datasets/lm_imgn/imgn'
            # xyz_root='/home/lyn/Desktop/GDR-Net-main/datasets/lm_imgn/xyz_crop_imgn/'
            # linemod each scene is an object
            with open(ann_file, "r") as f_ann:
                indices = [line.strip("\r\n").split()[-1]
                           for line in f_ann.readlines()]  # string ids

            for im_id in tqdm(indices):
               

                obj_name = im_id.split("/")[0]
                if obj_name == "benchviseblue":
                    obj_name = "benchvise"
                obj_id = ref.lm_full.obj2id[obj_name]
                # if self.filter_scene:
                #     if obj_name not in self.objs:
                #         continue
               
                pose_path = osp.join(scene_root, "{}-pose.txt".format(im_id))
                pose = np.loadtxt(pose_path, skiprows=1)  # 引入pose
                R = pose[:3, :3]
                t = pose[:3, 3]
              
                save_path = osp.join(xyz_root, f"{im_id}-nxyz.pkl")

                # render_obj_id = cls_indexes.index(obj_id)  # 0-based
                # render_obj_id = ref.lm_full.obj2id.index(obj_id)  # 0-based
                render_obj_id=obj_id-1

                # t=np.array([0,0,4])  #  #固定平移

                bgr_gl, depth_gl, nomal_img = self.get_renderer().render(
                    render_obj_id, IM_W, IM_H, self.cam, R, t, near, far)
                # bgr_gl, depth_gl= self.get_renderer().render(render_obj_id, IM_W, IM_H, K, R, t, near, far)

                # mask = (depth_gl > 0).astype("uint8")

                mask1 = (nomal_img != np.array([0, 0, 0])).astype("uint8")

                if mask1.sum() == 0:
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
                        show_ims = [
                            bgr_gl[:, :, [2, 1, 0]],
                            # get_emb_show(xyz_np),
                            get_emb_show(nomal_img),
                            get_emb_show(nxyz_crop),

                        ]
                        show_titles = ["bgr_gl", "nxyz", "nxyz_crop"]
                        grid_show(show_ims, show_titles, row=1, col=3)
                        break #跳过查看下一个类
                if not args.no_save:  # save file
                    mmcv.mkdir_or_exist(osp.dirname(save_path))
                    mmcv.dump(nxyz_info, save_path)
        if self.renderer is not None:
            self.renderer.close()

                  


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

    height = IM_H
    width = IM_W

    VIS = args.vis

    T_begin = time.perf_counter()
    setproctitle.setproctitle(
        f"gen_xyz_lm_train_pbr_{args.split}_{args.scene}")
    xyz_gen = XyzGen(args.split, args.scene)
    xyz_gen.main()
    T_end = time.perf_counter() - T_begin
    print("split", args.split, "scene", args.scene, "total time: ", T_end)
