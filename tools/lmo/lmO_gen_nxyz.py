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
from lib.meshrenderer.meshrenderer_phong_normals import Renderer
import ref

DATASETS_ROOT = osp.normpath(osp.join(PROJ_ROOT, "datasets"))

LM_OCC_OBJECTS = ["ape", "can", "cat", "driller", "duck", "eggbox", "glue", "holepuncher"]
SPLITS_LM = dict(

    lmo_train=dict(
        name="lmo_train",
        # use lm real all (8 objects) to train for lmo
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
        objs=LM_OCC_OBJECTS,  # selected objects
        ann_files=[
            osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/image_set/{}_{}.txt".format(_obj, "all"))
            for _obj in LM_OCC_OBJECTS
        ],
        image_prefixes=[
            osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/test/{:06d}".format(ref.lmo_full.obj2id[_obj]))
            for _obj in LM_OCC_OBJECTS
        ],
        xyz_prefixes=[
            osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/test/xyz_crop/{:06d}".format(ref.lmo_full.obj2id[_obj]))
            for _obj in LM_OCC_OBJECTS
        ],
        nxyz_prefixes=[
            osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/test/nxyz_crop/{:06d}".format(ref.lmo_full.obj2id[_obj]))
            for _obj in LM_OCC_OBJECTS
        ],
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_scene=True,
        filter_invalid=True,
        ref_key="lmo_full",
    ),
    lmo_test=dict(
        name="lmo_test",
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
        objs=LM_OCC_OBJECTS,
        ann_files=[osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/image_set/lmo_test.txt")],
        # NOTE: scene root
        image_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/test/{:06d}").format(2)],
        xyz_prefixes=[None],
        nxyz_prefixes=[None],
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_scene=False,
        filter_invalid=False,
        ref_key="lmo_full",
    ),
    lmo_bop_test=dict(
        name="lmo_bop_test",
        dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/"),
        models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
        objs=LM_OCC_OBJECTS,
        ann_files=[osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/image_set/lmo_bop_test.txt")],
        # NOTE: scene root
        image_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/test/{:06d}").format(2)],
        xyz_prefixes=[None],
        nxyz_prefixes=[None],
        scale_to_meter=0.001,
        with_masks=True,  # (load masks but may not use it)
        with_depth=True,  # (load depth path here, but may not use it)
        height=480,
        width=640,
        cache_dir=osp.join(PROJ_ROOT, ".cache"),
        use_cache=True,
        num_to_load=-1,
        filter_scene=False,
        filter_invalid=False,
        ref_key="lmo_full",
    ),
)
# single obj splits for lmo_test
for obj in ref.lmo_full.objects:
    for split in ["test"]:
        name = "lmo_{}_{}".format(obj, split)
        if split in ["train", "all"]:  # all is used to train lmo
            filter_invalid = True
        elif split in ["test"]:
            filter_invalid = False
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_LM:
            SPLITS_LM[name] = dict(
                name=name,
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
                objs=[obj],
                ann_files=[osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/image_set/lmo_test.txt")],
                # NOTE: scene root
                image_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/test/{:06d}").format(2)],
                xyz_prefixes=[None],
                nxyz_prefixes=[None],
                scale_to_meter=0.001,
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                height=480,
                width=640,
                cache_dir=osp.join(PROJ_ROOT, ".cache"),
                use_cache=True,
                num_to_load=-1,
                filter_scene=False,
                filter_invalid=False,
                ref_key="lmo_full",
            )

# single obj splits for lmo_bop_test
for obj in ref.lmo_full.objects:
    for split in ["test"]:
        name = "lmo_{}_bop_{}".format(obj, split)
        if split in ["train", "all"]:  # all is used to train lmo
            filter_invalid = True
        elif split in ["test"]:
            filter_invalid = False
        else:
            raise ValueError("{}".format(split))
        if name not in SPLITS_LM:
            SPLITS_LM[name] = dict(
                name=name,
                dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/"),
                models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/models"),
                objs=[obj],
                ann_files=[osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/image_set/lmo_bop_test.txt")],
                # NOTE: scene root
                image_prefixes=[osp.join(DATASETS_ROOT, "BOP_DATASETS/lmo/test/{:06d}").format(2)],
                xyz_prefixes=[None],
                nxyz_prefixes=[None],
                scale_to_meter=0.001,
                with_masks=True,  # (load masks but may not use it)
                with_depth=True,  # (load depth path here, but may not use it)
                height=480,
                width=640,
                cache_dir=osp.join(PROJ_ROOT, ".cache"),
                use_cache=True,
                num_to_load=-1,
                filter_scene=False,
                filter_invalid=False,
                ref_key="lmo_full",
            )

# ================ add single image dataset for debug =======================================
debug_im_ids = {"train": {obj: [] for obj in ref.lm_full.objects}, "test": {obj: [] for obj in ref.lm_full.objects}}
for obj in ref.lm_full.objects:
    for split in ["train", "test"]:
        cur_ann_file = osp.join(DATASETS_ROOT, f"BOP_DATASETS/lm/image_set/{obj}_{split}.txt")
        ann_files = [cur_ann_file]

        im_ids = []
        with open(cur_ann_file, "r") as f:
            for line in f:
                # scene_id(obj_id)/im_id
                im_ids.append("{}/{}".format(ref.lm_full.obj2id[obj], int(line.strip("\r\n"))))

        debug_im_ids[split][obj] = im_ids
        for debug_im_id in debug_im_ids[split][obj]:
            name = "lm_single_{}{}_{}".format(obj, debug_im_id.split("/")[1], split)
            if name not in SPLITS_LM:
                SPLITS_LM[name] = dict(
                    name=name,
                    dataset_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/"),
                    models_root=osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/models"),
                    objs=[obj],  # only this obj
                    ann_files=ann_files,
                    image_prefixes=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/test/{:06d}").format(ref.lm_full.obj2id[obj])
                    ],
                    xyz_prefixes=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/test/xyz_crop/{:06d}".format(ref.lm_full.obj2id[obj]))
                    ],
                    nxyz_prefixes=[
                        osp.join(DATASETS_ROOT, "BOP_DATASETS/lm/test/nxyz_crop/{:06d}".format(ref.lm_full.obj2id[obj]))
                    ],
                    scale_to_meter=0.001,
                    with_masks=True,  # (load masks but may not use it)
                    with_depth=True,  # (load depth path here, but may not use it)
                    height=480,
                    width=640,
                    cache_dir=osp.join(PROJ_ROOT, ".cache"),
                    use_cache=True,
                    num_to_load=-1,
                    filter_invalid=False,
                    filter_scene=True,
                    ref_key="lm_full",
                    debug_im_id=debug_im_id,  # NOTE: debug im id
                )
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
     
        self.ann_files = SPLITS_LM["lmo_train"]["ann_files"]  # idx files with image ids
        self.image_prefixes =SPLITS_LM["lmo_train"]["image_prefixes"]
        self.xyz_prefixes = SPLITS_LM["lmo_train"]["xyz_prefixes"]  #
        self.objs = SPLITS_LM["lmo_train"]["objs"] 
        self.nxyz_prefixes =SPLITS_LM["lmo_train"]["nxyz_prefixes"] 
        self.filter_scene = SPLITS_LM["lmo_train"].get("filter_scene", False)
        self.cat_ids = [cat_id for cat_id, obj_name in ref.lm_full.id2obj.items() if obj_name in self.objs]
        self.cat2label = {v: i for i, v in enumerate(self.cat_ids)}  # id_map
        self.height = SPLITS_LM["lmo_train"]["height"]  # 480
        self.width = SPLITS_LM["lmo_train"]["width"]  # 640
        self.models_root ="datasets/BOP_DATASETS/lm/models"  #修改一下

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
            model_paths = [osp.join(self.models_root, f"obj_{_id:06d}.ply") for _id in ref.lm_full.id2obj]
            self.renderer = Renderer(
                model_paths, vertex_tmp_store_folder=osp.join(
                    PROJ_ROOT, ".cache")
            )
        return self.renderer

    def main(self):

        assert len(self.ann_files) == len(self.image_prefixes), f"{len(self.ann_files)} != {len(self.image_prefixes)}"
        assert len(self.ann_files) == len(self.xyz_prefixes), f"{len(self.ann_files)} != {len(self.xyz_prefixes)}"
        for ann_file, scene_root, xyz_root,nxyz_root in zip(tqdm(self.ann_files), self.image_prefixes, self.xyz_prefixes,self.nxyz_prefixes):
            # linemod each scene is an object
            with open(ann_file, "r") as f_ann:
                indices = [line.strip("\r\n") for line in f_ann.readlines()]  # string ids
            gt_dict = mmcv.load(osp.join(scene_root, "scene_gt.json"))
            cam_dict = mmcv.load(osp.join(scene_root, "scene_camera.json"))
            for im_id in tqdm(indices):
                int_im_id = int(im_id)
                str_im_id = str(int_im_id)
                rgb_path = osp.join(scene_root, "rgb/{:06d}.png").format(int_im_id)
                assert osp.exists(rgb_path), rgb_path

                scene_id = int(rgb_path.split("/")[-3])
                scene_im_id = f"{scene_id}/{int_im_id}"

                K = np.array(cam_dict[str_im_id]["cam_K"], dtype=np.float32).reshape(3, 3)
                if self.filter_scene:
                    if scene_id not in self.cat_ids:
                        continue
                for anno_i, anno in enumerate(gt_dict[str_im_id]):
                    obj_id = anno["obj_id"]
                    if obj_id not in self.cat_ids:
                        continue
                    cur_label = self.cat2label[obj_id]  # 0-based label
                    R = np.array(anno["cam_R_m2c"], dtype="float32").reshape(3, 3)
                    t = np.array(anno["cam_t_m2c"], dtype="float32") / 1000.0    
                  
                    nxyz_path=osp.join(nxyz_root, f"{int_im_id:06d}_{anno_i:06d}-nxyz.pkl")
                    bgr_gl, depth_gl, nomal_img = self.get_renderer().render(
                        obj_id-1, self.width, self.height, K, R, t, near, far)
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
                        mmcv.mkdir_or_exist(osp.dirname(nxyz_path))
                        mmcv.dump(nxyz_info, nxyz_path)
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