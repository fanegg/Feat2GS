import os, subprocess, shlex, sys, gc
import time
import torch
import numpy as np
import shutil
import argparse
import gradio as gr
import uuid
# import spaces

# subprocess.run(shlex.split("pip install wheel/diff_gaussian_rasterization-0.0.0-cp310-cp310-linux_x86_64.whl"))
# subprocess.run(shlex.split("pip install wheel/simple_knn-0.0.0-cp310-cp310-linux_x86_64.whl"))
# subprocess.run(shlex.split("pip install wheel/curope-0.0.0-cp310-cp310-linux_x86_64.whl"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r")))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r", "dust3r")))
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from utils.dust3r_utils import compute_global_alignment, load_images, storePly, save_colmap_cameras, save_colmap_images

from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from demo_train_feat2gs import training
from demo_run_video import render_sets
GRADIO_CACHE_FOLDER = '/home/chenyue/tmp/temp/gradio_cache_folder'

from utils.feat_utils import FeatureExtractor
from dust3r.demo import _convert_scene_output_to_glb

import tempfile
tempfile.tempdir = "/home/chenyue/tmp/temp"
if not os.path.exists(tempfile.tempdir):
    os.makedirs(tempfile.tempdir)

#############################################################################################################################################


def get_dust3r_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    parser.add_argument("--model_path", type=str, default="submodules/mast3r/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="path to the model weights")    # hf: naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--niter", type=int, default=300)
    parser.add_argument("--focal_avg", type=bool, default=True)
    parser.add_argument("--n_views", type=int, default=3)
    parser.add_argument("--base_path", type=str, default=GRADIO_CACHE_FOLDER) 
    parser.add_argument("--feat_dim", type=int, default=256, help="PCA dimension. If None, PCA is not applied, and the original feature dimension is retained.")
    parser.add_argument("--feat_type", type=str, nargs='*', default=["dust3r",], help="Feature type(s). Multiple types can be specified for combination.")
    parser.add_argument("--vis_feat", action="store_true", default=True, help="Visualize features")
    parser.add_argument("--vis_key", type=str, default=None, help="Feature type to visualize (only for mast3r), e.g., 'decfeat' or 'desc'")
    parser.add_argument("--method", type=str, default='dust3r', help="Method of Initialization, e.g., 'dust3r' or 'mast3r'")

    return parser


# @spaces.GPU(duration=150)
def run_dust3r(inputfiles, input_path=None):

    if input_path is not None:
        imgs_path = './assets/example/' + input_path
        imgs_names = sorted(os.listdir(imgs_path))

        inputfiles = []
        for imgs_name in imgs_names:
            file_path = os.path.join(imgs_path, imgs_name)
            print(file_path)
            inputfiles.append(file_path)
        print(inputfiles)

    # ------ Step(1) DUSt3R initialization & Feature extraction ------
    # os.system(f"rm -rf {GRADIO_CACHE_FOLDER}")
    parser = get_dust3r_args_parser()
    opt = parser.parse_args()

    method = opt.method

    tmp_user_folder = str(uuid.uuid4()).replace("-", "")
    opt.img_base_path = os.path.join(opt.base_path, tmp_user_folder)
    img_folder_path = os.path.join(opt.img_base_path, "images")    
 
    model = AsymmetricCroCo3DStereo.from_pretrained(opt.model_path).to(opt.device)
    os.makedirs(img_folder_path, exist_ok=True)

    opt.n_views = len(inputfiles)  
    if opt.n_views == 1:
        raise gr.Error("The number of input images should be greater than 1.")
    print("Multiple images: ", inputfiles)
    # for image_file in inputfiles:
    #     image_path = image_file.name if hasattr(image_file, 'name') else image_file
    #     shutil.copy(image_path, img_folder_path)
    for image_path in inputfiles:
        if input_path is not None:
            shutil.copy(image_path, img_folder_path)
        else:
            shutil.move(image_path, img_folder_path)
    train_img_list = sorted(os.listdir(img_folder_path))
    assert len(train_img_list)==opt.n_views, f"Number of images in the folder is not equal to {opt.n_views}"
    images, ori_size = load_images(img_folder_path, size=512) 
    # images, ori_size, imgs_resolution = load_images(img_folder_path, size=512) 
    # resolutions_are_equal = len(set(imgs_resolution)) == 1
    # if resolutions_are_equal == False:
    #     raise gr.Error("The resolution of the input image should be the same.")
    print("ori_size", ori_size)
    start_time = time.time()
    ######################################################
    pairs = make_pairs(images, scene_graph='complete', prefilter=None, symmetrize=True)
    output = inference(pairs, model, opt.device, batch_size=opt.batch_size)

    scene = global_aligner(output, device=opt.device, mode=GlobalAlignerMode.PointCloudOptimizer)
    loss = compute_global_alignment(scene=scene, init="mst", niter=opt.niter, schedule=opt.schedule, lr=opt.lr, focal_avg=opt.focal_avg)
    scene = scene.clean_pointcloud()   

    imgs = to_numpy(scene.imgs)
    focals = scene.get_focals()
    poses = to_numpy(scene.get_im_poses())
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(1.0)))
    confidence_masks = to_numpy(scene.get_masks())
    intrinsics = to_numpy(scene.get_intrinsics())

    ######################################################
    end_time = time.time()
    print(f"Time taken for {opt.n_views} views: {end_time-start_time} seconds")
    
    output_colmap_path=img_folder_path.replace("images", f"sparse/0/{method}")
    
    # Feature extraction for per point(per pixel)
    extractor = FeatureExtractor(images, opt, method)
    feats = extractor(scene=scene)
    feat_type_str = '-'.join(extractor.feat_type)
    output_colmap_path = os.path.join(output_colmap_path, feat_type_str)
    os.makedirs(output_colmap_path, exist_ok=True)

    outfile = _convert_scene_output_to_glb(output_colmap_path, imgs, pts3d, confidence_masks, focals, poses, as_pointcloud=True, cam_size=0.03)
    feat_image_path = os.path.join(opt.img_base_path, "feat_dim0-9_dust3r.png")

    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), train_img_list)
    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    color_4_3dgs = np.concatenate([p[m] for p, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)
    feat_4_3dgs = np.concatenate([p[m] for p, m in zip(feats, confidence_masks)])
    storePly(os.path.join(output_colmap_path, f"points3D.ply"), pts_4_3dgs, color_4_3dgs, feat_4_3dgs)    

    del scene
    torch.cuda.empty_cache()
    gc.collect()

    return outfile, feat_image_path, opt, None, None


def run_feat2gs(opt, niter=2000):

    if opt is None:
        raise gr.Error("Please run Step 1 first!")
    
    try:
        if not os.path.exists(opt.img_base_path):
            raise ValueError(f"Input path does not exist: {opt.img_base_path}")
            
        if not os.path.exists(os.path.join(opt.img_base_path, "images")):
            raise ValueError("Input images not found. Please run Step 1 first")
        
        if not os.path.exists(os.path.join(opt.img_base_path, f"sparse/0/{opt.method}")):
            raise ValueError("DUSt3R output not found. Please run Step 1 first")
        
        # ------ Step(2) Readout 3DGS from features & Jointly optimize pose ------
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        parser.add_argument('--debug_from', type=int, default=-1)
        parser.add_argument("--test_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
        parser.add_argument("--start_checkpoint", type=str, default = None)
        parser.add_argument("--scene", type=str, default="demo")
        parser.add_argument("--n_views", type=int, default=3)
        parser.add_argument("--get_video", action="store_true")
        parser.add_argument("--optim_pose", type=bool, default=True)
        parser.add_argument("--feat_type", type=str, nargs='*', default=["dust3r",], help="Feature type(s). Multiple types can be specified for combination.")
        parser.add_argument("--method", type=str, default='dust3r', help="Method of Initialization, e.g., 'dust3r' or 'mast3r'")
        parser.add_argument("--feat_dim", type=int, default=256, help="Feture dimension after PCA . If None, PCA is not applied.")
        parser.add_argument("--model", type=str, default='Gft', help="Model of Feat2gs, 'G'='geometry'/'T'='texture'/'A'='all'")
        parser.add_argument("--dataset", default="demo", type=str)
        parser.add_argument("--resize", action="store_true", default=True, 
                        help="If True, resize rendering to square")
        
        args = parser.parse_args(sys.argv[1:])
        args.iterations = niter
        args.save_iterations.append(args.iterations)
        args.model_path = opt.img_base_path + '/output/'    
        args.source_path = opt.img_base_path
        # args.model_path = GRADIO_CACHE_FOLDER + '/output/'    
        # args.source_path = GRADIO_CACHE_FOLDER
        args.iteration = niter
        os.makedirs(args.model_path, exist_ok=True)
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

        output_ply_path = opt.img_base_path + f'/output/point_cloud/iteration_{args.iteration}/point_cloud.ply'
        # output_ply_path = GRADIO_CACHE_FOLDER+ f'/output/point_cloud/iteration_{args.iteration}/point_cloud.ply'

        torch.cuda.empty_cache()
        gc.collect()

        return output_ply_path, args, None

    except Exception as e:
        raise gr.Error(f"Step 2 failed: {str(e)}")


def run_render(opt, args, cam_traj='ellipse'):
    if opt is None or args is None:
        raise gr.Error("Please run Steps 1 and 2 first!")
    
    try:
        iteration_path = os.path.join(opt.img_base_path, f"output/point_cloud/iteration_{args.iteration}/point_cloud.ply")
        if not os.path.exists(iteration_path):
            raise ValueError("Training results not found. Please run Step 2 first")
        
        # ------ Step(3) Render video with camera trajectory ------
        parser = ArgumentParser(description="Testing script parameters")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        args.eval = True
        args.get_video = True
        args.n_views = opt.n_views
        args.cam_traj = cam_traj
        render_sets(
            model.extract(args),
            args.iteration,
            pipeline.extract(args),
            args,
        )

        output_video_path = opt.img_base_path + f'/output/videos/demo_{opt.n_views}_view_{args.cam_traj}.mp4'

        torch.cuda.empty_cache()
        gc.collect()

        return output_video_path
    
    except Exception as e:
        raise gr.Error(f"Step 3 failed: {str(e)}")


def process_example(inputfiles, input_path):
    dust3r_model, feat_image, dust3r_state, _, _ = run_dust3r(inputfiles, input_path=input_path)
    
    output_model, feat2gs_state, _ = run_feat2gs(dust3r_state, niter=2000)
    
    output_video = run_render(dust3r_state, feat2gs_state, cam_traj='interpolated')
    
    return dust3r_model, feat_image, output_model, output_video

def reset_dust3r_state():
    return None, None, None, None, None

def reset_feat2gs_state():
    return None, None, None

_TITLE = '''Feat2GS Demo'''
_DESCRIPTION = '''
<div style="display: flex; justify-content: center; align-items: center;">
    <div style="width: 100%; text-align: center; font-size: 30px;">
        <strong><span style="font-family: 'Comic Sans MS';"><span style="color: #E0933F">Feat</span><span style="color: #B24C33">2</span><span style="color: #E0933F">GS</span></span>: Probing Visual Foundation Models with Gaussian Splatting</strong>
    </div>
</div> 
<p></p>
<div align="center">
    <a style="display:inline-block" href="https://fanegg.github.io/Feat2GS/"><img src='https://img.shields.io/badge/Project-Website-green.svg'></a>&nbsp;
    <a style="display:inline-block" href="https://arxiv.org/abs/2412.09606"><img src="https://img.shields.io/badge/Arxiv-2412.09606-b31b1b.svg?logo=arXiv" alt='arxiv'></a>&nbsp;
    <a style="display:inline-block" href="https://youtu.be/4fT5lzcAJqo?si=_fCSIuXNBSmov2VA"><img src='https://img.shields.io/badge/Video-E33122?logo=Youtube'></a>&nbsp;
    <a style="display:inline-block" href="https://github.com/fanegg/Feat2GS"><img src="https://img.shields.io/badge/Code-black?logo=Github" alt='Code'></a>&nbsp;
    <a title="X" href="https://twitter.com/faneggchen" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/badge/@Yue%20Chen-black?logo=X" alt="X">
    </a>&nbsp;
    <a title="Bluesky" href="https://bsky.app/profile/fanegg.bsky.social" target="_blank" rel="noopener noreferrer" style="display: inline-block;">
        <img src="https://img.shields.io/badge/@Yue%20Chen-white?logo=Bluesky" alt="Bluesky">
    </a>
</div>
<p></p>
'''


# demo = gr.Blocks(title=_TITLE).queue()
demo = gr.Blocks(css=""".gradio-container {margin: 0 !important; min-width: 100%};""", title="Feat2GS Demo").queue()
with demo:
    dust3r_state = gr.State(None)
    feat2gs_state = gr.State(None)
    render_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("🚀 Quickstart", open=False):
                gr.Markdown("""
                1. **Input Images**
                * Upload 2 or more images of the same scene from different views
                * For best results, ensure images have good overlap

                2. **Step 1: DUSt3R Initialization & Feature Extraction**
                * Click "RUN Step 1" to process your images
                * This step estimates initial DUSt3R point cloud and camera poses, and extracts DUSt3R features for each pixel

                3. **Step 2: Readout 3DGS from Features**
                * Set the number of training iterations, larger number leads to better quality but longer time (default: 2000, max: 8000) 
                * Click "RUN Step 2" to optimize the 3D model

                4. **Step 3: Video Rendering**
                * Choose a camera trajectory
                * Click "RUN Step 3" to generate a video of your 3D model
                """)
            
            with gr.Accordion("💡 Tips", open=False):
                gr.Markdown("""
                * Processing time depends on image resolution and quantity
                * For optimal performance, test on high-end GPUs (A100/4090)
                * Use the mouse to interact with 3D models:
                  - Left button: Rotate
                  - Scroll wheel: Zoom
                  - Right button: Pan
                """)

    with gr.Row():
        with gr.Column(scale=1):
            # gr.Markdown('# ' + _TITLE)
            gr.Markdown(_DESCRIPTION)
    
    with gr.Row(variant='panel'):
        with gr.Tab("Input"):
            inputfiles = gr.File(file_count="multiple", label="images")
            input_path = gr.Textbox(visible=False, label="example_path")
            # button_gen = gr.Button("RUN")
            
    with gr.Row(variant='panel'):
        with gr.Tab("Step 1: DUSt3R initialization & Feature extraction"):
            dust3r_run = gr.Button("RUN Step 1")
            with gr.Column(scale=2):
                with gr.Group():
                    dust3r_model = gr.Model3D(
                        label="DUSt3R Output",
                        interactive=False,
                        # camera_position=[0.5, 0.5, 1],
                    )
                    feat_image = gr.Image(
                        label="Feature Visualization",
                        type="filepath"
                    )

    with gr.Row(variant='panel'):
        with gr.Tab("Step 2: Readout 3DGS from features & Jointly optimize pose"):
            niter = gr.Number(value=2000, precision=0, minimum=1000, maximum=8000, label="Training iterations")
            feat2gs_run = gr.Button("RUN Step 2")
            with gr.Column(scale=1):
                with gr.Group():
                    output_model = gr.Model3D(
                        label="3D Gaussian Splats Output, need more time to visualize",
                        interactive=False,
                        # camera_position=[0.5, 0.5, 1],
                    )
                    gr.Markdown(
                        """
                        <div class="model-description">
                           &nbsp;&nbsp;Use the left mouse button to rotate, the scroll wheel to zoom, and the right mouse button to move.
                        </div>
                        """
                    )    

    with gr.Row(variant='panel'):
        with gr.Tab("Step 3: Render video with camera trajectory"):
            cam_traj = gr.Dropdown(["arc", "spiral", "lemniscate", "wander", "ellipse", "interpolated"], value='ellipse', label="Camera trajectory")
            render_run = gr.Button("RUN Step 3")
            with gr.Column(scale=1):
                output_video = gr.Video(label="video", height=800)
   
    dust3r_run.click(
        fn=reset_dust3r_state,
        inputs=None,
        outputs=[dust3r_model, feat_image, dust3r_state, feat2gs_state, render_state],
        queue=False
    ).then(
        fn=run_dust3r,
        inputs=[inputfiles],
        outputs=[dust3r_model, feat_image, dust3r_state, feat2gs_state, render_state]
    )
    feat2gs_run.click(
        fn=reset_feat2gs_state,
        inputs=None,
        outputs=[output_model, feat2gs_state, render_state],
        queue=False
    ).then(
        fn=run_feat2gs,
        inputs=[dust3r_state, niter],
        outputs=[output_model, feat2gs_state, render_state]
    )
    render_run.click(run_render, inputs=[dust3r_state, feat2gs_state, cam_traj], outputs=[output_video])


    gr.Examples(
        examples=[
            ["./assets/example/plushies/1.jpg",]
        ],
        inputs=[input_path],
        outputs=[dust3r_model, feat_image, output_model, output_video],
        fn=lambda x: process_example(inputfiles=None, input_path=x),
        cache_examples=True,
        label='Examples'
    )

demo.launch(server_name="0.0.0.0", share=False)