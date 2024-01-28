# Download required files from torchvision
urls = [
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py",
    "https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py"
]
download_files(urls)


#example usage
!python train.py --api_key htpcxp3XQh7SsgMfjJns \
                --workspace basketball-formations \
                --project_name basketball-and-hoop \
                --version 5 \
                --hidden_layer 256 \
                --lr 0.005 \
                --num_epochs 1 \
                --threshold 0.6 \
                --video_url "https://www.youtube.com/watch?v=y8i6fsAXDZE"
