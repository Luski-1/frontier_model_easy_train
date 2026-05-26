import os
import shutil
import random

# ===================== 【请确认你的数据集路径】 =====================
DATA_ROOT = "/workspace/data/imagenet-256"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
EVAL_DIR = os.path.join(DATA_ROOT, "eval")
# 每个类别剪切的图片数量
MOVE_NUM = 5
# 支持的图片格式
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
# =================================================================


def create_eval_from_train():
    # 1. 检查train目录
    if not os.path.exists(TRAIN_DIR):
        print(f"错误：训练集目录不存在！{TRAIN_DIR}")
        return

    # 2. 创建eval根目录
    os.makedirs(EVAL_DIR, exist_ok=True)
    print(f"测试集目录创建成功：{EVAL_DIR}\n")

    # 3. 遍历所有合法类别（跳过隐藏文件夹）
    class_folders = [
        f
        for f in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, f)) and not f.startswith('.')
    ]

    print(f"检测到 {len(class_folders)} 个有效类别，开始处理...\n")
    # 记录不足的类别，最后汇总
    insufficient_classes = []

    # 4. 逐个处理类别
    for cls_name in class_folders:
        src_cls_dir = os.path.join(TRAIN_DIR, cls_name)
        dst_cls_dir = os.path.join(EVAL_DIR, cls_name)
        os.makedirs(dst_cls_dir, exist_ok=True)

        # 获取该类别所有图片
        all_images = [img for img in os.listdir(src_cls_dir) if img.lower().endswith(IMG_EXTENSIONS)]
        total_img = len(all_images)

        # 无图片
        if total_img == 0:
            print(f"类别【{cls_name}】：无任何图片，跳过")
            continue

        # 图片不足5张 → 提示并记录
        if total_img < MOVE_NUM:
            print(f"类别【{cls_name}】：图片不足 {MOVE_NUM} 张（仅剩 {total_img} 张），已全部移动")
            insufficient_classes.append((cls_name, total_img))
            selected_imgs = all_images
        else:
            # 图片充足 → 随机选5张
            selected_imgs = random.sample(all_images, MOVE_NUM)
            print(f"类别【{cls_name}】：已随机剪切 {MOVE_NUM} 张图片")

        # 执行剪切
        for img in selected_imgs:
            shutil.move(os.path.join(src_cls_dir, img), os.path.join(dst_cls_dir, img))

    # 5. 最终汇总：打印所有不足的类别
    print("\n" + "=" * 60)
    if insufficient_classes:
        print(f"【汇总】共有 {len(insufficient_classes)} 个类别图片不足 {MOVE_NUM} 张：")
        for cls_name, num in insufficient_classes:
            print(f"   - {cls_name}：仅 {num} 张")
    else:
        print(f"所有类别图片均充足，已成功剪切 {MOVE_NUM} 张/类")
    print("=" * 60)
    print("\n全部处理完成！测试集已创建完毕！")


if __name__ == "__main__":
    create_eval_from_train()
