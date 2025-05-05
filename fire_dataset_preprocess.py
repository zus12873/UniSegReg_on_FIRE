import os
import glob
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split

# 首先找出图像文件的实际扩展名
def find_image_extensions(directory):
    extensions = []
    if not os.path.isdir(directory):
        print(f"错误：目录 '{directory}' 不存在或不是一个目录。")
        return extensions
    try:
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                ext = os.path.splitext(file)[1].lower() # 转小写以兼容 .JPG 等
                if ext not in extensions and ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
                    extensions.append(ext)
    except Exception as e:
        print(f"查找扩展名时出错: {e}")
    return extensions

# 路径设置
fire_images_path = "FIRE/Images/"
# 指定要使用的固定掩码文件
fixed_mask_path = "FIRE/Masks/mask.png"  # 或者 "FIRE/Masks/feature_mask.png"
output_path = "FIRE/"

# 检查固定掩码文件是否存在
if not os.path.exists(fixed_mask_path):
    print(f"错误：指定的固定掩码文件 '{fixed_mask_path}' 不存在！请确保该文件存在。")
    exit(1)
else:
    print(f"找到固定掩码文件: {fixed_mask_path}")


# 创建目录
os.makedirs(output_path + "train", exist_ok=True)
os.makedirs(output_path + "valid", exist_ok=True)
os.makedirs(output_path + "test", exist_ok=True)
os.makedirs(output_path + "source_test", exist_ok=True)
print("已创建输出子目录：train, valid, test, source_test")

# 查找图像扩展名
print(f"正在 '{fire_images_path}' 中查找图像扩展名...")
extensions = find_image_extensions(fire_images_path)
if not extensions:
    print(f"警告：在 {fire_images_path} 目录中未找到支持的图像文件 (jpg, png, tif 等)!")
    # 即使找不到扩展名，也尝试查找 .jpg 文件，以防万一
    extensions = ['.jpg']


print(f"将尝试查找以下扩展名的图像: {extensions}")

# 获取所有图像文件
image_files = []
for ext in extensions:
    pattern = os.path.join(fire_images_path, f"*{ext}")
    found_files = glob.glob(pattern)
    # 如果扩展名是大写的，也尝试一下
    pattern_upper = os.path.join(fire_images_path, f"*{ext.upper()}")
    found_files_upper = glob.glob(pattern_upper)
    # 合并并去重
    all_found = list(set(found_files + found_files_upper))
    if all_found:
        image_files.extend(all_found)
        print(f"使用模式 '{pattern}' 和 '{pattern_upper}' 找到 {len(all_found)} 个文件")

# 去重最终列表
image_files = list(set(image_files))

if not image_files:
    print(f"错误：在 '{fire_images_path}' 未找到任何图像文件！请检查路径和文件格式。脚本将退出。")
    exit(1)

print(f"总共找到 {len(image_files)} 个图像文件")

# 数据集划分 (70% 训练, 15% 验证, 15% 测试)
try:
    train_files, test_valid_files = train_test_split(image_files, test_size=0.3, random_state=42)
    valid_files, test_files = train_test_split(test_valid_files, test_size=0.5, random_state=42)
except ValueError as e:
    print(f"错误：划分数据集时出错: {e}")
    print("请确保有足够的图像文件进行划分。")
    exit(1)


# 从训练集中选取一部分作为source_test
source_test_files = train_files[:min(10, len(train_files))]  # 选择10个或更少
print(f"划分完成：Train={len(train_files)}, Valid={len(valid_files)}, Test={len(test_files)}, SourceTest={len(source_test_files)}")

# --- 修改后的处理函数 ---
def process_file(image_path, destination_dir, mask_to_use):
    base_name = os.path.splitext(os.path.basename(image_path))[0] # 使用 splitext 获取纯文件名
    print(f"  处理: {os.path.basename(image_path)} -> {destination_dir} (使用掩码: {os.path.basename(mask_to_use)})")

    try:
        # 处理图像
        img_pil = Image.open(image_path)
        # 确保图像是 RGB 或灰度图，而不是 RGBA 或其他可能的多通道格式
        if img_pil.mode == 'RGBA':
            img_pil = img_pil.convert('RGB')
        elif img_pil.mode == 'P': # 处理调色板图像
             img_pil = img_pil.convert('RGB')

        img = np.array(img_pil)
        out_img_path = os.path.join(destination_dir, f"{base_name}_org.tif")
        Image.fromarray(img).save(out_img_path)

        # --- 使用固定的掩码路径 ---
        mask_path = mask_to_use

        # 处理掩码
        mask_pil = Image.open(mask_path)
        # 确保掩码是灰度图
        if mask_pil.mode != 'L':
            mask_pil = mask_pil.convert('L') # 转为灰度图
        mask = np.array(mask_pil)

        # 二值化掩码 (阈值设为0，假设背景为0，前景>0)
        if mask.max() > 0: # 避免全黑掩码处理错误
             mask = (mask > 0).astype(np.uint8) * 255
        else:
             mask = np.zeros_like(mask, dtype=np.uint8) # 全黑

        # 保存为npy格式
        out_mask_path = os.path.join(destination_dir, f"{base_name}_lab_b.npy")
        np.save(out_mask_path, mask)

        # 创建权重图像 (简单示例 - 使用边缘检测)
        weight = np.ones_like(mask).astype(np.float32)
        # 边缘区域权重更高
        if mask.max() > 0: # 只在非全黑掩码上计算边缘
            try:
                # Canny 需要 uint8 类型的单通道图像
                edges = cv2.Canny(mask, 50, 150)
                weight[edges > 0] = 2.0
            except cv2.error as e:
                 print(f"    警告：为 {base_name} 生成权重图时出错（可能是掩码格式问题）: {e}")
        else:
            print(f"    信息：掩码 {base_name} 为全黑，跳过边缘权重计算。")


        out_weight_path = os.path.join(destination_dir, f"{base_name}_weight.tif")
        # 保存权重图时确保是 L 模式（灰度）
        weight_img_pil = Image.fromarray((weight / weight.max() * 255).astype(np.uint8), mode='L')
        weight_img_pil.save(out_weight_path)

    except FileNotFoundError:
        print(f"  错误：文件未找到 - {image_path} 或 {mask_path}")
    except Exception as e:
        print(f"  处理文件 {os.path.basename(image_path)} 时出错: {str(e)}")

# 处理每个数据集
print("处理训练集...")
for img_path in train_files:
    process_file(img_path, output_path + "train", fixed_mask_path) # 传入固定掩码路径

print("处理验证集...")
for img_path in valid_files:
    process_file(img_path, output_path + "valid", fixed_mask_path) # 传入固定掩码路径

print("处理测试集...")
for img_path in test_files:
    process_file(img_path, output_path + "test", fixed_mask_path) # 传入固定掩码路径

# 复制source_test文件
print("处理source_test集...")
for img_path in source_test_files:
    process_file(img_path, output_path + "source_test", fixed_mask_path) # 传入固定掩码路径

print("数据预处理完成！")
