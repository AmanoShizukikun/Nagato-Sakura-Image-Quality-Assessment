import os
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

# 圖片輸入及輸出路徑
input_directory = "./data/input_images"
output_directory = "./data/quality_dataset"
min_quality = 10
max_quality = 101
quality_interval = 10

# 處理圖片
def process_images(input_dir, output_dir):
    quality_levels = list(range(min_quality, max_quality, quality_interval))
    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_single_image, file_name, input_dir, output_dir, quality_levels) for file_name in image_files]
        for future in futures:
            future.result()

# 處理單張圖片
def process_single_image(file_name, input_dir, output_dir, quality_levels):
    input_path = os.path.join(input_dir, file_name)
    try:
        with Image.open(input_path) as img:
            for quality in quality_levels:
                base_name, ext = os.path.splitext(file_name)
                output_file = f"{base_name}_q{quality}.jpg"
                output_path = os.path.join(output_dir, output_file)
                img.save(output_path, quality=quality)
    except Exception as e:
        print(f"處理圖片 {file_name} 時發生錯誤: {e}")

# 執行
process_images(input_directory, output_directory)
print("圖片處理完成！")