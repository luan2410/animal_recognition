import os
import sys
import io

# Sửa lỗi không in được tiếng Việt trên cmd
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Thư mục chứa các file cần đổi tên
# folder_path = r"D:\nam4\hk2\KT\animal_recognition\data\train\cat"
folder_path = r"D:\nam4\hk2\KT\animal_recognition\data\train\horse"
# Lấy danh sách file và sắp xếp
files = os.listdir(folder_path)
files.sort()

# Đổi tên file
for idx, filename in enumerate(files):
    # Tạo tên mới (giữ đuôi file gốc: .jpg, .png,...)
   # new_name = f"cat_{idx+1:04d}{os.path.splitext(filename)[1]}"
    new_name = f"horse_{idx+1:04d}{os.path.splitext(filename)[1]}"

    # Đường dẫn cũ và mới
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)

    # Đổi tên file
    os.rename(old_path, new_path)

print("Đổi tên hoàn tất!")
