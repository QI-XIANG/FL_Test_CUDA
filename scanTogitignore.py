import os
import subprocess

def add_to_gitignore(scan_path, gitignore_path=".gitignore", max_file_size_mb=2048, exclude_folders=None):
    """
    掃描指定路徑下的檔案與資料夾，將超過 max_file_size_mb 限制的檔案加入 .gitignore，
    並忽略指定的資料夾（例如 'dataset'），避免其內容被 Git 或 Git LFS 上傳。
    
    參數：
        scan_path (str): 要掃描的路徑
        gitignore_path (str): .gitignore 檔案路徑
        max_file_size_mb (int): 最大檔案大小（MB），預設為 2048 MB（2 GiB）
        exclude_folders (list): 要完全忽略的資料夾列表，預設為 None
    """
    # 將 MB 轉換為位元組 (1 MB = 1024 * 1024 位元組)
    max_file_size_bytes = max_file_size_mb * 1024 * 1024  # 2 GiB = 2,147,483,648 位元組
    ignore_list = []
    
    # 設定預設排除資料夾
    if exclude_folders is None:
        exclude_folders = ['dataset']  # 預設忽略 'dataset' 資料夾
    
    # 正規化掃描路徑
    scan_path = os.path.abspath(scan_path)
    
    try:
        # 讀取現有 .gitignore 條目
        existing_entries = set()
        if os.path.exists(gitignore_path):
            with open(gitignore_path, "r") as f:
                existing_entries = {line.strip() for line in f if line.strip() and not line.startswith('#')}

        # 獲取 Git LFS 追蹤的檔案
        lfs_tracked = set()
        try:
            result = subprocess.run(['git', 'lfs', 'ls-files'], capture_output=True, text=True, check=True)
            for line in result.stdout.splitlines():
                parts = line.split(' * ')
                if len(parts) > 1:
                    lfs_tracked.add(parts[1].strip())
        except subprocess.CalledProcessError:
            print("警告：無法檢查 Git LFS 追蹤檔案，假設無 LFS 追蹤。")

        # 掃描目錄樹
        for root, dirs, files in os.walk(scan_path):
            # 檢查是否在排除資料夾中
            rel_root = os.path.relpath(root, os.path.dirname(scan_path)).replace(os.sep, '/')
            if any(rel_root.startswith(ex_folder) for ex_folder in exclude_folders):
                continue  # 跳過排除資料夾下的檔案掃描

            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if (os.path.isfile(file_path) and 
                        os.path.getsize(file_path) > max_file_size_bytes):
                        # 計算相對路徑
                        relative_path = os.path.relpath(file_path, os.path.dirname(scan_path))
                        relative_path = relative_path.replace(os.sep, '/')
                        # 如果檔案已在 LFS 追蹤中，移除追蹤並加入 .gitignore
                        if relative_path in lfs_tracked:
                            print(f"警告：'{relative_path}' 已由 Git LFS 追蹤，將移除追蹤並加入 .gitignore")
                            subprocess.run(['git', 'lfs', 'untrack', relative_path], check=True)
                        if relative_path not in existing_entries:
                            ignore_list.append(relative_path)
                except OSError as e:
                    print(f"警告：無法處理 {file_path}：{e}")

        # 將排除資料夾加入 ignore_list
        for folder in exclude_folders:
            normalized_folder = folder.replace(os.sep, '/') + '/'
            if normalized_folder not in existing_entries:
                ignore_list.append(normalized_folder)

        # 如果有新條目，追加到 .gitignore
        if ignore_list:
            with open(gitignore_path, "a") as f:  # 使用追加模式 'a'
                if not existing_entries and os.stat(gitignore_path).st_size == 0:
                    f.write("# 自動生成的忽略列表（大檔案與指定資料夾）\n")
                for item in ignore_list:
                    f.write(f"{item}\n")
                    print(f"已將 '{item}' 加入 .gitignore。")
            # 更新 Git 索引以反映 .gitignore 變更
            subprocess.run(['git', 'update-index', '--assume-unchanged'] + [item for item in ignore_list if not item.endswith('/')])
        else:
            print("未發現需要加入 .gitignore 的新檔案或資料夾。")

    except PermissionError as e:
        print(f"錯誤：權限不足，無法存取 {gitignore_path}：{e}")
    except IOError as e:
        print(f"錯誤：無法寫入 {gitignore_path}：{e}")
    except subprocess.CalledProcessError as e:
        print(f"錯誤：執行 Git 命令失敗：{e}")
    except Exception as e:
        print(f"意外錯誤：{e}")

if __name__ == "__main__":
    # 獲取當前工作目錄
    current_path = os.getcwd()
    max_size = 100  # 最大檔案大小設為 100 MB
    add_to_gitignore(current_path, max_file_size_mb=max_size)