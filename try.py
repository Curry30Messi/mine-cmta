import subprocess


def get_git_commit_info():
    # 获取当前提交的哈希值
    commit_hash = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    # print(f"Current commit hash: {commit_hash}")




# 打印当前 Git 提交信息
get_git_commit_info()
