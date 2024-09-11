import os


def get_git_commit_hash(repo_path):
    try:
        head_file = os.path.join(repo_path, '.git', 'HEAD')
        with open(head_file, 'r') as f:
            ref = f.read().strip()

        if ref.startswith('ref: '):
            ref_path = os.path.join(repo_path, '.git', ref[5:])
            with open(ref_path, 'r') as f:
                commit_hash = f.read().strip()
            return commit_hash
        else:
            return ref
    except Exception as e:
        print(f"Exception: {e}")


repo_path = os.getcwd()  # 当前目录
commit_hash = get_git_commit_hash(repo_path)
print(commit_hash[:5])
