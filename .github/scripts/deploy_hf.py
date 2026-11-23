"""Copy this repo's Space files to the Hugging Face Space in one commit.

space/            -> the Space as it is served (app.py, index.html = the original page, ...)
docs/index.html   -> also served by the Space at /v3 (the same file GitHub Pages serves)
"""
import os
import pathlib

from huggingface_hub import CommitOperationAdd, HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
sha = os.environ.get("GITHUB_SHA", "")
ops = []
root = pathlib.Path("space")
for p in sorted(root.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts:
        ops.append(CommitOperationAdd(path_in_repo=p.relative_to(root).as_posix(), path_or_fileobj=str(p)))
page = pathlib.Path("docs/index.html")
if page.exists():
    ops.append(CommitOperationAdd(path_in_repo="v3.html", path_or_fileobj=str(page)))
ops.append(CommitOperationAdd(path_in_repo="DEPLOYED_FROM_GITHUB", path_or_fileobj=sha.encode()))
info = api.create_commit(
    repo_id=os.environ["SPACE_ID"],
    repo_type="space",
    operations=ops,
    commit_message=f"Deploy from GitHub {sha[:7]}",
)
print("Deployed:", info.commit_url)
