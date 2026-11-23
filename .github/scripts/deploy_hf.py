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

# Wait for the Space to come back up, then report what it serves (shown as an annotation on the run).
import json
import time
import urllib.request

space_host = os.environ["SPACE_ID"].lower().replace("/", "-").replace("_", "-") + ".hf.space"
stage, health = "", ""
deadline = time.time() + 600
time.sleep(20)
while time.time() < deadline:
    try:
        stage = api.get_space_runtime(os.environ["SPACE_ID"]).stage
    except Exception as e:  # noqa: BLE001
        stage = f"? {e}"
    if stage == "RUNNING":
        try:
            with urllib.request.urlopen(f"https://{space_host}/healthz", timeout=20) as r:
                health = r.read().decode()[:400]
            if not sha or json.loads(health).get("deployed_from") == sha[:12]:
                break
        except Exception as e:  # noqa: BLE001
            health = f"healthz error: {e}"
    elif "ERROR" in str(stage):
        break
    time.sleep(15)
print(f"::notice title=Space status::stage={stage} health={health}")
if stage != "RUNNING":
    raise SystemExit(f"Space did not start: {stage}")

# Pronunciation check self-test: send a short Italian clip (made by espeak-ng in the workflow) and report
# what Whisper heard and how long it took, for each model size.
clip = pathlib.Path("/tmp/asr_it.wav")
if clip.exists():
    deadline = time.time() + 480
    asr = {}
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"https://{space_host}/healthz", timeout=20) as r:
                asr = json.loads(r.read().decode()).get("asr", {})
            if asr.get("ready") or asr.get("error"):
                break
        except Exception as e:  # noqa: BLE001
            asr = {"healthz": str(e)}
        time.sleep(10)
    print(f"::notice title=ASR warm-up::{json.dumps(asr)}")
    data = clip.read_bytes()
    for model in os.environ.get("ASR_TEST_MODELS", "small,base,tiny").split(","):
        res = []
        for _ in range(2):
            t0 = time.time()
            try:
                req = urllib.request.Request(f"https://{space_host}/api/asr?lang=it&model={model}", data=data,
                                             headers={"Content-Type": "audio/wav"}, method="POST")
                with urllib.request.urlopen(req, timeout=180) as r:
                    out = json.loads(r.read().decode())
                res.append(f"{out.get('text')!r} server={out.get('secs')}s total={time.time() - t0:.1f}s")
            except Exception as e:  # noqa: BLE001
                res.append(f"error {e}")
        print(f"::notice title=ASR {model}::" + " | ".join(res))
