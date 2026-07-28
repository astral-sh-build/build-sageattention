import subprocess
from pathlib import Path

import modal

TEST_DIRECTORY = Path(__file__).parent.resolve()

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(uv_project_dir=str(TEST_DIRECTORY))
    .add_local_file(
        TEST_DIRECTORY / "test_sageattention.py",
        remote_path="/gpu-tests/test_sageattention.py",
    )
)

app = modal.App("astral-build-sageattention-gpu-tests")


@app.function(image=image, gpu="A10G", timeout=900)
def test() -> None:
    subprocess.run(
        ["python", "-m", "pytest", "-v", "/gpu-tests/test_sageattention.py"],
        check=True,
    )
