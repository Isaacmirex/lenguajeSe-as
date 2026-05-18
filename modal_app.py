"""
Deploy de la API de senas en Modal.com con GPU T4.

Configuracion clave:
- GPU T4 (~30 ms por inferencia)
- scaledown_window=30  -> container se apaga 30s tras el ultimo request
- min_containers=0     -> escala a 0 cuando nadie usa (no cobra)

Deploy:
    pip install modal
    modal token new          # autenticate en browser
    modal deploy modal_app.py

URL publica (ejemplo):
    https://<TU_USER>--senas-api-fastapi.modal.run
"""

from pathlib import Path
import modal

ROOT = Path(__file__).parent

# ------ imagen Docker en Modal ------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi[standard]>=0.110",
        "uvicorn[standard]>=0.27",
        "python-multipart>=0.0.7",
        "pyjwt>=2.8",
        "pillow>=10.0",
        "numpy>=1.26",
        "ultralytics>=8.3.0",
    )
    # Mete el modelo y el codigo dentro del container
    .add_local_file(
        local_path=str(ROOT / "runs" / "detect" / "senas_v1" / "weights" / "best.pt"),
        remote_path="/root/app/runs/detect/senas_v1/weights/best.pt",
        copy=True,
    )
    .add_local_file(str(ROOT / "api.py"), remote_path="/root/app/api.py", copy=True)
)

app = modal.App("senas-api")


@app.function(
    image=image,
    gpu="T4",
    scaledown_window=30,      # apaga el container tras 30 seg sin uso
    min_containers=0,         # cero replicas idle = $0 cuando nadie usa
    max_containers=2,         # tope para no escalar fuera de control
    timeout=300,
)
@modal.asgi_app()
def fastapi_app():
    """Sirve el FastAPI app que ya tenemos en api.py."""
    import sys
    sys.path.insert(0, "/root/app")
    import os
    os.chdir("/root/app")
    from api import app as fastapi_instance
    return fastapi_instance
