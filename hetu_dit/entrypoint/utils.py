import socket
import os


def get_loopback_host():
    try:
        socket.inet_pton(socket.AF_INET6, "::1")
        return "::1"  # IPv6 loopback is available
    except OSError:
        return "127.0.0.1"  # fallback to IPv4


def get_bind_host(host: str | None) -> str:
    if host:
        return host
    return os.environ.get("HETUDIT_HOST", "0.0.0.0")


def build_output_filename(
    model_class_name: str, task_id: str, output_type: str = "pil"
) -> str | None:
    if model_class_name in {"sd3", "sd3.5"} and output_type == "pil":
        return f"stable_diffusion_3_result_{task_id}.png"
    if model_class_name == "cogvideox":
        return f"cogvideox_{task_id}.mp4"
    if model_class_name == "flux" and output_type == "pil":
        return f"flux_result_{task_id}.png"
    if model_class_name == "hunyuandit" and output_type == "pil":
        return f"hunyuandit_result_{task_id}.png"
    if model_class_name == "hunyuanvideo":
        return f"hunyuan_video_{task_id}.mp4"
    return None
