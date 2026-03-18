from typing import Optional
from .version import __version__
from .cloud_vl import CloudVL

DEFAULT_ENDPOINT = "https://api.moondream.ai/v1"


def vl(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = DEFAULT_ENDPOINT,
    local: bool = False,
    **kwargs,
):
    """
    Factory function for creating a visual language model client.

    Args:
        api_key (str): Your API key for the remote (cloud) API.
        endpoint (str): The endpoint which you would like to call. Local is http://localhost:2020/v1 by default.
        local (bool): If True, use local GPU inference via Photon instead of the cloud API.
        **kwargs: Additional arguments forwarded to the backend (e.g. model, max_batch_size,
            kv_cache_pages, device for local mode).

    Returns:
        An instance of CloudVL or PhotonVL.
    """
    if local:
        from .photon_vl import PhotonVL
        return PhotonVL(api_key=api_key, **kwargs)
    model = kwargs.pop("model", None)
    return CloudVL(api_key=api_key, endpoint=endpoint, model=model, **kwargs)


__all__ = ["vl", "__version__"]
