from importlib.metadata import version as _pkg_version
from typing import Optional

from . import types
from .cloud_vl import CloudVL
from .finetune import ft

__version__ = _pkg_version("moondream")

DEFAULT_ENDPOINT = "https://api.moondream.ai/v1"


def photon_models() -> list[str]:
    """Return the models available to Photon local inference."""
    from kestrel.models import known_models

    return known_models()


def photon(
    model: str,
    *,
    api_key: Optional[str] = None,
    **runtime_config,
):
    """Create a local Photon model backed by Kestrel's bundled runtime."""
    from .photon_vl import PhotonVL

    return PhotonVL(
        api_key=api_key,
        model=model,
        **runtime_config,
    )


def vl(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = DEFAULT_ENDPOINT,
    local: bool = False,
    model: Optional[str] = None,
    **kwargs,
):
    """
    Factory function for creating a visual language model client.

    Args:
        api_key (str): Your API key for the remote (cloud) API.
        endpoint (str): The endpoint which you would like to call. Local is http://localhost:2020/v1 by default.
        local (bool): If True, delegate to ``photon()`` instead of the Cloud API.
        model (str): Model identifier. Required for local Photon inference.
        **kwargs: Additional arguments forwarded to the selected backend. In local mode,
            arguments are passed directly to Kestrel's ``RuntimeConfig``.

    Returns:
        An instance of CloudVL or PhotonVL.
    """
    if local:
        if model is None:
            raise TypeError("vl(local=True) requires model=<model identifier>")
        return photon(model, api_key=api_key, **kwargs)
    return CloudVL(api_key=api_key, endpoint=endpoint, model=model, **kwargs)


__all__ = ["ft", "photon", "photon_models", "vl", "__version__"]
