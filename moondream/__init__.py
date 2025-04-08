from typing import Optional
from .version import __version__
from .cloud_vl import CloudVL

DEFAULT_ENDPOINT = "https://api.moondream.ai/v1"


def vl(
    api_key: Optional[str] = None,
    endpoint: Optional[str] = DEFAULT_ENDPOINT,
    **kwargs,
):
    """
    Factory function for creating a visual language model client.

    Args:
        api_key (str): Your API key for the remote (cloud) API.
        endpoint (str): The endpoint which you would like to call. Local is http://localhost:2020/v1 by default.
        **kwargs.

    Returns:
        An instance of CloudVL.
    """
    return CloudVL(api_key=api_key, endpoint=endpoint, **kwargs)


__all__ = ["vl", "__version__"]
