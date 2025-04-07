from typing import Optional
from .version import __version__
from .cloud_vl import CloudVL

DEFAULT_API_URL = "https://api.moondream.ai/v1"


def vl(
    api_key: Optional[str] = None,
    api_url: Optional[str] = DEFAULT_API_URL,
    **kwargs,
):
    """
    Factory function for creating a visual language model client.

    Args:
        api_key (str): Your API key for the remote (cloud) API.
        api_url (str): The endpoint which you would like to call. Local is http://localhost:8000 by default.
        **kwargs.

    Returns:
        An instance of CloudVL.
    """
    return CloudVL(api_key=api_key, api_url=api_url, **kwargs)


__all__ = ["vl", "__version__"]
