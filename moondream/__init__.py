from .version import __version__
from .cloud_vl import CloudVL
from .local_vl import LocalVL


def vl(api_key: str = None, api_url: str = "https://api.moondream.ai/v1", **kwargs):
    """
    Factory function for creating a visual language model client.

    Args:
        api_key (str): Your API key for the remote (cloud) API. This is required unless local=True.
        local (bool): If True, returns a client that talks to the locally running server.
                      If False (default), returns a client that communicates with the cloud API.
        **kwargs: Additional keyword arguments that are passed to the underlying client.

    Returns:
        An instance of CloudVL (for remote mode) or LocalVL (for local mode).

    Raises:
        ValueError: If local is False and no API key is provided.
    """
    return CloudVL(api_key=api_key, api_url=api_url, **kwargs)


__all__ = ["vl", "__version__"]
