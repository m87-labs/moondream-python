from .cloud_vl import CloudVL


class LocalVL(CloudVL):
    """
    LocalVL is a client for the locally running Moondream server.

    In local mode, the API URL is fixed to "http://localhost:8000", and no API key is used.
    All methods (caption, query, detect, point) are inherited from CloudVL and operate
    against the local server endpoints.
    """

    def __init__(self, **kwargs):
        # Force the API URL to the local server and disable authentication.
        super().__init__(api_url="http://localhost:8000", api_key=None, **kwargs)
