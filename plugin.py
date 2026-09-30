from shared.utils.plugins import WAN2GPPlugin

from .ltx_tiled import apply_patches

PlugIn_Name = "LTX Tiled Refine"


class LTXTiledRefinePlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "1.0.0"
        self.description = ("LTX-2.5 Refine Details / Restore IC-LoRAs: per-step tiled fusion on 1024x576 windows, "
                            "as in Lightricks' TiledFusion workflows.")

    def setup_ui(self):
        apply_patches()
