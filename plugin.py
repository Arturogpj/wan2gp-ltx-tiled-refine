import json
import traceback

import gradio as gr

from shared.utils.plugins import WAN2GPPlugin

from .ltx_tiled import CONFIG_PATH, LOG, _load_config, apply_patches

PlugIn_Name = "LTX Tiled Refine"

TILE_CHOICES = [
    ("Quality - slowest (1080p: 9 tiles, 2K: 16)", "quality"),
    ("Balanced - recommended (1080p: 6 tiles, 2K: 9)", "balanced"),
    ("Fast - thinnest overlaps (1080p: 4 tiles, 2K: 9)", "fast"),
]


PHASE1_CHOICES = [
    ("Fast - one window when the draft is small (2K: ~2 min faster)", "fast"),
    ("Tiled - same windows as the final pass (old behaviour)", "tiled"),
]


def _save_setting(key, value):
    config = _load_config()
    config[key] = value
    with open(CONFIG_PATH, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)
    print(f"{LOG} {key} set to {value} (used from the next render that starts)")


def _save_tiles(mode):
    _save_setting("tiles", mode)


def _save_phase1(choice):
    _save_setting("phase1_single_window", choice != "tiled")


class LTXTiledRefinePlugin(WAN2GPPlugin):
    def __init__(self):
        super().__init__()
        self.name = PlugIn_Name
        self.version = "1.1.0"
        self.description = ("LTX-2.5 Refine Details / Restore IC-LoRAs: per-step tiled fusion on 1024x576 windows, "
                            "as in Lightricks' TiledFusion workflows.")
        self.request_component("guidance_phases")

    def setup_ui(self):
        apply_patches()

    def post_ui_setup(self, components: dict) -> dict:
        try:
            def create_tiles_dropdown():
                config = _load_config()
                with gr.Row() as row:
                    dropdown = gr.Dropdown(
                        choices=TILE_CHOICES,
                        value=config["tiles"],
                        label="LTX Refine Details tiles",
                        info="Only for the Refine Details / Restore LoRAs, at 1080p and above. Applies to the next render that starts.",
                        interactive=True,
                    )
                    phase1 = gr.Dropdown(
                        choices=PHASE1_CHOICES,
                        value="fast" if config.get("phase1_single_window", True) else "tiled",
                        label="LTX Refine phase 1 (2 Phases only)",
                        info="The low-res draft pass. The final pass is always tiled.",
                        interactive=True,
                    )
                dropdown.change(_save_tiles, inputs=[dropdown], outputs=None, show_progress="hidden")
                phase1.change(_save_phase1, inputs=[phase1], outputs=None, show_progress="hidden")
                return row

            if "guidance_phases" in components:
                self.insert_after(target_component_id="guidance_phases", new_component_constructor=create_tiles_dropdown)
        except Exception:
            traceback.print_exc()
        return {}
