import json
import os
import shutil
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
        self.version = "1.4.0"
        self.description = ("LTX-2.5 Refine Details / Restore IC-LoRAs: per-step tiled fusion on 1024x576 windows, "
                            "as in Lightricks' TiledFusion workflows.")
        self.request_component("guidance_phases")
        self.request_global("get_lora_dir")

    def setup_ui(self):
        apply_patches()

    def _install_presets(self):
        """Copy the bundled presets into WanGP's LTX-2 LoRA folder once; never overwrite a user's copy."""
        source = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")
        if not os.path.isdir(source):
            return
        try:
            get_lora_dir = getattr(self, "get_lora_dir", None)
            target = get_lora_dir("ltx2_25_22B_distilled") if callable(get_lora_dir) else os.path.join("loras", "ltx2")
        except Exception:
            target = os.path.join("loras", "ltx2")
        os.makedirs(target, exist_ok=True)
        for name in sorted(os.listdir(source)):
            destination = os.path.join(target, name)
            if not name.endswith(".json"):
                continue
            if not os.path.exists(destination):
                shutil.copyfile(os.path.join(source, name), destination)
                print(f"{LOG} added preset '{name[:-5]}' to {target}")
            else:
                self._keep_source_audio(destination)

    @staticmethod
    def _keep_source_audio(path):
        """Presets from 1.3.0 and older let LTX invent a new soundtrack; switch them to the source clip's own audio."""
        try:
            with open(path, "r", encoding="utf-8") as handle:
                settings = json.load(handle)
            if settings.get("postprocess_audio", None) != "":
                return
            settings["postprocess_audio"] = "control"
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(settings, handle, indent=4)
            print(f"{LOG} preset '{os.path.basename(path)[:-5]}' now keeps the source video's audio")
        except Exception:
            traceback.print_exc()

    def post_ui_setup(self, components: dict) -> dict:
        try:
            self._install_presets()
        except Exception:
            traceback.print_exc()
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
