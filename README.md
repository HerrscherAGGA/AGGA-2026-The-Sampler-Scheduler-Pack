# AGGA 2026: The Sampler & Scheduler Pack
### Intelligent Rendering Engine for Stable Diffusion (A1111 / Forge)

[![AGGA Engine](https://img.shields.io/badge/AGGA-Engine_2026-blueviolet?style=for-the-badge)](#) [![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge)](#) [![Architecture](https://img.shields.io/badge/Architecture-Latent_Injection-orange?style=for-the-badge)](#)
---

I'm back with something juicy. The truth is, I'm in the red and I simply refuse to buy another external hard drive just to fill it with thousands of LoRAs to fix broken checkpoints.

So, instead of more gigabytes, I brought you **Intelligence**.
Standard samplers (Euler, Heun, DPM) are mathematically pure but visually blind. They traverse noise linearly, unaware of what they are drawing. **AGGA** is different. It acts as an **Intelligent Middleware** between your prompt and the diffusion process.

### 🧠 Core Philosophy: The "Landing Phase"
This engine introduces three proprietary concepts to Stable Diffusion:

1.  **The Landing Phase:** Standard samplers often "wash out" fine texture in the final 15-20% of steps to eliminate noise. AGGA does the opposite: it detects high-frequency details (edges/texture) via gradient analysis and reinforces them just before the process finishes.
2.  **Energy Floor Protection:** We monitor the tensor standard deviation ($\sigma$). If the image energy drops below a threshold (usually 0.90), the engine mathematically injects contrast to prevent the "grey sludge" look common in merged models.
3.  **DNA Translation:** Mathematical bridging that allows different model architectures (Pony, SDXL, Illustrious) to interact without breaking the latent space.

---

## ⚠️ Installation (The "Auto-Inject" Method)

I bypassed the standard Extension format because it adds too much overhead. This is a **Direct Core Integration**. It works natively by injecting logic into the python backend of A1111 or Forge.

**How to install:**
1. Open a new cell in your Colab (or run locally in your python environment).
2. Copy the script below.
3. **Select your version** by changing the `BRANCH` variable (`modules-A1111` or `modules-Forge`).
4. Run it.


```python
# @title 🧬 Install AGGA 2026 Engine (Universal)
# @markdown Run this cell **AFTER** installing WebUI but **BEFORE** launching it.

import os
import requests
from pathlib import Path

# --- USER SELECTION ---
# @markdown Select your WebUI Architecture:
ARCHITECTURE = "A1111" # @param ["A1111", "Forge / Reforge"]

# --- CONFIGURATION ---
GITHUB_USER = "HerrscherAGGA"
REPO_NAME = "AGGA-2026-The-Sampler-Scheduler-Pack"

# Map selection
if ARCHITECTURE == "A1111":
    BRANCH = "modules-A1111"
else:
    BRANCH = "modules-Forge-%26-Reforge" 

# Construimos la URL Raw exacta
BASE_URL = f"https://raw.githubusercontent.com/{GITHUB_USER}/{REPO_NAME}/{BRANCH}"
AGGA_FILES = ['sd_agga_schedulers.py', 'sd_samplers_pseudo_hires.py', 'sd_samplers_pseudo_hires_loader.py']

# --- SMART PATH DETECTION ---
possible_paths = [
    Path('/content/stable-diffusion-webui'),                     # Standard A1111
    Path('/content/webui_forge_cu121_torch231/stable-diffusion-webui'), # Forge standard
    Path('/content/A1111'),                                      # Some notebooks
    Path('/content/gdrive/MyDrive/sd/stable-diffusion-webui'),   # Drive installations
    Path('/content/reforge/stable-diffusion-webui')              # Reforge specific
]

# Find the first path that actually exists
WEBUI_PATH = next((p for p in possible_paths if p.exists()), None)
TARGET_MOD = WEBUI_PATH / "modules" if WEBUI_PATH else None

# --- INSTALLER LOGIC ---
def install_engine():
    if not WEBUI_PATH or not TARGET_MOD.exists():
        print(f"❌ Error: WebUI folder not found.")
        print(f"   PLEASE RUN THE WEBUI INSTALLER CELL FIRST.")
        return

    print(f"🔍 WebUI detected at: {WEBUI_PATH}")
    print(f"🚀 Injecting AGGA 2026 Engine ({ARCHITECTURE} Mode)...")
    print(f"   Using Branch: {BRANCH}")
    
    success_count = 0
    
    # 1. Download Files
    for filename in AGGA_FILES:
        url = f"{BASE_URL}/{filename}"
        dest = TARGET_MOD / filename
        try:
            print(f"  ⬇️ Fetching {filename}...", end=" ")
            r = requests.get(url)
            if r.status_code == 404:
                print(f"❌ Failed (404). URL invalid:\n     {url}")
                return
            r.raise_for_status()
            dest.write_bytes(r.content)
            print("OK")
            success_count += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            return

    # 2. Inject Code (Patching)
    if success_count == len(AGGA_FILES):
        patches = [
            ("sd_schedulers.py", "import modules.sd_agga_schedulers"),
            ("sd_samplers_kdiffusion.py", "import modules.sd_samplers_pseudo_hires_loader")
        ]
        
        print("\n🛠️ Applying Neural Patches...")
        for core_file, import_line in patches:
            fpath = TARGET_MOD / core_file
            if fpath.exists():
                content = fpath.read_text()
                if import_line not in content:
                    with open(fpath, "a") as f: f.write(f"\n{import_line}\n")
                    print(f"  💉 Hook injected into {core_file}")
                else:
                    print(f"  ✨ {core_file} already active")
        
        print(f"\n✅ AGGA Engine Installed successfully.")

install_engine()
```
---

## 🛠️ The Sampler Dictionary

### 1. The "Pseudo-Hires" Series (General Purpose)
*Designed for speed and reliability. These replace Euler/Heun for daily use.*

* **Pseudo-Hires Soft:** The baseline. Pure Euler trajectory with a custom noise ramp. Best for soft anime and painterly styles where you want a balanced composition without artifacts.
* **Pseudo-Hires Sharp:** Introduces a **1.12x energy reinforcement** in the last third of generation (>65% steps). Ideal for defining metallic edges, mecha, and hard surfaces.
* **Pseudo-Hires Ultra:** Aggressive progressive boost scaling up to `1.18 + progress`. Use this for 8K texture prompts (skin pores, fabric weaving).
* **DPM++ 2M Pseudo-Hires:** A modified second-order solver. It keeps the structural stability of DPM++ 2M but adds a "detail kick" at the end to prevent the plastic/smooth look common in standard DPM.

### 2. AGGA Flash v9 (Universal Fusion)
*A 4-in-1 Smart Engine that changes its mathematics based on step count and prompt commands.*

* **Auto-Rescue Mode (<15 steps):** Analyzes image entropy at Step 0.
    * *Too flat/grey?* -> Switches to Euler Ancestral to inject life.
    * *Too chaotic?* -> Switches to DPM++ 2M to enforce structure.
    * *Balanced?* -> Uses Native Turbo mode for speed.
* **High-Res Mode (>15 steps):** Activates the classic Flash V2 logic (Sharpness injection) for crisp details.
* **Fusion Mode (Prompt Control):** You can mix two sampler engines in a single generation using these prompts:

| Prompt Command | Base Engine (Structure) | End Engine (Texture) | Best For... |
| :--- | :--- | :--- | :--- |
| `fsn_euler_dpm` | Euler | DPM++ 2M | Complex illustrations, abstract backgrounds. |
| `fsn_native_flash` | Native | Flash V2 | **The Classic Look.** Latex, armor, high contrast. |
| `fsn_dpm_euler` | DPM++ 2M | Euler | Soft female portraits, oil painting style. |
| `fsn_native_dpm` | Native | DPM++ 2M | "The Tank." Unbreakable hands and complex poses. |

> **Tip:** Use `split_30` or `split_80` in your prompt to decide at what percentage the engine switches.

### 3. The "Native" Series (Detail & Structure)
* **AGGA Detail-Native:** Works as an internal "Refiner." In the last 40% of generation, it "tricks" the model with a lower sigma value to force it to hallucinate micro-details (pores, dust) without changing the overall shape.
* **AGGA Structural-Detail (The Hybrid):** **RECOMMENDED FOR REALISM.**
    * *First 45%: DPM++ 2M* -> Ensures perfect anatomy and coherent composition.
    * *Last 55%: AGGA Detail* -> Injects texture and "crunch."
    * *Result:* Perfect bodies with realistic skin texture.

### 4. AGGA Universal Bridge (The Model Translator)
*Solves the "Fried Image" problem when mixing Pony, SDXL, and Illustrious.*

Instead of linear weighting, this sampler uses a **Quartic Stability Curve** (`f(p) = 1.0 - (2p - 1.0)^4`) to translate the statistical "DNA" (Mean/Std Dev) of one model to another.

**Usage:** Select the `AGGA Lora universal Bridge` sampler and add commands to your prompt.

**Direction Commands (Where are you going?):**
* `Hacia_Pony` (Targeting Pony V6 style)
* `Hacia_Noobai`
* `Hacia_Illustrious_V2`
* `Hacia_Velvette`
* `Hacia_SDXL`

**Power Modes:**
* `Modo_Safe`: 1:1 Technical compatibility. Prevents errors/artifacts without altering colors.
* `Modo_Fuerte`: Aggressive injection. High contrast, deep blacks.
* `Modo_Neutral`: Disables DNA translation, enables only the Sharpness Injector.

> **Example:** You are using a **Juggernaut (SDXL)** checkpoint but want to use a **Pony LoRA**.
> **Prompt:** `<lora:MyPonyLora:1> Hacia_Pony, Modo_Safe`

---

## 📐 Scheduler Guide (Time Warping)

The scheduler controls *time*—how long the sampler spends on each noise level.

* **AGGA Smart v2:** The "Set and Forget" option. It auto-selects the curve:
    * 1-8 Steps: **Turbo/DMD** (Aggressive).
    * 9-19 Steps: **AYS** (Align Your Steps).
    * 20+ Steps: **Dynamic Rho**.
* **Style-Anchor:** Slows down "time" in the middle sigmas (where style concepts are formed), forcing the sampler to process aesthetics more deeply.
* **Double-Anchor (The Sniper):** Features a "time dip" at 82% of the process. This is mathematically tuned to fix small faces and eyes in wide shots.
* **Pixel-Staircase v2:** Groups steps into "terraces." It holds the noise level constant for 2-3 steps, allowing pixel quantization to settle before moving down. **Essential for Pixel Art.**

---

## 🏆 Recommended Workflows

| Goal | Sampler | Scheduler | Why? |
| :--- | :--- | :--- | :--- |
| **Photorealism** | `Structural-Detail` | `Dynamic Rho` | DPM base fixes anatomy; Native finish fixes skin. |
| **Style/Artistic** | `Style-Repair Ultra` | `Style-Ultra` | Recovers lighting/volumetrics in flat 2.5D models. |
| **Pixel Art** | `Pixel-Master` | `Pixel Staircase v2` | Prevents the "smoothing" effect of normal schedulers. |
| **Turbo/Lightning** | `DMD-Turbo Landing` | `AGGA DMD Power` | Converges in 6-8 steps with a clean finish. |
| **Broken LoRAs** | `Universal Bridge` | `AGGA UNIVERSAL BRIDGE` | Fixes "fried" images caused by incompatible LoRAs. |

---

*(c) 2026 AGGA Engine. Open Source Intelligence.*
