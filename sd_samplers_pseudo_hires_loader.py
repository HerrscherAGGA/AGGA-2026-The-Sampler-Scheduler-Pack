from modules import sd_samplers_common
from modules.sd_samplers_kdiffusion import (
    KDiffusionSampler,
    samplers_data_k_diffusion,
)
from modules.sd_samplers_pseudo_hires import (
    sample_pseudo_hires_soft,
    sample_pseudo_hires_sharp,
    sample_pseudo_hires_ultra,
    sample_dpmpp_2m_pseudo_hires,
    sample_pseudo_hires_flash_v2,
    sample_pseudo_hires_detail,
    sample_agga_dmd_turbo,
    sample_agga_herrscher_native,
    sample_agga_detail_native,
    sample_agga_hyper_detail_hybrid,
    sample_agga_structural_detail,
    sample_agga_style_repair_pro,
    sample_agga_style_repair_prompt_aware,
    sample_agga_style_repair_ultra,
    sample_agga_lora_bridge,
    sample_agga_lora_bridge_stable,
    sample_agga_lora_bridge_sharp,
    sample_agga_lora_bridge_ultra_sharp,
    sample_agga_universal_bridge, 
    sample_agga_pixel_master,
)

def register():
    existing = [s.name for s in samplers_data_k_diffusion]
    
    # Lista de tuplas (Nombre, Función, Alias)
    agga_samplers = [
        ("AGGA Pseudo-HiRes Soft", sample_pseudo_hires_soft, ["AGGA_ph_soft"]),
        ("AGGA Pseudo-HiRes Sharp", sample_pseudo_hires_sharp, ["AGGA_ph_sharp"]),
        ("AGGA Pseudo-HiRes Ultra", sample_pseudo_hires_ultra, ["AGGA_ph_ultra"]),
        ("AGGA Pseudo-HiRes DPM++ 2M", sample_dpmpp_2m_pseudo_hires, ["agga_ph_dpm2m"]),
        ("AGGA Pseudo-HiRes Flash v2", sample_pseudo_hires_flash_v2, ["agga_ph_flash_v2"]),
        ("AGGA Pseudo-HiRes Detail", sample_pseudo_hires_detail, ["agga_ph_detail"]),
        ("AGGA DMD-Turbo Landing", sample_agga_dmd_turbo, ["agga_dmd_turbo"]),
        ("AGGA Herrscher-Native", sample_agga_herrscher_native, ["agga_herrscher_v6"]),
        ("AGGA-Detail-Native", sample_agga_detail_native, ["agga_dn"]),
        ("AGGA Structural-Detail (Hybrid)", sample_agga_structural_detail, ["agga_struct_detail"]),
        ("AGGA Hyper-Detail Hybrid", sample_agga_hyper_detail_hybrid, ["agga_hyper_detail_hybrid"]),
        ("AGGA Style-Repair (Test)", sample_agga_style_repair_pro, ["agga_sr"]),
        ("AGGA Style-Repair (Prompt-Aware)", sample_agga_style_repair_prompt_aware, ["agga_sr_prompt"]),
        ("AGGA Style-Repair Ultra", sample_agga_style_repair_ultra, ["agga_sru"]),
        ("AGGA PDXL-Lora Adapted",       sample_agga_lora_bridge,         ["agga_LA"]),
        ("AGGA PDXL-Lora Adapted Stable", sample_agga_lora_bridge_stable,  ["agga_LAS"]),
        ("AGGA PDXL-Lora Adapted Sharp", sample_agga_lora_bridge_sharp,   ["agga_LA_Sharp"]),
        ("AGGA PDXL-Lora Adapted Ultra-Sharp", sample_agga_lora_bridge_ultra_sharp,   ["agga_LA_US"]),
        ("AGGA Lora universal Bridge", sample_agga_universal_bridge,   ["agga_L_UB"]),
        ("AGGA Pixel-Master (Test)", sample_agga_pixel_master, ["agga_pixel"]),
    ]

    count = 0
    for name, func, aliases in agga_samplers:
        if name not in existing:
            # Creamos el SamplerData compatible con A1111/Forge
            # El truco para el módulo: usar lambda para instanciar KDiffusionSampler tardíamente
            s_data = sd_samplers_common.SamplerData(
                name,
                lambda model, f=func: KDiffusionSampler(f, model),
                aliases=aliases,
                options={}
            )
            samplers_data_k_diffusion.append(s_data)
            count += 1

    print(f"[AGGA Module] {count} Samplers registered successfully.")

# Ejecutar automáticamente al ser importado por sd_samplers_kdiffusion.py
register()