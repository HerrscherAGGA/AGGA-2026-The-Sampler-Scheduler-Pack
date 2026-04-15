import modules.sd_schedulers as sd_schedulers
from modules.sd_samplers_pseudo_hires import (
    get_sigmas_agga_dmd, 
    get_sigmas_log_linear, 
    get_sigmas_dynamic_rho, 
    get_sigmas_pseudo_native,
    get_sigmas_agga_smart,
    get_sigmas_style_anchor,
    get_sigmas_ultra_anchor,
    get_sigmas_agga_ays_anchor,
    get_sigmas_agga_double_anchor,
    get_sigmas_agga_pixel_staircase,
    get_sigmas_agga_pixel_staircase_v2,
    get_sigmas_agga_lora_universal_bridge,
)

def register():
    new_data = [
        ('agga_dmd_p', 'AGGA DMD Power', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_agga_dmd(n, sigma_min, sigma_max, device), 7.0),
        
        ('agga_log', 'AGGA Log-Linear', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_log_linear(n, sigma_min, sigma_max, device), -1),
        
        ('agga_dyn_rho', 'AGGA Dynamic Rho', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_dynamic_rho(n, sigma_min, sigma_max, device), -1),
        
        ('agga_pseudo', 'AGGA Pseudo-Native', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_pseudo_native(n, sigma_min, sigma_max, device), -1),

        ('agga_smart', 'AGGA Smart-Automatic', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_agga_smart(n, sigma_min, sigma_max, device), -1),

        ('agga_style_anchor', 'AGGA Style-Anchor', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_style_anchor(n, sigma_min, sigma_max, device), -1),
        ('agga_style_Ultra', 'AGGA Style-Ultra', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_ultra_anchor(n, sigma_min, sigma_max, device), -1),
        ('agga_double_anchor', 'AGGA Double-Anchor', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_agga_double_anchor(n, sigma_min, sigma_max, device), -1),
        ('agga_ays_anchor', 'AGGA AYS-Anchor', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_agga_ays_anchor(n, sigma_min, sigma_max, device), -1),
        ('agga_pixel', 'AGGA Pixel Staircase', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_agga_pixel_staircase(n, sigma_min, sigma_max, device), -1),
        ('agga_pixel_v2', 'AGGA Pixel Staircase V2', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_agga_pixel_staircase_v2(n, sigma_min, sigma_max, device), -1),
        ('agga_LUB', 'AGGA UNIVERSAL BRIDGE', 
         lambda n, sigma_min, sigma_max, device, **k: get_sigmas_agga_lora_universal_bridge(n, sigma_min, sigma_max, device), -1),
    ]

    for name, label, func, rho in new_data:
        if any(x.name == name for x in sd_schedulers.schedulers): continue
        
        sched = sd_schedulers.Scheduler(name, label, func, default_rho=rho)
        sd_schedulers.schedulers.append(sched)
        sd_schedulers.schedulers_map.update({sched.name: sched, sched.label: sched})
    
    print(f"[AGGA Module] {len(new_data)} Schedulers registered successfully.")

# Ejecutar automáticamente al ser importado por sd_schedulers.py
register()
