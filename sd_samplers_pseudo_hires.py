import torch
from modules import shared
import torch.nn.functional as F
import inspect

# =====================================================
# SIGMAS BASE (COMPARTIDOS)
# =====================================================

def pseudo_hires_sigmas(n, sigma_min, sigma_max, device):
    ramp = torch.linspace(0, 1, n, device=device)

    sigmas = sigma_max * torch.exp(-ramp * 4.5)

    p0 = int(n * 0.45)
    p1 = int(n * 0.80)

    sigmas[p0:p1] = sigmas[p0]
    sigmas[p1:] = torch.linspace(sigmas[p1], sigma_min, n - p1, device=device)

    sigmas[-2] = sigmas[-1]
    return sigmas


# =====================================================
# PSEUDO-HIRES SOFT
# =====================================================

@torch.no_grad()
def sample_pseudo_hires_soft(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    **kwargs
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps

    shared.state.job_count = 1
    shared.state.job_no = 0

    for i in range(total_steps):
        shared.state.sampling_step = i + 1

        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma

        x = x + d * dt

        if callback is not None:
            callback({
                "x": x,
                "i": i,
                "sigma": sigma,
                "sampling_step": i + 1,          
                "sampling_steps": total_steps    
            })

    return x


# =====================================================
# PSEUDO-HIRES SHARP
# =====================================================

@torch.no_grad()
def sample_pseudo_hires_sharp(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    **kwargs
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps

    shared.state.job_count = 1
    shared.state.job_no = 0

    for i in range(total_steps):
        shared.state.sampling_step = i + 1

        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma

        if i > total_steps * 0.65:
            x = x + d * dt * 1.12
        else:
            x = x + d * dt
       
        if callback is not None:
            callback({
                "x": x,
                "i": i,
                "sigma": sigma,
                "sampling_step": i + 1,          
                "sampling_steps": total_steps    
            })

    return x


# =====================================================
# PSEUDO-HIRES ULTRA
# =====================================================

@torch.no_grad()
def sample_pseudo_hires_ultra(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    **kwargs
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps

    shared.state.job_count = 1
    shared.state.job_no = 0

    for i in range(total_steps):
        shared.state.sampling_step = i + 1

        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma

        # ULTRA: refuerzo progresivo
        if i > total_steps * 0.55:
            boost = 1.18 + (i / total_steps) * 0.10
            x = x + d * dt * boost
        else:
            x = x + d * dt

        if callback is not None:
            callback({
                "x": x,
                "i": i,
                "sigma": sigma,
                "sampling_step": i + 1,          
                "sampling_steps": total_steps    
            })

    return x

# =====================================================
# DPM++ 2M PSEUDO-HIRES 
# =====================================================

@torch.no_grad()
def sample_dpmpp_2m_pseudo_hires(
    model,
    x,
    sigmas,
    extra_args=None,
    callback=None,
    **kwargs
):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])

    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps

    shared.state.job_count = 1
    shared.state.job_no = 0

    old_denoised = None  

    for i in range(total_steps):
        shared.state.sampling_step = i + 1

        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        denoised = model(x, sigma * s_in, **extra_args)

        if old_denoised is None:
            d = (x - denoised) / sigma
            x = x + d * dt
        else:
            d = (x - denoised) / sigma
            d_old = (x - old_denoised) / sigmas[max(i-1, 0)] if old_denoised is not None else d

            correction = 0.5 * (d + d_old) * dt
            x = x + correction * 1.0  # puedes ajustar el factor

            # Boost progresivo (similar a Ultra, pero más suave para feeling DPM)
            if i > total_steps * 0.60:
                boost = 1.12 + (i / total_steps) * 0.08  # más conservador que Ultra
                x = x + d * dt * boost

        old_denoised = denoised  

        if callback is not None:
            callback({
                "x": x,
                "i": i,
                "sigma": sigma,
                "sampling_step": i + 1,
                "sampling_steps": total_steps
            })

    return x

# =====================================================
# AGGA SMART-COMBO V9 (Universal Fusion)
# =====================================================
@torch.no_grad()
def sample_agga_smart_combo(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    
    # ---------------------------------------------------------
    # 1. Prompt Reader
    # ---------------------------------------------------------
    prompt_tags = ""
    try:
        # Buscamos en la pila de ejecución el objeto 'p' de A1111
        for frame_info in inspect.stack():
            if 'p' in frame_info.frame.f_locals:
                p_obj = frame_info.frame.f_locals['p']
                if hasattr(p_obj, 'prompt'):
                    prompt_tags = (str(p_obj.prompt) + " " + str(p_obj.all_prompts)).lower()
                    break
    except:
        pass

    # ---------------------------------------------------------
    # 2. CONFIGURACIÓN DE ESTRATEGIA
    # ---------------------------------------------------------
    
    # Valores por defecto
    strategy = "AUTO"
    engine_1 = "NONE"
    engine_2 = "NONE"
    split_ratio = 0.5  # 50% por defecto
    
    # A) DETECCIÓN DE MODO FUSIÓN (Prioridad Alta)
    if "fsn_" in prompt_tags:
        strategy = "FUSION"
        
        # Detectar combinaciones
        if "euler_dpm" in prompt_tags:
            engine_1, engine_2 = "EULER_A", "DPM2"
            desc = "Euler A >> DPM++ 2M"
        elif "native_flash" in prompt_tags:
            engine_1, engine_2 = "NATIVE", "FLASH"
            desc = "Native >> Flash V2"
        elif "dpm_euler" in prompt_tags:
            engine_1, engine_2 = "DPM2", "EULER_A"
            desc = "DPM++ 2M >> Euler A"
        elif "native_dpm" in prompt_tags:
            engine_1, engine_2 = "NATIVE", "DPM2"
            desc = "Native >> DPM++ 2M"
        else:
            # Fusión por defecto si solo pone 'fsn_'
            engine_1, engine_2 = "EULER_A", "DPM2"
            desc = "Standard Fusion"

        # Detectar punto de corte personalizado (ej: split_70)
        # Busca palabras como 'split_30', 'split_80'
        import re
        match = re.search(r'split_(\d+)', prompt_tags)
        if match:
            split_val = int(match.group(1))
            split_ratio = split_val / 100.0
            
        shared.state.textinfo = f"AGGA FUSION: [{desc}] @ {int(split_ratio*100)}%"

    # B) MODO AUTOMÁTICO (Si no hay fusión)
    else:
        if total_steps > 15:
            strategy = "HIGH_RES"
            engine_1 = "FLASH" # Solo usa un motor
            shared.state.textinfo = "AGGA High-Res: [Flash V2]"
        else:
            strategy = "SMART_LOW"
            engine_1 = "SMART" # El motor se decide en el paso 0
            shared.state.textinfo = "AGGA Smart: Analyzing..."

    # ---------------------------------------------------------
    # 3. BUCLE DE GENERACIÓN
    # ---------------------------------------------------------
    
    # Estados internos
    old_denoised = None
    h_last = None
    current_engine = engine_1
    
    # Helper para DPM
    def t_fn(sigma): return -sigma.log()

    for i in range(total_steps):
        shared.state.sampling_step = i + 1
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        
        # --- Predicción ---
        denoised = model(x, sigma * s_in, **extra_args)
        
        # --- LÓGICA DE CAMBIO DE MOTOR (Fusión) ---
        if strategy == "FUSION":
            switch_step = int(total_steps * split_ratio)
            
            if i < switch_step:
                new_engine = engine_1
            else:
                new_engine = engine_2
            
            # Si cambiamos de motor, limpiamos la memoria del sampler anterior
            if new_engine != current_engine:
                old_denoised = None # Reset vital para DPM
                current_engine = new_engine

        # --- LÓGICA SMART (Análisis en paso 0) ---
        if strategy == "SMART_LOW" and i == 0:
            energy = denoised.std()
            if energy < 0.88:
                current_engine = "EULER_A"
                tag = "Rescue: Euler A"
            elif energy > 1.25:
                current_engine = "DPM2"
                tag = "Rescue: DPM++ 2M"
            else:
                current_engine = "NATIVE"
                tag = "Native Optimized"
            shared.state.textinfo = f"AGGA Smart: [{tag}]"

        # -----------------------------------------------------
        # EJECUCIÓN DE MOTORES
        # -----------------------------------------------------
        
        # === MOTOR: DPM++ 2M (Exacto) ===
        if current_engine == "DPM2":
            t, t_next = t_fn(sigma), t_fn(sigma_next)
            h = t_next - t
            
            if old_denoised is None or sigma_next == 0:
                x = (sigma_next / sigma) * x - (-h).expm1() * denoised
            else:
                h_last = -sigmas[i-1].log() + sigmas[i].log()
                r = h / h_last
                denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
                x = (sigma_next / sigma) * x - (-h).expm1() * denoised_d

        # === MOTOR: EULER ANCESTRAL (Exacto) ===
        elif current_engine == "EULER_A":
            sigma_up = (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5
            sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
            d = (x - denoised) / sigma
            
            x = x + d * (sigma_down - sigma)
            if sigma_next > 0:
                x = x + torch.randn_like(x) * sigma_up

        # === MOTOR: NATIVE (Turbo Stable) ===
        elif current_engine == "NATIVE":
            if i < total_steps - 1:
                x = denoised + (x - denoised) * (sigma_next / sigma)
            else:
                x = denoised

        # === MOTOR: FLASH V2 (Texturizado) ===
        elif current_engine == "FLASH":
            dt = sigma_next - sigma
            if old_denoised is None:
                d = (x - denoised) / sigma
                x = x + d * dt * 1.05
            else:
                d = (x - denoised) / sigma
                d_old = (x - old_denoised) / sigmas[max(i-1, 0)]
                x = x + (0.6 * d + 0.4 * d_old) * dt
            
            # Boost Lógico de Flash
            if i > total_steps * 0.50:
                progress = (i - total_steps * 0.50) / (total_steps * 0.50)
                boost = 1.08 + progress * 0.18 
                x = x + d * dt * boost
            if i == total_steps - 1:
                x = x + (denoised - x) * 0.15

        old_denoised = denoised
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})
        if shared.state.interrupted: break

    return torch.clamp(x, -5.0, 5.0)
        
# =====================================================
# AGGA PSEUDO-HIRES DETAIL 
# =====================================================

@torch.no_grad()
def sample_pseudo_hires_detail(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    
    prev_x = x.clone()

    def get_detail_mask(latent):
        # Calculamos gradientes de forma más eficiente
        dy = torch.abs(latent[:, :, 1:, :] - latent[:, :, :-1, :])
        dx = torch.abs(latent[:, :, :, 1:] - latent[:, :, :, :-1])
        
        # Padding para recuperar el tamaño original (B, C, H, W)
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        
        grad = torch.sqrt(dx**2 + dy**2).mean(dim=1, keepdim=True)
        # Normalización robusta: No usamos .max() para evitar amplificar ruido en zonas planas
        grad = torch.clamp(grad * 2.0, 0.0, 1.0) 
        return grad
    
    for i in range(total_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma # Dirección del gradiente
        
        progress = i / total_steps
        
        if progress < 0.4:
            multiplier = 1.05
        elif progress < 0.7:
            multiplier = 1.10 + (progress * 0.10)
        else:
            multiplier = 1.0 
            
        x_next = x + d * dt * multiplier

        if progress >= 0.7:
            detail_mask = get_detail_mask(x)

            refine_strength = 0.05 * (1.0 - progress)
            x_next = x_next + (denoised - x_next) * refine_strength * detail_mask
            
            sharpen_strength = 0.03 * detail_mask
            x_next = x_next + (x_next - x) * sharpen_strength

        x = x_next
        
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})

    return torch.clamp(x, -5.0, 5.0)

# =====================================================
# AGGA DMD-TURBO LANDING - Optimizado para guías LCM/DMD2 (pocos pasos)
# =====================================================

@torch.no_grad()
def sample_agga_dmd_turbo(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps

    for i in range(total_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma
        
        if i < total_steps - 1:
            x = x + d * dt
            
        else:
            x = x + (denoised - x) * 0.85

            blurred_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
            sharpen_map = x - blurred_x
            
            x = x + sharpen_map * 0.15

        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})

    return torch.clamp(x, -5.0, 5.0)

# =====================================================
# AGGA Herrscher-Native   
# =====================================================
@torch.no_grad()
def sample_agga_herrscher_native(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    
    # Reducimos momentum para que sea más ágil en pocos pasos (8 steps)
    momentum_beta = 0.50 
    momentum = None
    
    for i in range(total_steps):
        sigma = sigmas[i]
        sigma_next = sigmas[i+1]
        dt = sigma_next - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma
        
        if momentum is None:
            momentum = d
        else:
            momentum = momentum_beta * momentum + (1 - momentum_beta) * d
        
        x_next = x + momentum * dt
        
        progress = i / total_steps
        if 0.2 < progress < 0.9:
            mag = x_next.std()
            if mag < 1.0:
                scale_factor = (1.0 / (mag + 1e-6)) * 0.05
                x_next = x_next * (1.0 + scale_factor)
            
            x_next = x_next - (x_next.mean() * 0.02)

        x = x_next
       
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})

    return torch.clamp(x, -5.0, 5.0)

# =====================================================
# AGGA-Detail-Native
# =====================================================
@torch.no_grad()
def sample_agga_detail_native(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    shared.state.textinfo = "LCM Detail V4: Hires-Refiner Mode"
    
    for i in range(total_steps):
        shared.state.sampling_step = i
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        progress = i / total_steps

        denoised = model(x, sigma * s_in, **extra_args)
        
        if i > 0:
        
            dy = torch.abs(denoised[:, :, 1:, :] - denoised[:, :, :-1, :])
            dx = torch.abs(denoised[:, :, :, 1:] - denoised[:, :, :, :-1])
            dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
            dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
            detail_mask = torch.clamp((dx + dy).mean(dim=1, keepdim=True) * 2.5, 0.0, 1.0)
            
            if progress > 0.60:
            
                sigma_refine = sigma * 0.95
                refine_denoised = model(x, sigma_refine * s_in, **extra_args)
                
                denoised = denoised + (refine_denoised - denoised) * (0.4 * detail_mask)
            else:
                
                blurred = F.avg_pool2d(denoised, 3, 1, 1)
                denoised = denoised + (denoised - blurred) * (0.12 * detail_mask)

        if i < total_steps - 1:
            x = denoised + (x - denoised) * (sigma_next / sigma)
        else:
            x = denoised
        
        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})
        if shared.state.interrupted: break

    return torch.clamp(x, -5.0, 5.0)

# =====================================================
# AGGA Hyper-Detail Hybrid
# =====================================================
@torch.no_grad()
def sample_agga_hyper_detail_hybrid(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    
    split_step = int(total_steps * 0.66)
    
    for i in range(total_steps):
        shared.state.sampling_step = i
        sigma, sigma_next = sigmas[i], sigmas[i + 1]

        denoised = model(x, sigma * s_in, **extra_args)
        
        dy = torch.abs(denoised[:, :, 1:, :] - denoised[:, :, :-1, :])
        dx = torch.abs(denoised[:, :, :, 1:] - denoised[:, :, :, :-1])
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        detail_mask = torch.clamp((dx + dy).mean(dim=1, keepdim=True) * 2.5, 0.0, 1.0)

        if i >= split_step:
            shared.state.textinfo = f"Phase 2: Power Detail (Step {i+1})"
            sigma_refine = sigma * 0.95
            refine_denoised = model(x, sigma_refine * s_in, **extra_args)
            denoised = denoised + (refine_denoised - denoised) * (0.45 * detail_mask)
        
        else:
            shared.state.textinfo = f"Phase 1: V2 Structure (Step {i+1})"
            blurred = F.avg_pool2d(denoised, 3, 1, 1)
            denoised = denoised + (denoised - blurred) * (0.12 * detail_mask)

        if i < total_steps - 1:
            x = denoised + (x - denoised) * (sigma_next / sigma)

            x_std = x.std()
            if x_std < 1.0:
                x = x * (1.0 + (1.0 - x_std) * 0.05)
        else:

            x = denoised

        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})
        if shared.state.interrupted: break

    return torch.clamp(x, -5.0, 5.0)
    
# =====================================================
# AGGA Style Repair
# =====================================================
@torch.no_grad()
def sample_agga_style_repair_pro(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    
    BOCETADO_STEPS = int(total_steps * 0.40) 
    prev_x = x.clone()

    for i in range(total_steps):
        shared.state.sampling_step = i
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        progress = i / total_steps
        
        denoised = model(x, sigma * s_in, **extra_args)  
      
        dy = torch.abs(denoised[:, :, 1:, :] - denoised[:, :, :-1, :])
        dx = torch.abs(denoised[:, :, :, 1:] - denoised[:, :, :, :-1])
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        style_mask = torch.clamp(torch.sqrt(dx**2 + dy**2).mean(dim=1, keepdim=True) * 2.8, 0.0, 1.0)

        if i < BOCETADO_STEPS:
            shared.state.textinfo = f"Phase 1: Creative Sketching (Step {i+1})"

            boost = 1.20 + (i / BOCETADO_STEPS) * 0.25
            d = (x - denoised) / sigma
            x = x + d * (sigma_next - sigma) * boost
        
        else:
            shared.state.textinfo = f"Phase 2: Style Repair (Step {i+1})"
                                    
            strength = 0.05 + progress * 0.15
            x = x + (denoised - x) * strength * style_mask
            
            if i % 3 == 0 and i < total_steps - 1:
            
                sigma_style = sigma * (1.12 + 0.05 * (progress - 0.4))
                denoised_style = model(x, sigma_style * s_in, **extra_args)
            
                style_delta = (denoised_style - denoised) * 0.04 * style_mask
                x = x + style_delta

            x = x + (x - prev_x) * 0.035 * style_mask

        if i < total_steps - 1 and i >= BOCETADO_STEPS:

            x = denoised + (x - denoised) * (sigma_next / sigma)

            x_std = x.std()
            if x_std < 1.08:
                x = x * (1.08 / (x_std + 1e-6))
        elif i == total_steps - 1:

            x = denoised + (x - denoised) * 0.07

        prev_x = x.clone()
        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})
        if shared.state.interrupted: break

    return torch.clamp(x, -5.0, 5.0)

# =====================================================
# AGGA Style Repair Ultra
# =====================================================
@torch.no_grad()
def sample_agga_style_repair_ultra(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    
    BOCETADO_STEPS = int(total_steps * 0.45) 
    prev_x = x.clone()

    for i in range(total_steps):
        shared.state.sampling_step = i
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        progress = i / total_steps
        
        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma
        
        dy = torch.abs(denoised[:, :, 1:, :] - denoised[:, :, :-1, :])
        dx = torch.abs(denoised[:, :, :, 1:] - denoised[:, :, :, :-1])
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        
        style_mask = torch.clamp(torch.sqrt(dx**2 + dy**2).mean(dim=1, keepdim=True) * 2.8, 0.0, 1.0)

        if i < BOCETADO_STEPS:
            x = x + d * dt 
        
        else:
        
            speed_factor = torch.exp(-torch.abs(dt) * 2.5).item()
            u_boost = 1.18 + progress * 0.12
            
            if i % 2 == 0 or speed_factor > 0.6:
            
                sigma_style = sigma * (1.15 + 0.05 * (progress - 0.4))
                denoised_style = model(x, sigma_style * s_in, **extra_args)
                
                atm_mask = torch.clamp(style_mask + 0.20, 0.0, 1.0)
                
                style_delta = (denoised_style - denoised) * 0.05 * atm_mask * (1.0 + speed_factor)
                x = x + style_delta * u_boost

            x = x + d * dt * u_boost
            
            sharpen = 0.038 + (speed_factor * 0.015)
            x = x + (x - prev_x) * sharpen * style_mask

        if i == total_steps - 1:
            x = denoised + (x - denoised) * 0.08

        prev_x = x.clone()
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})
        if shared.state.interrupted: break

    return torch.clamp(x, -5.5, 5.5)

# =====================================================
# AGGA STRUCTURAL-DETAIL (Hybrid V5)
# =====================================================
@torch.no_grad()
def sample_agga_structural_detail(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps

    old_d = None
    
    split_idx = int(total_steps * 0.45)

    for i in range(total_steps):
        shared.state.sampling_step = i + 1
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma

        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma

        if i < split_idx:

            if old_d is None:

                x = x + d * dt
            else:

                x = x + 0.5 * (d + old_d) * dt

        else:

            dy = torch.abs(denoised[:, :, 1:, :] - denoised[:, :, :-1, :])
            dx = torch.abs(denoised[:, :, :, 1:] - denoised[:, :, :, :-1])
            dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
            dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')

            detail_mask = torch.clamp((dx + dy).mean(dim=1, keepdim=True) * 2.2, 0.0, 1.0)
            
            progress_phase2 = (i - split_idx) / (total_steps - split_idx)
            boost_amount = 1.0 + (progress_phase2 * 0.15)
            
            final_dt = dt * (1.0 + (boost_amount - 1.0) * detail_mask)
            
            x = x + d * final_dt
                        
            if progress_phase2 > 0.6:
                x = x + (x - x.clone()) * 0.04 * detail_mask

        old_d = d

        if callback is not None:
            callback({
                "x": x,
                "i": i,
                "sigma": sigma,
                "sampling_step": i + 1,
                "sampling_steps": total_steps
            })

    return torch.clamp(x, -5.0, 5.0)

# =====================================================
# AGGA STYLE-REPAIR (Prompt-Aware Edition)
# =====================================================
@torch.no_grad()
def sample_agga_style_repair_prompt_aware(model, x, sigmas, extra_args=None, callback=None, **kwargs):

    if hasattr(model, 'need_last_noise_uncond'):
        model.need_last_noise_uncond = True
    
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    
    BOCETADO_STEPS = int(total_steps * 0.40)
    prev_x = x.clone()

    for i in range(total_steps):
        shared.state.sampling_step = i
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma
        progress = i / total_steps
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        style_force_vector = None
        
        last_noise_uncond = getattr(model, 'last_noise_uncond', None)
        
        if last_noise_uncond is not None and i > BOCETADO_STEPS:
            # Reconstruimos la imagen "Incondicional" (x0_uncond) desde el ruido
            # x0 = x - sigma * noise
            uncond_denoised = x - sigma * last_noise_uncond
            
            style_force_vector = denoised - uncond_denoised

        dy = torch.abs(denoised[:, :, 1:, :] - denoised[:, :, :-1, :])
        dx = torch.abs(denoised[:, :, :, 1:] - denoised[:, :, :, :-1])
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        style_mask = torch.clamp(torch.sqrt(dx**2 + dy**2).mean(dim=1, keepdim=True) * 2.8, 0.0, 1.0)

        # --- FASE 1: BOCETADO ---
        if i < BOCETADO_STEPS:

            boost = 1.20 + (i / BOCETADO_STEPS) * 0.20
            d = (x - denoised) / sigma
            x = x + d * dt * boost
        
        # --- FASE 2: RECUPERACIÓN DE ESTILO DIRIGIDA ---
        else:

            if style_force_vector is not None:

                prompt_guidance = style_force_vector * style_mask * 0.25
                
                x = x + prompt_guidance * torch.abs(dt)

            x = x + (x - prev_x) * 0.035 * style_mask

            d = (x - denoised) / sigma
            x = x + d * dt

        if i >= BOCETADO_STEPS:
            x_std = x.std()
            if x_std > 1.15: # Límite de seguridad
                x = x * (1.15 / x_std)

        prev_x = x.clone()
        
        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})

    return torch.clamp(x, -5.0, 5.0)

# =====================================================
# AGGA PIXEL-MASTER V10 (Spatial-Only)
# =====================================================
@torch.no_grad()
def sample_agga_pixel_master(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    
    # AJUSTES V10
    # Factor de reducción. 
    # 2.0 = Bloques visibles (SNES)
    # 3.0 - 4.0 = Bloques muy grandes (Atari/NES)
    # Si con 2.0 se ve borroso, SUBE a 3.0 o 4.0.
    block_size = 1.88
    
    for i in range(total_steps):
        shared.state.sampling_step = i
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma
        
        denoised = model(x, sigma * s_in, **extra_args)
        
        d = (x - denoised) / sigma
        x = x + d * dt

        if i == total_steps - 1:

            latents = denoised
            
            h, w = latents.shape[-2:]
            small_h, small_w = int(h / block_size), int(w / block_size)
            
            latents_pixelated = F.interpolate(latents, size=(small_h, small_w), mode='area')
            
            latents_pixelated = F.interpolate(latents_pixelated, size=(h, w), mode='nearest')
            
            x = latents_pixelated

        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})

    return x


# =====================================================
# LORA BRIDGE (PDXL TO VELVETTE_V4/ NoobAI)
# =====================================================

@torch.no_grad()
def sample_agga_lora_bridge(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    
    # ADN de Pony V6 (Valores reales extraídos)
    PONY_STD_TARGET = 0.016593
    DELTA_MEAN = -0.0104 
    
    for i in range(total_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        progress = i / total_steps
        
        # 1. Predicción y Limpieza Anti-NaNs
        denoised = model(x, sigma * s_in, **extra_args)
        denoised = torch.nan_to_num(denoised, nan=0.0, posinf=4.0, neginf=-4.0)
        
        # 2. Monitor de Salud del Tensor
        mag = denoised.std()
        
        # --- EL SUELO DE ENERGÍA AGGA ---
        # Si la energía cae demasiado (gris), forzamos recuperación
        if mag < 0.90:
            denoised = denoised * (0.95 / (mag + 1e-6))

        # 3. Inyección de ADN con Factor 1.3 (Blindada)
        influence = 1.0 - (2.0 * progress - 1.0)**4
        # El ratio de escala se calcula con precisión de seguridad
        scale_ratio = torch.clamp(torch.tensor(PONY_STD_TARGET / (mag + 1e-6)), 0.8, 1.25)
        
        # Aplicamos la fórmula de ADN:
        # $$denoised = denoised \cdot (1 + (ratio - 1) \cdot influence \cdot 1.3) + (\Delta\mu \cdot influence)$$
        denoised = denoised * (1.0 + (scale_ratio - 1.0) * influence * 1.3)
        denoised = denoised + (DELTA_MEAN * influence * 0.7) # Reducido un 30% para evitar el colapso

        # 4. Salto de Consistencia con Suelo de Precisión
        safe_sigma = max(sigma.item(), 1e-4)
        d = (x - denoised) / safe_sigma
        x = x + d * dt 

        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})

    return torch.clamp(x, -6.0, 6.0)

@torch.no_grad()
def sample_agga_lora_bridge_stable(model, x, sigmas, extra_args=None, callback=None, **kwargs):
   
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    
    # ADN de Pony V6 (Valores reales extraídos)
    PONY_STD_TARGET = 0.016593
    DELTA_MEAN = -0.0104 
    
    for i in range(total_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        progress = i / total_steps
        
        # 1. Predicción y Sanitización Instantánea (FP16 Safe)
        denoised = model(x, sigma * s_in, **extra_args)
        denoised = torch.nan_to_num(denoised, nan=0.0, posinf=4.0, neginf=-4.0)
        
        # 2. Monitor de Salud y Suelo de Energía
        mag = denoised.std()
        
        # Si la energía cae por debajo de 0.90 (zona de peligro gris), inyectamos soporte.
        # Esto es lo que salvó tu imagen en el paso 08.
        if mag < 0.90:
            denoised = denoised * (0.95 / (mag + 1e-6))

        # 3. Inyección de ADN Pony
        # Curva de influencia tipo campana
        influence = 1.0 - (2.0 * progress - 1.0)**4
        
        # Cálculo del ratio con límites estrictos (basado en tus logs)
        scale_ratio = torch.clamp(torch.tensor(PONY_STD_TARGET / (mag + 1e-6)), 0.8, 1.25)
        
        # Aplicamos el ADN con tu factor agresivo de 1.3, ahora que es seguro
        denoised = denoised * (1.0 + (scale_ratio - 1.0) * influence * 1.3)
        denoised = denoised + (DELTA_MEAN * influence * 0.7)

        # 4. Salto de Consistencia Blindado
        # Evita la división por cero al final que causa la "niebla"
        safe_sigma = max(sigma.item(), 1e-4)
        d = (x - denoised) / safe_sigma
        x = x + d * dt 

        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})

    return torch.clamp(x, -6.0, 6.0)

@torch.no_grad()
def sample_agga_lora_bridge_sharp(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    
    PONY_STD_TARGET = 0.016593
    DELTA_MEAN = -0.0104 
    
    for i in range(total_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        progress = i / total_steps
        
        # 1. Predicción y Sanitización
        denoised = model(x, sigma * s_in, **extra_args)
        denoised = torch.nan_to_num(denoised, nan=0.0, posinf=4.0, neginf=-4.0)
        
        # 2. Monitor de Salud
        mag = denoised.std()

        # --- INYECTOR DE NITIDEZ INTELIGENTE ---
        # Condición doble:
        # a) Solo en fase de textura (30% al 85%)
        # b) SOLO si el tensor está "✅ ESTABLE" (mag >= 0.90)
        if 0.3 < progress < 0.85 and mag >= 0.90:
            # Aislamiento de altas frecuencias (bordes, pestañas, texturas finas)
            blurred = F.avg_pool2d(denoised, kernel_size=3, stride=1, padding=1)
            high_freq = denoised - blurred
            # Inyectamos un 15% extra de nitidez
            denoised = denoised + (high_freq * 0.15)
        
        # --- SUELO DE ENERGÍA ---
        # Si no está estable, aplicamos el rescate normal
        if mag < 0.90:
            denoised = denoised * (0.95 / (mag + 1e-6))

        # 3. Inyección de ADN Pony (Igual que la estable)
        influence = 1.0 - (2.0 * progress - 1.0)**4
        scale_ratio = torch.clamp(torch.tensor(PONY_STD_TARGET / (mag + 1e-6)), 0.8, 1.25)
        denoised = denoised * (1.0 + (scale_ratio - 1.0) * influence * 1.3)
        denoised = denoised + (DELTA_MEAN * influence * 0.7)

        # 4. Salto de Consistencia Blindado
        safe_sigma = max(sigma.item(), 1e-4)
        d = (x - denoised) / safe_sigma
        x = x + d * dt 

        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})

    return torch.clamp(x, -6.0, 6.0)


@torch.no_grad()
def sample_agga_lora_bridge_ultra_sharp(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    
    # Valores de ADN
    PONY_STD_TARGET = 0.016593
    DELTA_MEAN = -0.0104
    # NUEVO: Factor de "Temperatura" o Vibrancia (1.15 = 15% extra de pop)
    VIBRANCY_FACTOR = 1.15 
    
    for i in range(total_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        progress = i / total_steps
        
        # 1. Predicción y Sanitización
        denoised = model(x, sigma * s_in, **extra_args)
        denoised = torch.nan_to_num(denoised, nan=0.0, posinf=4.5, neginf=-4.5)
        
        # 2. Monitor de Salud
        mag = denoised.std()

        # --- FASE DE INYECCIÓN ACTIVA (30% al 85%) ---
        if 0.3 < progress < 0.85 and mag >= 0.88:
            # A) INYECTOR DE NITIDEZ (Bordes)
            blurred = F.avg_pool2d(denoised, kernel_size=3, stride=1, padding=1)
            high_freq = denoised - blurred
            denoised = denoised + (high_freq * 0.18) # Ligeramente más agresivo (18%)
            
            # B) NUEVO: INYECTOR DE VIBRANCIA (Temperatura/Color)
            # Calculamos la media actual del tensor
            current_mean = denoised.mean(dim=(1, 2, 3), keepdim=True)
            # Centramos el color alrededor de la media
            centered_color = denoised - current_mean
            # Estiramos la saturación (boost de vibrancia) sin mover el brillo medio
            boosted_color = centered_color * VIBRANCY_FACTOR
            # Restauramos la media
            denoised = boosted_color + current_mean
        
        # --- SUELO DE ENERGÍA (Anti-Gris) ---
        if mag < 0.88:
             denoised = denoised * (0.95 / (mag + 1e-6))

        # 3. Inyección de ADN Pony (Estructural)
        influence = 1.0 - (2.0 * progress - 1.0)**4
        scale_ratio = torch.clamp(torch.tensor(PONY_STD_TARGET / (mag + 1e-6)), 0.8, 1.3)
        
        # Aplicamos el ADN con tu factor 1.3
        denoised = denoised * (1.0 + (scale_ratio - 1.0) * influence * 1.3)
        # Aplicamos el desplazamiento de negros
        denoised = denoised + (DELTA_MEAN * influence * 0.7)

        # 4. Salto de Consistencia Blindado
        safe_sigma = max(sigma.item(), 1e-4)
        d = (x - denoised) / safe_sigma
        x = x + d * dt 

        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})

    return torch.clamp(x, -6.0, 6.0)

# =====================================================
# AGGA UNIVERSAL BRIDGE 
# =====================================================
DNA_LIBRARY = {
    "PONY":           {"std": 0.016593, "mean": -0.000042},
    "NOOBAI":         {"std": 0.019757, "mean": 0.010395},
    "ILLUSTRIOUS_V1": {"std": 0.021907, "mean": 0.012189},
    "ILLUSTRIOUS_V2": {"std": 0.022665, "mean": 0.011841},
    "VELVETTE":       {"std": 0.019734, "mean": 0.010366},
    "ONEOBSESSION":   {"std": 0.019625, "mean": 0.010283},
    "SDXL":           {"std": 0.017260, "mean": 0.000001},
    "ANIMAGINE":      {"std": 0.017165, "mean": 0.000001},
}

@torch.no_grad()
def sample_agga_universal_bridge(model, x, sigmas, extra_args=None, callback=None, **kwargs):   
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    
    source_dna, target_dna = None, None
    p_scale, p_mean = 1.15, 0.85 

    for i in range(total_steps):
        shared.state.sampling_step = i + 1
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        progress = i / total_steps
        
        denoised = model(x, sigma * s_in, **extra_args)
        denoised = torch.nan_to_num(denoised, nan=0.0, posinf=4.0, neginf=-4.0)
        mag = denoised.std()
        
        if i == 0:
            # 1. AUTO-DETECCIÓN DE BASE
            source_key = min(DNA_LIBRARY, key=lambda k: abs(DNA_LIBRARY[k]["std"] - mag.item()))
            source_dna = DNA_LIBRARY[source_key]
            target_dna = source_dna 
            
            # 2. HACK DE DETECCIÓN DE PROMPT (Fix A1111/Reforge)
            prompt = ""
            for frame_info in inspect.stack():
                if 'p' in frame_info.frame.f_locals:
                    p_obj = frame_info.frame.f_locals['p']
                    if hasattr(p_obj, 'prompt'):
                        prompt = str(p_obj.prompt).lower()
                        break
            
            # 3. MATRIZ DE COMANDOS
            detected_hacia = "Nativo (Auto)"
            detected_desde = "Auto-Detectado"
            for key in DNA_LIBRARY.keys():
                k_low = key.lower()
                if f"hacia_{k_low}" in prompt: 
                    target_dna = DNA_LIBRARY[key]
                    detected_hacia = key
                if f"desde_{k_low}" in prompt: 
                    source_dna = DNA_LIBRARY[key]
                    source_key = key
                    detected_desde = "Forzado (Manual)"

            # 4. SELECTOR DE POTENCIA
            modo_desc = "Estándar"
            if "modo_fuerte" in prompt: 
                p_scale, p_mean = 1.35, 0.75
                modo_desc = "Fuerte (Contraste AGGA)"
            elif "modo_safe" in prompt: 
                p_scale, p_mean = 1.0, 1.0
                modo_desc = "Safe (Sincronización 1:1)"
            elif "modo_ultra" in prompt: 
                p_scale, p_mean = 1.50, 0.60
                modo_desc = "Ultra (Fuerza Bruta)"
            elif "modo_neutral" in prompt: 
                p_scale, p_mean = 0.0, 0.0
                modo_desc = "Neutral (Solo Post-Proceso)"

            # --- PANEL DE CONTROL AGGA: MOSTRANDO EL PODER ---
            print(f"\n" + "═"*60)
            print(f" 🚀 AGGA UNIVERSAL BRIDGE V10 - MOTOR ACTIVO")
            print(f" " + "─"*58)
            print(f" 📡 BASE: {source_key} [{detected_desde}]")
            print(f" 🎯 LORA: {detected_hacia} (Hacia)")
            print(f" ⚡ MODO: {modo_desc}")
            print(f" 📈 PARÁMETROS: Scale {p_scale} | Mean Shift {p_mean}")
            
            # Extraemos solo tus comandos del prompt para mostrarlos
            comandos = [w for w in prompt.split() if any(x in w for x in ["hacia_", "desde_", "modo_"])]
            if comandos: print(f" 📝 COMANDOS DETECTADOS: {', '.join(comandos)}")
            
            print(f" 🛠️  MÉTODO: Traducción Matricial de ADN (Campana de Gauss)")
            print(f"═"*60 + "\n")

        # 5. INYECTOR DE NITIDEZ (Tus 0.16 clásicos)
        if 0.35 < progress < 0.85 and mag >= 0.90:
            blurred = F.avg_pool2d(denoised, kernel_size=3, stride=1, padding=1)
            denoised = denoised + ((denoised - blurred) * 0.16)

        # 6. TRADUCCIÓN DE ADN MATRICIAL (Campana de Gauss)
        influence = 1.0 - (2.0 * progress - 1.0)**4
        scale_ratio = torch.clamp(torch.tensor(target_dna["std"] / (mag + 1e-6)), 0.6, 1.4)
        
        denoised = denoised * (1.0 + (scale_ratio - 1.0) * influence * p_scale)
        denoised = denoised + ((target_dna["mean"] - source_dna["mean"]) * influence * p_mean)

        # 7. SUELO DE ENERGÍA Y SALTO
        if mag < 0.88: denoised = denoised * (0.95 / (mag + 1e-6))
        x = x + ((x - denoised) / max(sigma.item(), 1e-4)) * dt

        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})

    return torch.clamp(x, -6.0, 6.0)

# =====================================================
# AGGA AUTO-INTELLIGENT SCHEDULER V2 
# =====================================================

def get_sigmas_agga_smart(n, sigma_min, sigma_max, device):
       
    # 1. ESCÁNER DE PROMPT (Para overrides manuales)
    prompt_tags = ""
    try:
        for frame_info in inspect.stack():
            if 'p' in frame_info.frame.f_locals:
                p_obj = frame_info.frame.f_locals['p']
                if hasattr(p_obj, 'prompt'):
                    prompt_tags = (str(p_obj.prompt) + " " + str(p_obj.all_prompts)).lower()
                    break
    except:
        pass

    # 2. SELECTOR MANUAL (Prioridad Absoluta)
    if "sched_ays" in prompt_tags: 
        print(" AGGA SCHEDULER: Forzado a [AYS Anchor]")
        return get_sigmas_agga_ays_anchor(n, sigma_min, sigma_max, device)
    
    if "sched_turbo" in prompt_tags: 
        print(" AGGA SCHEDULER: Forzado a [DMD Turbo]")
        return get_sigmas_agga_dmd(n, sigma_min, sigma_max, device)
    
    if "sched_repair" in prompt_tags: 
        print(" AGGA SCHEDULER: Forzado a [Double Anchor]")
        return get_sigmas_agga_double_anchor(n, sigma_min, sigma_max, device)

    # 3. LÓGICA AUTOMÁTICA POR ZONAS (La magia)
    
    # ZONA 1: LIGHTNING (1-4 Pasos)
    # Aquí necesitamos Rho explosivo para que la imagen aparezca de golpe.
    if n <= 4:
        # Usamos DMD pero con una curva logarítmica simulada para convergencia instantánea
        return get_sigmas_agga_dmd(n, sigma_min, sigma_max, device)

    # ZONA 2: TURBO (5-8 Pasos) -> Tu favorito para Flash V2
    # La curva DMD (Rho 7) es perfecta aquí porque mantiene el ruido alto al principio
    # y cae en picada al final, ideal para samplers agresivos.
    elif n <= 8:
        return get_sigmas_agga_dmd(n, sigma_min, sigma_max, device)

    # ZONA 3: HYBRID SWEET-SPOT (9-19 Pasos) -> La zona de FUSIÓN
    # Aquí es donde AYS (Align Your Steps) brilla. Es una curva optimizada por NVIDIA
    # que distribuye los pasos "donde más importan". Ideal para 'fsn_euler_dpm'.
    elif n < 20:
        return get_sigmas_agga_ays_anchor(n, sigma_min, sigma_max, device)

    # ZONA 4: HIGH FIDELITY (20-49 Pasos)
    # Dynamic Rho empieza suave y se vuelve preciso.
    # Da el mejor acabado de piel y texturas para 'sample_pseudo_hires_flash_v2'.
    elif n < 50:
        return get_sigmas_dynamic_rho(n, sigma_min, sigma_max, device)

    # ZONA 5: DEEP REPAIR / WALLPAPER (50+ Pasos)
    # Si pones tantos pasos, es porque quieres arreglar algo roto o hacer upscaling.
    # Usamos Double Anchor para fijar la estructura.
    else:
        return get_sigmas_agga_double_anchor(n, sigma_min, sigma_max, device)
                
# =====================================================
# AGGA Schedule   
# =====================================================

def get_sigmas_agga_dmd(n, sigma_min, sigma_max, device):
   
    rho = 7.0 
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_log_linear(n, sigma_min, sigma_max, device):
    """Ideal para AGGA Detail: Distribución logarítmica perfecta."""
    steps = torch.linspace(0, 1, n, device=device)
    sigmas = torch.exp(torch.log(torch.tensor(sigma_max)) * (1 - steps) + torch.log(torch.tensor(sigma_min)) * steps)
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_dynamic_rho(n, sigma_min, sigma_max, device, rho_start=5.0, rho_end=9.0):
    """Curva de potencia variable para máxima estabilidad."""
    ramp = torch.linspace(0, 1, n, device=device)
   
    v_rho = rho_start + (rho_end - rho_start) * ramp
    
    sigmas = []
    for i in range(n):
        r = v_rho[i]
        m_inv_r = sigma_min ** (1 / r)
        M_inv_r = sigma_max ** (1 / r)
        sig = (M_inv_r + ramp[i] * (m_inv_r - M_inv_r)) ** r
        sigmas.append(sig)
    
    sigmas = torch.stack(sigmas)
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_pseudo_native(n, sigma_min, sigma_max, device):
    """Tu lógica original de rampas Soft/Sharp ahora como Scheduler universal."""
    return pseudo_hires_sigmas(n, sigma_min, sigma_max, device)

def get_sigmas_style_anchor(n, sigma_min, sigma_max, device):
   
    ramp = torch.linspace(0, 1, n, device=device)
    
    ramp_anchored = ramp + 0.15 * torch.sin(ramp * 3.1415)
    
    sigmas = sigma_max * ((sigma_min / sigma_max) ** ramp_anchored)
    
    sigmas[-2] = sigmas[-2] * 0.9 
    
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_ultra_anchor(n, sigma_min, sigma_max, device):
    
    ramp = torch.linspace(0, 1, n, device=device)
    
    ramp_hybrid = ramp + 0.32 * torch.sin(ramp * torch.pi)
    
    sigmas = sigma_max * ((sigma_min / sigma_max) ** ramp_hybrid)
    
    sigmas[-3:] = torch.linspace(sigmas[-3].item(), sigma_min, 3, device=device)
    
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_agga_double_anchor(n, sigma_min, sigma_max, device):
    
    t = torch.linspace(0, 1, n, device=device)

    style_warp = 0.35 * torch.sin(t * torch.pi)
    
    eye_sniper = 0.15 * torch.exp(-180 * (t - 0.82)**2)
    
    dmd_tail = 0.10 * (t**4) 

    t_warped = t + style_warp + eye_sniper + dmd_tail
    t_warped = t_warped / t_warped[-1] # Normalización de la línea de tiempo

    sigmas = sigma_max * ((sigma_min / sigma_max) ** (t_warped**1.1))
    
    s_start = torch.log10(sigmas[-4])
    s_end = torch.log10(torch.as_tensor(sigma_min, device=device))
    
    sigmas[-4:] = torch.logspace(s_start, s_end, 4, device=device)
    
    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_agga_pixel_staircase(n, sigma_min, sigma_max, device):
    
    t = torch.linspace(0, 1, n, device=device)
    sigmas = torch.exp(torch.log(torch.tensor(sigma_max)) * (1 - t) + torch.log(torch.tensor(sigma_min)) * t)
    
    for i in range(len(sigmas)):
        if i % 3 != 0 and i > 0:
            # Bajada minúscula para evitar dt=0 (que rompe el sampler)
            sigmas[i] = sigmas[i-1] * 0.98 
            
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_agga_pixel_staircase_v2(
    n,
    sigma_min,
    sigma_max,
    device,
    hold_steps=3,
    decay=0.985
):
    t = torch.linspace(0, 1, n, device=device)
    s_max = torch.as_tensor(sigma_max, device=device)
    s_min = torch.as_tensor(sigma_min, device=device)

    sigmas = torch.exp(
        torch.log(s_max) * (1 - t) +
        torch.log(s_min) * t
    )

    for i in range(1, len(sigmas)):
        if i % hold_steps != 0:

            sigmas[i] = sigmas[i - 1] * decay

    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_agga_lora_universal_bridge(n, sigma_min, sigma_max, device):
    t = torch.linspace(0, 1, n, device=device)
    
    t_warped = t + 0.28 * torch.sin(t * torch.pi)
    
    sigmas = sigma_max * ((sigma_min / sigma_max) ** (t_warped**1.15))

    sigmas[-2] = sigmas[-2] * 0.90
    
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_agga_ays_anchor(n, sigma_min, sigma_max, device):
    ays_base = [14.615, 6.315, 3.771, 2.181, 1.342, 0.862, 0.555, 0.380, 0.234, 0.113, 0.029]
    t = torch.linspace(0, 1, n, device=device)
    t_warped = t + 0.15 * torch.sin(t * torch.pi) 
    sigmas_base = torch.tensor(ays_base, device=device).log()
    indices = t_warped * (len(ays_base) - 1)
    idx_floor = indices.long().clamp(0, len(ays_base)-2)
    idx_ceil = (idx_floor + 1).clamp(0, len(ays_base)-1)
    alpha = indices - idx_floor
    
    log_sigmas = sigmas_base[idx_floor] * (1 - alpha) + sigmas_base[idx_ceil] * alpha
    sigmas = torch.exp(log_sigmas)
    
    return torch.cat([sigmas, sigmas.new_zeros([1])])