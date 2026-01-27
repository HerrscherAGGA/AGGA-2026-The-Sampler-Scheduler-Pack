import torch
import torch.nn.functional as F
from modules import shared
import inspect

# =====================================================
# SIGMAS & SCHEDULER LOGIC
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

def get_sigmas_agga_dmd(n, sigma_min, sigma_max, device):
    rho = 7.0 
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_log_linear(n, sigma_min, sigma_max, device):
    steps = torch.linspace(0, 1, n, device=device)
    s_max = torch.tensor(sigma_max, device=device)
    s_min = torch.tensor(sigma_min, device=device)
    sigmas = torch.exp(torch.log(s_max) * (1 - steps) + torch.log(s_min) * steps)
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_dynamic_rho(n, sigma_min, sigma_max, device, rho_start=5.0, rho_end=9.0):
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
    
    # Fix tensor types
    s_min_t = torch.as_tensor(sigma_min, device=device)
    current_last = sigmas[-3]
    
    sigmas[-3:] = torch.linspace(current_last.item(), s_min_t.item(), 3, device=device)
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_agga_double_anchor(n, sigma_min, sigma_max, device):
    t = torch.linspace(0, 1, n, device=device)
    style_warp = 0.35 * torch.sin(t * torch.pi)
    eye_sniper = 0.15 * torch.exp(-180 * (t - 0.82)**2)
    dmd_tail = 0.10 * (t**4) 
    t_warped = t + style_warp + eye_sniper + dmd_tail
    t_warped = t_warped / t_warped[-1]
    sigmas = sigma_max * ((sigma_min / sigma_max) ** (t_warped**1.1))
    
    s_start = torch.log10(sigmas[-4])
    s_end = torch.log10(torch.as_tensor(sigma_min, device=device))
    sigmas[-4:] = torch.logspace(s_start, s_end, 4, device=device)
    return torch.cat([sigmas, sigmas.new_zeros([1])])

def get_sigmas_agga_pixel_staircase(n, sigma_min, sigma_max, device):
    t = torch.linspace(0, 1, n, device=device)
    s_max = torch.as_tensor(sigma_max, device=device)
    s_min = torch.as_tensor(sigma_min, device=device)
    sigmas = torch.exp(torch.log(s_max) * (1 - t) + torch.log(s_min) * t)
    
    for i in range(len(sigmas)):
        if i % 3 != 0 and i > 0:
            sigmas[i] = sigmas[i-1] * 0.98 
    return torch.cat([sigmas, sigmas.new_zeros([1])])

import torch

def get_sigmas_agga_pixel_staircase_v2(
    n,
    sigma_min,
    sigma_max,
    device,
    hold_steps=3,
    decay=0.985
):
    """
    AGGA Pixel Staircase Scheduler
    - hold_steps: cuántos pasos se 'congelan'
    - decay: cuánto cae el sigma en pasos congelados
    """

    t = torch.linspace(0, 1, n, device=device)
    s_max = torch.as_tensor(sigma_max, device=device)
    s_min = torch.as_tensor(sigma_min, device=device)

    # Base log schedule (estable)
    sigmas = torch.exp(
        torch.log(s_max) * (1 - t) +
        torch.log(s_min) * t
    )

    for i in range(1, len(sigmas)):
        if i % hold_steps != 0:
            # Paso retenido (edge locking)
            sigmas[i] = sigmas[i - 1] * decay

    return torch.cat([sigmas, sigmas.new_zeros([1])])


def get_sigmas_agga_smart(n, sigma_min, sigma_max, device):
    if n <= 8:
        return get_sigmas_agga_dmd(n, sigma_min, sigma_max, device)
    else:
        return get_sigmas_dynamic_rho(n, sigma_min, sigma_max, device)


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

def get_sigmas_agga_lora_universal_bridge(n, sigma_min, sigma_max, device):
    t = torch.linspace(0, 1, n, device=device)
    
    t_warped = t + 0.28 * torch.sin(t * torch.pi)
    
    sigmas = sigma_max * ((sigma_min / sigma_max) ** (t_warped**1.15))

    sigmas[-2] = sigmas[-2] * 0.90
    
    return torch.cat([sigmas, sigmas.new_zeros([1])])

# =====================================================
# SAMPLER FUNCTIONS
# =====================================================

@torch.no_grad()
def sample_pseudo_hires_soft(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    for i in range(total_steps):
        shared.state.sampling_step = i + 1
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma
        x = x + d * dt
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})
    return x

@torch.no_grad()
def sample_pseudo_hires_sharp(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    for i in range(total_steps):
        shared.state.sampling_step = i + 1
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma
        if i > total_steps * 0.65:
            x = x + d * dt * 1.12
        else:
            x = x + d * dt
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})
    return x

@torch.no_grad()
def sample_pseudo_hires_ultra(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    for i in range(total_steps):
        shared.state.sampling_step = i + 1
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma
        if i > total_steps * 0.55:
            boost = 1.18 + (i / total_steps) * 0.10
            x = x + d * dt * boost
        else:
            x = x + d * dt
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})
    return x

@torch.no_grad()
def sample_dpmpp_2m_pseudo_hires(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    old_denoised = None  
    for i in range(total_steps):
        shared.state.sampling_step = i + 1
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        denoised = model(x, sigma * s_in, **extra_args)
        if old_denoised is None:
            d = (x - denoised) / sigma
            x = x + d * dt
        else:
            d = (x - denoised) / sigma
            d_old = (x - old_denoised) / sigmas[max(i-1, 0)] if old_denoised is not None else d
            correction = 0.5 * (d + d_old) * dt
            x = x + correction * 1.0 
            if i > total_steps * 0.60:
                boost = 1.12 + (i / total_steps) * 0.08
                x = x + d * dt * boost
        old_denoised = denoised
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})
    return x

@torch.no_grad()
def sample_pseudo_hires_flash_v2(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    old_denoised = None
    for i in range(total_steps):
        shared.state.sampling_step = i + 1
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        denoised = model(x, sigma * s_in, **extra_args)
        if old_denoised is None:
            d = (x - denoised) / sigma
            x = x + d * dt * 1.05
        else:
            d = (x - denoised) / sigma
            d_old = (x - old_denoised) / sigmas[max(i-1, 0)]
            x = x + (0.6 * d + 0.4 * d_old) * dt
        if i > total_steps * 0.50:
            progress = (i - total_steps * 0.50) / (total_steps * 0.50)
            boost = 1.08 + progress * 0.18
            x = x + d * dt * boost
        if i == total_steps - 1:
            x = x + (denoised - x) * 0.15
        old_denoised = denoised
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})
    return x

@torch.no_grad()
def sample_pseudo_hires_detail(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    def get_detail_mask(latent):
        dy = torch.abs(latent[:, :, 1:, :] - latent[:, :, :-1, :])
        dx = torch.abs(latent[:, :, :, 1:] - latent[:, :, :, :-1])
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        grad = torch.sqrt(dx**2 + dy**2).mean(dim=1, keepdim=True)
        grad = torch.clamp(grad * 2.0, 0.0, 1.0) 
        return grad
    for i in range(total_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        dt = sigma_next - sigma
        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma
        progress = i / total_steps
        if progress < 0.4: multiplier = 1.05
        elif progress < 0.7: multiplier = 1.10 + (progress * 0.10)
        else: multiplier = 1.0
        x_next = x + d * dt * multiplier
        if progress >= 0.7:
            detail_mask = get_detail_mask(x)
            refine_strength = 0.05 * (1.0 - progress)
            x_next = x_next + (denoised - x_next) * refine_strength * detail_mask
            sharpen_strength = 0.03 * detail_mask
            x_next = x_next + (x_next - x) * sharpen_strength
        x = x_next
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})
    return torch.clamp(x, -5.0, 5.0)

@torch.no_grad()
def sample_agga_dmd_turbo(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    for i in range(total_steps):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
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
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})
    return torch.clamp(x, -5.0, 5.0)

@torch.no_grad()
def sample_agga_herrscher_native(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    momentum_beta = 0.50 
    momentum = None
    for i in range(total_steps):
        sigma, sigma_next = sigmas[i], sigmas[i+1]
        dt = sigma_next - sigma
        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma
        if momentum is None: momentum = d
        else: momentum = momentum_beta * momentum + (1 - momentum_beta) * d
        x_next = x + momentum * dt
        progress = i / total_steps
        if 0.2 < progress < 0.9:
            mag = x_next.std()
            if mag < 1.0:
                scale_factor = (1.0 / (mag + 1e-6)) * 0.05
                x_next = x_next * (1.0 + scale_factor)
            x_next = x_next - (x_next.mean() * 0.02)
        x = x_next
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i + 1, "sampling_steps": total_steps})
    return torch.clamp(x, -5.0, 5.0)

@torch.no_grad()
def sample_agga_detail_native(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
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
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})
        if shared.state.interrupted: break
    return torch.clamp(x, -5.0, 5.0)

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
            sigma_refine = sigma * 0.95
            refine_denoised = model(x, sigma_refine * s_in, **extra_args)
            denoised = denoised + (refine_denoised - denoised) * (0.45 * detail_mask)
        else:
            blurred = F.avg_pool2d(denoised, 3, 1, 1)
            denoised = denoised + (denoised - blurred) * (0.12 * detail_mask)
        if i < total_steps - 1:
            x = denoised + (x - denoised) * (sigma_next / sigma)
            x_std = x.std()
            if x_std < 1.0: x = x * (1.0 + (1.0 - x_std) * 0.05)
        else:
            x = denoised
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})
        if shared.state.interrupted: break
    return torch.clamp(x, -5.0, 5.0)

# =====================================================
# AGGA STYLE-REPAIR (Prompt-Aware Edition)
# =====================================================
@torch.no_grad()
def sample_agga_style_repair_prompt_aware(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    # 1. HACK CLAVE: Pedimos al backend que guarde el ruido negativo (Uncond)
    # Esto es vital. Sin esto, no podemos saber "qué es lo que el modelo quiere ocultar".
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
        
        # 2. Predicción estándar (Ya tiene el CFG aplicado)
        # Esto es: "Uncond + (Cond - Uncond) * CFG_Scale"
        denoised = model(x, sigma * s_in, **extra_args)
        
        # 3. RECUPERACIÓN DEL VECTOR OCULTO (Lógica CFG++)
        # Intentamos recuperar la imagen "Incondicional" (la tendencia natural del modelo, ej: 2D)
        # para compararla con la "Denoised" (tu prompt, ej: 3D).
        style_force_vector = None
        
        # Verificamos si el modelo guardó el ruido incondicional (Uncond)
        last_noise_uncond = getattr(model, 'last_noise_uncond', None)
        
        if last_noise_uncond is not None and i > BOCETADO_STEPS:
            # Reconstruimos la imagen "Incondicional" (x0_uncond) desde el ruido
            # x0 = x - sigma * noise
            uncond_denoised = x - sigma * last_noise_uncond
            
            # EL VECTOR DE LA VERDAD:
            # La diferencia entre lo que obtuviste (denoised) y lo que el modelo quería darte (uncond)
            # es la esencia pura de tu prompt.
            style_force_vector = denoised - uncond_denoised

        # 4. Máscara de Estilo (Tu algoritmo clásico 2.8x)
        dy = torch.abs(denoised[:, :, 1:, :] - denoised[:, :, :-1, :])
        dx = torch.abs(denoised[:, :, :, 1:] - denoised[:, :, :, :-1])
        dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
        dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
        style_mask = torch.clamp(torch.sqrt(dx**2 + dy**2).mean(dim=1, keepdim=True) * 2.8, 0.0, 1.0)

        # --- FASE 1: BOCETADO ---
        if i < BOCETADO_STEPS:
            # Boost estructural simple para definir formas
            boost = 1.20 + (i / BOCETADO_STEPS) * 0.20
            d = (x - denoised) / sigma
            x = x + d * dt * boost
        
        # --- FASE 2: RECUPERACIÓN DE ESTILO DIRIGIDA ---
        else:
            # A. Inyección de Prompt Vector (La magia nueva)
            if style_force_vector is not None:
                # Si el modelo tiene un estilo oculto 3D, este vector apuntará fuerte hacia él.
                # Lo inyectamos SOLO donde hay detalle (style_mask).
                # Multiplicador 0.25 es fuerte pero seguro porque está enmascarado.
                prompt_guidance = style_force_vector * style_mask * 0.25
                
                # Sumamos esa guía a la imagen actual. Esto fuerza el estilo.
                x = x + prompt_guidance * torch.abs(dt)

            # B. Tu lógica clásica de reparación (Anti-Blur)
            x = x + (x - prev_x) * 0.035 * style_mask
            
            # C. Paso de Euler estándar para avanzar
            d = (x - denoised) / sigma
            x = x + d * dt

        # 5. Protección de Varianza (Evita que el CFG extra queme la imagen)
        if i >= BOCETADO_STEPS:
            x_std = x.std()
            if x_std > 1.15: # Límite de seguridad
                x = x * (1.15 / x_std)

        prev_x = x.clone()
        
        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})

    return torch.clamp(x, -5.0, 5.0)

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
            boost = 1.20 + (i / BOCETADO_STEPS) * 0.25
            d = (x - denoised) / sigma
            x = x + d * (sigma_next - sigma) * boost
        else:
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
            if x_std < 1.08: x = x * (1.08 / (x_std + 1e-6))
        elif i == total_steps - 1:
            x = denoised + (x - denoised) * 0.07
        prev_x = x.clone()
        if callback: callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})
        if shared.state.interrupted: break
    return torch.clamp(x, -5.0, 5.0)

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
        
        # 1. GENERACIÓN (Limpia y segura)
        denoised = model(x, sigma * s_in, **extra_args)
        
        # Pasos intermedios normales
        d = (x - denoised) / sigma
        x = x + d * dt

        # 2. INTERVENCIÓN FINAL (Solo Estructura)
        # Solo en el último paso aplicamos la cuadrícula.
        if i == total_steps - 1:
            # Tomamos la imagen final
            latents = denoised
            
            # Calculamos el tamaño reducido
            h, w = latents.shape[-2:]
            small_h, small_w = int(h / block_size), int(w / block_size)
            
            # DOWN: Usamos 'area' para promediar los colores correctamente
            # (Esto imita la lógica de 'average_box' de pyxelate)
            latents_pixelated = F.interpolate(latents, size=(small_h, small_w), mode='area')
            
            # UP: Usamos 'nearest' para crear los bordes duros
            latents_pixelated = F.interpolate(latents_pixelated, size=(h, w), mode='nearest')
            
            # Sustitución directa (Sin tocar colores, solo forma)
            x = latents_pixelated

        if callback:
            callback({"x": x, "i": i, "sigma": sigma, "sampling_step": i, "sampling_steps": total_steps})

    return x
    
# =====================================================
# AGGA STRUCTURAL-DETAIL (Hybrid V5)
# =====================================================
@torch.no_grad()
def sample_agga_structural_detail(model, x, sigmas, extra_args=None, callback=None, **kwargs):
    extra_args = extra_args or {}
    s_in = x.new_ones([x.shape[0]])
    total_steps = len(sigmas) - 1
    shared.state.sampling_steps = total_steps
    
    # Variables de estado para DPM++ 2M
    old_d = None
    
    # Punto de corte: 45% Estructura pura -> 55% Detalle AGGA
    split_idx = int(total_steps * 0.45)

    for i in range(total_steps):
        shared.state.sampling_step = i + 1
        sigma = sigmas[i]
        sigma_next = sigmas[i + 1]
        dt = sigma_next - sigma
        
        # 1. Predicción del modelo
        denoised = model(x, sigma * s_in, **extra_args)
        d = (x - denoised) / sigma

        # ==========================================
        # FASE 1: ESTRUCTURA SAGRADA (DPM++ 2M Puro)
        # ==========================================
        if i < split_idx:
            # Algoritmo DPM++ 2M estándar para máxima coherencia anatómica
            if old_d is None:
                # Paso Euler inicial (necesario para arrancar)
                x = x + d * dt
            else:
                # Paso DPM++ 2M (promedio de derivadas)
                # Esto suaviza la trayectoria y evita deformaciones
                x = x + 0.5 * (d + old_d) * dt
                
        # ==========================================
        # FASE 2: INYECCIÓN DE DETALLE AGGA (Native)
        # ==========================================
        else:
            # Calculamos la máscara de detalle (tu algoritmo de detección de bordes)
            dy = torch.abs(denoised[:, :, 1:, :] - denoised[:, :, :-1, :])
            dx = torch.abs(denoised[:, :, :, 1:] - denoised[:, :, :, :-1])
            dy = F.pad(dy, (0, 0, 0, 1), mode='replicate')
            dx = F.pad(dx, (0, 1, 0, 0), mode='replicate')
            # Sensibilidad ajustada a 2.2x para no quemar la imagen
            detail_mask = torch.clamp((dx + dy).mean(dim=1, keepdim=True) * 2.2, 0.0, 1.0)
            
            # Boost progresivo: Empieza en 1.0 y sube hasta 1.15 al final
            # Solo se aplica en las zonas de detalle (detail_mask)
            progress_phase2 = (i - split_idx) / (total_steps - split_idx)
            boost_amount = 1.0 + (progress_phase2 * 0.15)
            
            # Interpolamos el dt: Normal en zonas planas, Boosted en zonas de detalle
            final_dt = dt * (1.0 + (boost_amount - 1.0) * detail_mask)
            
            # Aplicamos el paso (usamos Euler aquí para respuesta directa al boost)
            x = x + d * final_dt
            
            # Micro-Sharpening (solo en los últimos pasos para textura "crisp")
            if progress_phase2 > 0.6:
                x = x + (x - x.clone()) * 0.04 * detail_mask

        # Guardamos la derivada para el siguiente paso DPM
        old_d = d

        if callback is not None:
            callback({
                "x": x,
                "i": i,
                "sigma": sigma,
                "sampling_step": i + 1,
                "sampling_steps": total_steps
            })

    # Clamp de seguridad final
    return torch.clamp(x, -5.0, 5.0)
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