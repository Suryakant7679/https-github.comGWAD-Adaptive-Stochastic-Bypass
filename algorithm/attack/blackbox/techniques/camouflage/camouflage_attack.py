import torch
import numpy as np
from algorithm.attack.base import Query

#print("🔥 CAMOUFLAGE V5 (HSJA-STEALTH ENGINE) LOADED")

class CamouflageAttack(object):
    def __init__(self, device, model=None, q_budgets=None, stop=True, data_type='cifar10', cfg=None):
        self.device = device
        self.model = model
        self.q_budgets = q_budgets if q_budgets is not None else [10000]

        # --- HSJA ADAPTIVE PARAMETERS ---
        self.alpha = 0.05       # Small steps to bypass Screener
        self.gamma = 1.0        # Trajectory jitter to bypass HoDS
        self.ref_pool = []

    def set_adaptive(self, adapt_type, move_rate, batch_size, x2_pool):
        if x2_pool is not None:
            self.ref_pool = x2_pool[:5000]
        print(f"[*] Camouflage V5 Synced. Pool Size: {len(self.ref_pool)}")

    def query(self, x):
        if x.dim() == 3: x = x.unsqueeze(0)
        with torch.no_grad():
            q = Query(t='attack', x=x)
            return self.model(q)

    def get_ref(self):
        if len(self.ref_pool) == 0: return None
        idx = np.random.randint(0, len(self.ref_pool))
        return self.ref_pool[idx].to(self.device)

    def core(self, x_start, y):
        x_adv = x_start.clone().detach().squeeze(0)
        target_class = y.item()
        T = int(self.q_budgets[0])
        
        # Initial slight push to start the search
        x_adv = x_adv + 0.005 * torch.sign(torch.randn_like(x_adv)).to(self.device)
        
        for t in range(T):
            # 1. HSJA LOGIC: Random Directional Probe
            # Instead of a burst of queries, we use a single directional probe per step
            u = torch.randn_like(x_adv).to(self.device)
            u = u / (torch.norm(u) + 1e-8)
            
            x_probe = x_adv + self.alpha * u
            x_probe = torch.clamp(x_probe, 0, 1)
            
            out = self.query(x_probe)
            pred = torch.argmax(out, dim=1).item()
            
            # 2. ADAPTIVE STEP
            if pred != target_class:
                # Success: Move to the new adversarial point
                x_adv = x_probe
            else:
                # Failure: Shift trajectory randomly (Diversification)
                x_adv = x_adv + (self.alpha * self.gamma) * torch.randn_like(x_adv).to(self.device)

            # 3. CAMOUFLAGE FLUSH (The "Ghost" Logic)
            # Every 3rd query, we send a completely different benign image 
            # to keep the HoDS detector's FIFO buffer "dirty"
            if t % 3 == 0:
                x_ref = self.get_ref()
                if x_ref is not None:
                    _ = self.query(x_ref)

            x_adv = torch.clamp(x_adv, 0, 1)

            # 4. SUCCESS EXIT
            # We check if we've actually flipped the label
            if pred != target_class and t > 5:
                print(f"[!] HSJA-CAMOUFLAGE SUCCESS at query {t}")
                return x_adv.unsqueeze(0), [t]

        return x_adv.unsqueeze(0), [T]

    def untarget(self, x, y):
        adv, iters = self.core(x, y)
        return adv, iters, 0, 0