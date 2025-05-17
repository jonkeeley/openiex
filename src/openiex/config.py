from dataclasses import dataclass
import numpy as np

@dataclass
class SystemConfig:
    bed_height: float              # meters
    column_radius: float           # meters
    Lambda: float                  # ionic capacity [mol/L resin]
    epsilon_i: float               # interstitial void fraction
    epsilon_p: float               # pore void fraction
    Nz: int = 25                   # number of axial segments

    @property
    def A(self):
        return np.pi * self.column_radius**2

    @property
    def dz(self):
        return self.bed_height / (self.Nz - 1)

    @property
    def vol_column(self):
        """Total packed bed volume (resin + interstitial)"""
        return self.A * self.bed_height
    
    @property
    def vol_interstitial(self):
        """Fluid-phase volume only (mobile phase in interstitial space)"""
        return self.A * self.bed_height * self.epsilon_i