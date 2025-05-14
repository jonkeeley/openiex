from .species import Ion, Protein
from .config import SystemConfig

class ExchangeSystem:
    def __init__(self, ions, proteins, config: SystemConfig):
        self.ions = ions
        self.proteins = proteins
        self.species = {**ions, **proteins}
        self.config = config
        self.K_eq = {}
        self.k_ads = {}
        self.k_des = {}

    def set_equilibrium(self, a, b, K_eq_val, k_ads_val):
        self.K_eq[(a, b)] = K_eq_val
        self.k_ads[(a, b)] = k_ads_val
        self.k_des[(a, b)] = k_ads_val / K_eq_val
        self.K_eq[(b, a)] = 1.0 / K_eq_val
        self.k_ads[(b, a)] = k_ads_val / K_eq_val
        self.k_des[(b, a)] = k_ads_val

    def check_equilibria(self):
        missing = []

        # Check ion–ion (each unordered pair only once)
        ion_list = list(self.ions.keys())
        for i in range(len(ion_list)):
            for j in range(i + 1, len(ion_list)):
                a, b = ion_list[i], ion_list[j]
                if (a, b) not in self.K_eq:
                    missing.append((a, b))

        # Check protein–ion in the order (protein, ion)
        for p in self.proteins:
            for i in self.ions:
                if (p, i) not in self.K_eq:
                    missing.append((p, i))

        if missing:
            print("Missing equilibrium definitions for:")
            for pair in missing:
                print(f"  {pair}")
        else:
            print("All required equilibria are defined.")

    def to_dict(self):
        """Convert to a JSON‐serializable structure."""
        return {
            "config": vars(self.config),
            "ions": {
                name: {"D": ion.D, "Kd": ion.Kd, "unit": ion.unit}
                for name, ion in self.ions.items()
            },
            "proteins": {
                name: {
                    "D": p.D, "Kd": p.Kd,
                    "unit": p.unit,
                    "sigma": p.sigma, "nu": p.nu
                }
                for name, p in self.proteins.items()
            },
            "equilibria": {
                f"{a}|{b}": {
                    "K_eq": self.K_eq[(a,b)],
                    "k_ads": self.k_ads[(a,b)]
                }
                for (a,b) in self.K_eq
            }
        }

    @classmethod
    def from_dict(cls, data: dict):
        """Reconstruct an ExchangeSystem from its dict form."""
        cfg = SystemConfig(**data["config"])

        ions = {
            name: Ion(name, **vals)
            for name, vals in data["ions"].items()
        }
        proteins = {
            name: Protein(name, **vals)
            for name, vals in data["proteins"].items()
        }
        sys = cls(ions, proteins, cfg)

        # re‑apply equilibria
        for key, params in data["equilibria"].items():
            a, b = key.split("|")
            K_eq = params["K_eq"]
            k_ads = params["k_ads"]
            sys.set_equilibrium(a, b, K_eq, k_ads)
        return sys