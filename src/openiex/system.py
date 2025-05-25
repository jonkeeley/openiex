import numpy as np
from .species import Ion, Protein, Inert
from .config import SystemConfig

class ExchangeSystem:
    def __init__(
        self,
        ions: dict[str, Ion],
        proteins: dict[str, Protein],
        inert: dict[str, Inert],
        config: SystemConfig
    ):
        self.ions = ions
        self.proteins = proteins
        self.inert = inert
        self.species = {**ions, **proteins, **inert}
        self.config = config
        self.ln_K_eq = {}
        self.ln_k_ads = {}
        self.ln_k_des = {}

    def set_equilibrium(self, a: str, b: str, ln_K_eq_val: float, ln_k_ads_val: float):
        self.ln_K_eq[(a, b)]  = ln_K_eq_val
        self.ln_k_ads[(a, b)] = ln_k_ads_val
        self.ln_k_des[(a, b)] = ln_k_ads_val - ln_K_eq_val
        self.ln_K_eq[(b, a)]  = 1.0 - ln_K_eq_val
        self.ln_k_ads[(b, a)] = ln_k_ads_val - ln_K_eq_val
        self.ln_k_des[(b, a)] = ln_k_ads_val

    def check_equilibria(self):
        missing = []
        ion_list = list(self.ions.keys())
        # ion-ion
        for i in range(len(ion_list)):
            for j in range(i + 1, len(ion_list)):
                a, b = ion_list[i], ion_list[j]
                if (a, b) not in self.ln_K_eq:
                    missing.append((a, b))
        # protein-ion
        for p in self.proteins:
            for i in self.ions:
                if (p, i) not in self.ln_K_eq:
                    missing.append((p, i))
        if missing:
            print("Missing equilibrium definitions for:")
            for pair in missing:
                print(f"  {pair}")
        else:
            print("All required equilibria are defined.")

    def to_dict(self) -> dict:
        """Serialize the entire system, including extinction coefficients."""
        return {
            "config": vars(self.config),
            "ions": {
                name: {
                    "D": ion.D,
                    "K_d": ion.K_d,
                    "unit": ion.unit,
                    "mol_cond": ion.mol_cond,
                    "ext_coeff_260": ion.ext_coeff_260,
                    "ext_coeff_280": ion.ext_coeff_280
                }
                for name, ion in self.ions.items()
            },
            "proteins": {
                name: {
                    "D": p.D,
                    "K_d": p.K_d,
                    "unit": p.unit,
                    "mol_cond": p.mol_cond,
                    "sigma": p.sigma,
                    "nu": p.nu,
                    "ext_coeff_260": p.ext_coeff_260,
                    "ext_coeff_280": p.ext_coeff_280
                }
                for name, p in self.proteins.items()
            },
            "inert": {
                name: {
                    "D": inv.D,
                    "K_d": inv.K_d,
                    "unit": inv.unit,
                    "mol_cond": inv.mol_cond,
                    "ext_coeff_260": inv.ext_coeff_260,
                    "ext_coeff_280": inv.ext_coeff_280
                }
                for name, inv in self.inert.items()
            },
            "equilibria": {
                f"{a}|{b}": {"ln_K_eq": self.ln_K_eq[(a, b)], "ln_k_ads": self.ln_k_ads[(a, b)]}
                for (a, b) in self.ln_K_eq
            }
        }

    def to_math_dict(self) -> dict:
        """
        Like to_dict(), but strips out non-binding attributes:
        unit, mol_cond, extinction coefficients.
        """
        d = self.to_dict()
        for group in ("ions", "proteins", "inert"):
            for vals in d.get(group, {}).values():
                for drop in ("unit", "mol_cond", "ext_coeff_260", "ext_coeff_280"):
                    vals.pop(drop, None)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ExchangeSystem":
        """Reconstruct the system from its serialized form."""
        cfg = SystemConfig(**data["config"])
        ions = {
            name: Ion(
                name,
                D=vals["D"],
                K_d=vals["K_d"],
                unit=vals["unit"],
                mol_cond=vals["mol_cond"],
                ext_coeff_260=vals.get("ext_coeff_260", 0.0),
                ext_coeff_280=vals.get("ext_coeff_280", 0.0)
            )
            for name, vals in data["ions"].items()
        }
        proteins = {
            name: Protein(
                name,
                D=vals["D"],
                K_d=vals["K_d"],
                sigma=vals["sigma"],
                nu=vals["nu"],
                unit=vals["unit"],
                mol_cond=vals["mol_cond"],
                ext_coeff_260=vals.get("ext_coeff_260", 0.0),
                ext_coeff_280=vals.get("ext_coeff_280", 0.0)
            )
            for name, vals in data["proteins"].items()
        }
        inert = {
            name: Inert(
                name,
                D=vals["D"],
                K_d=vals["K_d"],
                unit=vals["unit"],
                mol_cond=vals["mol_cond"],
                ext_coeff_260=vals.get("ext_coeff_260", 0.0),
                ext_coeff_280=vals.get("ext_coeff_280", 0.0)
            )
            for name, vals in data["inert"].items()
        }
        sys = cls(ions, proteins, inert, cfg)
        for key, params in data.get("equilibria", {}).items():
            a, b = key.split("|")
            sys.set_equilibrium(a, b, params["ln_K_eq"], params["ln_k_ads"])
        return sys
