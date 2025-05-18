import json
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
        self.ions               = ions
        self.proteins           = proteins
        self.inert              = inert
        self.species            = {**ions, **proteins, **inert}
        self.config             = config

        # SMA mass‐action storage
        self.K_eq               = {}
        self.k_ads              = {}
        self.k_des              = {}

        # LDF storage (only protein→ion allowed)
        self.k_ldf              = {}

        # Per‐pair kinetic model: "SMA" or "LDF"
        self.pair_kinetic_model = {}
        # Global default ("SMA" or "LDF")
        self.kinetic_model      = "SMA"

    def set_kinetic_model(self, model: str):
        model = model.upper()
        if model not in ("SMA","LDF"):
            raise ValueError("Must be 'SMA' or 'LDF'")
        self.kinetic_model = model

    def set_equilibrium(
        self,
        a: str,
        b: str,
        K_eq_val: float,
        k_rate_val: float,
        kinetic_model: str = None
    ):
        """
        Define K_eq and one k_rate for the pair (a,b).
         - If model=="SMA": symmetric k_ads/k_des for both (a,b) and (b,a).
         - If model=="LDF": only allowed if a∈proteins and b∈ions; single-direction k_ldf[(a,b)].
        """
        model = (kinetic_model or self.kinetic_model).upper()
        if model not in ("SMA","LDF"):
            raise ValueError("kinetic_model must be 'SMA' or 'LDF'")

        # Thermodynamics always stored bidirectionally
        self.K_eq[(a,b)] = K_eq_val
        self.K_eq[(b,a)] = 1.0 / K_eq_val

        # Record which kinetics to use here
        self.pair_kinetic_model[(a,b)] = model

        if model == "SMA":
            # mass‐action forward/backward
            self.k_ads[(a,b)] = k_rate_val
            self.k_des[(a,b)] = k_rate_val / K_eq_val
            self.pair_kinetic_model[(b,a)] = "SMA"
            self.k_ads[(b,a)] = k_rate_val / K_eq_val
            self.k_des[(b,a)] = k_rate_val
            # remove any leftover LDF
            self.k_ldf.pop((a,b), None)
            self.k_ldf.pop((b,a), None)

        else:  # model == "LDF"
            # only allow protein→ion
            if a not in self.proteins or b not in self.ions:
                raise ValueError("LDF only allowed for protein→ion pairs (j,i)")
            self.k_ldf[(a,b)] = k_rate_val
            # do NOT set (b,a), and do NOT set any SMA rates here
            self.k_ads.pop((a,b), None)
            self.k_des.pop((a,b), None)
            # ensure reverse kinetic_model stays SMA if previously set
            self.pair_kinetic_model[(b,a)] = "LDF"

    def check_equilibria(self):
        missing = []
        ion_list = list(self.ions.keys())
        # ion-ion
        for i in range(len(ion_list)):
            for j in range(i + 1, len(ion_list)):
                a, b = ion_list[i], ion_list[j]
                if (a, b) not in self.K_eq:
                    missing.append((a, b))
        # protein-ion
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

    def to_dict(self) -> dict:
        """Serialize the entire system, including extinction coefficients."""
        return {
            "config": vars(self.config),
            "ions": {
                name: {
                    "D": ion.D,
                    "Kd": ion.Kd,
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
                    "Kd": p.Kd,
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
                    "Kd": inv.Kd,
                    "unit": inv.unit,
                    "mol_cond": inv.mol_cond,
                    "ext_coeff_260": inv.ext_coeff_260,
                    "ext_coeff_280": inv.ext_coeff_280
                }
                for name, inv in self.inert.items()
            },
            "equilibria": {
                f"{a}|{b}": {"K_eq": self.K_eq[(a, b)], "rate": self.k_rate[(a, b)]}
                for (a, b) in self.K_eq
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
                Kd=vals["Kd"],
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
                Kd=vals["Kd"],
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
                Kd=vals["Kd"],
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
            sys.set_equilibrium(a, b, params["K_eq"], params["k_rate"])
        return sys
