class Species:
    """
    Base class for all species in ion-exchange simulations, defining shared physical and optical properties.

    Attributes:
        name (str): Identifier for the species.
        D (float): Axial dispersion coefficient in m²/s.
        K_d (float): Pore accessibility fraction (dimensionless).
        unit (str): Concentration unit, one of 'M' or 'particles/mL'.
        mol_cond (float): Molar conductivity in S·cm²/mol (for conductivity plots only).
        ext_coeff_260 (float): Extinction coefficient at 260 nm [M⁻¹·cm⁻¹].
        ext_coeff_280 (float): Extinction coefficient at 280 nm [M⁻¹·cm⁻¹].
    """
    ACCEPTED_UNITS = {"M", "particles/mL"}

    def __init__(
        self,
        name: str,
        D: float,
        K_d: float = 1.0,
        unit: str = "M",
        mol_cond: float = 0.0,
        ext_coeff_260: float = 0.0,
        ext_coeff_280: float = 0.0,
    ):
        if unit not in self.ACCEPTED_UNITS:
            raise ValueError(
                f"Invalid unit '{unit}' for species '{name}'. "
                f"Allowed units: {sorted(self.ACCEPTED_UNITS)}"
            )
        self.name = name
        self.D = D
        self.K_d = K_d
        self.unit = unit
        self.mol_cond = mol_cond
        self.ext_coeff_260 = ext_coeff_260
        self.ext_coeff_280 = ext_coeff_280

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} {self.name}: D={self.D} m²/s, "
            f"K_d={self.K_d}, unit={self.unit}>"
        )


class Ion(Species):
    """
    Competitive salt ions: in AEX, these are anions; in CEX, these are cations.
    """
    pass


class Protein(Species):
    """
    Protein or large binding species (e.g., free DNA) in ion-exchange simulations.
    Models steric binding characteristics via SMA parameters.

    Additional Attributes:
        sigma (float): Number of binding sites sterically hindered by each protein (dimensionless).
        nu (float): Number of binding sites on average each protein exchanges with (dimensionless).
    """
    def __init__(
        self,
        name: str,
        D: float,
        sigma: float,
        nu: float,
        K_d: float = 1.0,
        unit: str = "particles/mL",
        mol_cond: float = 0.0,
        ext_coeff_260: float = 0.0,
        ext_coeff_280: float = 0.0,
    ):
        super().__init__(name, D, K_d, unit, mol_cond, ext_coeff_260, ext_coeff_280)
        self.sigma = sigma
        self.nu = nu

    def __repr__(self):
        return (
            f"<Protein {self.name}: D={self.D} m²/s,  sigma={self.sigma}, "
            f"nu={self.nu}, K_d={self.K_d}, unit={self.unit}>"
        )


class Inert(Species):
    """
    Inert species (e.g., buffer components, cations in AEX or anions in CEX).
    Optional for accurate contributions to overall conductivity or UV absorbance.
    Does not participate in binding kinetics but will be included in solver state.
    Note: adding inert species increases solver workload.
    """
    pass
