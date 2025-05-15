class Species:
    ACCEPTED_UNITS = {"M", "particles/mL"}
    def __init__(
            self, 
            name: str, 
            D: float, 
            Kd: float = 1, 
            unit: str = "M", 
            mol_cond: float = 0, 
            ext_coeff_260: float = 0, 
            ext_coeff_280: float = 0
        ):
        if unit not in Species.ACCEPTED_UNITS:
            raise ValueError(
                f"Invalid unit '{unit}' for species '{name}'. "
                f"Allowed units are: {list(Species.ACCEPTED_UNITS)}"
            )
        self.name = name
        self.D = D # Effective axial dispersion coefficients (m^2/s)
        self.Kd = Kd # Pore accessibility fraction ()
        self.unit = unit # Only relevant for plotting and feed conditions
        self.mol_cond = mol_cond # S·cm²/mol (for plotting conductivity)
        self.ext_coeff_260 = ext_coeff_260 # M⁻¹cm⁻¹
        self.ext_coeff_280 = ext_coeff_280 # M⁻¹cm⁻¹

class Ion(Species):
    pass

class Protein(Species):
    def __init__(
        self,
        name: str,
        D: float,
        sigma: float,
        nu: float,
        Kd: float = 1, 
        unit: str = "M", 
        mol_cond: float = 0, 
        ext_coeff_260: float = 0, 
        ext_coeff_280: float = 0
    ):
        super().__init__(name, D, Kd, unit, mol_cond, ext_coeff_260, ext_coeff_280)
        self.sigma = sigma
        self.nu    = nu

class Inert(Species):
    pass 