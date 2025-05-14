class Species:
    ACCEPTED_UNITS = {"M", "particles/mL"}
    def __init__(self, name, D, Kd, unit="M"):
        if unit not in Species.ACCEPTED_UNITS:
            raise ValueError(
                f"Invalid unit '{unit}' for species '{name}'. "
                f"Allowed units are: {list(Species.ACCEPTED_UNITS)}"
            )
        self.name = name
        self.D = D # Effective axial dispersion coefficients (m^2/s)
        self.Kd = Kd # Pore accessibility fraction ()
        self.unit = unit # Only relevant for plotting and feed conditions

class Ion(Species):
    pass

class Protein(Species):
    def __init__(self, name, D, Kd, sigma, nu):
        super().__init__(name, D, Kd)
        self.sigma = sigma # Steric factors for proteins ()
        self.nu = nu # Number of displaced counterions per bound protein ()