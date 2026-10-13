"""Parameter registry: product archetypes x regions, with sources.

Units: kg CO2e per kg product unless noted. WARM factors converted from
MTCO2e/short ton by dividing by 0.9072.
"""
from dataclasses import dataclass

SHORT_TON = 0.9072
SCC_USD_PER_KG = 0.190          # US EPA (2023) social cost of carbon, $190/tCO2e
COLD_KWH_PER_M3_DAY = 55.8 / 365  # Evans et al. (2014) chilled-store mean SEC
STORE_DENSITY_KG_M3 = 250.0
LAST_MILE_KM = 20.0
VAN_EF_KG_PER_KGKM = 0.15 / 1000   # light van, kg CO2e per kg-km at full load
VAN_UTILIZATION = 0.57 * (1 - 0.275)

# WARM v16 alternative-pathway factors (all food types), kg CO2e / kg
EOL_COMBUSTION = -0.13 / SHORT_TON
EOL_AD = -0.04 / SHORT_TON
EOL_COMPOST = -0.15 / SHORT_TON


@dataclass(frozen=True)
class Archetype:
    name: str
    ef_prod: float        # Poore & Nemecek (2018) global mean
    eol_landfill: float   # WARM v16 landfill, by food type
    p_base: float         # household waste share of purchases (WRAP)
    horizon: int          # clearance window, days (FoodKeeper; Haijema et al. 2026)
    price: float          # USD per kg, full price
    unit_kg: float = 1.0


@dataclass(frozen=True)
class Region:
    name: str
    grid: float           # kg CO2e / kWh
    landfill: float       # share of retail food waste by pathway
    combustion: float
    ad: float
    compost: float

    def eol(self, a: Archetype) -> float:
        return (self.landfill * a.eol_landfill + self.combustion * EOL_COMBUSTION
                + self.ad * EOL_AD + self.compost * EOL_COMPOST)


ARCHETYPES = {
    "dairy": Archetype("dairy", 3.15, 0.48 / SHORT_TON, 0.19, 3, 2.5),
    "bakery": Archetype("bakery", 1.57, 0.99 / SHORT_TON, 0.31, 2, 4.0),
    "produce": Archetype("produce", 0.79, 0.23 / SHORT_TON, 0.35, 3, 4.0),
    "poultry": Archetype("poultry", 9.87, 0.49 / SHORT_TON, 0.20, 2, 8.0),
}

# Grid: CLP/HKE 2024 weighted; MEE 2024 No.33 (Guangdong 2022); DESNZ 2024; eGRID2023; EEA 2023.
# Pathway mixes: HK EPD (2023) ~6% food waste recycled; others are representative scenario mixes.
REGIONS = {
    "Hong Kong": Region("Hong Kong", 0.42, 0.94, 0.00, 0.06, 0.00),
    "Guangdong": Region("Guangdong", 0.44, 0.15, 0.80, 0.05, 0.00),
    "UK": Region("UK", 0.207, 0.20, 0.35, 0.45, 0.00),
    "US": Region("US", 0.350, 0.60, 0.15, 0.10, 0.15),
    "EU": Region("EU", 0.215, 0.25, 0.30, 0.25, 0.20),
}


def ops_per_kg(region: Region, days: int) -> float:
    cold = COLD_KWH_PER_M3_DAY / STORE_DENSITY_KG_M3 * days * region.grid
    van = LAST_MILE_KM * VAN_EF_KG_PER_KGKM / VAN_UTILIZATION
    return cold + van
