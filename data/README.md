# Data Directory

This directory contains galaxy catalogs used for JANUS-Z cosmological analysis.

## Current Catalog (v17)

### `jwst_extended_catalog_v17.csv`

Main catalog with 200 high-redshift galaxies.

**Columns**:

| Column | Description | Units |
|--------|-------------|-------|
| `ID` | Galaxy identifier | — |
| `z` | Redshift | — |
| `z_err` | Redshift uncertainty | — |
| `z_type` | Redshift type (spec/phot) | — |
| `log_Mstar` | Log stellar mass | log(M☉) |
| `sigma_Mstar` | Stellar mass uncertainty | dex |
| `log_SFR` | Log star formation rate | log(M☉/yr) |
| `metallicity_12OH` | Gas-phase metallicity | 12+log(O/H) |
| `has_AGN` | AGN flag | 0/1 |
| `is_dusty` | NIRCam-dark flag (A_V > 3) | 0/1 |
| `protocluster` | Proto-cluster membership | name or "field" |
| `sigma_v` | Velocity dispersion | km/s (-1 if N/A) |
| `Survey` | Source survey | — |
| `Reference` | Literature reference | — |

**Sample statistics**:
- Total galaxies: 200
- Redshift range: 6.63 < z < 14.32
- Spectroscopic: 55 (27.5%)
- Photometric: 145 (72.5%)
- With metallicity: 55
- Dusty/NIRCam-dark: 4

**Sources**:
- JADES DR4 (Bunker+2025)
- EXCELS (Carnall+2025, Cullen+2025)
- GLASS (Morishita+2025)
- A3COSMOS (Nov 2025)
- CEERS (Finkelstein+2024)
- UNCOVER (Bezanson+2024)
- COSMOS-Web

## Historical Catalogs

### `jwst_extended_catalog_v16.csv`
150 galaxies (archived from v16)

### `jwst_108_galaxies.csv`
108 galaxies (archived from v15)

---

## Data Usage

```python
import pandas as pd

# Load catalog
df = pd.read_csv('data/jwst_extended_catalog_v17.csv')

# Filter spectroscopic
spec_z = df[df['z_type'] == 'spec']

# Filter dusty galaxies
dusty = df[df['is_dusty'] == 1]
```

---

## License

Data compiled from public JWST/ALMA surveys. Individual measurements subject to original publication terms.
