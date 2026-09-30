"""The installed EnergyPlus reads a BSDF the same way round for diffuse light as for the beam.

Goes into the pyenergyplus fork as test/test_cfs_transposed.py and runs in its
build workflow on every platform, after the wheel is installed. Standard library
only; the EnergyPlus submodule next to this folder supplies the example and the
weather.

EnergyPlus' own example CmplxGlz_Daylighting_SouthVB45deg.idf, January, with the
beam removed from the weather (DNI 0, GHI = DHI) and ground reflectance 0, so the
only light on the window is sky diffuse. EnergyPlus' sky transmittance is then
transmitted solar / (incident sky diffuse x glazed area). The test recomputes it
from the file's front transmittance matrix CFS_Glz_1_TfSol, with the columns as
the incident direction (the reading EnergyPlus' beam path uses) and with the rows
(the unpatched diffuse path), and requires EnergyPlus to match the columns.

Unpatched EnergyPlus 25.2.0 and 26.1.0 give 0.0940 (the rows); the columns give
0.1122. See the BuildingModelsGenerator wiki page "EnergyPlus reads the blind
transposed for diffuse light".
"""
import math
import os
import re
import tempfile
import unittest
from pathlib import Path

from pyenergyplus.api import EnergyPlusAPI

EPLUS = Path(__file__).resolve().parent.parent / 'EnergyPlus'
# The submodule in the fork's workflow; CFSFIX_IDF / CFSFIX_EPW point elsewhere for a local check
IDF = Path(os.environ.get('CFSFIX_IDF', EPLUS / 'testfiles' / 'CmplxGlz_Daylighting_SouthVB45deg.idf'))
EPW = Path(os.environ.get('CFSFIX_EPW', EPLUS / 'weather' / 'USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw'))
WINDOW = 'Win_421'
TRANSMITTED = 'Surface Window Transmitted Solar Radiation Rate'
SKY = 'Surface Outside Face Incident Sky Diffuse Solar Radiation Rate per Area'

# Klems full basis: ring edges (deg) and patches per ring
EDGES = (0, 5, 15, 25, 35, 45, 55, 65, 75, 90)
SIZES = (1, 8, 16, 20, 24, 24, 24, 16, 12)


def fields(body, keep_blank=False):
    values = [x.strip() for x in re.sub(r'!.*', '', body).split(',')]
    return values if keep_blank else [x for x in values if x]


def sky_transmittances(idf):
    """(columns as incident, rows as incident), each averaged over the sky patches
    as EnergyPlus does: up or level counts as sky, weighted by projected solid angle."""
    f = fields(re.search(r'Matrix:TwoDimension,\s*CFS_Glz_1_TfSol\s*,(.*?);', idf, re.S).group(1))
    n = int(f[0])
    values = [float(x) for x in f[2:2 + n * n]]
    T = [values[r * n:(r + 1) * n] for r in range(n)]            # T[row][column], as written
    lam, up = [], []
    for k, count in enumerate(SIZES):
        lo, hi = math.radians(EDGES[k]), math.radians(EDGES[k + 1])
        theta = 0.0 if k == 0 else math.radians(0.5 * (EDGES[k] + EDGES[k + 1]))
        for j in range(count):
            lam.append(math.pi * (math.sin(hi) ** 2 - math.sin(lo) ** 2) / count)
            up.append(-math.sin(theta) * math.sin(math.radians(j * 360.0 / count)))  # phi = 270 deg is up
    sky = [i for i in range(n) if up[i] >= -1e-9]
    col = [sum(lam[i] * T[i][j] for i in range(n)) for j in range(n)]   # down column j
    row = [sum(T[j][m] * lam[m] for m in range(n)) for j in range(n)]   # along row j
    mean = lambda t: sum(t[j] * lam[j] for j in sky) / sum(lam[j] for j in sky)
    return mean(col), mean(row)


def glazed_area(idf):
    f = fields(re.search(r'FenestrationSurface:Detailed,\s*' + WINDOW + r'\s*,(.*?);', idf, re.S).group(1),
               keep_blank=True)
    count = int(f[7])                        # after type, construction, base surface, 3 optional, multiplier
    v = [tuple(float(x) for x in f[8 + 3 * i:11 + 3 * i]) for i in range(count)]
    cross = [0.0, 0.0, 0.0]
    for a, b in zip(v, v[1:] + v[:1]):
        cross[0] += a[1] * b[2] - a[2] * b[1]
        cross[1] += a[2] * b[0] - a[0] * b[2]
        cross[2] += a[0] * b[1] - a[1] * b[0]
    return 0.5 * math.sqrt(sum(c * c for c in cross))


def diffuse_only(source, target):
    out = []
    for i, line in enumerate(source.read_text(encoding='latin-1').splitlines()):
        f = line.split(',')
        if i >= 8 and re.match(r'^\d{4}$', f[0]):
            f[14], f[17] = '0', '0'          # direct normal radiation, illuminance
            f[13], f[16] = f[15], f[18]      # global = diffuse
        out.append(','.join(f))
    target.write_text('\n'.join(out) + '\n', encoding='latin-1')


def sky_only_idf(idf):
    text = re.sub(r'Site:GroundReflectance,.*?;', '', idf, flags=re.S)
    text = re.sub(r'\bRunPeriod,.*?;', '', text, flags=re.S)
    return text + ('\nSite:GroundReflectance,0,0,0,0,0,0,0,0,0,0,0,0;\n'
                   'RunPeriod,January,1,1,,1,31,,,No,No,No,No,No;\n')


class SkyTransmittanceReadsColumns(unittest.TestCase):

    def test_sky_transmittance_follows_the_columns(self):
        idf = IDF.read_text(encoding='latin-1')
        columns, rows = sky_transmittances(idf)
        area = glazed_area(idf)
        self.assertGreater(abs(columns - rows) / columns, 0.05, 'the example no longer tells the readings apart')

        api = EnergyPlusAPI()
        state = api.state_manager.new_state()
        api.exchange.request_variable(state, TRANSMITTED, WINDOW)
        api.exchange.request_variable(state, SKY, WINDOW)
        sums = {'transmitted': 0.0, 'incident': 0.0}
        handles = {}

        def each_timestep(s):
            if not api.exchange.api_data_fully_ready(s) or api.exchange.warmup_flag(s):
                return
            if not handles:
                handles['t'] = api.exchange.get_variable_handle(s, TRANSMITTED, WINDOW)
                handles['e'] = api.exchange.get_variable_handle(s, SKY, WINDOW)
            e = api.exchange.get_variable_value(s, handles['e'])
            if e > 0:
                sums['transmitted'] += api.exchange.get_variable_value(s, handles['t'])
                sums['incident'] += e * area

        api.runtime.callback_end_zone_timestep_after_zone_reporting(state, each_timestep)
        with tempfile.TemporaryDirectory() as work:
            work = Path(work)
            (work / 'in.idf').write_text(sky_only_idf(idf), encoding='latin-1')
            diffuse_only(EPW, work / 'diffuse_only.epw')
            code = api.runtime.run_energyplus(state, ['-w', str(work / 'diffuse_only.epw'), '-d', str(work),
                                                      str(work / 'in.idf')])
            self.assertEqual(code, 0, (work / 'eplusout.err').read_text(errors='replace')[-3000:])
        self.assertGreater(sums['incident'], 0.0)
        measured = sums['transmitted'] / sums['incident']
        print(f'\nsky transmittance: EnergyPlus {measured:.4f}, columns {columns:.4f}, rows {rows:.4f}')
        self.assertAlmostEqual(measured, columns, delta=0.005 * columns,
                               msg=f'EnergyPlus {measured:.4f} is not the column reading {columns:.4f} '
                                   f'(rows {rows:.4f}): the diffuse path is still transposed')


if __name__ == '__main__':
    unittest.main()
