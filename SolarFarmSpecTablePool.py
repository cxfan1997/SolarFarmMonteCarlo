PANEL_FRAC       = 0.5
PANEL_HGT        = 1.0
PANEL_TILT       = 20.0
ALB_FRONT        = 0.1
ALB_BACK         = 0.6
ALB_SURFACE      = (0.00, 1.01, 0.03)
EMIS_FRONT       = 0.70
EMIS_BACK        = 0.99
EMIS_SURFACE     = (0.80, 1.01, 0.03)
SOLAR_ZENITH     = (0.0, 89.1, 3.0)
SOLAR_AZIMUTH    = (0.0, 360.1, 6.0)

NPHOTONS         = 100000
MAX_PROCESSES    = 50

FILE_OUTPUT      = 'solarfarm_spec.nc'

from multiprocessing import Pool
from datetime import datetime

import numpy as np
import xarray as xa

from SolarFarmMonteCarlo import (
  doDownwardRadiation2D, doPanelEmission2D, doGroundEmission2D
)
from Utils import waitResults

def getCoord(coord_spec, coord_name):
  """
  Create a DataArray object based on the given coordinate specification.
  If the specification is a float, the DataArray will contain a single value.
  If the specification is a tuple of length 2 or 3, the DataArray will contain a range of values.

  Parameters:
  - coord_spec: The specification for the coordinate. It can be a float or a tuple of length 2 or 3.
  - coord_name: The name of the coordinate.

  Returns:
  - A DataArray object representing the coordinate.

  Raises:
  - ValueError: If coord_spec is not a float or a tuple of length 2 or 3.
  """
  if (type(coord_spec) is float):
    return xa.DataArray([coord_spec], dims=coord_name)
  elif (type(coord_spec) is tuple and len(coord_spec) > 1 and len(coord_spec) < 4):
    return xa.DataArray(np.arange(*coord_spec), dims=coord_name)
  else:
    raise ValueError('coord_spec must be a float or a tuple of length 2 or 3.')
  
VAR_ALBG = getCoord(ALB_SURFACE, 'surface_albedo')
VAR_SOLZEN = getCoord(SOLAR_ZENITH, 'solar_zenith_angle')
VAR_SOLAZ = getCoord(SOLAR_AZIMUTH, 'solar_azimuth_angle')
VAR_EMIS = getCoord(EMIS_SURFACE, 'surface_emissivity')
PARAMS_SW = (
  1.0, PANEL_HGT, 1.0 / PANEL_FRAC, np.radians(PANEL_TILT),
  ALB_FRONT, ALB_BACK, np.nan
)
PARAMS_LW = (
  1.0, PANEL_HGT, 1.0 / PANEL_FRAC, np.radians(PANEL_TILT),
  1-EMIS_FRONT, 1-EMIS_BACK, np.nan
)

def generateSWDirLookupTable():
  """
  Generate lookup table for direct shortwave fluxes.

  Returns:
  A dictionary containing the following keys:
    - 'frac_swdir_refl': A DataArray containing the fraction of direct shortwave flux reflected by the surface.
    - 'frac_swdir_grnd': A DataArray containing the fraction of direct shortwave flux absorbed by the surface.
    - 'frac_swdir_upnl': A DataArray containing the fraction of direct shortwave flux transmitted through the surface without being absorbed or scattered.
    - 'frac_swdir_dpnl': A DataArray containing the fraction of direct shortwave flux transmitted through the surface and then scattered back into the atmosphere.
  """
  tempvar = xa.DataArray(
    np.full((len(VAR_SOLZEN), len(VAR_SOLAZ), len(VAR_ALBG)), np.nan),
    dims=['solar_zenith_angle', 'solar_azimuth_angle', 'surface_albedo'],
    coords={'surface_albedo': VAR_ALBG,
            'solar_zenith_angle': VAR_SOLZEN,
            'solar_azimuth_angle': VAR_SOLAZ}
  )
  frac_swdir_refl = tempvar.copy()
  frac_swdir_upnl = tempvar.copy()
  frac_swdir_dpnl = tempvar.copy()
  frac_swdir_grnd = tempvar.copy()
  
  results = []
  with Pool(processes=MAX_PROCESSES) as pool:
    for solzen in VAR_SOLZEN:
      for solaz in VAR_SOLAZ:
        for albg in VAR_ALBG:
          param = PARAMS_SW[:-1] + (albg,)
          results.append(pool.apply_async(
            doDownwardRadiation2D, args=(NPHOTONS, *param,
                                   -np.cos(np.deg2rad(solzen)),
                                   np.deg2rad(solaz-180.0), False)
          ))
    pool.close()
    waitResults(results)

  for i, solzen in enumerate(VAR_SOLZEN):
    for j, solaz in enumerate(VAR_SOLAZ):
      for k, albg in enumerate(VAR_ALBG):
        index = i*len(VAR_SOLAZ)*len(VAR_ALBG) + j*len(VAR_ALBG) + k
        frac_swdir_refl[i,j,k] = results[index].get()['n_upward'] / NPHOTONS
        frac_swdir_upnl[i,j,k] = results[index].get()['n_front'] / NPHOTONS
        frac_swdir_dpnl[i,j,k] = results[index].get()['n_back'] / NPHOTONS
        frac_swdir_grnd[i,j,k] = results[index].get()['n_ground'] / NPHOTONS
  
  return {
    'frac_swdir_refl': frac_swdir_refl, 'frac_swdir_grnd': frac_swdir_grnd,
    'frac_swdir_upnl': frac_swdir_upnl, 'frac_swdir_dpnl': frac_swdir_dpnl
  }

def generateSWDifLookupTable():
  """
  Generate a lookup table for diffuse shortwave fluxes.

  Returns:
    dict: A dictionary containing the following keys:
      - 'frac_swdif_refl': The fraction of shortwave diffuse flux reflected from the surface.
      - 'frac_swdif_grnd': The fraction of shortwave diffuse flux absorbed by the ground.
      - 'frac_swdif_upnl': The fraction of shortwave diffuse flux absorbed by the upper panel.
      - 'frac_swdif_dpnl': The fraction of shortwave diffuse flux absorbed by the lower panel.
  """
  tempvar = xa.DataArray(
    np.full(len(VAR_ALBG), np.nan),
    dims=['surface_albedo'],
    coords={'surface_albedo': VAR_ALBG}
  )
  frac_swdif_refl = tempvar.copy()
  frac_swdif_upnl = tempvar.copy()
  frac_swdif_dpnl = tempvar.copy()
  frac_swdif_grnd = tempvar.copy()
  
  results = []
  with Pool(processes=MAX_PROCESSES) as pool:
    for albg in VAR_ALBG:
      param = PARAMS_SW[:-1] + (albg,)
      results.append(pool.apply_async(
        doDownwardRadiation2D, args=(NPHOTONS, *param, None, None, False)
      ))
    pool.close()
    waitResults(results)
  
  for i, albg in enumerate(VAR_ALBG):
    frac_swdif_refl[i] = results[i].get()['n_upward'] / NPHOTONS
    frac_swdif_upnl[i] = results[i].get()['n_front'] / NPHOTONS
    frac_swdif_dpnl[i] = results[i].get()['n_back'] / NPHOTONS
    frac_swdif_grnd[i] = results[i].get()['n_ground'] / NPHOTONS

  return {
    'frac_swdif_refl': frac_swdif_refl, 'frac_swdif_grnd': frac_swdif_grnd,
    'frac_swdif_upnl': frac_swdif_upnl, 'frac_swdif_dpnl': frac_swdif_dpnl
  }

def generateLWLookupTable():
  """
  Generate a lookup table for longwave fluxes.

  Returns:
    dict: A dictionary containing the following keys:
      - 'frac_lwatm_refl': The fraction of longwave flux originating from downward LW radiation from the atmosphere that is reflected by the atmosphere.
      - 'frac_lwatm_grnd': The fraction of longwave flux originating from downward LW radiation from the atmosphere that is absorbed by the ground.
      - 'frac_lwatm_upnl': The fraction of longwave flux originating from downward LW radiation from the atmosphere that is absorbed by the upper panel.
      - 'frac_lwatm_dpnl': The fraction of longwave flux originating from downward LW radiation from the atmosphere that is absorbed by the lower panel.
      - 'frac_lwgrnd_refl': The fraction of longwave flux originating from the ground emission that is reflected by the atmosphere.
      - 'frac_lwgrnd_grnd': The fraction of longwave flux originating from the ground emission that is absorbed by the ground.
      - 'frac_lwgrnd_upnl': The fraction of longwave flux originating from the ground emission that is absorbed by the upper panel.
      - 'frac_lwgrnd_dpnl': The fraction of longwave flux originating from the ground emission that is absorbed by the lower panel.
      - 'frac_lwupnl_refl': The fraction of longwave flux originating from the upper panel emission that is reflected by the atmosphere.
      - 'frac_lwupnl_grnd': The fraction of longwave flux originating from the upper panel emission that is absorbed by the ground.
      - 'frac_lwupnl_upnl': The fraction of longwave flux originating from the upper panel emission that is absorbed by the upper panel.
      - 'frac_lwupnl_dpnl': The fraction of longwave flux originating from the upper panel emission that is absorbed by the lower panel.
      - 'frac_lwdpnl_refl': The fraction of longwave flux originating from the lower panel emission that is reflected by the atmosphere.
      - 'frac_lwdpnl_grnd': The fraction of longwave flux originating from the lower panel emission that is absorbed by the ground.
      - 'frac_lwdpnl_upnl': The fraction of longwave flux originating from the lower panel emission that is absorbed by the upper panel.
      - 'frac_lwdpnl_dpnl': The fraction of longwave flux originating from the lower panel emission that is absorbed by the lower panel.
  """
  tempvar = xa.DataArray(
    np.full(len(VAR_EMIS), np.nan),
    dims=['surface_emissivity'],
    coords={'surface_emissivity': VAR_EMIS}
  )
  frac_lwatm_refl = tempvar.copy()
  frac_lwatm_upnl = tempvar.copy()
  frac_lwatm_dpnl = tempvar.copy()
  frac_lwatm_grnd = tempvar.copy()
  frac_lwgrnd_refl = tempvar.copy()
  frac_lwgrnd_upnl = tempvar.copy()
  frac_lwgrnd_dpnl = tempvar.copy()
  frac_lwgrnd_grnd = tempvar.copy()
  frac_lwupnl_refl = tempvar.copy()
  frac_lwupnl_upnl = tempvar.copy()
  frac_lwupnl_dpnl = tempvar.copy()
  frac_lwupnl_grnd = tempvar.copy()
  frac_lwdpnl_refl = tempvar.copy()
  frac_lwdpnl_upnl = tempvar.copy()
  frac_lwdpnl_dpnl = tempvar.copy()
  frac_lwdpnl_grnd = tempvar.copy()

  results = []
  with Pool(processes=MAX_PROCESSES) as pool:
    for emis in VAR_EMIS:
      param = PARAMS_LW[:-1] + (1-emis,)
      results.append(pool.apply_async(
        doDownwardRadiation2D, args=(NPHOTONS, *param, None, None, False)
      ))
    pool.close()
    waitResults(results)
  
  for i, emis in enumerate(VAR_EMIS):
    frac_lwatm_refl[i] = results[i].get()['n_upward'] / NPHOTONS
    frac_lwatm_upnl[i] = results[i].get()['n_front'] / NPHOTONS
    frac_lwatm_dpnl[i] = results[i].get()['n_back'] / NPHOTONS
    frac_lwatm_grnd[i] = results[i].get()['n_ground'] / NPHOTONS

  results = []
  with Pool(processes=MAX_PROCESSES) as pool:
    for emis in VAR_EMIS:
      param = PARAMS_LW[:-1] + (1-emis,)
      results.append(pool.apply_async(
        doPanelEmission2D, args=(NPHOTONS, *param, None, None, True, False)
      ))
    pool.close()
    waitResults(results)

  for i, emis in enumerate(VAR_EMIS):
    frac_lwupnl_refl[i] = results[i].get()['n_upward'] / NPHOTONS
    frac_lwupnl_upnl[i] = results[i].get()['n_front'] / NPHOTONS
    frac_lwupnl_dpnl[i] = results[i].get()['n_back'] / NPHOTONS
    frac_lwupnl_grnd[i] = results[i].get()['n_ground'] / NPHOTONS

  results = []
  with Pool(processes=MAX_PROCESSES) as pool:
    for emis in VAR_EMIS:
      param = PARAMS_LW[:-1] + (1-emis,)
      results.append(pool.apply_async(
        doPanelEmission2D, args=(NPHOTONS, *param, None, None, False, False)
      ))
    pool.close()
    waitResults(results)

  for i, emis in enumerate(VAR_EMIS):
    frac_lwdpnl_refl[i] = results[i].get()['n_upward'] / NPHOTONS
    frac_lwdpnl_upnl[i] = results[i].get()['n_front'] / NPHOTONS
    frac_lwdpnl_dpnl[i] = results[i].get()['n_back'] / NPHOTONS
    frac_lwdpnl_grnd[i] = results[i].get()['n_ground'] / NPHOTONS

  results = []
  with Pool(processes=MAX_PROCESSES) as pool:
    for emis in VAR_EMIS:
      param = PARAMS_LW[:-1] + (1-emis,)
      results.append(pool.apply_async(
        doGroundEmission2D, args=(NPHOTONS, *param, None, None)
      ))
    pool.close()
    waitResults(results)
  
  for i, emis in enumerate(VAR_EMIS):
    frac_lwgrnd_refl[i] = results[i].get()['n_upward'] / NPHOTONS
    frac_lwgrnd_upnl[i] = results[i].get()['n_front'] / NPHOTONS
    frac_lwgrnd_dpnl[i] = results[i].get()['n_back'] / NPHOTONS
    frac_lwgrnd_grnd[i] = results[i].get()['n_ground'] / NPHOTONS

  return {
    'frac_lwatm_refl': frac_lwatm_refl, 'frac_lwatm_grnd': frac_lwatm_grnd,
    'frac_lwatm_upnl': frac_lwatm_upnl, 'frac_lwatm_dpnl': frac_lwatm_dpnl,
    'frac_lwgrnd_refl': frac_lwgrnd_refl, 'frac_lwgrnd_grnd': frac_lwgrnd_grnd,
    'frac_lwgrnd_upnl': frac_lwgrnd_upnl, 'frac_lwgrnd_dpnl': frac_lwgrnd_dpnl,
    'frac_lwupnl_refl': frac_lwupnl_refl, 'frac_lwupnl_grnd': frac_lwupnl_grnd,
    'frac_lwupnl_upnl': frac_lwupnl_upnl, 'frac_lwupnl_dpnl': frac_lwupnl_dpnl,
    'frac_lwdpnl_refl': frac_lwdpnl_refl, 'frac_lwdpnl_grnd': frac_lwdpnl_grnd,
    'frac_lwdpnl_upnl': frac_lwdpnl_upnl, 'frac_lwdpnl_dpnl': frac_lwdpnl_dpnl
  }

def generateDummyLookupTable(alb:float=0.15, emis:float=0.85):
  """
  Generate a dummy lookup table with a constant albedo for testing purposes.

  Parameters:
  - alb: The albedo value.
  - emis: The emissivity value.

  Returns:
  - dict: A full dictionary that contains fields for each lookup table.
  """
  tempvar = xa.DataArray(
    np.full((len(VAR_SOLZEN), len(VAR_SOLAZ), len(VAR_ALBG)), np.nan),
    dims=['solar_zenith_angle', 'solar_azimuth_angle', 'surface_albedo'],
    coords={'surface_albedo': VAR_ALBG,
            'solar_zenith_angle': VAR_SOLZEN,
            'solar_azimuth_angle': VAR_SOLAZ}
  )
  frac_swdir_refl = tempvar.copy()
  frac_swdir_upnl = tempvar.copy()
  frac_swdir_dpnl = tempvar.copy()
  frac_swdir_grnd = tempvar.copy()

  tempvar = xa.DataArray(
    np.full(len(VAR_ALBG), np.nan),
    dims=['surface_albedo'],
    coords={'surface_albedo': VAR_ALBG}
  )
  frac_swdif_refl = tempvar.copy()
  frac_swdif_upnl = tempvar.copy()
  frac_swdif_dpnl = tempvar.copy()
  frac_swdif_grnd = tempvar.copy()

  tempvar = xa.DataArray(
    np.full(len(VAR_EMIS), np.nan),
    dims=['surface_emissivity'],
    coords={'surface_emissivity': VAR_EMIS}
  )
  frac_lwatm_refl = tempvar.copy()
  frac_lwatm_upnl = tempvar.copy()
  frac_lwatm_dpnl = tempvar.copy()
  frac_lwatm_grnd = tempvar.copy()
  frac_lwgrnd_refl = tempvar.copy()
  frac_lwgrnd_upnl = tempvar.copy()
  frac_lwgrnd_dpnl = tempvar.copy()
  frac_lwgrnd_grnd = tempvar.copy()
  frac_lwupnl_refl = tempvar.copy()
  frac_lwupnl_upnl = tempvar.copy()
  frac_lwupnl_dpnl = tempvar.copy()
  frac_lwupnl_grnd = tempvar.copy()
  frac_lwdpnl_refl = tempvar.copy()
  frac_lwdpnl_upnl = tempvar.copy()
  frac_lwdpnl_dpnl = tempvar.copy()
  frac_lwdpnl_grnd = tempvar.copy()

  print('Building direct SW lookup table...')
  frac_swdir_refl[:,:,:] = alb
  frac_swdir_upnl[:,:,:] = (1 - alb) * 4 / 5
  frac_swdir_dpnl[:,:,:] = (1 - alb) * 0 / 5
  frac_swdir_grnd[:,:,:] = (1 - alb) * 1 / 5

  print('Building diffuse SW lookup table...')
  frac_swdif_refl[:] = alb
  frac_swdif_upnl[:] = (1 - alb) * 4 / 5
  frac_swdif_dpnl[:] = (1 - alb) * 0 / 5
  frac_swdif_grnd[:] = (1 - alb) * 1 / 5

  print('Building LW atm lookup table...')
  frac_lwatm_refl[:] = (1 - emis)
  frac_lwatm_upnl[:] = emis * 1 / 2
  frac_lwatm_dpnl[:] = emis * 0 / 2
  frac_lwatm_grnd[:] = emis * 1 / 2

  print('Building LW ground lookup table...')
  frac_lwgrnd_refl[:] = 0.5
  frac_lwgrnd_upnl[:] = 0.0
  frac_lwgrnd_dpnl[:] = 0.5
  frac_lwgrnd_grnd[:] = 0.0

  print('Building LW upnl lookup table...')
  frac_lwupnl_refl[:] = 1.0
  frac_lwupnl_upnl[:] = 0.0
  frac_lwupnl_dpnl[:] = 0.0
  frac_lwupnl_grnd[:] = 0.0

  print('Building LW dpnl lookup table...')
  frac_lwdpnl_refl[:] = 0.0
  frac_lwdpnl_upnl[:] = 0.0
  frac_lwdpnl_dpnl[:] = 0.0
  frac_lwdpnl_grnd[:] = 1.0

  return {
    'frac_swdir_refl': frac_swdir_refl, 'frac_swdir_grnd': frac_swdir_grnd,
    'frac_swdir_upnl': frac_swdir_upnl, 'frac_swdir_dpnl': frac_swdir_dpnl,
    'frac_swdif_refl': frac_swdif_refl, 'frac_swdif_grnd': frac_swdif_grnd,
    'frac_swdif_upnl': frac_swdif_upnl, 'frac_swdif_dpnl': frac_swdif_dpnl,
    'frac_lwatm_refl': frac_lwatm_refl, 'frac_lwatm_grnd': frac_lwatm_grnd,
    'frac_lwatm_upnl': frac_lwatm_upnl, 'frac_lwatm_dpnl': frac_lwatm_dpnl,
    'frac_lwgrnd_refl': frac_lwgrnd_refl, 'frac_lwgrnd_grnd': frac_lwgrnd_grnd,
    'frac_lwgrnd_upnl': frac_lwgrnd_upnl, 'frac_lwgrnd_dpnl': frac_lwgrnd_dpnl,
    'frac_lwupnl_refl': frac_lwupnl_refl, 'frac_lwupnl_grnd': frac_lwupnl_grnd,
    'frac_lwupnl_upnl': frac_lwupnl_upnl, 'frac_lwupnl_dpnl': frac_lwupnl_dpnl,
    'frac_lwdpnl_refl': frac_lwdpnl_refl, 'frac_lwdpnl_grnd': frac_lwdpnl_grnd,
    'frac_lwdpnl_upnl': frac_lwdpnl_upnl, 'frac_lwdpnl_dpnl': frac_lwdpnl_dpnl
  }

if __name__ == '__main__':
  from argparse import ArgumentParser
  import os, sys
  parser = ArgumentParser()
  parser.add_argument('--dummy', action='store_true', help='Generate dummy lookup table')
  args = parser.parse_args()

  if args.dummy:
    print('Generating dummy lookup table...')
    data = generateDummyLookupTable()
    print('Saving dummy lookup table... => solarfarm_spec_dummy.nc')
    ds = xa.Dataset(data)
    ds.to_netcdf('solarfarm_spec_dummy.nc')
    exit()
  
  if os.path.exists(FILE_OUTPUT):
    print(f'File {FILE_OUTPUT} already exists. Loading...')
    ds = xa.open_dataset(FILE_OUTPUT)
  else:
    ds = xa.Dataset()
    ds.attrs['title'] = 'Solar Farm Configuration Data'
    ds.attrs['version'] = '0.1'
    ds.attrs['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ds.attrs['solar_panel_fraction'] = PANEL_FRAC
    ds.attrs['solar_panel_height'] = PANEL_HGT
    ds.attrs['solar_panel_tilt'] = PANEL_TILT

  if 'frac_swdir_refl' in ds:
    print('Direct SW lookup table already exists. Skipping...')
  else:
    print('Generating direct SW lookup table...')
    data = generateSWDirLookupTable()
    ds.update(data)
    ds.attrs['solar_panel_albedo_front'] = ALB_FRONT
    ds.attrs['solar_panel_albedo_back'] = ALB_BACK
    ds.attrs['swdir_nphotons'] = NPHOTONS
    ds.attrs['swdir_navgs'] = 1
    print(f'Saving lookup table... => {FILE_OUTPUT}')
    ds.to_netcdf(FILE_OUTPUT)

  NPHOTONS *= 10

  if 'frac_swdif_refl' in ds:
    print('Diffuse SW lookup table already exists. Skipping...')
  else:
    print('Generating diffuse SW lookup table...')
    data = generateSWDifLookupTable()
    ds.update(data)
    ds.attrs['swdif_nphotons'] = NPHOTONS
    ds.attrs['swdif_navgs'] = 1
    print(f'Saving lookup table... => {FILE_OUTPUT}')
    ds.to_netcdf(FILE_OUTPUT)
  
  if 'frac_lwatm_refl' in ds:
    print('LW lookup table already exists. Skipping...')
  else:
    print('Generating LW lookup table...')
    data = generateLWLookupTable()
    ds.update(data)
    ds.attrs['solar_panel_emissivity_front'] = EMIS_FRONT
    ds.attrs['solar_panel_emissivity_back'] = EMIS_BACK
    ds.attrs['lw_nphotons'] = NPHOTONS
    ds.attrs['lw_navgs'] = 1
    print(f'Saving lookup table... => {FILE_OUTPUT}')
    ds.to_netcdf(FILE_OUTPUT)

  if not 'history' in ds.attrs:
    ds.attrs['history'] = []
  ds.attrs['history'] = ds.attrs['history'] + [datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': python ' + ' '.join(sys.argv)]
  ds.to_netcdf(FILE_OUTPUT)