import sys, time, os
from datetime import datetime

from mpi4py import MPI
import numpy as np
import xarray as xa

from SolarFarmMonteCarlo import (
  doDownwardRadiation2D, doPanelEmission2D, doGroundEmission2D
)

# ========== USER-DEFINED PARAMETERS ==========
PANEL_FRAC       = 0.5
PANEL_HGT        = 1.0
PANEL_TILT       = 20.0
ALB_FRONT        = 0.1
ALB_BACK         = 0.8
ALB_SURFACE      = np.arange(0.0, 1.01, 0.02)
EMIS_FRONT       = 0.80
EMIS_BACK        = 0.96
EMIS_SURFACE     = np.arange(0.8, 1.01, 0.02)
SOLAR_ZENITH     = np.arange(0.0, 85.0, 1.0)
SOLAR_AZIMUTH    = np.arange(0.0, 360.1, 2.0)

NPHOTONS         = 100000
NAVGS            = 10
FILE_OUTPUT      = 'solarfarm_spec.nc'

DO_SW_DIR        = True
DO_SW_DIF        = False
DO_LW            = False
# =============================================

def getCoord(coord_spec, coord_name):
  """
  Create a DataArray object based on the given coordinate specification.
  If the specification is a float, the DataArray will contain a single value.
  If the specification is a tuple of length 2 or 3, the DataArray will contain a range of values.
  If the specification is a numpy array, the DataArray will contain the values in the array.

  Parameters:
  - coord_spec: The specification for the coordinate. It can be a float, a tuple of length 2 or 3, or a numpy array.
  - coord_name: The name of the coordinate.

  Returns:
  - A DataArray object representing the coordinate.

  Raises:
  - ValueError: If coord_spec is not a float or a tuple of length 2 or 3.
  """
  if (type(coord_spec) is np.array):
    return xa.DataArray(coord_spec, dims=coord_name)
  elif (type(coord_spec) is float):
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
COMM = MPI.COMM_WORLD
MPI_SIZE = COMM.Get_size()
MPI_RANK = COMM.Get_rank()

def printMPIMessage(message:str):
  """
  Prints a message with the current time and MPI rank.

  Args:
  - message (str): The message to be printed.

  Returns:
  - None
  """
  time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
  print(f'{time} [{MPI_RANK:4d}] {message}')
  sys.stdout.flush()

def generateSWDirLookupTable():
  """
  Generate lookup table for direct shortwave fluxes.

  Returns:
  A dictionary containing the following keys:
    - 'frac_swdir_refl' (xarray.DataArray): fraction of direct shortwave flux reflected to the atmosphere.
    - 'frac_swdir_grnd' (xarray.DataArray): fraction of direct shortwave flux absorbed by the ground.
    - 'frac_swdir_upnl' (xarray.DataArray): fraction of direct shortwave flux absorbed by the upper panel.
    - 'frac_swdir_dpnl' (xarray.DataArray): fraction of direct shortwave flux absorbed by the lower panel.
  """
  if MPI_RANK==0:
    printMPIMessage("Creating lookup table for direct shortwave fluxes...")
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
    printMPIMessage("Total number of simulations: {:,}".format(
      len(VAR_SOLZEN) * len(VAR_SOLAZ) * len(VAR_ALBG)
    ))
  COMM.Barrier()

  params = []
  for solzen in VAR_SOLZEN:
    for solaz in VAR_SOLAZ:
      for albg in VAR_ALBG:
          params += [(
            NPHOTONS, *PARAMS_SW[:-1], albg,
            -np.cos(np.deg2rad(solzen)), np.deg2rad(solaz-180.0), False
          )] * NAVGS

  samples_per_rank = len(params) / MPI_SIZE
  start_index = int(MPI_RANK * samples_per_rank)
  end_index = int((MPI_RANK + 1) * samples_per_rank)
  if MPI_RANK == MPI_SIZE - 1:
    end_index = len(params)
  params = params[start_index:end_index]
  printMPIMessage(f'Tasked with {len(params)} simulations. ({start_index} - {end_index})')

  results = np.full((len(params), 4), np.nan)
  for i, param in enumerate(params):
    printMPIMessage(f'Running simulation {i+1}/{len(params)}...')
    start_time = time.time()
    d = doDownwardRadiation2D(*param)
    end_time = time.time()
    time_used = end_time - start_time
    printMPIMessage(f'Time used for simulation {i+1}: {time_used:.2f} seconds')
    results[i,0] = d['n_upward'] / NPHOTONS
    results[i,1] = d['n_front'] / NPHOTONS
    results[i,2] = d['n_back'] / NPHOTONS
    results[i,3] = d['n_ground'] / NPHOTONS

  printMPIMessage("Tasks completed.")
  results = COMM.gather(results, root=0)

  if MPI_RANK == 0:
    printMPIMessage("Gathering results...")
    results = np.concatenate(results, axis=0)
    for i, solzen in enumerate(VAR_SOLZEN):
      for j, solaz in enumerate(VAR_SOLAZ):
        for k, albg in enumerate(VAR_ALBG):
          inds = (i*len(VAR_SOLAZ)*len(VAR_ALBG)+j*len(VAR_ALBG)+k) * NAVGS
          inde = inds + NAVGS
          frac_swdir_refl[i,j,k] = np.mean(results[inds:inde,0])
          frac_swdir_upnl[i,j,k] = np.mean(results[inds:inde,1])
          frac_swdir_dpnl[i,j,k] = np.mean(results[inds:inde,2])
          frac_swdir_grnd[i,j,k] = np.mean(results[inds:inde,3])

    return {
      'frac_swdir_refl': frac_swdir_refl, 'frac_swdir_grnd': frac_swdir_grnd,
      'frac_swdir_upnl': frac_swdir_upnl, 'frac_swdir_dpnl': frac_swdir_dpnl
    }
  else:
    return None
  
def generateSWDifLookupTable():
  """
  Generate a lookup table for diffuse shortwave fluxes.

  Returns:
    dict: A dictionary containing the following keys:
      - 'frac_swdif_refl' (xarray.DataArray): The fraction of shortwave diffuse flux reflected to the atmosphere.
      - 'frac_swdif_grnd' (xarray.DataArray): The fraction of shortwave diffuse flux absorbed by the ground.
      - 'frac_swdif_upnl' (xarray.DataArray): The fraction of shortwave diffuse flux absorbed by the upper panel.
      - 'frac_swdif_dpnl' (xarray.DataArray): The fraction of shortwave diffuse flux absorbed by the lower panel.
  """
  if MPI_RANK==0:
    printMPIMessage("Creating lookup table for direct shortwave fluxes...")
    tempvar = xa.DataArray(
      np.full(len(VAR_ALBG), np.nan),
      dims=['surface_albedo'],
      coords={'surface_albedo': VAR_ALBG}
    )
    frac_swdif_refl = tempvar.copy()
    frac_swdif_upnl = tempvar.copy()
    frac_swdif_dpnl = tempvar.copy()
    frac_swdif_grnd = tempvar.copy()
    printMPIMessage("Total number of simulations: {:,}".format(len(VAR_ALBG)))
  COMM.Barrier()

  params = []
  for albg in VAR_ALBG:
    params += [(
      NPHOTONS, *PARAMS_SW[:-1], albg,
      None, None, False
    )] * NAVGS

  samples_per_rank = len(params) / MPI_SIZE
  start_index = int(MPI_RANK * samples_per_rank)
  end_index = int((MPI_RANK + 1) * samples_per_rank)
  if MPI_RANK == MPI_SIZE - 1:
    end_index = len(params)
  params = params[start_index:end_index]
  printMPIMessage(f'Tasked with {len(params)} simulations. ({start_index} - {end_index})')

  results = np.full((len(params), 4), np.nan)
  for i, param in enumerate(params):
    printMPIMessage(f'Running simulation {i+1}/{len(params)}...')
    start_time = time.time()
    d = doDownwardRadiation2D(*param)
    end_time = time.time()
    time_used = end_time - start_time
    printMPIMessage(f'Time used for simulation {i+1}: {time_used:.2f} seconds')
    results[i,0] = d['n_upward'] / NPHOTONS
    results[i,1] = d['n_front'] / NPHOTONS
    results[i,2] = d['n_back'] / NPHOTONS
    results[i,3] = d['n_ground'] / NPHOTONS

  printMPIMessage("Tasks completed.")
  results = COMM.gather(results, root=0)

  if MPI_RANK == 0:
    printMPIMessage("Gathering results...")
    results = np.concatenate(results, axis=0)
    for k, albg in enumerate(VAR_ALBG):
      inds = k * NAVGS
      inde = inds + NAVGS
      frac_swdif_refl[k] = np.mean(results[inds:inde,0])
      frac_swdif_upnl[k] = np.mean(results[inds:inde,1])
      frac_swdif_dpnl[k] = np.mean(results[inds:inde,2])
      frac_swdif_grnd[k] = np.mean(results[inds:inde,3])

    return {
      'frac_swdif_refl': frac_swdif_refl, 'frac_swdif_grnd': frac_swdif_grnd,
      'frac_swdif_upnl': frac_swdif_upnl, 'frac_swdif_dpnl': frac_swdif_dpnl
    }
  else:
    return None

def generateLWLookupTable():
  """
  Generate a lookup table for longwave fluxes.

  Returns:
    dict: A dictionary containing the following keys:
      - 'frac_lwatm_refl': The fraction of longwave flux originating from downward LW radiation from the atmosphere that goes out to the atmosphere.
      - 'frac_lwatm_grnd': The fraction of longwave flux originating from downward LW radiation from the atmosphere that is absorbed by the ground.
      - 'frac_lwatm_upnl': The fraction of longwave flux originating from downward LW radiation from the atmosphere that is absorbed by the upper panel.
      - 'frac_lwatm_dpnl': The fraction of longwave flux originating from downward LW radiation from the atmosphere that is absorbed by the lower panel.
      - 'frac_lwgrnd_refl': The fraction of longwave flux originating from the ground emission that goes out to the atmosphere.
      - 'frac_lwgrnd_grnd': The fraction of longwave flux originating from the ground emission that is absorbed by the ground.
      - 'frac_lwgrnd_upnl': The fraction of longwave flux originating from the ground emission that is absorbed by the upper panel.
      - 'frac_lwgrnd_dpnl': The fraction of longwave flux originating from the ground emission that is absorbed by the lower panel.
      - 'frac_lwupnl_refl': The fraction of longwave flux originating from the upper panel emission that goes out to the atmosphere.
      - 'frac_lwupnl_grnd': The fraction of longwave flux originating from the upper panel emission that is absorbed by the ground.
      - 'frac_lwupnl_upnl': The fraction of longwave flux originating from the upper panel emission that is absorbed by the upper panel.
      - 'frac_lwupnl_dpnl': The fraction of longwave flux originating from the upper panel emission that is absorbed by the lower panel.
      - 'frac_lwdpnl_refl': The fraction of longwave flux originating from the lower panel emission that goes out to the atmosphere.
      - 'frac_lwdpnl_grnd': The fraction of longwave flux originating from the lower panel emission that is absorbed by the ground.
      - 'frac_lwdpnl_upnl': The fraction of longwave flux originating from the lower panel emission that is absorbed by the upper panel.
      - 'frac_lwdpnl_dpnl': The fraction of longwave flux originating from the lower panel emission that is absorbed by the lower panel.
  """
  if MPI_RANK==0:
    printMPIMessage("Creating lookup table for direct shortwave fluxes...")
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
    printMPIMessage("Total number of simulations: {:,}".format(len(VAR_EMIS) * 4))
  COMM.Barrier()

  params = []
  for emis in VAR_EMIS:
    params += [(
      doDownwardRadiation2D,
      NPHOTONS, *PARAMS_LW[:-1], 1-emis,
      None, None, False
    )] * NAVGS
    params += [(
      doPanelEmission2D,
      NPHOTONS, *PARAMS_LW[:-1], 1-emis,
      None, None, True, False
    )] * NAVGS
    params += [(
      doPanelEmission2D,
      NPHOTONS, *PARAMS_LW[:-1], 1-emis,
      None, None, False, False
    )] * NAVGS
    params += [(
      doGroundEmission2D,
      NPHOTONS, *PARAMS_LW[:-1], 1-emis,
      None, None
    )] * NAVGS

  samples_per_rank = len(params) / MPI_SIZE
  start_index = int(MPI_RANK * samples_per_rank)
  end_index = int((MPI_RANK + 1) * samples_per_rank)
  if MPI_RANK == MPI_SIZE - 1:
    end_index = len(params)
  params = params[start_index:end_index]
  printMPIMessage(f'Tasked with {len(params)} simulations. ({start_index} - {end_index})')

  results = np.full((len(params), 4), np.nan)
  for i, param in enumerate(params):
    printMPIMessage(f'Running simulation {i+1}/{len(params)}...')
    start_time = time.time()
    d = param[0](*param[1:])
    end_time = time.time()
    time_used = end_time - start_time
    printMPIMessage(f'Time used for simulation {i+1}: {time_used:.2f} seconds')
    results[i,0] = d['n_upward'] / NPHOTONS
    results[i,1] = d['n_front'] / NPHOTONS
    results[i,2] = d['n_back'] / NPHOTONS
    results[i,3] = d['n_ground'] / NPHOTONS

  printMPIMessage("Tasks completed.")
  results = COMM.gather(results, root=0)

  if MPI_RANK == 0:
    printMPIMessage("Gathering results...")
    results = np.concatenate(results, axis=0)
    for i, emis in enumerate(VAR_EMIS):
      inds = 4 * i * NAVGS
      inde = inds + NAVGS
      frac_lwatm_refl[i] = np.mean(results[inds:inde,0])
      frac_lwatm_upnl[i] = np.mean(results[inds:inde,1])
      frac_lwatm_dpnl[i] = np.mean(results[inds:inde,2])
      frac_lwatm_grnd[i] = np.mean(results[inds:inde,3])
      inds = (4 * i + 1) * NAVGS
      inde = inds + NAVGS
      frac_lwupnl_refl[i] = np.mean(results[inds:inde,0])
      frac_lwupnl_upnl[i] = np.mean(results[inds:inde,1])
      frac_lwupnl_dpnl[i] = np.mean(results[inds:inde,2])
      frac_lwupnl_grnd[i] = np.mean(results[inds:inde,3])
      inds = (4 * i + 2) * NAVGS
      inde = inds + NAVGS
      frac_lwdpnl_refl[i] = np.mean(results[inds:inde,0])
      frac_lwdpnl_upnl[i] = np.mean(results[inds:inde,1])
      frac_lwdpnl_dpnl[i] = np.mean(results[inds:inde,2])
      frac_lwdpnl_grnd[i] = np.mean(results[inds:inde,3])
      inds = (4 * i + 3) * NAVGS
      inde = inds + NAVGS
      frac_lwgrnd_refl[i] = np.mean(results[inds:inde,0])
      frac_lwgrnd_upnl[i] = np.mean(results[inds:inde,1])
      frac_lwgrnd_dpnl[i] = np.mean(results[inds:inde,2])
      frac_lwgrnd_grnd[i] = np.mean(results[inds:inde,3])

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
  else:
    return None


if __name__ == '__main__':
  if MPI_RANK == 0:
    if os.path.exists(FILE_OUTPUT):
      ds = xa.load_dataset(FILE_OUTPUT)
    else:
      ds = xa.Dataset()
      ds.attrs['title'] = 'Solar Farm Configuration Data'
      ds.attrs['version'] = '0.1'
      ds.attrs['date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
      ds.attrs['solar_panel_fraction'] = PANEL_FRAC
      ds.attrs['solar_panel_height'] = PANEL_HGT
      ds.attrs['solar_panel_tilt'] = PANEL_TILT

  if DO_SW_DIR:
    data = generateSWDirLookupTable()
    if MPI_RANK == 0:
      ds.update(data)
      ds.attrs['solar_panel_albedo_front'] = ALB_FRONT
      ds.attrs['solar_panel_albedo_back'] = ALB_BACK
      ds.attrs['swdir_nphotons'] = NPHOTONS
      ds.attrs['swdir_navgs'] = NAVGS
      printMPIMessage(f'Saving lookup table... => {FILE_OUTPUT}')
      ds.to_netcdf(FILE_OUTPUT)

  if DO_SW_DIF:
    data = generateSWDifLookupTable()
    if MPI_RANK == 0:
      ds.update(data)
      ds.attrs['swdif_nphotons'] = NPHOTONS
      ds.attrs['swdif_navgs'] = NAVGS
      printMPIMessage(f'Saving lookup table... => {FILE_OUTPUT}')
      ds.to_netcdf(FILE_OUTPUT)

  if DO_LW:
    data = generateLWLookupTable()
    if MPI_RANK == 0:
      ds.update(data)
      ds.attrs['solar_panel_emissivity_front'] = EMIS_FRONT
      ds.attrs['solar_panel_emissivity_back'] = EMIS_BACK
      ds.attrs['lw_nphotons'] = NPHOTONS
      ds.attrs['lw_navgs'] = NAVGS
      printMPIMessage(f'Saving lookup table... => {FILE_OUTPUT}')
      ds.to_netcdf(FILE_OUTPUT)

  if not 'history' in ds.attrs:
    ds.attrs['history'] = []
  ds.attrs['history'] = ds.attrs['history'] + [datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ': python ' + ' '.join(sys.argv)]
  ds.to_netcdf(FILE_OUTPUT)