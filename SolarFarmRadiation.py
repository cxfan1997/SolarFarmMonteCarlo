from multiprocessing import Pool
from SolarFarmMonteCarlo import doDownwardRadiation2D, doPanelEmission2D, doGroundEmission2D
from Utils import waitResults
from dataclasses import dataclass
  
@dataclass
class SolarFarmRadiation:
  nphotons_SWdirect:int       = int(1e7)
  nphotons_SWdiffuse:int      = int(1e7)
  nphotons_LW:int             = int(1e7)
  npartitions:int             = 100
  panel_area_fraction:float   = 0.45
  panel_height:float          = 1.0
  panel_tilt_angle:float      = 20.0
  panel_conversion_rate:float = 0.18
  panel_albedo_front:float    = 0.21
  panel_albedo_back:float     = 0.6
  ground_albedo:float         = 0.3
  panel_temp_front:float      = 320.15
  panel_temp_back:float       = 320.15
  ground_temp:float           = 310.15
  panel_emis_front:float      = 0.83
  panel_emis_back:float       = 0.97
  ground_emis:float           = 0.92

  @property
  def w(self):
    return 1.0 / self.panel_area_fraction
  
  @property
  def paramSW(self):
    return (
      1.0, self.panel_height, self.w, math.radians(self.panel_tilt_angle),
      self.panel_albedo_front, self.panel_albedo_back, self.ground_albedo
    )
  
  @property
  def paramLW(self):
    return (
      1.0, self.panel_height, self.w, math.radians(self.panel_tilt_angle),
      1-self.panel_emis_front, 1-self.panel_emis_back, 1-self.ground_emis
    )
  
  def SWSolver(self, fdwn_dir:float, fdwn_dif:float, mu0:float=None, phi0:float=None):
    print('Solving direct solar radiation...')
    frac_swdir_upwd = 0.0
    frac_swdir_grnd = 0.0
    frac_swdir_upnl = 0.0
    frac_swdir_dpnl = 0.0
    results = []
    nphotons = int(self.nphotons_SWdirect / self.npartitions)
    with Pool() as p:
      for _ in range(self.npartitions):
        results.append(p.apply_async(
          doDownwardRadiation2D, args=(nphotons, *self.paramSW, mu0, phi0, False)
        ))
      p.close()
      waitResults(results)
    for r in results:
      frac_swdir_upwd += r.get()['n_upward'] / nphotons / self.npartitions
      frac_swdir_grnd += r.get()['n_ground'] / nphotons / self.npartitions
      frac_swdir_upnl += r.get()['n_front']   / nphotons / self.npartitions
      frac_swdir_dpnl += r.get()['n_back']   / nphotons / self.npartitions

    print('Solving diffuse solar radiation...')
    frac_swdif_upwd = 0.0
    frac_swdif_grnd = 0.0
    frac_swdif_upnl = 0.0
    frac_swdif_dpnl = 0.0
    results = []
    nphotons = int(self.nphotons_SWdiffuse / self.npartitions)
    with Pool() as p:
      for _ in range(self.npartitions):
        results.append(p.apply_async(
          doDownwardRadiation2D, args=(nphotons, *self.paramSW, None, None, False)
        ))
      p.close()
      waitResults(results)
    for r in results:
      frac_swdif_upwd += r.get()['n_upward'] / nphotons / self.npartitions
      frac_swdif_grnd += r.get()['n_ground'] / nphotons / self.npartitions
      frac_swdif_upnl += r.get()['n_front']   / nphotons / self.npartitions
      frac_swdif_dpnl += r.get()['n_back']   / nphotons / self.npartitions

    out = {
      'frac_swdir_upwd': frac_swdir_upwd,
      'frac_swdir_grnd': frac_swdir_grnd,
      'frac_swdir_upnl': frac_swdir_upnl,
      'frac_swdir_dpnl': frac_swdir_dpnl,
      'flux_swdn_dir': fdwn_dir,
      'flux_swdn_dif': fdwn_dif,
      'flux_swup_dir': fdwn_dir * frac_swdir_upwd,
      'flux_swup_dif': fdwn_dif * frac_swdif_upwd,
      'flux_swabs_grnd_dir': fdwn_dir * frac_swdir_grnd,
      'flux_swabs_grnd_dif': fdwn_dif * frac_swdif_grnd,
      'flux_swabs_upnl_dir': fdwn_dir * frac_swdir_upnl,
      'flux_swabs_upnl_dif': fdwn_dif * frac_swdif_upnl,
      'flux_swabs_dpnl_dir': fdwn_dir * frac_swdir_dpnl,
      'flux_swabs_dpnl_dif': fdwn_dif * frac_swdif_dpnl
    }
    out['flux_swdn'] = out['flux_swdn_dir'] + out['flux_swdn_dif']
    out['flux_swup'] = out['flux_swup_dir'] + out['flux_swup_dif']
    out['flux_swabs_grnd'] = out['flux_swabs_grnd_dir'] + out['flux_swabs_grnd_dif']
    out['flux_swabs_upnl'] = out['flux_swabs_upnl_dir'] + out['flux_swabs_upnl_dif']
    out['flux_swabs_dpnl'] = out['flux_swabs_dpnl_dir'] + out['flux_swabs_dpnl_dif']
    out['flux_electricity'] = out['flux_swabs_upnl'] * self.panel_conversion_rate
    return out
  
  def stefanBoltzmann(self, temp:float, emis:float):
    return 5.67e-8 * emis * temp**4
  
  def LWSolver(self, fdwn:float):
    print('Solving downward LW radiation...')
    frac_lwatm_upwd = 0.0
    frac_lwatm_grnd = 0.0
    frac_lwatm_upnl = 0.0
    frac_lwatm_dpnl = 0.0
    results = []
    nphotons = int(self.nphotons_LW / self.npartitions)
    with Pool() as p:
      for _ in range(self.npartitions):
        results.append(p.apply_async(
          doDownwardRadiation2D, args=(nphotons, *self.paramLW, None, None, False)
        ))
      p.close()
      waitResults(results)
    for r in results:
      frac_lwatm_upwd += r.get()['n_upward'] / nphotons / self.npartitions
      frac_lwatm_grnd += r.get()['n_ground'] / nphotons / self.npartitions
      frac_lwatm_upnl += r.get()['n_front']   / nphotons / self.npartitions
      frac_lwatm_dpnl += r.get()['n_back']   / nphotons / self.npartitions

    print('Solving panel emission from front surface...')
    frac_lwupnl_upwd = 0.0
    frac_lwupnl_grnd = 0.0
    frac_lwupnl_upnl = 0.0
    frac_lwupnl_dpnl = 0.0
    results = []
    nphotons = int(self.nphotons_LW / self.npartitions)
    with Pool() as p:
      for _ in range(self.npartitions):
        results.append(p.apply_async(
          doPanelEmission2D, args=(nphotons, *self.paramLW, None, None, True, False)
        ))
      p.close()
      waitResults(results)
    for r in results:
      frac_lwupnl_upwd += r.get()['n_upward'] / nphotons / self.npartitions
      frac_lwupnl_grnd += r.get()['n_ground'] / nphotons / self.npartitions
      frac_lwupnl_upnl += r.get()['n_front']   / nphotons / self.npartitions
      frac_lwupnl_dpnl += r.get()['n_back']   / nphotons / self.npartitions

    print('Solving panel emission from back surface...')
    frac_lwdpnl_upwd = 0.0
    frac_lwdpnl_grnd = 0.0
    frac_lwdpnl_upnl = 0.0
    frac_lwdpnl_dpnl = 0.0
    results = []
    nphotons = int(self.nphotons_LW / self.npartitions)
    with Pool() as p:
      for _ in range(self.npartitions):
        results.append(p.apply_async(
          doPanelEmission2D, args=(nphotons, *self.paramLW, None, None, False, False)
        ))
      p.close()
      waitResults(results)
    for r in results:
      frac_lwdpnl_upwd += r.get()['n_upward'] / nphotons / self.npartitions
      frac_lwdpnl_grnd += r.get()['n_ground'] / nphotons / self.npartitions
      frac_lwdpnl_upnl += r.get()['n_front']   / nphotons / self.npartitions
      frac_lwdpnl_dpnl += r.get()['n_back']   / nphotons / self.npartitions

    print('Solving ground emission...')
    frac_lwgrnd_upwd = 0.0
    frac_lwgrnd_grnd = 0.0
    frac_lwgrnd_upnl = 0.0
    frac_lwgrnd_dpnl = 0.0
    results = []
    nphotons = int(self.nphotons_LW / self.npartitions)
    with Pool() as p:
      for _ in range(self.npartitions):
        results.append(p.apply_async(
          doGroundEmission2D, args=(nphotons, *self.paramLW, None, None,
                                   0, 1.0 / self.panel_area_fraction, False)
        ))
      p.close()
      waitResults(results)
    for r in results:
      frac_lwgrnd_upwd += r.get()['n_upward'] / nphotons / self.npartitions
      frac_lwgrnd_grnd += r.get()['n_ground'] / nphotons / self.npartitions
      frac_lwgrnd_upnl += r.get()['n_front']   / nphotons / self.npartitions
      frac_lwgrnd_dpnl += r.get()['n_back']   / nphotons / self.npartitions
    
    out = {
      'flux_lwdn': fdwn,
      'flux_lwemis_upnl': self.stefanBoltzmann(self.panel_temp_front, self.panel_emis_front) * self.panel_area_fraction,
      'flux_lwemis_dpnl': self.stefanBoltzmann(self.panel_temp_back, self.panel_emis_back) * self.panel_area_fraction,
      'flux_lwemis_grnd': self.stefanBoltzmann(self.ground_temp, self.ground_emis),
    }
    out['flux_lwup'] = (
      out['flux_lwdn'] * frac_lwatm_upwd
      + out['flux_lwemis_upnl'] * frac_lwupnl_upwd
      + out['flux_lwemis_dpnl'] * frac_lwdpnl_upwd
      + out['flux_lwemis_grnd'] * frac_lwgrnd_upwd
    )
    out['flux_lwabs_upnl'] = (
      out['flux_lwdn'] * frac_lwatm_upnl
      + out['flux_lwemis_upnl'] * frac_lwupnl_upnl
      + out['flux_lwemis_dpnl'] * frac_lwdpnl_upnl
      + out['flux_lwemis_grnd'] * frac_lwgrnd_upnl
    )
    out['flux_lwabs_dpnl'] = (
      out['flux_lwdn'] * frac_lwatm_dpnl
      + out['flux_lwemis_upnl'] * frac_lwupnl_dpnl
      + out['flux_lwemis_dpnl'] * frac_lwdpnl_dpnl
      + out['flux_lwemis_grnd'] * frac_lwgrnd_dpnl
    )
    out['flux_lwabs_grnd'] = (
      out['flux_lwdn'] * frac_lwatm_grnd
      + out['flux_lwemis_upnl'] * frac_lwupnl_grnd
      + out['flux_lwemis_dpnl'] * frac_lwdpnl_grnd
      + out['flux_lwemis_grnd'] * frac_lwgrnd_grnd
    )
    out['flux_lwnet_upnl'] = out['flux_lwabs_upnl'] - out['flux_lwemis_upnl']
    out['flux_lwnet_dpnl'] = out['flux_lwabs_dpnl'] - out['flux_lwemis_dpnl']
    out['flux_lwnet_grnd'] = out['flux_lwabs_grnd'] - out['flux_lwemis_grnd']
    return out

if __name__ == '__main__':
  import math
  model = SolarFarmRadiation()
  # model.panel_area_fraction = 1e-6
  # model.ground_temp = 314.15
  sw = model.SWSolver(720.0, 80.0, -math.cos(math.radians(32.0)))
  lw = model.LWSolver(330.0)
  print('=' * 80)
  print('Total downward solar radiation: {:.2f} W/m^2'.format(sw['flux_swdn']))
  print('Total reflected solar radiation: {:.2f} W/m^2'.format(sw['flux_swup']))
  print('Total absorbed solar radiation by ground: {:.2f} W/m^2'.format(sw['flux_swabs_grnd']))
  print('Total absorbed solar radiation by front panel: {:.2f} W/m^2'.format(sw['flux_swabs_upnl']))
  print('Total absorbed solar radiation by back panel: {:.2f} W/m^2'.format(sw['flux_swabs_dpnl']))
  print('=' * 80)
  print('Total downward LW radiation: {:.2f} W/m^2'.format(lw['flux_lwdn']))
  print('Total emission from ground: {:.2f} W/m^2'.format(lw['flux_lwemis_grnd']))
  print('Total emission from front panel: {:.2f} W/m^2'.format(lw['flux_lwemis_upnl']))
  print('Total emission from back panel: {:.2f} W/m^2'.format(lw['flux_lwemis_dpnl']))
  print('Total upward LW radiation: {:.2f} W/m^2'.format(lw['flux_lwup']))
  print('Total absorbed LW radiation by ground: {:.2f} W/m^2'.format(lw['flux_lwabs_grnd']))
  print('Total absorbed LW radiation by front panel: {:.2f} W/m^2'.format(lw['flux_lwabs_upnl']))
  print('Total absorbed LW radiation by back panel: {:.2f} W/m^2'.format(lw['flux_lwabs_dpnl']))
  print('Total net LW radiation by ground: {:.2f} W/m^2'.format(lw['flux_lwnet_grnd']))
  print('Total net LW radiation by front panel: {:.2f} W/m^2'.format(lw['flux_lwnet_upnl']))
  print('Total net LW radiation by back panel: {:.2f} W/m^2'.format(lw['flux_lwnet_dpnl']))
  print('=' * 80)
  print('Total net energy by ground: {:.2f} W/m^2'.format(sw['flux_swabs_grnd'] + lw['flux_lwnet_grnd']))
  print('Total net energy by front panel: {:.2f} W/m^2'.format(sw['flux_swabs_upnl'] + lw['flux_lwnet_upnl'] - sw['flux_electricity']))
  print('Total net energy by back panel: {:.2f} W/m^2'.format(sw['flux_swabs_dpnl'] + lw['flux_lwnet_dpnl']))
  print('Power Generation: {:.2f} W/m^2'.format(sw['flux_electricity']))
  print('Net surface energy balance: {:.2f} W/m^2'.format(
    sw['flux_swdn'] + lw['flux_lwdn'] - sw['flux_swup'] - lw['flux_lwup']
  ))
  print('=' * 80)