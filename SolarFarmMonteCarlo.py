import math
import random

def emitPhoton2D(xmin:float, xmax:float, ymin:float, ymax:float, direction:int):
  '''
  Emit a photon packet from a random location with constraints on the emission region
  and direction. Returns the location and velocity of the photon.

  Parameters:
    xmin (float): The minimum x-coordinate of the emission region
    xmax (float): The maximum x-coordinate of the emission region
    ymin (float): The minimum y-coordinate of the emission region
    ymax (float): The maximum y-coordinate of the emission region
    direction (int): The direction of the photon emission. 1 for up, -1 for down.

  Returns:
    tuple: A tuple containing the following variables in sequence.
      x (float): The x-coordinate of the photon.
      y (float): The y-coordinate of the photon.
      vx (float): The x-component of the photon velocity.
      vy (float): The y-component of the photon velocity.
  '''

  # sample initial location with uniform distribution
  x = random.uniform(xmin, xmax)
  y = random.uniform(ymin, ymax)

  # random direction (zenith angle with cos^2 distribution)
  mu = math.sqrt(random.uniform(0, 1))
  phi = random.uniform(0, 2*math.pi)

  # determine the y-component of the velocity
  vy = direction * mu

  # determine the x-component of the velocity
  # vx = math.sqrt(1 - mu**2)
  # if (phi > math.pi):
  #   vx = -vx
  vx = math.sqrt(1 - mu**2) * math.cos(phi)

  return (x, y, vx, vy)

def intersectPanel2D(x:float, y:float, vx:float, vy:float,
                     panel_length:float, panel_height:float,
                     scene_length:float, panel_tilt_rad:float):
  '''
  Check if the photon intersects tilted panels.
  The two adjacent panel arrays are assumed to form a parallelogram
  with the following vertices:
    (0, 0),
    (scene_length, 0),
    (scene_length + panel_length * cos(panel_tilt_rad), panel_length * sin(panel_tilt_rad)),
    (panel_length * cos(panel_tilt_rad), panel_length * sin(panel_tilt_rad)).

  Parameters:
  x (float): The x-coordinate of the photon's current position.
  y (float): The y-coordinate of the photon's current position.
  vx (float): The x-component of the photon's velocity vector.
  vy (float): The y-component of the photon's velocity vector.
  panel_length (float): The length of the panel.
  panel_height (float): The height of the panel.
  scene_length (float): The length of the solar farm
                        (flat panel length + space between panel arrays).
  panel_tilt_rad (float): The tilt angle of the panel in radians.

  Returns:
  dict: A dictionary containing the collision status and the destination coordinates.
    - 'collision' (int): The collision status, where
      * 0 indicates no collision
      * 1 indicates a collision with the front side of the panel, and
      * -1 indicates a collision with the back side of the panel.
    - 'dest' (tuple): The destination coordinates (x, y) after the collision.
  '''

  # check if photon is between the panel arrays
  # if not, move them to the boundary of the panel arrays
  # ymin and ymax are the y-coordinates of the bottom and top boundaries
  ymin = 0; ymax = panel_length * math.sin(panel_tilt_rad)
  if (y < ymin and vy > 0):
    # photon is below the panels and moving upwards => bottom boundary
    x += (ymin - y) * vx / vy
    y = ymin
  elif (y > ymax and vy < 0):
    # photon is above the panels and moving downwards => top boundary
    x += (ymax - y) * vx / vy
    y = ymax
  elif ((y < ymin and vy < 0) or (y > ymax and vy > 0)):
    # photon will never reach the panels
    return {'collision': 0, 'dest': (x, y)}
  
  # xmin and xmax are the x-coordinates of the left and right boundaries
  # at the top / bottom of the panel arrays
  if (panel_tilt_rad > 1e-3):
    xmin = y / math.tan(panel_tilt_rad)
  else:
    xmin = 0
  xmax = xmin + scene_length

  # normalize the horizontal coordinate
  # we work within one parallelogram formed by two panel arrays
  # nx is the normalizing factor
  nx = 0
  if (x < xmin):
    nx = math.ceil((xmin - x) / scene_length)
  if (x > xmax):
    nx = math.floor((xmax - x) / scene_length)
  x += scene_length * nx

  # special case: small tilt angle
  if (panel_tilt_rad <= 1e-3):
    if (x < panel_length):
      return {'collision': 1 if vy < 0 else -1, 'dest': (x - scene_length * nx, y)}
    else:
      return {'collision': 0, 'dest': (x - scene_length * nx, y)}
  
  # calculate the final destination as if the panel is absent
  if (vy > 0):
    # photon is moving upwards => top boundary
    ydest = ymax
    xdest = x + (ymax - y) * vx / vy
  else:
    # photon is moving downwards => bottom boundary
    ydest = ymin
    xdest = x + (ymin - y) * vx / vy
  
  # check if the photon intersects the panel by calculating the x-coordinate
  # of the photon's current position.
  # If the photon is still inside the parallelogram, no collision occurs.
  # xmin and xmax are the x-coordinates of the left and right boundaries
  # at the top / bottom of the panel arrays
  xmin = ydest / math.tan(panel_tilt_rad)
  xmax = xmin + scene_length
  if (xdest >= xmin and xdest <= xmax):
    return {'collision': 0, 'dest': (xdest - scene_length * nx, ydest)}
  
  # calcualte the collision point
  # solve the problem: two lines intersect
  #    solar panel: y = tan(panel_tilt_rad) * x
  #    photon path: y = vy / vx * (x - x0) + y0
  if (xdest > xmax):
    # photon hits the right boundary (i.e., front side of the panel)
    # here we make the panel cross (0,0) to simplify the calculation
    x -= scene_length
    d1 = vy * x * math.cos(panel_tilt_rad) - vx * y * math.cos(panel_tilt_rad) 
    d2 = vy * math.cos(panel_tilt_rad) - vx * math.sin(panel_tilt_rad)
    xdest = d1 / d2
    ydest = math.tan(panel_tilt_rad) * xdest
    xdest += scene_length
    return {'collision': 1, 'dest': (xdest - scene_length * nx, ydest)}
  else:
    # photon hits the left boundary (i.e., back side of the panel)
    d1 = vy * x * math.cos(panel_tilt_rad) - vx * y * math.cos(panel_tilt_rad) 
    d2 = vy * math.cos(panel_tilt_rad) - vx * math.sin(panel_tilt_rad)
    xdest = d1 / d2
    ydest = math.tan(panel_tilt_rad) * xdest
    return {'collision': -1, 'dest': (xdest - scene_length * nx, ydest)}
  
def tracePhoton2D(
    x:float, y:float, vx:float, vy:float,
    panel_length:float, panel_height:float,
    scene_length:float, panel_tilt_rad:float,
    albedo_front_panel:float, albedo_back_panel:float, albedo_ground:float
):
  """
  Traces the path of a photon in a 2D solar farm model.

  Args:
    x (float): The x-coordinate of the photon's initial position.
    y (float): The y-coordinate of the photon's initial position.
    vx (float): The x-component of the photon's initial velocity.
    vy (float): The y-component of the photon's initial velocity.
    panel_length (float): The length of the solar panel.
    panel_height (float): The height of the solar panel.
    scene_length (float): The width of the solar farm (flat panel length + space between panel arrays).
    panel_tilt_rad (float): The angle of inclination of the solar panel in radians.
    albedo_front_panel (float): The albedo of the front panel.
    albedo_back_panel (float): The albedo of the back panel.
    albedo_ground (float): The albedo of the ground.

  Returns:
    dict: A dictionary containing the photon's trajectory history and the outcome of the photon's path.
      - 'hist' (list): A list of tuples representing the photon's position and velocity at each step.
      - 'outcome' (int): The outcome of the photon's path. Possible values:
        - 0: Photon reached the sky without any collision.
        - 1: Photon was absorbed by the ground.
        - 2: Photon was absorbed by the front panel.
        - 3: Photon was absorbed by the back panel.
  """
  hist = [(x, y, vx, vy)]

  # y-coordinate of the ground
  h_ground = panel_length * math.sin(panel_tilt_rad) / 2 - panel_height

  while True:
    r = intersectPanel2D(
      x, y, vx, vy, panel_length, panel_height,
      scene_length, panel_tilt_rad
    )
    x = r['dest'][0]; y = r['dest'][1]

    # photon hits the panel
    if (r['collision'] == 0):                   # no collision
      if (vy > 0):                              # upward to the sky
        outcome = 0
        break
      else:
        x = x + (h_ground - y) * vx / vy
        y = h_ground
        if (random.uniform(0, 1) > albedo_ground): # absorbed by the ground
          outcome = 1; vx = 0; vy = 0
          break
        angle_offset = 0                           # reflected by the ground

    # photon hits the front panel
    if (r['collision'] == 1):
      if (random.uniform(0, 1) > albedo_front_panel): # absorbed by the front panel
        outcome = 2; vx = 0; vy = 0
        break
      angle_offset = panel_tilt_rad                   # reflected by the front panel

    # photon hits the back panel
    elif (r['collision'] == -1):
      if (random.uniform(0, 1) > albedo_back_panel): # absorbed by the back panel
        outcome = 3; vx = 0; vy = 0
        break
      angle_offset = math.pi + panel_tilt_rad        # reflected by the back panel

    # determine a new direction
    # apply rotation if the photon comes off from the tilted panel
    _, _, vx, vy = emitPhoton2D(0, 0, 0, 0, 1)
    vx2 = vx * math.cos(angle_offset) - vy * math.sin(angle_offset)
    vy2 = vx * math.sin(angle_offset) + vy * math.cos(angle_offset)
    vx = vx2; vy = vy2
    hist.append((x, y, vx, vy))

    # move the photon away from the boundary a little bit
    x += vx * 1e-3
    y += vy * 1e-3

  hist.append((x, y, vx, vy))
  return {
    'hist': hist,
    'outcome': outcome,
  }

def doDownwardRadiation2D(
  nphotons:int, panel_length:float, panel_height:float,
  scene_length:float, panel_tilt_rad:float,
  albedo_front_panel:float, albedo_back_panel:float, albedo_ground:float,
  mu0:float=None, phi0:float=None, output_hist:bool=False
):
  """
  Simulates the downward radiation on a 2D solar farm and
  calculates the number of photons absorbed by different surfaces.

  Parameters:
  - nphotons (int): The number of photons to simulate.
  - panel_length (float): The length of the solar panel.
  - panel_height (float): The height of the solar panel.
  - scene_length (float): The length of the solar farm scene
                          (flat panel length + space between panel arrays).
  - panel_tilt_rad (float): The tilt angle of the solar panel in radians.
  - albedo_front_panel (float): The albedo (1 - emissivity) of the front surface of the solar panel.
  - albedo_back_panel (float): The albedo (1 - emissivity) of the back surface of the solar panel.
  - albedo_ground (float): The albedo (1 - emissivity) of the ground.
  - mu0 (float, optional): The cosine of the incident angle of the photons. Defaults to None (random).
  - phi0 (float, optional): The azimuthal angle of the incident photons. Defaults to None (random).
  - output_hist (bool, optional): Whether to output the photon history. Defaults to False.

  Returns:
  A dictionary containing the following keys:
  - 'n_upward' (int): The number of photons that go upward to the sky.
  - 'n_ground' (int): The number of photons absorbed by the ground.
  - 'n_front' (int): The number of photons absorbed by the front surface of the panel.
  - 'n_back' (int): The number of photons absorbed by the back surface of the panel.
  - 'hist' (list): A list of photon histories if output_hist is True.
  """
  # Counters
  n_upward = 0; n_ground = 0; n_front = 0; n_back = 0; hist = []

  for _ in range(nphotons):
    # Emit a photon from the top of the solar farm scene
    x, y, vx, vy = emitPhoton2D(
      0, scene_length,
      panel_length*math.sin(panel_tilt_rad)+1e-1,
      panel_length*math.sin(panel_tilt_rad)+1e-1,
      -1
    )

    # Set the incident angle of the photons
    if (mu0 is not None):
      vy = mu0
      vx = math.sqrt(1 - mu0**2)
      if (phi0 is not None):
        vx *= math.cos(phi0)

    # Trace the photon's path
    r = tracePhoton2D(
      x, y, vx, vy,
      panel_length, panel_height, scene_length,
      panel_tilt_rad, albedo_front_panel,
      albedo_back_panel, albedo_ground
    )

    # Update the counters
    if (r['outcome'] == 0):
      n_upward += 1
    elif (r['outcome'] == 1):
      n_ground += 1
    elif (r['outcome'] == 2):
      n_front += 1
    elif (r['outcome'] == 3):
      n_back += 1

    # Append the photon history
    if (output_hist):
      hist.append(r['hist'])

  return {
    'n_upward': n_upward,
    'n_ground': n_ground,
    'n_front': n_front,
    'n_back': n_back,
    'hist': hist,
  }

def doPanelEmission2D(
  nphotons:int, panel_length:float, panel_height:float,
  scene_length:float, panel_tilt_rad:float,
  albedo_front_panel:float, albedo_back_panel:float, albedo_ground:float,
  mu0:float=None, phi0:float=None, emit_up:bool=True, output_hist:bool=False
):
  """
  Simulates the emission of photons from a 2D solar panel and calculates the number of photons absorbed by different surfaces.

  Parameters:
  - nphotons (int): The number of photons to simulate.
  - panel_length (float): The length of the solar panel.
  - panel_height (float): The height of the solar panel.
  - scene_length (float): The length of the solar farm scene (flat panel length + space between panel arrays).
  - panel_tilt_rad (float): The angle of the solar panel with respect to the ground in radians.
  - albedo_front_panel (float): The albedo (1 - emissivity) of the front surface of the solar panel.
  - albedo_back_panel (float): The albedo (1 - emissivity) of the back surface of the solar panel.
  - albedo_ground (float): The albedo (1 - emissivity) of the ground.
  - mu0 (float, optional): The cosine of the incident angle of the photons. Defaults to None (random).
  - phi0 (float, optional): The azimuthal angle of the incident photons. Defaults to None (random).
  - emit_up (bool, optional): Whether to emit photons upward from the front panel or downward from the back panel.
                              Defaults to True (upward from the front panel).
  - output_hist (bool, optional): Whether to output the photon history. Defaults to False.

  Returns:
  A dictionary containing the number of photons absorbed by different surfaces:
  - n_upward (int): The number of photons absorbed by the sky.
  - n_ground (int): The number of photons absorbed by the ground.
  - n_front (int): The number of photons absorbed by the front surface of the panel.
  - n_back (int): The number of photons absorbed by the back surface of the panel.
  - hist (list): A list of photon histories if output_hist is True.
  """
  # Counters
  n_upward = 0; n_ground = 0; n_front = 0; n_back = 0; hist = []

  for _ in range(nphotons):
    # Emit a photon from the front or back panel
    x, y, vx, vy = emitPhoton2D(0, panel_length, 0, 0, 1 if emit_up else -1)

    # Set the emission angle of the photons
    if (mu0 is not None):
      vy = mu0 * (1 if emit_up else -1)
      vx = math.sqrt(1 - mu0**2)
      if (phi0 is not None):
        vx *= math.cos(phi0)

    # Rotate the photon's direction if the panel is tilted
    x2 = x * math.cos(panel_tilt_rad) - y * math.sin(panel_tilt_rad)
    y2 = x * math.sin(panel_tilt_rad) + y * math.cos(panel_tilt_rad)
    vx2 = vx * math.cos(panel_tilt_rad) - vy * math.sin(panel_tilt_rad)
    vy2 = vx * math.sin(panel_tilt_rad) + vy * math.cos(panel_tilt_rad)

    # Move the photon away from the panel
    x = x2 + vx2 * 1e-3; y = y2 + vy2 * 1e-3
    vx = vx2; vy = vy2

    # Trace the photon's path
    r = tracePhoton2D(
      x, y, vx, vy,
      panel_length, panel_height,
      scene_length, panel_tilt_rad,
      albedo_front_panel, albedo_back_panel, albedo_ground
    )

    # Update the counters
    if (r['outcome'] == 0):
      n_upward += 1
    elif (r['outcome'] == 1):
      n_ground += 1
    elif (r['outcome'] == 2):
      n_front += 1
    elif (r['outcome'] == 3):
      n_back += 1

    # Append the photon history
    if (output_hist):
      hist.append(r['hist'])

  return {
    'n_upward': n_upward,
    'n_ground': n_ground,
    'n_front': n_front,
    'n_back': n_back,
    'hist': hist,
  }

def doGroundEmission2D(
  nphotons:int, panel_length:float, panel_height:float,
  scene_length:float, panel_tilt_rad:float,
  albedo_front_panel:float, albedo_back_panel:float, albedo_ground:float,
  mu0:float=None, phi0:float=None,
  emit_xmin:float=None, emit_xmax:float=None,
  output_hist:bool=False
):
  """
  Simulates the emission of photons from the ground in a 2D solar farm and
  calculates the number of photons absorbed by different surfaces.

  Parameters:
  - nphotons (int): The number of photons to simulate.
  - panel_length (float): The length of the solar panel.
  - panel_height (float): The height of the solar panel.
  - scene_length (float): The length of the solar farm scene
                          (flat panel length + space between panel arrays).
  - panel_tilt_rad (float): The angle of the solar panels with respect to the ground in radians.
  - albedo_front_panel (float): The albedo (1 - emissivity) of the front surface of the panel.
  - albedo_back_panel (float): The albedo (1 - emissivity) of the back surface of the panel.
  - albedo_ground (float): The albedo (1 - emissivity) of the ground.
  - mu0 (float, optional): The cosine of the incident angle of the photons. Defaults to None (random).
  - phi0 (float, optional): The azimuthal angle of the incident photons. Defaults to None (random).
  - xmin (float, optional): The minimum x-coordinate of the emission region. Defaults to None (no limit).
  - xmax (float, optional): The maximum x-coordinate of the emission region. Defaults to None (no limit).
  - output_hist (bool, optional): Whether to output the photon history. Defaults to False.

  Returns:
  A dictionary containing the number of photons absorbed by different surfaces:
  - n_upward (int): The number of photons absorbed by the sky.
  - n_ground (int): The number of photons absorbed by the ground.
  - n_front (int): The number of photons absorbed by the front surface of the panel.
  - n_back (int): The number of photons absorbed by the back surface of the panel.
  - hist (list): A list of photon histories if output_hist is True.
  """
  # Calculate the y-coordinate of the ground
  h_ground = panel_length * math.sin(panel_tilt_rad) / 2 - panel_height

  # Counters
  n_upward = 0; n_ground = 0; n_front = 0; n_back = 0; hist = []

  for _ in range(nphotons):
    # Emit a photon from the ground
    x, y, vx, vy = emitPhoton2D(
      emit_xmin if emit_xmin is not None else 0,
      emit_xmax if emit_xmax is not None else scene_length,
      h_ground, h_ground, 1
    )

    # Set the emission angle of the photons
    if (mu0 is not None):
      vy = mu0
      vx = math.sqrt(1 - mu0**2)
      if (phi0 is not None):
        vx *= math.cos(phi0)

    # Trace the photon's path
    r = tracePhoton2D(
      x, y, vx, vy,
      panel_length, panel_height, scene_length, panel_tilt_rad,
      albedo_front_panel, albedo_back_panel, albedo_ground
    )

    # Update the counters
    if (r['outcome'] == 0):
      n_upward += 1
    elif (r['outcome'] == 1):
      n_ground += 1
    elif (r['outcome'] == 2):
      n_front += 1
    elif (r['outcome'] == 3):
      n_back += 1

    # Append the photon history
    if (output_hist):
      hist.append(r['hist'])

  return {
    'n_upward': n_upward,
    'n_ground': n_ground,
    'n_front': n_front,
    'n_back': n_back,
    'hist': hist,
  }

def getShade2D(
  panel_length:float, panel_height:float,
  scene_length:float, panel_tilt_rad:float,
  mu:float, phi:float
):
  """
  Calculate the shaded regions in a 2D solar farm.

  Args:
    panel_length (float): Length of the solar panel.
    panel_height (float): Height of the solar panel.
    scene_length (float): Length of the solar farm scene
                          (flat panel length + space between panel arrays).
    panel_tilt_rad (float): Angle of inclination of the solar panel in radians.
    mu (float): Cosine of the solar zenith angle of the photons.
    phi (float): Solar azimuth angle in radians.

  Returns:
    list: A list of tuples representing the shaded regions.

  """
  # Calculate the y-coordinate of the ground
  h_ground = panel_length * math.sin(panel_tilt_rad) / 2 - panel_height

  # Calculate the coordinates of the solar panel at the center of the scene
  x1, y1 = 0, 0
  x2, y2 = panel_length * math.cos(panel_tilt_rad), panel_length * math.sin(panel_tilt_rad)

  # Direct sunlight direction
  vy = -abs(mu)
  vx = math.sqrt(1 - vy**2) * math.cos(phi)

  # Calculate the intersection points of the direct sunlight with the ground
  # from the tip of the solar panel
  y3 = h_ground; y4 = h_ground
  if (abs(vy) > 1e-3):
    x3 = x1 + (y3 - y1) * vx / vy
    x4 = x2 + (y4 - y2) * vx / vy
  else:
    x3 = x1; x4 = x2

  # Check if the shadow covers the entire scene length (solar panel and space between arrays)
  if (x4 - x3 < scene_length):
    # Normalize the x-coordinates
    x3 = x3 % scene_length; x4 = x4 % scene_length
    if (x3 > x4):
      # The shadow is fragmented
      return [(0, x4), (x3, scene_length)]
    else:
      # The shadow is continuous
      return [(x3, x4)]
  else:
    # The shadow covers the entire scene length
    return [(0, scene_length)]
  
def plotPanels2D(
  ax, x_panel_centers:list, y_panel_centers:list,
  panel_length:float, panel_height:float, panel_tilt_rad:float
):
  """
  Plot a panel on the given axes.

  Parameters:
  - ax: The axes object to plot on.
  - x_panel_centers: A list of x-coordinates for the panel centers.
  - y_panel_centers: A list of y-coordinates for the panel centers.
  - panel_length: The length of the panel.
  - panel_height: The height of the panel.
  - panel_tilt_rad: The angle of the panel in radians.

  Returns:
  - ax: The matplotlib axes object.
  """
  # Add a small offset to distinguish the front and back panels
  eps = 1e-2

  for xc, yc in zip(x_panel_centers, y_panel_centers):
    x1 = xc - panel_length * math.cos(panel_tilt_rad) / 2
    x2 = xc + panel_length * math.cos(panel_tilt_rad) / 2
    y1 = yc - panel_length * math.sin(panel_tilt_rad) / 2
    y2 = yc + panel_length * math.sin(panel_tilt_rad) / 2

    # Panel surface (front panel red, back panel brown)
    ax.plot([x1, x2], [y1+eps, y2+eps], color='red', ls='-')
    ax.plot([x1, x2], [y1-eps, y2-eps], color='brown', ls='-')

    # Panel support at the center of panels
    ax.plot([xc, xc], [yc-panel_height, yc], color='k', ls='-')

  return ax

def plotSingleRayTracing2D(
  trajectory_history:list, panel_length:float, panel_height:float, 
  scene_length:float, panel_tilt_rad:float, ax=None
):
  """
  Plots the scene based on the given parameters.

  Parameters:
  - trajectory_history (list): A list of tuples representing the history of positions and velocities.
  - panel_length (float): The length of the panels.
  - panel_height (float): The height of the panels.
  - scene_length (float): The length of the solar farm scene
                          (flat panel length + space between panel arrays).
  - panel_tilt_rad (float): The angle of the panels in radians.
  - ax: The axes object to plot on. If None, a new figure will be created.

  Returns:
  - ax: The matplotlib axes object.
  """
  import matplotlib.pyplot as plt
  import numpy as np

  if (ax is None):
    fig, ax = plt.subplots()

  # scene boundary
  xmax = 0; xmin = 0

  # panel centers
  xc = panel_length*math.cos(panel_tilt_rad)/2
  yc = panel_length*math.sin(panel_tilt_rad)/2

  for j in range(len(trajectory_history)-1):
    x, y, vx, vy = trajectory_history[j]
    xdest, ydest, _, _ = trajectory_history[j+1]
    xmax = max(xmax, x, xdest)
    xmin = min(xmin, x, xdest)

    # source
    ax.plot([x], [y], 'bo')

    # ray
    ax.quiver([x], [y], [vx], [vy], color='purple',
              angles='xy', scale_units='xy', scale=panel_height * 5)
    ax.plot([x, xdest], [y, ydest], 'b-')

    # destination
    ax.plot([xdest], [ydest], 'b*')

  # solar panels - as many as necessary
  n1 = math.floor(xmin / scene_length)
  n2 = math.ceil(xmax / scene_length) + 1
  xc_list = np.arange(xc + n1 * scene_length, xc + n2 * scene_length, scene_length)
  yc_list = [yc] * len(xc_list)
  plotPanels2D(ax, xc_list, yc_list, panel_length, panel_height, panel_tilt_rad)

  # parallelogram formed by panel arrays
  ax.axhline(y=0, color='gray', ls='--')
  ax.axhline(y=yc * 2, color='gray', ls='--')

  # ground
  ax.axhline(y=yc - panel_height, color='k', ls='-', lw=2)

  # plot decoration
  ax.set_aspect('equal')

  return ax

def plotDirectSolar2D(
  nphotons:int, panel_length:float, panel_height:float,
  scene_length:float, panel_tilt_rad:float,
  mu:float, phi:float, ax=None
):
  """
  Plot the direct solar radiation on the solar farm.

  Args:
    panel_length (float): Length of the solar panel.
    panel_height (float): Height of the solar panel.
    scene_length (float): Length of the solar farm scene
                          (flat panel length + space between panel arrays).
    panel_tilt_rad (float): Angle of inclination of the solar panel in radians.
    mu (float): Cosine of the solar zenith angle of the photons.
    phi (float): Solar azimuth angle in radians.
    ax: The axes object to plot on. If None, a new figure will be created.

  Returns:
    fig: The matplotlib figure object.
    ax: The matplotlib axes object.
  """
  import matplotlib.pyplot as plt
  import numpy as np

  histlist = []
  for i in range(nphotons):
    # emit photons uniformly across three solar panel arrays
    x, y, vx, vy = emitPhoton2D(
      3 * scene_length * i / nphotons, 3 * scene_length * i / nphotons,
      panel_length*math.sin(panel_tilt_rad)+1e-1, panel_length*math.sin(panel_tilt_rad)+1e-1, -1
    )

    # set the incident angle of the photons
    if (mu is not None):
      vy = mu
      vx = math.sqrt(1 - mu**2)
      if (phi is not None):
        vx *= math.cos(phi)

    # albedo set to 0 to stop reflection
    r = tracePhoton2D(
      x, y, vx, vy,
      panel_length, panel_height,
      scene_length, panel_tilt_rad,
      0.0, 0.0, 0.0
    )

    # store the photon travel history
    histlist.append(r['hist'][:2])

  if (ax is None):
    fig, ax = plt.subplots()

  # scene boundary
  xmax = 0; xmin = 0

  # panel centers
  xc = panel_length*math.cos(panel_tilt_rad)/2
  yc = panel_length*math.sin(panel_tilt_rad)/2

  for hist in histlist:
    x, y, vx, vy = hist[0]
    xdest, ydest, _, _ = hist[1]
    xmax = max(xmax, x, xdest)
    xmin = min(xmin, x, xdest)

    # ray
    ax.plot([x, xdest], [y, ydest], 'b-')

  # solar panels - as many as necessary
  n1 = math.floor(xmin / scene_length)
  n2 = math.ceil(xmax / scene_length) + 1
  xc_list = np.arange(xc + n1 * scene_length, xc + n2 * scene_length, scene_length)
  yc_list = [yc] * len(xc_list)
  plotPanels2D(ax, xc_list, yc_list, panel_length, panel_height, panel_tilt_rad)

  # parallelogram formed by panel arrays
  ax.axhline(y=0, color='gray', ls='--')
  ax.axhline(y=yc * 2, color='gray', ls='--')

  # ground
  ax.axhline(y=yc - panel_height, color='k', ls='-', lw=2)

  # plot decoration
  # ax.set_aspect('equal')

  return ax

# def emitPhoton3D(
#   xmin:float, xmax:float, ymin:float, ymax:float,
#   zmin:float, zmax:float, direction:int
# ):
#   '''
#   Emit a photon from the specified location and direction
#   Returns the location and velocity of the photon
#   direction = 1: up, -1: down
#   '''

#   # sample initial location
#   x = random.uniform(xmin, xmax)
#   y = random.uniform(ymin, ymax)
#   z = random.uniform(zmin, zmax)

#   # random direction (zenith angle with cos^2 distribution)
#   mu  = math.sqrt(random.uniform(0, 1))
#   phi = random.uniform(0, 2*math.pi)
#   vz = direction * mu
#   vx = math.sqrt(1 - mu**2) * math.cos(phi)
#   vy = math.sqrt(1 - mu**2) * math.sin(phi)

#   return (x, y, z, vx, vy, vz)
  
if __name__=='__main__':
  panel_tilt_rad = math.radians(20)
  panel_length = 1
  panel_height = 1
  scene_length = 10
  albedo_front_panel = 0.1
  albedo_back_panel = 0.6
  albedo_ground = 0.3
  nphotons = int(1e6)
  # nphotons = 10
  mu0 = -1.0
  output_hist = False
  
  r = doDownwardRadiation2D(
    nphotons, panel_length, panel_height, scene_length, panel_tilt_rad,
    albedo_front_panel, albedo_back_panel, albedo_ground,
    mu0=mu0, output_hist=output_hist
  )
  n_upward = r['n_upward']
  n_ground = r['n_ground']
  n_front = r['n_front']
  n_back = r['n_back']

  print(f'Downward radiation on a 2D solar farm')
  print(f'n_total = {nphotons}')
  print(f'n_upward = {n_upward} ({n_upward/nphotons*100:.2f}%)')
  print(f'n_ground = {n_ground} ({n_ground/nphotons*100:.2f}%)')
  print(f'n_front = {n_front} ({n_front/nphotons*100:.2f}%)')
  print(f'n_back = {n_back} ({n_back/nphotons*100:.2f}%)')