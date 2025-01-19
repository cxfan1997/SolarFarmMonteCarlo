from math import floor, sin, cos, acos, pi

def SolarAngles(lat:float, lon:float, declin:float, jday:float):
  """
  Calculate solar zenith and azimuth angles.

  Parameters:
  lat (float): Latitude in radians.
  lon (float): Longitude in radians.
  declin (float): Solar declination angle in radians.
  jday (float): Julian day. A day spans from 0.0 to 1.0. A year ranges from 0.0 to 365.0.

  Returns:
  zenith, azimuth (tuple of size 2): A tuple containing the solar zenith angle and solar azimuth angle in degrees.
  """

  # Calculate solar zenith angle
  hra = (jday-floor(jday)) * 2.0 * pi + lon
  coszen = sin(lat) * sin(declin) - cos(lat) * cos(declin) * cos(hra)
  coszen = min(1.0, max(-1.0, coszen))
  zenith = acos(coszen) * 180.0 / pi

  # Calculate solar azimuth angle
  cosaz = (sin(declin) * cos(lat) +
           cos(declin) * sin(lat) * cos(hra))
  if zenith > 0.0 and zenith < 180.0:
    cosaz /= sin(zenith * pi / 180.0)
  else:
    cosaz = 0.0
  cosaz = min(1.0, max(-1.0, cosaz))
  azimuth = acos(cosaz) * 180.0 / pi
  if (hra+2*pi) % (2*pi) > pi: azimuth = 360.0 - azimuth

  return zenith, azimuth