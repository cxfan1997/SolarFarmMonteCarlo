from time import sleep

MAX_PRGBAR_WIDTH   = 60

def printMessage(message:str):
  """Print a message to stdout.
  
  Args:
    message (str): Message to print.
  """
  length = len(message)
  if length > MAX_PRGBAR_WIDTH:
    print(message)
  else:
    print(message + ' ' * (MAX_PRGBAR_WIDTH - length))

def printProgressBar(i, n):
  """Print a progress bar to stdout.

  Args:
    i (int): Current iteration.
    n (int): Total number of iterations.
  """
  num_width = len(str(n))
  frac = i / n
  prg_width = MAX_PRGBAR_WIDTH - 2 * num_width - 11
  n1 = int(frac * prg_width)
  n2 = prg_width - n1
  s1 = '{{:{}d}}/{{:{}d}}'.format(num_width, num_width).format(i, n)
  s2 = '[{}{}] {:5.1f}%'.format('#' * n1, '-' * n2, frac * 100)
  print('{} {}'.format(s1, s2), end='\r')

def waitResults(results:list):
  """Wait for all processes in a pool to finish.

  Args:
    results (list): List of results from a multiprocessing pool.
  """
  count = 0
  n = len(results)
  while count < n:
    count = sum([results[i].ready() for i in range(n)])
    printProgressBar(count, n)
    sleep(1)
  print('')