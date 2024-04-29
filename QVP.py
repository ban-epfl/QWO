import numpy as np


def QVP_grasp(V, depth = 3, parameters = None, pi=None):

  rtime = time.perf_counter()
  n = V.shape[0]
  theta = V @ V.T
  blanket_sizes = (np.array(supp(V @ V.T))).sum(axis = 1)
  pi = initial_order(V)
  order = Order(n, pi=pi, theta=theta)
  Q = V[order.order]
  Q = build_Q(Q, V, n-1, 0, order)

  for i in range(n):
    y = order.get(i)
    y_parents = find_parents(y, Q, V, order)
    order.set_parents(y, y_parents)
    order.bump_edges(len(y_parents)) 

  while dfs(Q, V, depth - 1, score, set(), [], order, history_sparsity, blanket_sizes):
    sys.stdout.write("\rGRaSP edge count: %i    " % order.get_edges())
    sys.stdout.flush()

  rtime = time.perf_counter() - rtime

  sys.stdout.write("\nGRaSP completed in: %.2fs \n" % rtime)
  sys.stdout.flush()

  learned_B = np.zeros((n, n))
  for i in range(n):
    for j in order.get_parents(i):
      learned_B[j][i] = 1

  return (learned_B).T

def dfs(Q, V, depth: int, score, flipped: set, history: List[set], order):

  cache = [{}, {}, {}, 0, 0]

  indices = list(range(order.len()))
  random.shuffle(indices)

  for i in indices:
    y = order.get(i)
    y_parents = order.get_parents(y)
    random.shuffle(y_parents)

    for x in y_parents:
      covered = set([x] + order.get_parents(x)) == set(y_parents)

      if len(history) > 0 and not covered:
        continue

      j = order.index(x)

      for k in range(j, i + 1):
        z = order.get(k)
        cache[0][k] = z
        cache[1][k] = order.get_parents(z)[:]
        cache[2][k] = order.get_theta_sum(z)
      cache[3] = order.get_edges()
      cache[4] = order.get_total_theta_sum()

      tuck(i, j, order)
      score_bump = update(Q, V, i, j, score, order, cache)

      # because things that should be zero sometimes are not
      if score_bump > 0:
        return True

      # ibid
      if score_bump > 1e-3:
        flipped = flipped ^ set([
                        tuple(sorted([x, z]))
                        for z in order.get_parents(x)
                        if order.index(z) < i
                    ])

        if len(flipped) > 0 and flipped not in history:
          history.append(flipped)
          if depth > 0 and dfs(Q, V, depth - 1, flipped, history, order):
            return True
          del history[-1]

      for k in range(j, i + 1):
        z = cache[0][k]
        order.set(k, z)
        order.set_parents(z, cache[1][k])
        order.set_theta_sum(z, cache[2][k])
      order.set_edges(cache[3])
      order.set_total_theta_sum(cache[4])
      Q = build_Q(Q, V, i, j, order)

  return False

def QVP_Hill_Climbing(V, dist_limit, pi=None):

  rtime = time.perf_counter()
  n = V.shape[0]
  theta = V @ V.T
  pi = initial_order(V)
  order = Order(n, pi=pi, theta=theta)
  Q = V[order.order]
  Q = build_Q(Q, V, n-1, 0, order)

  for i in range(n):
    y = order.get(i)
    y_parents = find_parents(y, Q, V, order)
    order.set_parents(y, y_parents)
    order.bump_edges(len(y_parents))

  while score_increase(Q, V, order, dist_limit):
    sys.stdout.write("\rGRaSP edge count: %i    " % order.get_edges())
    sys.stdout.flush()

  rtime = time.perf_counter() - rtime

  sys.stdout.write("\nGRaSP completed in: %.2fs \n" % rtime)
  sys.stdout.flush()

  learned_B = np.zeros((n, n))
  for i in range(n):
    for j in order.get_parents(i):
      learned_B[j][i] = 1

  return (learned_B).T

def score_increase(Q, V, order, dist_limit):

  cache = [{}, {}, {}, 0, 0]
  num_variables = order.len()
  for i in range(num_variables):
    end = min(num_variables - i, dist_limit+1)
    for j in range(1, end):
      for k in range(i, i+j+1):
        z = order.get(k)
        cache[0][k] = z
        cache[1][k] = order.get_parents(z)[:]
        cache[2][k] = order.get_theta_sum(z)
      cache[3] = order.get_edges()
      cache[4] = order.get_total_theta_sum()

      order.swap(i, i+j)
      score_bump = update(Q, V, i, j, order, cache)
      print(order.order, i, i+j)

      # because things that should be zero sometimes are not
      if score_bump > 0:
        return True

      for k in range(i, i+j+1):
        z = cache[0][k]
        order.set(k, z)
        order.set_parents(z, cache[1][k])
        order.set_theta_sum(z, cache[2][k])
      order.set_edges(cache[3])
      order.set_total_theta_sum(cache[4])
      Q = build_Q(Q, V, i+j, i, order)

  return False

def update(Q:np.ndarray, V:np.ndarray, i: int, j: int, score, order, cache):

  Q = build_Q(Q, V, i, j, order)
  for k in range(j, i + 1):
    y = order.get(k)
    y_parents = find_parents(y, Q, V, order)
    order.set_parents(y, y_parents) 

  order.update_edges()

  old_edge_count = cache[3]
  old_score = - old_edge_count
  new_score = - order.get_edges()
  return new_score - old_score 

def build_Q(Q, V, i, j, order):
  # Construct again the indexes of Q between i and j (including both)
  out = copy.deepcopy(Q)
  for t in range(i, j-1, -1):
    out[t] = V[order.get(t)]
    for s in range(t+1, len(V)):
      out[t] = (out[t] - vector_projection(V[order.get(t)], out[s]))
    out[t] = out[t]/np.linalg.norm(out[t], 2)
  return out

def tuck(i: int, j: int, order):
  ancestors = []
  get_ancestors(order.get(i), ancestors, order)
  shift = 0
  for k in range(j + 1, i + 1):
    if order.get(k) in ancestors:
      order.insert(j + shift, order.pop(k))
      shift += 1

def get_ancestors(y: int, ancestors: List[int], order):
  ancestors.append(y)
  for x in order.get_parents(y):
    if x not in ancestors:
      get_ancestors(x, ancestors, order) 

def find_parents(i, Q, V, order):
  ind = order.index(i)
  if ind == 0:
    return []
  parents = []
  for j in range(ind):
    if not orthogonal_check(Q[ind], V[order.get(j)]):
      parents.append(order.get(j))
  return parents

def orthogonal_check(u, v):
  return abs(np.dot(u, v)) < THRESHOLD

def find_Fisher_threshold(N, n):
  alpha = 2 / n**2
  phi_inv = norm.ppf(1 - alpha/2, loc=0, scale=1)/np.sqrt(N - n - 1)
  threshold = 1 - (2 / (1 + np.exp(2 * phi_inv)))
  return threshold

def initial_order(V, method = "size of markov blanket"):
  if(method == "size of markov blanket"):
    x = (np.array(supp(V @ V.T))).sum(axis = 1)
    return np.argsort(x)[::-1]

def supp(A):
  return [[1 if abs(x)>THRESHOLD else 0 for x in row] for row in A]
