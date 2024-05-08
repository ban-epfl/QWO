import numpy as np
import random


class Order:
    def __init__(self, n, pi, theta):
      self.order = list(pi)
      self.theta = theta
      if(self.order == list(np.arange(n))):
        random.shuffle(self.order)
      self.parents = {}
      self.theta_sum = {}
      self.edges = 0
      self.total_theta_sum = 0

      for i in range(n):
        y = self.order[i]
        self.parents[y] = []
        self.theta_sum[y] = 0

    def get(self, i):
        return self.order[i]

    def set(self, i, y):
        self.order[i] = y

    def index(self, y):
        return self.order.index(y)

    def insert(self, i, y):
        self.order.insert(i, y)

    def pop(self, i=-1):
        return self.order.pop(i)

    def get_parents(self, y):
        return self.parents[y]

    def set_parents(self, y, y_parents):
        self.parents[y] = y_parents

    def get_theta_sum(self, y):
        return self.theta_sum[y]

    def set_theta_sum(self, y, y_theta_sum):
        self.theta_sum[y] = y_theta_sum

    def get_edges(self):
        return self.edges

    def set_edges(self, edges):
        self.edges = edges

    def get_total_theta_sum(self):
        return self.total_theta_sum

    def set_total_theta_sum(self, total_theta_sum):
      self.total_theta_sum = total_theta_sum

    def update_edges(self):
      edges = 0
      for i in self.order:
        edges += len(self.parents[i])
      self.edges = edges

    def update_theta_sums(self, i, j):
      for t in range(j, i+1):
        y = self.order[i]
        theta_sum = 0
        for p in self.get_parents(y):
          theta_sum += abs(self.theta[p, y])
        self.set_theta_sum(y, theta_sum)
      total_theta_sum = 0
      for i in range(len(self.order)):
        total_theta_sum += self.theta_sum[i]
      self.total_theta_sum = total_theta_sum

    def bump_edges(self, bump):
        self.edges += bump

    def len(self):
        return len(self.order)