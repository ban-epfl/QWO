# Description: This file contains the Order class which is used to represent the order of the variables during the search process.

class Order:
    def __init__(self, n, pi):
      self.order = list(pi)
      self.parents = {}
      self.edges = 0

      for i in range(n):
        y = self.order[i]
        self.parents[y] = []

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

    def get_edges(self):
        return self.edges

    def set_edges(self, edges):
        self.edges = edges

    def update_edges(self):
      edges = 0
      for i in self.order:
        edges += len(self.parents[i])
      self.edges = edges

    def bump_edges(self, bump):
        self.edges += bump

    def len(self):
        return len(self.order)
    
    def swap(self, i, j):
      t = self.order[i]
      self.order[i] = self.order[j]
      self.order[j] = t