from math import inf
from collections import deque
'''
Ford-Fulkerson implementation and general network flow solver base class. Implemented based
on William Fiset youtube video: https://www.youtube.com/watch?v=LdOnanfc5TM&list=PLDV1Zeh2NRsDj3NzHbbFIC58etjZhiGcG&index=1
'''

class Edge:
    def __init__(self, start: int, to: int, capacity: float) -> None:
        '''
        Simple class for directed edges with a capacity associated to them.
        :param start: Starting index of the edge.
        :param to: Ending index of the edge.
        :param capacity: Capacity of the edge.
        '''
        self.start = start
        self.to = to
        self.capacity = capacity
        self.flow = 0

    def isResidual(self) -> bool:
        return self.capacity == 0

    def remainingCapacity(self) -> float:
        return self.capacity - self.flow

    def augment(self, bottleneck: float) -> None:
        '''
        After finding an augmenting path, the edge (and its residual edge) update by the bottleneck value of the path.
        :param bottleneck: The bottleneck value found on the path.
        :return: None
        '''
        self.flow += bottleneck

        '''
        Note that residual is not created in the Edge initialization, rather it 
        is created when adding the edge to the solver graph.
        '''
        try:
            self.residual.flow -= bottleneck
        except AttributeError:
            print('The residual edge was not found. This is likely due to running the augment method outside the solver class.')

    def toString(self):
        return f'Edge {self.start} -> {self.to} | flow = {self.flow:>5} | capacity = {self.capacity:>5} | is residual: {self.isResidual()}'






class SolverBase:
    def __init__(self, n, s, t):
        '''
        This underlying solver can be extended to use different methods for finding augmenting paths. In particular,
        the solver method is left unimplemented to allow the user to choose a method for finding the paths.
        :param n: The number of edges in the network, indexed by integers from 0 to n-1.
        :param s: The index of the starting node.
        :param t: The index of the target node.
        '''
        self.MaxFlow = 0
        self.n = n
        self.s = s
        self.t = t
        self.initializeEmptyFlowGraph()
        self.visited = [0 for _ in range(n)]
        self.visitedToken = 1
        self.solved = False

    def initializeEmptyFlowGraph(self):
        self.graph = [[] for _ in range(self.n)]

    def addEdge(self, start, to, capacity):
        '''
        Adds an edge to the graph. Creates its residual edge and adds that as well.
        :param start:
        :param to:
        :param capacity:
        :return:
        '''
        E1 = Edge(start, to, capacity)
        E2 = Edge(to, start, 0)
        E1.residual = E2
        E2.residual = E1
        self.graph[start].append(E1)
        self.graph[to].append(E2)

    def visit(self, i):
        '''
        Updates the node with a visited token, denoting which augmenting path it was last a part of.
        It is used to check if there is a cycle when creating the augmenting paths.
        :param i:
        :return:
        '''
        self.visited[i] = self.visitedToken

    def getGraph(self):
        self.execute()
        return self.graph

    def getMaxFlow(self):
        self.execute()
        return self.MaxFlow

    def execute(self):
        if self.solved:
            return
        self.solved = True
        self.solve()

    def getSolution(self):
        self.execute()
        for node in self.graph:
            for edge in node:
                print(edge.toString())

    def solve(self):
        #This will be overriden in the subclass that extends this class.
        pass

    def markNodesAsUnvisited(self):
        self.visitedToken += 1




class FordFulkersonDFSSolver(SolverBase):
    '''
    Extends the SolverBase method to use a depth first search method for finding the augmenting paths.
    A link for depth first search can be found here: https://www.youtube.com/watch?v=AfSk24UTFS8&t=2407s
    '''
    def __init__(self, n: int, s: int, t: int):
        super().__init__(n,s,t)

    def solve(self) -> None:
        '''
        Repeatedly finds augmenting paths until there are none left (this is when the flow f is equal to 0)
        :return: None
        '''
        f = -1
        while f != 0:
            f = self.dfs(self.s, inf)
            self.visitedToken += 1
            self.MaxFlow += f


    def dfs(self, node: int, flow: float):
        '''
        A depth first search along valid paths,
        :param node:
        :param flow:
        :return: the bottleneck value along the augmenting path is returned
        '''

        #if at sink node, return the flow along the augmenting path
        if node == self.t:
            return flow

        #
        self.visit(node)
        for edge in self.graph[node]:
            if edge.remainingCapacity() > 0 and self.visited[edge.to] != self.visitedToken:
                bottleNeck = self.dfs(edge.to, min(flow, edge.remainingCapacity()))

                if bottleNeck > 0:
                    edge.augment(bottleNeck)
                    return bottleNeck
        return 0

class EdmondsKarpSolver(SolverBase):
    def __init__(self, n, s, t):
        super().__init__(n, s, t)

    def solve(self):
        f = -1
        while f != 0:
            self.markNodesAsUnvisited()
            f = self.bfs()
            self.MaxFlow += f


    def bfs(self):
        q = deque()
        self.visit(self.s)
        q.append(self.s)

        prev = [None for _ in range(self.n)]
        while q:
            node = q.popleft()
            if node == self.t:
                break

            for edge in self.graph[node]:
                cap = edge.remainingCapacity()
                if cap > 0 and (not self.visited[edge.to] == self.visitedToken):
                    self.visit(edge.to)
                    prev[edge.to] = edge
                    q.append(edge.to)

        if prev[self.t] is None:
            return 0

        bottleNeck = inf
        edge = prev[self.t]
        while edge is not None:
            bottleNeck = min(bottleNeck, edge.remainingCapacity())
            edge = prev[edge.start]

        edge = prev[t]
        while edge is not None:
            edge.augment(bottleNeck)
            edge = prev[edge.start]

        return bottleNeck

















n = 7
s, t = 0, 6
solver = EdmondsKarpSolver(n, s, t)


solver.addEdge(s, 1, 9)
solver.addEdge(s, 3, 12)
solver.addEdge(1, 2, 6)
solver.addEdge(1, 3, 9)
solver.addEdge(1, 4, 4)
solver.addEdge(1, 3, 5)
solver.addEdge(2, 3, 2)
solver.addEdge(2, 4, 6)
solver.addEdge(2, 5, 3)
solver.addEdge(3, 2, 4)
solver.addEdge(3, t, 7)
solver.addEdge(4, 5, 2)
solver.addEdge(4, t, 8)
solver.addEdge(5, t, 5)


print(solver.getMaxFlow())
solver.getSolution()










