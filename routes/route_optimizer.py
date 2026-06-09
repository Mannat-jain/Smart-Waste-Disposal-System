"""Route Optimizer — Dijkstra + Genetic Algorithm."""
import heapq, math, random, logging
from dataclasses import dataclass
logger = logging.getLogger(__name__)

@dataclass
class Bin:
    bin_id: str; lat: float; lon: float; fill_level: float; zone: str
    @property
    def needs_collection(self): return self.fill_level >= 60.0

def haversine_km(a, b):
    R=6371.0; dlat=math.radians(b.lat-a.lat); dlon=math.radians(b.lon-a.lon)
    h=math.sin(dlat/2)**2+math.cos(math.radians(a.lat))*math.cos(math.radians(b.lat))*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

class Graph:
    def __init__(self, bins):
        self.id_to_bin = {b.bin_id:b for b in bins}
        self.adj = {b.bin_id:[] for b in bins}
        for i,a in enumerate(bins):
            for b in bins[i+1:]:
                d=haversine_km(a,b)
                self.adj[a.bin_id].append((d,b.bin_id))
                self.adj[b.bin_id].append((d,a.bin_id))

    def dijkstra(self, start):
        dist={bid:float("inf") for bid in self.id_to_bin}; dist[start]=0.0
        heap=[(0.0,start)]
        while heap:
            d,u=heapq.heappop(heap)
            if d>dist[u]: continue
            for cost,v in self.adj[u]:
                if d+cost<dist[v]: dist[v]=d+cost; heapq.heappush(heap,(dist[v],v))
        return dist

class GeneticOptimizer:
    def __init__(self,graph,stops,depot,pop_size=30,generations=50,mutation_rate=0.15):
        self.graph=graph; self.stops=stops; self.depot=depot
        self.pop_size=pop_size; self.generations=generations; self.mr=mutation_rate

    def _cost(self, order):
        full=[self.depot]+order+[self.depot]
        return sum(haversine_km(self.graph.id_to_bin[full[i]],self.graph.id_to_bin[full[i+1]])
                   for i in range(len(full)-1))

    def _cx(self,p1,p2):
        n=len(p1); a,b=sorted(random.sample(range(n),2))
        child=[None]*n; child[a:b]=p1[a:b]
        fill=[g for g in p2 if g not in child]; j=0
        for i in range(n):
            if child[i] is None: child[i]=fill[j]; j+=1
        return child

    def run(self):
        if not self.stops: return [],0.0
        pop=[random.sample(self.stops,len(self.stops)) for _ in range(self.pop_size)]
        best=min(pop,key=self._cost); best_cost=self._cost(best)
        for _ in range(self.generations):
            new=[min(random.sample(pop,min(5,len(pop))),key=self._cost)
                 for _ in range(self.pop_size)]
            children=[]
            for i in range(0,len(new)-1,2):
                for p in [self._cx(new[i],new[i+1]),self._cx(new[i+1],new[i])]:
                    r=p[:];
                    if len(r)>=2 and random.random()<self.mr:
                        a,b=random.sample(range(len(r)),2); r[a],r[b]=r[b],r[a]
                    children.append(r)
            pop=children or new
            gb=min(pop,key=self._cost); gc=self._cost(gb)
            if gc<best_cost: best,best_cost=gb,gc
        return best,round(best_cost,2)

class RouteOptimizer:
    def __init__(self,bins,depot):
        self.bins=bins; self.depot=depot
        self.graph=Graph([depot]+bins)

    def optimize(self,threshold=60.0):
        eligible=[b for b in self.bins if b.fill_level>=threshold]
        stop_ids=[b.bin_id for b in eligible]
        greedy=sorted(stop_ids,key=lambda bid:self.graph.id_to_bin[bid].fill_level,reverse=True)
        def cost(order):
            full=[self.depot.bin_id]+order+[self.depot.bin_id]
            return sum(haversine_km(self.graph.id_to_bin[full[i]],self.graph.id_to_bin[full[i+1]])
                       for i in range(len(full)-1))
        baseline_km=cost(greedy)
        best_route,best_km=GeneticOptimizer(self.graph,stop_ids,self.depot.bin_id).run()
        savings=round((1-best_km/baseline_km)*100,1) if baseline_km else 0.0
        return {"route":[self.depot.bin_id]+best_route+[self.depot.bin_id],
                "total_km":best_km,"baseline_km":round(baseline_km,2),
                "savings_pct":savings,"eligible_bins":stop_ids}

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    depot=Bin("DEPOT",51.500,-0.120,0,"depot")
    bins=[Bin("bin_A",51.507,-0.127,87,"A"),Bin("bin_B",51.510,-0.130,41,"B"),
          Bin("bin_C",51.514,-0.125,73,"C"),Bin("bin_D",51.518,-0.118,91,"D"),
          Bin("bin_E",51.505,-0.115,28,"E"),Bin("bin_F",51.512,-0.135,62,"F")]
    print(RouteOptimizer(bins,depot).optimize())
