import math
import numpy as np

class Task:
    def __init__(self, data):
        self.id = data['id']
        self.lat = data['lat_app']
        self.lng = data['lng_app']
        self.pub_time = data['appearance_time']
        # Valid time e-p. Default 40s from table?
        # Data might have finishing_time but that's historical.
        # Paper says "Valid time of tasks e-p (s) ... 40".
        self.valid_time = 40
        self.exp_time = self.pub_time + self.valid_time
        self.value = data.get('payment', 1.0)

class Worker:
    def __init__(self, data):
        self.id = data['id']
        self.lat = data['lat']
        self.lng = data['lng']
        self.radius = data['radius'] # Reachable distance
        # Paper says "Reachable distance ... 1km" default.
        # Data has radius. Let's use data radius or override.
        # self.radius = 1.0 # km
        
        self.online_time = data['appearance_time']
        # Available time off-on. Default 1h?
        # Paper says "Available time ... 1h".
        self.avail_time = 3600
        self.offline_time = self.online_time + self.avail_time
        
        self.current_lat = self.lat
        self.current_lng = self.lng
        self.current_time = self.online_time
        self.assigned_tasks = [] # List of Tasks
        self.schedule = [] # List of Tasks to do

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def travel_time(dist_km, speed_kmh=30):
    # Assume average speed 30 km/h?
    # Paper doesn't specify speed, but typical urban speed.
    return (dist_km / speed_kmh) * 3600 # seconds

class GreedyAssignment:
    def __init__(self):
        self.workers = []
        self.tasks = []
        
    def add_worker(self, worker):
        self.workers.append(worker)
        
    def add_task(self, task):
        self.tasks.append(task)
        
    def assign(self, current_time):
        # Assign unassigned tasks to available workers greedily
        # Sort tasks by value or expiration?
        # Greedy: "assigns each worker the maximal valid task set ... until all tasks assigned or workers exhausted"
        # Simple greedy: For each task, find best worker.
        
        unassigned_tasks = [t for t in self.tasks if current_time < t.exp_time]
        available_workers = [w for w in self.workers if current_time < w.offline_time]
        
        assignments = []
        
        # Simple matching
        for t in unassigned_tasks:
            best_w = None
            min_dist = float('inf')
            
            for w in available_workers:
                # Check constraints
                dist = haversine(w.current_lat, w.current_lng, t.lat, t.lng)
                if dist > w.radius:
                    continue
                
                arr_time = w.current_time + travel_time(dist)
                if arr_time > t.exp_time:
                    continue
                if arr_time > w.offline_time:
                    continue
                
                # Check if worker is free?
                # "assume that each worker can perform at most one task at a time"
                # "single task assignment mode"
                # But they can have a sequence.
                # Greedy usually just appends.
                
                # For simplicity, let's assume worker is available if they have no current task
                # or we append to schedule.
                # If we append, we need to check if it fits after current schedule.
                
                # Let's assume simple greedy: assign to closest worker who can do it.
                if dist < min_dist:
                    min_dist = dist
                    best_w = w
            
            if best_w:
                # Assign
                best_w.schedule.append(t)
                # Update worker state (simplified)
                # In reality, worker moves.
                # Here we just mark it.
                assignments.append((best_w.id, t.id))
                self.tasks.remove(t)
                # Update worker time/loc
                best_w.current_time += travel_time(min_dist)
                best_w.current_lat = t.lat
                best_w.current_lng = t.lng
                
                
        return assignments

class DependencyGraphAssignment:
    def __init__(self):
        self.workers = []
        self.tasks = []
        
    def add_worker(self, worker):
        self.workers.append(worker)
        
    def add_task(self, task):
        self.tasks.append(task)
        
    def compute_reachable_tasks(self, worker, tasks, current_time):
        # RS_w
        reachable = []
        for t in tasks:
            if current_time > t.exp_time:
                continue
            dist = haversine(worker.current_lat, worker.current_lng, t.lat, t.lng)
            if dist > worker.radius:
                continue
            
            arr_time = worker.current_time + travel_time(dist)
            if arr_time > t.exp_time:
                continue
            if arr_time > worker.offline_time:
                continue
            
            reachable.append(t)
        return reachable

    def construct_graph(self, workers, tasks, current_time):
        # 1. Compute RS for all workers
        rs_map = {}
        for w in workers:
            rs_map[w.id] = set(t.id for t in self.compute_reachable_tasks(w, tasks, current_time))
            
        # 2. Build edges
        adj = {w.id: [] for w in workers}
        nodes = [w.id for w in workers]
        
        for i in range(len(workers)):
            u = workers[i]
            for j in range(i + 1, len(workers)):
                v = workers[j]
                # If share tasks
                if not rs_map[u.id].isdisjoint(rs_map[v.id]):
                    adj[u.id].append(v.id)
                    adj[v.id].append(u.id)
                    
        return adj, rs_map

    def get_connected_components(self, adj):
        visited = set()
        components = []
        for node in adj:
            if node not in visited:
                component = []
                stack = [node]
                visited.add(node)
                while stack:
                    u = stack.pop()
                    component.append(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            stack.append(v)
                components.append(component)
        return components

    def dfsearch(self, worker_ids, task_ids, current_time):
        # Exact search for optimal assignment in a component
        # worker_ids: list of worker IDs in component
        # task_ids: list of task IDs reachable by these workers
        
        # State: (index_in_workers, current_assignments)
        # We want to maximize assigned tasks
        
        # Map IDs to objects
        ws = [w for w in self.workers if w.id in worker_ids]
        ts = [t for t in self.tasks if t.id in task_ids]
        
        # Precompute reachable for these workers within this set of tasks
        # To speed up
        
        best_assignment = []
        max_assigned = 0
        
        # Simple backtracking
        # For each worker, choose a task (or sequence) from reachable tasks
        # Constraint: One task per worker (simplified) or Sequence?
        # Paper says "Sequence".
        # But "assume that each worker can perform at most one task at a time".
        # And "single task assignment mode".
        # If we just assign one task per worker per step, it's matching.
        # If sequence, it's harder.
        # Let's assume Matching for now (Sequence length 1).
        
        # This is Maximum Bipartite Matching with constraints (or Max Weight if values differ).
        # Since we want to maximize number of tasks, it's Max Cardinality Matching.
        # But workers can only do reachable tasks.
        # And tasks can only be done by one worker.
        
        # We can use Network Flow or Hopcroft-Karp.
        # Or just greedy with backtracking if small.
        
        # Let's use a simple greedy heuristic for the component for speed,
        # or a slightly better one.
        
        # Actually, if I implement "Greedy" inside the component, it's same as global greedy.
        # The benefit of components is to run EXACT algorithm.
        # Exact matching can be done with max_flow.
        
        # I'll implement a Max Flow based matching for the component.
        # Source -> Workers (cap 1) -> Tasks (cap 1) -> Sink (cap 1).
        # Edges (w, t) if t in RS_w.
        
        # This solves the "assign one task per worker" problem optimally.
        # If sequence is needed, it's harder.
        # Paper says "assign each worker a fixed task sequence".
        # But in "Adaptive Algorithm", "For each idle worker, the first task ... is executed".
        # So we only need to decide the immediate next task?
        # But the decision depends on future.
        # "planning a dynamically valid task sequence".
        
        # If I stick to matching (sequence len 1), it's efficient.
        return self.max_flow_matching(ws, ts, current_time)

    def max_flow_matching(self, workers, tasks, current_time):
        # Source: s, Sink: t
        # Nodes: s, w1..wn, t1..tm, t
        # Edges: s->w (1), w->task (1), task->t (1)
        
        # Build graph
        capacity = {}
        graph = {}
        
        s = 'source'
        t_sink = 'sink'
        
        graph[s] = []
        graph[t_sink] = []
        
        w_ids = [w.id for w in workers]
        t_ids = [t.id for t in tasks]
        
        for wid in w_ids:
            graph[s].append(wid)
            graph[wid] = []
            capacity[(s, wid)] = 1
            capacity[(wid, s)] = 0
            
        for tid in t_ids:
            graph[tid] = []
            graph[tid].append(t_sink)
            capacity[(tid, t_sink)] = 1
            capacity[(t_sink, tid)] = 0
            
        # Edges w->t
        for w in workers:
            reachable = self.compute_reachable_tasks(w, tasks, current_time)
            for task in reachable:
                if task.id in t_ids: # Should be
                    graph[w.id].append(task.id)
                    graph[task.id].append(w.id) # Residual
                    capacity[(w.id, task.id)] = 1
                    capacity[(task.id, w.id)] = 0 # Residual start 0
                    
        # Max flow (Edmonds-Karp or BFS)
        flow = 0
        assignments = []
        
        while True:
            parent = {node: None for node in graph}
            queue = [s]
            path_found = False
            while queue:
                u = queue.pop(0)
                if u == t_sink:
                    path_found = True
                    break
                for v in graph.get(u, []):
                    if parent[v] is None and capacity.get((u, v), 0) > 0:
                        parent[v] = u
                        queue.append(v)
            
            if not path_found:
                break
                
            path_flow = 1
            flow += path_flow
            v = t_sink
            while v != s:
                u = parent[v]
                capacity[(u, v)] -= path_flow
                capacity[(v, u)] += path_flow
                v = u
                
        # Extract assignments
        for w in workers:
            for task_id in graph[w.id]:
                if task_id != s and task_id != t_sink:
                    if capacity.get((w.id, task_id), 0) == 0 and capacity.get((task_id, w.id), 0) == 1:
                        assignments.append((w.id, task_id))
                        
        return assignments

    def assign(self, current_time):
        # 1. Construct WDG
        # Only consider available workers and unassigned tasks
        avail_workers = [w for w in self.workers if current_time < w.offline_time] # And idle?
        unassigned_tasks = [t for t in self.tasks if current_time < t.exp_time]
        
        if not avail_workers or not unassigned_tasks:
            return []
            
        adj, rs_map = self.construct_graph(avail_workers, unassigned_tasks, current_time)
        
        # 2. Partition
        components = self.get_connected_components(adj)
        
        all_assignments = []
        
        # 3. Solve for each component
        for comp_worker_ids in components:
            # Gather tasks reachable by this component
            comp_task_ids = set()
            for wid in comp_worker_ids:
                comp_task_ids.update(rs_map[wid])
            
            # Solve
            assignments = self.dfsearch(comp_worker_ids, list(comp_task_ids), current_time)
            all_assignments.extend(assignments)
            
        # Update state
        assigned_task_ids = set()
        for wid, tid in all_assignments:
            assigned_task_ids.add(tid)
            # Update worker
            w = next(w for w in self.workers if w.id == wid)
            t = next(t for t in self.tasks if t.id == tid)
            
            w.schedule.append(t)
            dist = haversine(w.current_lat, w.current_lng, t.lat, t.lng)
            w.current_time += travel_time(dist)
            w.current_lat = t.lat
            w.current_lng = t.lng
            
        # Remove assigned tasks
        self.tasks = [t for t in self.tasks if t.id not in assigned_task_ids]
            
        return all_assignments
