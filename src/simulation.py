import pickle
import heapq
from assignment import Task, Worker, GreedyAssignment, DependencyGraphAssignment
import datetime

class Simulator:
    def __init__(self, data_path, method='greedy'):
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
            
        self.workers_data = self.data['workers']
        self.requests_data = self.data['requests']
        self.start_ts = self.data['start_ts']
        self.end_ts = self.data['end_ts']
        
        # Simulation starts at 9:00 (start_ts + 1h)
        self.sim_start_time = self.start_ts + 3600
        self.sim_end_time = self.end_ts
        
        self.events = []
        # Add worker events
        for i, w in enumerate(self.workers_data):
            if self.sim_start_time <= w['appearance_time'] < self.sim_end_time:
                # Use i as tie breaker
                heapq.heappush(self.events, (w['appearance_time'], 'worker', i, w))
                
        # Add request events
        for i, r in enumerate(self.requests_data):
            if self.sim_start_time <= r['appearance_time'] < self.sim_end_time:
                heapq.heappush(self.events, (r['appearance_time'], 'request', i, r))
                
        if method == 'greedy':
            self.solver = GreedyAssignment()
        elif method == 'dta':
            self.solver = DependencyGraphAssignment()
        else:
            raise ValueError("Unknown method")
            
        self.total_assigned = 0
        self.total_tasks = 0
        
    def run(self):
        print(f"Starting simulation from {self.sim_start_time} to {self.sim_end_time}")
        print(f"Total events: {len(self.events)}")
        
        current_time = self.sim_start_time
        
        while self.events:
            t, type, _, obj = heapq.heappop(self.events)
            current_time = t
            
            if type == 'worker':
                w = Worker(obj)
                self.solver.add_worker(w)
            elif type == 'request':
                task = Task(obj)
                self.solver.add_task(task)
                self.total_tasks += 1
                
            # Run assignment
            # In real-time, we might batch or run on every event.
            # Greedy runs on every event or periodically?
            # "When a worker or task appears ... PA is calculated"
            assignments = self.solver.assign(current_time)
            if assignments:
                self.total_assigned += len(assignments)
                # print(f"Time {current_time}: Assigned {len(assignments)} tasks")
                
            if len(self.events) % 1000 == 0:
                print(f"Remaining events: {len(self.events)}, Assigned: {self.total_assigned}/{self.total_tasks}")
                
        print(f"Simulation finished.")
        print(f"Total Tasks: {self.total_tasks}")
        print(f"Total Assigned: {self.total_assigned}")
        print(f"Assignment Rate: {self.total_assigned/self.total_tasks:.4f}")

if __name__ == "__main__":
    sim = Simulator('data/processed/yueche_data.pkl', method='dta')
    sim.run()
