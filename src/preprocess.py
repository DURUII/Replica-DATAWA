import numpy as np
import pickle
import os
from data_loader import DataLoader
import datetime

class Preprocessor:
    def __init__(self, data_dir, grid_size=0.005, delta_t=5, k=3):
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.delta_t = delta_t
        self.k = k
        self.loader = DataLoader(data_dir)

    def get_grid_index(self, lat, lng, min_lat, min_lng, lat_steps, lng_steps):
        lat_idx = int((lat - min_lat) / self.grid_size)
        lng_idx = int((lng - min_lng) / self.grid_size)
        lat_idx = min(lat_idx, lat_steps - 1)
        lng_idx = min(lng_idx, lng_steps - 1)
        return lat_idx, lng_idx

    def process(self, worker_file, request_file, start_hour_utc, end_hour_utc, date_str):
        # Load data
        workers = self.loader.load_workers(worker_file)
        requests = self.loader.load_requests(request_file)

        # Define bounds (fixed or dynamic)
        # For consistency, let's use dynamic bounds from the data
        lats = [w['lat'] for w in workers] + [r['lat_app'] for r in requests]
        lngs = [w['lng'] for w in workers] + [r['lng_app'] for r in requests]
        min_lat, max_lat = min(lats), max(lats)
        min_lng, max_lng = min(lngs), max(lngs)
        
        lat_steps = int(np.ceil((max_lat - min_lat) / self.grid_size))
        lng_steps = int(np.ceil((max_lng - min_lng) / self.grid_size))
        num_grids = lat_steps * lng_steps
        
        print(f"Grid: {lat_steps}x{lng_steps} = {num_grids} cells")

        # Time filtering
        # date_str format: 'YYYY-MM-DD'
        base_time = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        start_ts = base_time.timestamp() + start_hour_utc * 3600
        end_ts = base_time.timestamp() + end_hour_utc * 3600
        
        print(f"Filtering data from {start_ts} to {end_ts}")
        
        filtered_requests = [r for r in requests if start_ts <= r['appearance_time'] < end_ts]
        print(f"Filtered requests: {len(filtered_requests)} out of {len(requests)}")

        # Create time series
        # Total duration
        duration = end_ts - start_ts
        num_vectors = int(duration / (self.k * self.delta_t))
        
        # Matrix C: (Num_Vectors, Num_Grids, k)
        C = np.zeros((num_vectors, num_grids, self.k), dtype=int)
        
        for r in filtered_requests:
            t = r['appearance_time'] - start_ts
            if t < 0 or t >= duration:
                continue
            
            # Find vector index
            # Each vector covers k * delta_t
            vec_idx = int(t / (self.k * self.delta_t))
            if vec_idx >= num_vectors:
                continue
                
            # Find dimension index within vector
            # t % (k * delta_t) gives time within the vector
            # dim_idx = int((t % (k * delta_t)) / delta_t)
            time_in_vec = t - vec_idx * (self.k * self.delta_t)
            dim_idx = int(time_in_vec / self.delta_t)
            
            # Find grid index
            lat_idx, lng_idx = self.get_grid_index(r['lat_app'], r['lng_app'], min_lat, min_lng, lat_steps, lng_steps)
            grid_idx = lat_idx * lng_steps + lng_idx
            
            C[vec_idx, grid_idx, dim_idx] = 1

        return {
            'C': C,
            'min_lat': min_lat,
            'min_lng': min_lng,
            'lat_steps': lat_steps,
            'lng_steps': lng_steps,
            'grid_size': self.grid_size,
            'delta_t': self.delta_t,
            'k': self.k,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'workers': workers, # Save all workers (they have appearance time)
            'requests': filtered_requests
        }

if __name__ == "__main__":
    # Yueche: 8:00 - 11:00 (UTC 0:00 - 3:00)
    # Date: 2016-11-01
    preprocessor = Preprocessor('/Users/durui/Code/Replica-DATAWA/data/raw')
    data = preprocessor.process('CN01_W', 'CN01_R', 0, 3, '2016-11-01')
    
    os.makedirs('data/processed', exist_ok=True)
    with open('data/processed/yueche_data.pkl', 'wb') as f:
        pickle.dump(data, f)
    print("Saved processed data to data/processed/yueche_data.pkl")
