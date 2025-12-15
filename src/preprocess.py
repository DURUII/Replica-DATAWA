import numpy as np
import pickle
import os
from data_loader import DataLoader
import datetime
import math

class Preprocessor:
    def __init__(self, data_dir, grid_size=0.005, delta_t=5, k=3, grid_size_m=None):
        self.data_dir = data_dir
        self.grid_size = grid_size
        self.grid_size_m = grid_size_m
        self.delta_t = delta_t
        self.k = k
        self.loader = DataLoader(data_dir)
        self.use_projection = False
        self.transformer = None
        self.min_x = None
        self.min_y = None

    def get_grid_index(self, lat, lng, min_lat, min_lng, lat_steps, lng_steps):
        if self.use_projection and self.transformer is not None and self.min_x is not None and self.min_y is not None:
            x, y = self.transformer.transform(lng, lat)
            lat_idx = int((y - self.min_y) / self.grid_size_m)
            lng_idx = int((x - self.min_x) / self.grid_size_m)
        else:
            lat_idx = int((lat - min_lat) / self.grid_size_lat_deg)
            lng_idx = int((lng - min_lng) / self.grid_size_lng_deg)
        lat_idx = max(0, min(lat_idx, lat_steps - 1))
        lng_idx = max(0, min(lng_idx, lng_steps - 1))
        return lat_idx, lng_idx

    def process(self, worker_file, request_file, start_hour_utc, end_hour_utc, date_str):
        # Load data
        workers = self.loader.load_workers(worker_file)
        requests = self.loader.load_requests(request_file)

        # Time filtering
        # date_str format: 'YYYY-MM-DD'
        base_time = datetime.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=datetime.timezone.utc)
        start_ts = base_time.timestamp() + start_hour_utc * 3600
        end_ts = base_time.timestamp() + end_hour_utc * 3600
        
        print(f"Filtering data from {start_ts} to {end_ts}")
        
        filtered_requests = [r for r in requests if start_ts <= r['appearance_time'] <= end_ts]
        print(f"Filtered requests: {len(filtered_requests)} out of {len(requests)}")
        
        filtered_workers = [w for w in workers if start_ts <= w['appearance_time'] <= end_ts]
        print(f"Filtered workers: {len(filtered_workers)} out of {len(workers)}")

        # Define global bounds using all requests and workers (fixed grid graph)
        req_lats = [r['lat_app'] for r in requests]
        req_lngs = [r['lng_app'] for r in requests]
        wrk_lats = [w['lat'] for w in workers]
        wrk_lngs = [w['lng'] for w in workers]
        all_lats = req_lats + wrk_lats
        all_lngs = req_lngs + wrk_lngs
        min_lat, max_lat = min(all_lats), max(all_lats)
        min_lng, max_lng = min(all_lngs), max(all_lngs)
        projection_epsg = None
        if self.grid_size_m is not None:
            try:
                from pyproj import Transformer
                center_lat = (min_lat + max_lat) / 2.0
                center_lng = (min_lng + max_lng) / 2.0
                zone = int(math.floor((center_lng + 180) / 6) + 1)
                north = center_lat >= 0
                projection_epsg = (32600 + zone) if north else (32700 + zone)
                self.transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{projection_epsg}", always_xy=True)
                xs = []
                ys = []
                for la, lo in zip(all_lats, all_lngs):
                    x, y = self.transformer.transform(lo, la)
                    xs.append(x); ys.append(y)
                min_x, max_x = float(np.min(xs)), float(np.max(xs))
                min_y, max_y = float(np.min(ys)), float(np.max(ys))
                self.min_x, self.min_y = min_x, min_y
                self.use_projection = True
                lat_steps = max(1, int(np.ceil((max_y - min_y) / self.grid_size_m)) + 1)
                lng_steps = max(1, int(np.ceil((max_x - min_x) / self.grid_size_m)) + 1)
                meters_per_deg_lat = 111000.0
                meters_per_deg_lng = 111000.0 * np.cos(np.deg2rad(center_lat))
                self.grid_size_lat_deg = self.grid_size_m / meters_per_deg_lat
                self.grid_size_lng_deg = self.grid_size_m / meters_per_deg_lng
            except Exception:
                self.use_projection = False
                self.transformer = None
                lat_center = (min_lat + max_lat) / 2.0
                meters_per_deg_lat = 111000.0
                meters_per_deg_lng = 111000.0 * np.cos(np.deg2rad(lat_center))
                self.grid_size_lat_deg = self.grid_size_m / meters_per_deg_lat
                self.grid_size_lng_deg = self.grid_size_m / meters_per_deg_lng
                lat_steps = max(1, int(np.ceil((max_lat - min_lat) / self.grid_size_lat_deg)) + 1)
                lng_steps = max(1, int(np.ceil((max_lng - min_lng) / self.grid_size_lng_deg)) + 1)
        else:
            self.grid_size_lat_deg = self.grid_size
            self.grid_size_lng_deg = self.grid_size
            lat_steps = max(1, int(np.ceil((max_lat - min_lat) / self.grid_size_lat_deg)) + 1)
            lng_steps = max(1, int(np.ceil((max_lng - min_lng) / self.grid_size_lng_deg)) + 1)
        num_grids = lat_steps * lng_steps
        print(f"Grid: {lat_steps}x{lng_steps} = {num_grids} cells")

        # Create time series
        # Total duration
        duration = end_ts - start_ts
        num_vectors = int(np.ceil(duration / (self.k * self.delta_t)))
        
        # Matrix C: (Num_Vectors, Num_Grids, k)
        C_count = np.zeros((num_vectors, num_grids, self.k), dtype=int)
        # Matrix S: (Num_Vectors, Num_Grids, k) - Supply
        S = np.zeros((num_vectors, num_grids, self.k), dtype=int)
        
        for r in filtered_requests:
            t = r['appearance_time'] - start_ts
            if t < 0 or t > duration:
                continue
            
            # Find vector index
            # Each vector covers k * delta_t
            vec_idx = int(t / (self.k * self.delta_t))
            vec_idx = min(vec_idx, num_vectors - 1)
                
            # Find dimension index within vector
            # t % (k * delta_t) gives time within the vector
            # dim_idx = int((t % (k * delta_t)) / delta_t)
            time_in_vec = t - vec_idx * (self.k * self.delta_t)
            time_in_vec = min(time_in_vec, self.k * self.delta_t - 1e-6)
            dim_idx = int(time_in_vec / self.delta_t)
            
            # Find grid index
            lat_idx, lng_idx = self.get_grid_index(r['lat_app'], r['lng_app'], min_lat, min_lng, lat_steps, lng_steps)
            grid_idx = lat_idx * lng_steps + lng_idx
            
            C_count[vec_idx, grid_idx, dim_idx] += 1

        for w in filtered_workers:
            t = w['appearance_time'] - start_ts
            if t < 0 or t > duration:
                continue
                
            vec_idx = int(t / (self.k * self.delta_t))
            vec_idx = min(vec_idx, num_vectors - 1)
                
            time_in_vec = t - vec_idx * (self.k * self.delta_t)
            time_in_vec = min(time_in_vec, self.k * self.delta_t - 1e-6)
            dim_idx = int(time_in_vec / self.delta_t)
            
            lat_idx, lng_idx = self.get_grid_index(w['lat'], w['lng'], min_lat, min_lng, lat_steps, lng_steps)
            grid_idx = lat_idx * lng_steps + lng_idx
            
            S[vec_idx, grid_idx, dim_idx] += 1

        C = (C_count > 0).astype(int)

        # Spatial adjacency mask (4-neighborhood)
        adj_mask = np.zeros((num_grids, num_grids), dtype=int)
        for i in range(lat_steps):
            for j in range(lng_steps):
                idx = i * lng_steps + j
                if i > 0:
                    adj_mask[idx, (i - 1) * lng_steps + j] = 1
                if i < lat_steps - 1:
                    adj_mask[idx, (i + 1) * lng_steps + j] = 1
                if j > 0:
                    adj_mask[idx, i * lng_steps + (j - 1)] = 1
                if j < lng_steps - 1:
                    adj_mask[idx, i * lng_steps + (j + 1)] = 1

        return {
            'C': C,
            'C_count': C_count,
            'S': S,
            'min_lat': min_lat,
            'min_lng': min_lng,
            'min_x': self.min_x,
            'min_y': self.min_y,
            'lat_steps': lat_steps,
            'lng_steps': lng_steps,
            'grid_shape': (lat_steps, lng_steps),
            'grid_size': self.grid_size_m if self.grid_size_m is not None else self.grid_size,
            'grid_size_lat_deg': self.grid_size_lat_deg,
            'grid_size_lng_deg': self.grid_size_lng_deg,
            'cell_size_m': self.grid_size_m if self.grid_size_m is not None else None,
            'projection_epsg': projection_epsg,
            'delta_t': self.delta_t,
            'k': self.k,
            'start_ts': start_ts,
            'end_ts': end_ts,
            'workers': workers, # Save all workers (they have appearance time)
            'requests': filtered_requests,
            'adj_mask': adj_mask
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
