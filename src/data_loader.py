import os

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_workers(self, filename):
        workers = []
        path = os.path.join(self.data_dir, 'worker', filename)
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                # 6i9echdcg6xB6kvrhi7ahccdf@rF3qpx 1 1 1477929732 30.66908 104.10336 ...
                worker = {
                    'id': parts[0],
                    'platform_id': int(parts[1]),
                    'radius': float(parts[2]),
                    'appearance_time': int(parts[3]),
                    'lat': float(parts[4]),
                    'lng': float(parts[5]),
                    # ... ignore the rest for now
                }
                workers.append(worker)
        return workers

    def load_requests(self, filename):
        requests = []
        path = os.path.join(self.data_dir, 'request', filename)
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                # 49655ijgj4tx7oyuiljgbik7b-zD.kus 1 1477982146 1477987052 30.6993 104.107 30.689 104.056 6.3315 29.3
                request = {
                    'id': parts[0],
                    'platform_id': int(parts[1]),
                    'appearance_time': int(parts[2]),
                    'finishing_time': int(parts[3]), # This might be historical finish time
                    'lat_app': float(parts[4]),
                    'lng_app': float(parts[5]),
                    'lat_end': float(parts[6]),
                    'lng_end': float(parts[7]),
                    'distance': float(parts[8]),
                    'payment': float(parts[9])
                }
                requests.append(request)
        return requests

if __name__ == "__main__":
    loader = DataLoader('/Users/durui/Code/Replica-DATAWA/data/raw')
    workers = loader.load_workers('CN01_W')
    requests = loader.load_requests('CN01_R')
    print(f"Loaded {len(workers)} workers and {len(requests)} requests.")
