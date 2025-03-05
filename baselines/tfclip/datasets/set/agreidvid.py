import os
import os.path as osp
import json
import numpy as np
import re
from utils.serialization import read_json
from tqdm import tqdm
import glob

class infostruct(object):
    pass

class AGReIDVid(object):
    
    def __init__(self, cfg, root='./data/Mars/Mars', min_seq_len=0):
        self.cfg = cfg
        self.root = root
        
        self.subset = cfg.DATASETS.SUBSET
        self.split_train_json_path = osp.join(self.root, self.subset, 'split_train.json')
        
        self.train_dir = '/path/to/your/train'
        self.query_dir = '/path/to/your/case1_aerial_to_ground/query'
        self.gallery_dir = '/path/to/your/case1_aerial_to_ground/gallery'
        
        # Check if we should use direct loading
        self.use_direct_loading = self.query_dir is not None and self.gallery_dir is not None
        
        if self.use_direct_loading:
            print("=> Using direct loading from directories")
            print(f"   Query dir: {self.query_dir}")
            print(f"   Gallery dir: {self.gallery_dir}")
            
            # Load directly from directories
            train = self._load_from_directory(self.train_dir)
            query = self._load_from_directory(self.query_dir)
            gallery = self._load_from_directory(self.gallery_dir)
            
        else:
           assert False, "Direct loading not implemented yet"
           
        # Calculate statistics
        num_train_pids = train.get('num_pids', 0)
        num_train_tracklets = train.get('num_tracklets', 0)
        num_train_imgs = sum(train.get('num_imgs_per_tracklet', [0]))
        num_train_cams = train.get('num_cams', 0)
        num_train_vids = train.get('num_tracks', 0)

        num_query_pids = query['num_pids']
        num_query_tracklets = query['num_tracklets']
        num_query_imgs = sum(query['num_imgs_per_tracklet'])

        num_gallery_pids = gallery['num_pids']
        num_gallery_tracklets = gallery['num_tracklets']
        num_gallery_imgs = sum(gallery['num_imgs_per_tracklet'])
        
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs
        num_imgs_per_tracklet = train.get('num_imgs_per_tracklet', []) + query['num_imgs_per_tracklet'] + gallery['num_imgs_per_tracklet']

        min_num = np.min(num_imgs_per_tracklet) if num_imgs_per_tracklet else 0
        max_num = np.max(num_imgs_per_tracklet) if num_imgs_per_tracklet else 0
        avg_num = np.mean(num_imgs_per_tracklet) if num_imgs_per_tracklet else 0

        num_total_pids = num_train_pids + num_gallery_pids
        num_total_tracklets = num_train_tracklets + num_query_tracklets + num_gallery_tracklets

        print("=> Dataset loaded")
        print("Dataset statistics:")
        print("  -------------------------------------------")
        print("  subset   | # ids | # tracklets | # images")
        print("  -------------------------------------------")
        print("  train    | {:5d} | {:10d} | {:8d}".format(num_train_pids, num_train_tracklets, num_train_imgs))
        print("  query    | {:5d} | {:10d} | {:8d}".format(num_query_pids, num_query_tracklets, num_query_imgs))
        print("  gallery  | {:5d} | {:10d} | {:8d}".format(num_gallery_pids, num_gallery_tracklets, num_gallery_imgs))
        print("  -------------------------------------------")
        print("  total    | {:5d} | {:10d} | {:8d}".format(num_total_pids, num_total_tracklets, num_total_imgs))
        print("  number of images per tracklet: {} ~ {}, average {:.1f}".format(min_num, max_num, avg_num))
        print("  -------------------------------------------")
        
        self.train = train.get('tracklets', [])
        self.query = query['tracklets']
        self.gallery = gallery['tracklets']

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_gallery_pids = num_gallery_pids

        self.queryinfo = infostruct()
        self.queryinfo.pid = query['pids']
        self.queryinfo.camid = query['camid']
        self.queryinfo.tranum = num_query_imgs

        self.galleryinfo = infostruct()
        self.galleryinfo.pid = gallery['pids']
        self.galleryinfo.camid = gallery['camid']
        self.galleryinfo.tranum = num_gallery_imgs

        self.num_train_cams = num_train_cams
        self.num_train_vids = num_train_vids
    
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.root):
            raise RuntimeError("'{}' is not available".format(self.root))
        if not osp.exists(self.split_train_json_path):
            raise RuntimeError("'{}' is not available".format(self.split_train_json_path))
        if not osp.exists(self.split_query_json_path):
            raise RuntimeError("'{}' is not available".format(self.split_query_json_path))
        if not osp.exists(self.split_gallery_json_path):
            raise RuntimeError("'{}' is not available".format(self.split_gallery_json_path))
    
    def _load_from_split_json(self, json_path):
        """Load split from json file"""
        return read_json(json_path)
    
    def _load_from_directory(self, directory):
        """
        Load data directly from a directory structure:
        directory/
            ├── person_id_1/
            │   ├── tracklet_1/
            │   │   ├── frame_1.jpg
            │   │   ├── frame_2.jpg
            │   │   └── ...
            │   ├── tracklet_2/
            │   │   └── ...
            │   └── ...
            ├── person_id_2/
            │   └── ...
            └── ...
        """
        print(f"Loading data from directory: {directory}")
        
        # Initialize return structure
        result = {
            'tracklets': [],
            'num_imgs_per_tracklet': [],
            'pids': [],
            'camid': [],
            'original_pids': []  # Store original PIDs for reference
        }
        
        # Get all person ID folders
        person_folders = sorted([d for d in glob.glob(os.path.join(directory, '*')) if os.path.isdir(d)])
        
        # Extract original person IDs
        original_pids = [int(os.path.basename(folder)) for folder in person_folders]
        result['original_pids'] = original_pids
        
        # Create mapping from original PIDs to consecutive indices (0, 1, 2, ...)
        pid_mapping = {pid: idx for idx, pid in enumerate(sorted(original_pids))}
        result['pid_mapping'] = pid_mapping
        
        # Keep track of all pids and camids
        all_pids = set()
        all_camids = set()
        
        for person_folder in tqdm(person_folders, desc="Processing identities"):
            # Extract person ID from folder name
            original_person_id = int(os.path.basename(person_folder))
            # Map to consecutive index
            mapped_person_id = pid_mapping[original_person_id]
            
            all_pids.add(mapped_person_id)
            
            # Get all tracklet folders for this person
            tracklet_folders = sorted([d for d in glob.glob(os.path.join(person_folder, '*')) if os.path.isdir(d)])
            
            for tracklet_folder in tracklet_folders:
                # Get all image files in this tracklet
                img_paths = sorted(glob.glob(os.path.join(tracklet_folder, '*.jpg')))
                
                if len(img_paths) == 0:
                    continue
                
                try:
                    image_name = os.path.basename(img_paths[0])
                    camid = int(image_name.split("C")[1].split("E")[0])
                except Exception as e:
                    print(f"Error extracting camera ID from {img_paths[0]}: {e}")
                    camid = 0  # Default camera ID
                    
                if camid not in all_camids:
                    all_camids.add(camid)
                    
                # Add this tracklet to the result with the MAPPED person ID
                result['tracklets'].append((tuple(img_paths), mapped_person_id, camid, 1))
                result['num_imgs_per_tracklet'].append(len(img_paths))
        
        # Update the result dictionary with statistics
        result['pids'] = sorted(list(all_pids))
        result['camid'] = sorted(list(all_camids))
        result['num_pids'] = len(all_pids)
        result['num_cams'] = len(all_camids)
        result['num_tracklets'] = len(result['tracklets'])
        result['num_tracks'] = result['num_tracklets']
        
        print(f"Loaded {result['num_tracklets']} tracklets of {result['num_pids']} identities")
        
        # Print ID mapping information
        print(f"Original ID range: min={min(original_pids)}, max={max(original_pids)}, count={len(original_pids)}")
        print(f"Mapped ID range: min={min(result['pids'])}, max={max(result['pids'])}, count={len(result['pids'])}")
        
        return result
    

if __name__ == '__main__':
    # test
    dataset = AGReIDVid()