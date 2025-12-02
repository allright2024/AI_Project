import random
from torch.utils.data import Sampler

class NoDuplicateTitleBatchSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        self.title_to_indices = {}
        for idx, example in enumerate(data_source):
            title = getattr(example, 'title', None)
            if title is None:
                title = f"__no_title_{idx}__"
            
            if title not in self.title_to_indices:
                self.title_to_indices[title] = []
            self.title_to_indices[title].append(idx)

    def __iter__(self):
        title_pool = {t: list(idxs) for t, idxs in self.title_to_indices.items()}
        for t in title_pool:
            random.shuffle(title_pool[t])
            
        active_titles = list(title_pool.keys())
        random.shuffle(active_titles)
        
        batch = []
        
        while active_titles:
            current_batch_titles = set()
            batch = []
            
            candidates = list(active_titles)
            random.shuffle(candidates)
            
            titles_to_remove_from_active = []
            
            for title in candidates:
                if len(batch) >= self.batch_size:
                    break
                
                if title_pool[title]:
                    idx = title_pool[title].pop()
                    batch.append(idx)
                    current_batch_titles.add(title)
                    
                    if not title_pool[title]:
                        titles_to_remove_from_active.append(title)
            
            for t in titles_to_remove_from_active:
                active_titles.remove(t)
                
            if len(batch) == self.batch_size:
                yield batch
            elif len(batch) > 0 and not self.drop_last:
                yield batch
            
            
            
            if len(batch) < self.batch_size:
                break

    def __len__(self):
        return len(self.data_source) // self.batch_size
