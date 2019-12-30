import time
import faiss
import torch
import numpy as np


def compute_features(dataloader, model, args, use_predict_fn=False, concat_vid=False, keep_dim=False):
    cargs = args
    if cargs.verbose:
        print('Compute features')
    start = time.time()
    model.eval()
    features = []
    # discard the label information in the dataloader
    for i, data_arr in enumerate(dataloader):
        pose_data = data_arr[0]
        with torch.no_grad():
            data = pose_data.to(args.device)
            if use_predict_fn:
                pose_features = model.predict(data).data.to('cpu', non_blocking=True).numpy()
                pose_features = pose_features.reshape(-1, 1)
            else:
                pose_features = model(data)
                if isinstance(pose_features, (list, tuple)):
                    pose_features = pose_features[0]
                pose_features = pose_features.data.to('cpu', non_blocking=True).numpy()

        if concat_vid:  # Concatenate each clip's video features to its pose embedding
            vid_features = data_arr[1]
            batch_features = np.concatenate([pose_features, vid_features], axis=1)
        else:
            batch_features = pose_features

        features.append(batch_features)

        # measure elapsed time
        batch_time = time.time() - start
        start = time.time()

        if cargs.verbose and (i % 200) == 0:
            print('{0} / {1}\t'
                  'Time: {batch_time:.3f})'
                  .format(i, len(dataloader), batch_time=batch_time))
    features = np.concatenate(features)
    if keep_dim:
        n, c, t, v = data.size()
        features = features.reshape(features.shape[0], -1, v)
    return features


class Kmeans:
    def __init__(self, k, knn=1):
        self.k = k
        self.knn = knn
        self.centroids = None
        self.labels = None
        self.images_lists = []
        self.dists = None

    def cluster(self, data,  verbose=False):
        """Performs k-means clustering.
            Args:
            x_data (np.array N * dim): data to cluster
        """
        start = time.time()
        # cluster the data
        labels, loss, self.centroids, self.dists = run_kmeans(data, self.k, verbose, self.knn)
        self.labels = labels
        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[labels[i]].append(i)

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - start))

        return loss, self.dists


def run_kmeans(x, nmb_clusters, verbose=False, knn=1):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    nmb_clusters = int(nmb_clusters)
    clus = faiss.Clustering(d, nmb_clusters)
    clus.niter = 30
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    dists, labels = index.search(x, knn)
    losses = faiss.vector_to_array(clus.obj)
    centroids = faiss.vector_float_to_array(clus.centroids).reshape(nmb_clusters, d)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in labels], losses[-1], centroids, dists
