import numpy as np


class DenseToSparse:
    def __init__(self):
        pass

    def dense_to_sparse(self, *args):
        pass

    def __repr__(self):
        pass


class UniformSampling(DenseToSparse):
    name = "uar"
    def __init__(self, num_samples, max_depth=np.inf):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def __repr__(self):
        return "%s{ns=%d,md=%f}" % (self.name, self.num_samples, self.max_depth)

    def dense_to_sparse(self, depth):
        """
        Samples pixels with `num_samples`/#pixels probability in `depth`.
        Only pixels with a maximum depth of `max_depth` are considered.
        If no `max_depth` is given, samples in all pixels
        """
        mask_keep = depth > 0
        if self.max_depth is not np.inf:
            mask_keep = np.bitwise_and(mask_keep, depth <= self.max_depth)
        n_keep = np.count_nonzero(mask_keep)
        if n_keep == 0:
            return mask_keep
        else:
            prob = float(self.num_samples) / n_keep
            return np.bitwise_and(mask_keep, np.random.uniform(0, 1, depth.shape) < prob)


class LidarRadarSampling(DenseToSparse):
    name = "lidar_radar"
    def __init__(self, num_samples, max_depth=np.inf):
        DenseToSparse.__init__(self)
        self.num_samples = num_samples
        self.max_depth = max_depth

    def __repr__(self):
        return "%s{ns=%d,md=%f}" % (self.name, self.num_samples, self.max_depth)

    def dense_to_sparse(self, lidar_depth, radar_depth):
        """
        Samples pixels with `num_samples`/#pixels probability in `depth`.
        Only pixels with a maximum depth of `max_depth` are considered.
        If no `max_depth` is given, samples in all pixels
        """
        # Convert to numpy array first
        lidar_depth = np.squeeze(lidar_depth.cpu().numpy().transpose(1, 2, 0))
        radar_depth = np.squeeze(radar_depth.cpu().numpy().transpose(1, 2, 0))

        # h, w, _ = lidar_depth.shape
        # h_lin = np.linspace(0, h-1, h)
        # w_lin = np.linspace(0, w-1, w)
        # h_grid, w_grid = np.meshgrid(h_lin, w_lin, indexing="ij")
        # coord_map = np.concatenate((h_grid[..., None], w_grid[..., None]), axis=-1)

        # Find the locations radar > 0
        radar_coord_tmp = np.where(radar_depth > 0)
        lidar_coord_tmp = np.where(lidar_depth > 0)

        # Concatenate to coordinate map
        radar_coord = np.concatenate((radar_coord_tmp[0][..., None], radar_coord_tmp[1][..., None]), axis=-1)
        lidar_coord = np.concatenate((lidar_coord_tmp[0][..., None], lidar_coord_tmp[1][..., None]), axis=-1)

        radar_expand = np.expand_dims(radar_coord, axis=1)
        lidar_expand = np.expand_dims(lidar_coord, axis=0)

        # Compute the pair-wise distance => (100 v.s. 3000)
        dist = np.sqrt(np.sum((radar_expand - lidar_expand) ** 2, axis=-1))

        # Get the top 2 nearest points and get unique
        lidar_candidates = np.unique(np.argsort(dist, axis=-1)[:, :2])
        mask = lidar_coord[lidar_candidates, :]
        output_mask = np.zeros(lidar_depth.shape)
        output_mask[mask[:, 0], mask[:, 1]] = 1

        return output_mask