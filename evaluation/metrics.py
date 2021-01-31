import torch
import math
import numpy as np


def log10(x):
    """Convert a new tensor with the base-10 logarithm of the elements of x. """
    return torch.log(x) / math.log(10)


# Object to record the evaluation results
class Result(object):
    def __init__(self):
        self.irmse, self.imae = 0, 0
        self.mse, self.rmse, self.mae = 0, 0, 0
        self.absrel, self.lg10 = 0, 0
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def set_to_worst(self):
        self.irmse, self.imae = np.inf, np.inf
        self.mse, self.rmse, self.mae = np.inf, np.inf, np.inf
        self.absrel, self.lg10 = np.inf, np.inf
        self.delta1, self.delta2, self.delta3 = 0, 0, 0
        self.data_time, self.gpu_time = 0, 0

    def update(self, irmse, imae, mse, rmse, mae, absrel, lg10, delta1, delta2, delta3, gpu_time, data_time):
        self.irmse, self.imae = irmse, imae
        self.mse, self.rmse, self.mae = mse, rmse, mae
        self.absrel, self.lg10 = absrel, lg10
        self.delta1, self.delta2, self.delta3 = delta1, delta2, delta3
        self.data_time, self.gpu_time = data_time, gpu_time

    def evaluate(self, output, target):
        valid_mask = target>0
        output = output[valid_mask]
        target = target[valid_mask]

        abs_diff = (output - target).abs()

        self.mse = float((torch.pow(abs_diff, 2)).mean())
        self.rmse = math.sqrt(self.mse)
        self.mae = float(abs_diff.mean())
        self.lg10 = float((log10(output) - log10(target)).abs().mean())
        self.absrel = float((abs_diff / target).mean())

        maxRatio = torch.max(output / target, target / output)
        self.delta1 = float((maxRatio < 1.25).float().mean())
        self.delta2 = float((maxRatio < 1.25 ** 2).float().mean())
        self.delta3 = float((maxRatio < 1.25 ** 3).float().mean())
        self.data_time = 0
        self.gpu_time = 0

        inv_output = 1 / output
        inv_target = 1 / target
        abs_inv_diff = (inv_output - inv_target).abs()
        self.irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
        self.imae = float(abs_inv_diff.mean())


#  Object to record evaluation results in different distance intervals
class Result_multidist(object):
    def __init__(self):
        # initialize the end points of the dist intervals
        self.dist_interval = [10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]
        # Initialize result object of each interval
        self.result_lst = [Result() for i in range(len(self.dist_interval))]
        # Initialize valid label for every distance interval
        self.valid_label = [1 for _ in range(len(self.dist_interval))]

    # Set each result to worst
    def set_to_worst(self):
        for res in self.result_lst:
            res.set_to_worst()

    # Update the results given another multi-distance result object
    def update(self, result):
        # Check if result is multidist object
        assert isinstance(result, Result_multidist)

        # Iterate through all dist interval to perform the update
        for idx, res in enumerate(result):
            self.result_lst[idx].update(
                res.irmse, res.imae, res.mse,
                res.rmse, res.mae, res.absrel, res.log10,
                res.delta1, res.delta2, res.delta3
            )

    # Evaluate the results
    def evaluate(self, output, target):
        # Compute shared valid mask first
        valid_mask = target>0

        # Iterate through all the distance intervals and evaluate
        for idx, interval in enumerate(self.dist_interval):
            # First interval => min=0
            if idx == 0:
                dist_min = 0.
                dist_max = interval
            # Last interval => max=inf
            elif idx == len(self.dist_interval)-1:
                dist_min = self.dist_interval[idx - 1]
                dist_max = np.inf
            else:
                dist_min = self.dist_interval[idx - 1]
                dist_max = interval

            # Compute distance-aware valid mask
            # ToDo: Fix the corner case that no points lies in the interval.
            # ToDo: How to balance the point counts in different distance range.
            dist_valid_mask = (target >= dist_min) & (target <= dist_max)
            valid_mask_final = dist_valid_mask & valid_mask

            # change valid label to 0 if no point in the distance range
            if torch.sum(valid_mask_final) == 0:
                self.valid_label[idx] = 0

            output_masked = output[valid_mask_final]
            target_masked = target[valid_mask_final]

            abs_diff = (output_masked - target_masked).abs()

            self.result_lst[idx].mse = float((torch.pow(abs_diff, 2)).mean())
            self.result_lst[idx].rmse = math.sqrt(self.result_lst[idx].mse)
            self.result_lst[idx].mae = float(abs_diff.mean())
            self.result_lst[idx].lg10 = float((log10(output_masked) - log10(target_masked)).abs().mean())
            self.result_lst[idx].absrel = float((abs_diff / target_masked).mean())

            maxRatio = torch.max(output_masked / target_masked, target_masked / output_masked)
            self.result_lst[idx].delta1 = float((maxRatio < 1.25).float().mean())
            self.result_lst[idx].delta2 = float((maxRatio < 1.25 ** 2).float().mean())
            self.result_lst[idx].delta3 = float((maxRatio < 1.25 ** 3).float().mean())

            inv_output = 1 / output_masked
            inv_target = 1 / target_masked
            abs_inv_diff = (inv_output - inv_target).abs()
            self.result_lst[idx].irmse = math.sqrt((torch.pow(abs_inv_diff, 2)).mean())
            self.result_lst[idx].imae = float(abs_inv_diff.mean())


class AverageMeter_multidist(object):
    def __init__(self):
        self.dist_interval = [10., 20., 30., 40., 50., 60., 70., 80., 90., 100.]
        self.avg_lst = [AverageMeter() for i in range(len(self.dist_interval))]
        self.reset()

    def reset(self):
        # Reset every average meter
        for avg in self.avg_lst:
            avg.reset()

    def update(self, result, n=1):
        # Fetch the result list
        assert isinstance(result, Result_multidist)
        assert self.dist_interval == result.dist_interval
        result_lst = result.result_lst

        for idx, res in enumerate(result_lst):
            # Skip the invalid distanve intervals
            if result.valid_label[idx] == 0:
                pass
            else:
                self.avg_lst[idx].update(res, n)

    def average(self):
        avg = Result_multidist()
        # Iterate through all the avg_lst and perform average
        for idx, avg_obj in enumerate(self.avg_lst):
            res = avg_obj.average()
            avg.result_lst[idx].update(
                res.irmse, res.imae, res.mse,
                res.rmse, res.mae, res.absrel, res.lg10,
                res.delta1, res.delta2, res.delta3
            )

        return avg


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.count = 0.0

        self.sum_irmse, self.sum_imae = 0, 0
        self.sum_mse, self.sum_rmse, self.sum_mae = 0, 0, 0
        self.sum_absrel, self.sum_lg10 = 0, 0
        self.sum_delta1, self.sum_delta2, self.sum_delta3 = 0, 0, 0
        self.sum_data_time, self.sum_gpu_time = 0, 0

    def update(self, result, gpu_time, data_time, n=1):
        self.count += n

        self.sum_irmse += n*result.irmse
        self.sum_imae += n*result.imae
        self.sum_mse += n*result.mse
        self.sum_rmse += n*result.rmse
        self.sum_mae += n*result.mae
        self.sum_absrel += n*result.absrel
        self.sum_lg10 += n*result.lg10
        self.sum_delta1 += n*result.delta1
        self.sum_delta2 += n*result.delta2
        self.sum_delta3 += n*result.delta3
        self.sum_data_time += n*data_time
        self.sum_gpu_time += n*gpu_time

    def average(self):
        avg = Result()
        avg.update(
            self.sum_irmse / self.count, self.sum_imae / self.count,
            self.sum_mse / self.count, self.sum_rmse / self.count, self.sum_mae / self.count, 
            self.sum_absrel / self.count, self.sum_lg10 / self.count,
            self.sum_delta1 / self.count, self.sum_delta2 / self.count, self.sum_delta3 / self.count,
            self.sum_gpu_time / self.count, self.sum_data_time / self.count)
        return avg