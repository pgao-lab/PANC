import importlib
import os
from collections import OrderedDict
from lib.test.evaluation.environment import env_settings
import time
import cv2 as cv
import torch
from lib.utils.lmdb_utils import decode_img
from pathlib import Path
import numpy as np
import math

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def trackerlist(name: str, parameter_name: str, dataset_name: str, run_ids=None, display_name: str = None,
                result_only=False):
    """Generate list of trackers.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_ids: A single or list of run_ids.
        display_name: Name to be displayed in the result plots.
    """
    if run_ids is None or isinstance(run_ids, int):
        run_ids = [run_ids]
    return [Tracker(name, parameter_name, dataset_name, run_id, display_name, result_only) for run_id in run_ids]


class Tracker:
    """Wraps the tracker for evaluation and running purposes.
    args:
        name: Name of tracking method.
        parameter_name: Name of parameter file.
        run_id: The run id.
        display_name: Name to be displayed in the result plots.
    """

    def __init__(self, name: str, parameter_name: str, dataset_name: str, run_id: int = None, display_name: str = None,
                 result_only=False):
        assert run_id is None or isinstance(run_id, int)

        self.name = name
        self.parameter_name = parameter_name
        self.dataset_name = dataset_name
        self.run_id = run_id
        self.display_name = display_name

        env = env_settings()
        if self.run_id is None:
            self.results_dir = '{}/{}/{}'.format(env.results_path, self.name, self.parameter_name)
        else:
            self.results_dir = '{}/{}/{}_{:03d}'.format(env.results_path, self.name, self.parameter_name, self.run_id)
        if result_only:
            self.results_dir = '{}/{}'.format(env.results_path, self.name)

        tracker_module_abspath = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                              '..', 'tracker', '%s.py' % self.name))
        if os.path.isfile(tracker_module_abspath):
            tracker_module = importlib.import_module('lib.test.tracker.{}'.format(self.name))
            self.tracker_class = tracker_module.get_tracker_class()
        else:
            self.tracker_class = None

    def create_tracker(self, params):
        tracker = self.tracker_class(params, self.dataset_name)
        return tracker

    def run_sequence(self, seq, debug=None):
        """Run tracker on sequence.
        args:
            seq: Sequence to run the tracker on.
            visualization: Set visualization flag (None means default value specified in the parameters).
            debug: Set debug level (None means default value specified in the parameters).
            multiobj_mode: Which mode to use for multiple objects.
        """
        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)

        params.debug = debug_

        # Get init information
        init_info = seq.init_info()

        tracker = self.create_tracker(params)

        output = self._track_sequence(tracker, seq, init_info)
        return output

    def _track_sequence1(self, tracker, seq, init_info):

        output = {'target_bbox': [],
                  'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        # Initialize
        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        init_default = {'target_bbox': init_info.get('init_bbox'),
                        'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            image = self._read_image(frame_path)

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output

            if len(seq.ground_truth_rect) > 1:
                info['gt_bbox'] = seq.ground_truth_rect[frame_num]
            out = tracker.track(image, info)
            print(out)
            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        for key in ['target_bbox', 'all_boxes', 'all_scores']:
            if key in output and len(output[key]) <= 1:
                output.pop(key)

        return output

    def _track_sequence(self, tracker, seq, init_info):

        output = {'target_bbox': [], 'time': []}
        if tracker.params.save_all_boxes:
            output['all_boxes'] = []
            output['all_scores'] = []

        def _store_outputs(tracker_out: dict, defaults=None):
            defaults = {} if defaults is None else defaults
            for key in output.keys():
                val = tracker_out.get(key, defaults.get(key, None))
                if key in tracker_out or val is not None:
                    output[key].append(val)

        image = self._read_image(seq.frames[0])

        start_time = time.time()
        out = tracker.initialize(image, init_info)
        if out is None:
            out = {}

        prev_output = OrderedDict(out)
        if 'target_bbox' not in prev_output:
            prev_output['target_bbox'] = init_info.get('init_bbox', None)  # 使用初始框作为 target_bbox
        init_default = {'target_bbox': prev_output['target_bbox'], 'time': time.time() - start_time}
        if tracker.params.save_all_boxes:
            init_default['all_boxes'] = out['all_boxes']
            init_default['all_scores'] = out['all_scores']

        _store_outputs(out, init_default)

        last_preturb = np.zeros((image.shape[0], image.shape[1], image.shape[2]))
        adversarial_noises = []
        denosed_noises = []
        output_dir = '/hy-tmp/f_4_1_5_xiaorong/'

        for frame_num, frame_path in enumerate(seq.frames[1:], start=1):
            print(seq.name)
            image = self._read_image(frame_path)
            image_clean = image

            start_time = time.time()

            info = seq.frame_info(frame_num)
            info['previous_output'] = prev_output
            last_gt = prev_output['target_bbox']
            heavy_noise = np.random.randint(-1, 2, (image.shape[0], image.shape[1], image.shape[2])) * 128
            image_noise = image + heavy_noise
            image_noise = np.clip(image_noise, 0, 255).astype(np.uint8)
            noise_sample = image_noise - 128
            clean_sample_init = image.astype(np.float32) - 128

            outputs_orig = tracker.track(image)
            outputs_target = tracker.track(image_noise)

            target_score = overlap_ratio(np.array(outputs_orig['target_bbox']), np.array(outputs_target['target_bbox']))

            adversarial_sample = image.astype(np.float32) - 128

            if target_score < 0.8:
                n_steps = 0
                epsilon = 0.05
                delta = 0.05
                weight = 0.5
                para_rate = 0.9
                while True:
                    clean_sample = clean_sample_init + weight * last_preturb

                    trial_sample = clean_sample + forward_perturbation(
                        epsilon * get_diff(clean_sample, noise_sample), adversarial_sample, noise_sample)
                    trial_sample = np.clip(trial_sample, -128, 127)

                    outputs_adv = tracker.track((trial_sample + 128).astype(np.uint8))

                    threshold_1 = overlap_ratio(np.array(outputs_orig['target_bbox']),
                                                np.array(outputs_adv['target_bbox']))

                    threshold_2 = overlap_ratio(np.array(last_gt), np.array(outputs_adv['target_bbox']))
                    threshold = para_rate * threshold_1 + (1 - para_rate) * threshold_2

                    adversarial_sample = trial_sample
                    break

                while True:
                    d_step = 0
                    while True:
                        d_step += 1
                        trial_samples = []
                        score_sum = []

                        for i in range(10):
                            trial_sample = adversarial_sample + orthogonal_perturbation(delta, adversarial_sample,
                                                                                        noise_sample)

                            trial_sample = np.clip(trial_sample, -128, 127)

                            outputs_adv = tracker.track((trial_sample + 128).astype(np.uint8))
                            score_1 = overlap_ratio(np.array(outputs_orig['target_bbox']),
                                                    np.array(outputs_adv['target_bbox']))
                            score_2 = overlap_ratio(np.array(last_gt), np.array(outputs_adv['target_bbox']))

                            score = para_rate * score_1 + (1 - para_rate) * score_2

                            score_sum.append(score)
                            trial_samples.append(trial_sample)

                        d_score = np.mean(np.array(score_sum) <= threshold)
                        if d_score > 0.0:
                            if d_score < 0.3:
                                delta /= 0.9
                            elif d_score > 0.7:
                                delta *= 0.9

                            adversarial_sample = trial_samples[np.argmin(score_sum)]

                            threshold = score_sum[np.argmin(score_sum)]
                            break
                        elif d_step >= 5 or delta > 0.3:
                            break
                        else:
                            delta /= 0.9

                    e_step = 0
                    while True:
                        trial_sample = adversarial_sample + forward_perturbation(
                            epsilon * get_diff(adversarial_sample, noise_sample), adversarial_sample, noise_sample)
                        trial_sample = np.clip(trial_sample, -128, 127)

                        outputs_adv = tracker.track((trial_sample + 128).astype(np.uint8))

                        l2_norm = np.mean(get_diff(clean_sample_init, trial_sample))
                        threshold_1 = overlap_ratio(np.array(outputs_orig['target_bbox']),
                                                    np.array(outputs_adv['target_bbox']))
                        threshold_2 = overlap_ratio(np.array(last_gt), np.array(outputs_adv['target_bbox']))
                        threshold_sum = para_rate * threshold_1 + (1 - para_rate) * threshold_2
                        if threshold_sum <= threshold:
                            adversarial_sample = trial_sample
                            epsilon *= 0.9
                            threshold = threshold_sum
                            break

                        elif e_step >= 30 or l2_norm > 10000:
                            break

                        else:
                            epsilon /= 0.9
                    n_steps += 1

                    if threshold <= target_score or l2_norm > 10000:
                        adversarial_sample = np.clip(adversarial_sample, -128, 127)
                        last_preturb = adversarial_sample - clean_sample
                        image = (adversarial_sample + 128).astype(np.uint8)
                        break
            else:
                adversarial_sample = image + last_preturb
                adversarial_sample = np.clip(adversarial_sample, 0, 255)
                image = adversarial_sample.astype(np.uint8)
            outputs_advvv = tracker.track(image)
            iou_adv = overlap_ratio(np.array(outputs_orig['target_bbox']),
                                    np.array(outputs_advvv['target_bbox']))
            pred_bbox_pre = outputs_advvv['target_bbox']

            adversarial_image = image
            clean_image = image_clean
            noise_adversarial = adversarial_image - image_clean
            if adversarial_image is None:
                raise FileNotFoundError("无法加载对抗性样本图像。请检查文件路径。")

            original_iou = iou_adv

            iou_threshold = original_iou

            add_noise_patch_size = 2 ** round(math.log2(min(image_clean.shape[0] // 4, image_clean.shape[1] // 4)))
            denoise_initial_patch_size = add_noise_patch_size // 1

            denoise_iterations = 5

            attack_instance = PatchTrackingAttack(
                adversarial_image=adversarial_image,
                clean_image=clean_image,
                tracker=tracker,
                original_bbox=outputs_orig['target_bbox'],
                iou_threshold=iou_threshold,
                add_noise_patch_size=add_noise_patch_size,
                denoise_initial_patch_size=denoise_initial_patch_size,
                denoise_iterations=denoise_iterations
            )
            #

            min_tracking_result, denosed_image = attack_instance.attack()

            if min_tracking_result is None:
                min_tracking_result = outputs_advvv

            noise_denosed = denosed_image - image_clean

            adversarial_noises.append(np.mean(np.abs(noise_adversarial)))
            denosed_noises.append(np.mean(np.abs(noise_denosed)))

            image = denosed_image
            final_out = min_tracking_result
            final_iou = overlap_ratio(np.array(outputs_orig['target_bbox']),
                                      np.array(final_out['target_bbox']))
            if final_iou <= iou_threshold:
                out = min_tracking_result
                OUT_IOU = final_iou
            else:
                out = outputs_advvv
                OUT_IOU = iou_threshold

            prev_output = OrderedDict(out)
            _store_outputs(out, {'time': time.time() - start_time})

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        mean_adversarial_noise = np.mean(adversarial_noises)
        median_adversarial_noise = np.median(adversarial_noises)
        mean_denosed_noise = np.mean(denosed_noises)
        median_denosed_noise = np.median(denosed_noises)
        video_name = seq.name  # 获取视频名称
        results_file = os.path.join(output_dir, f'{video_name}_results.txt')
        with open(results_file, 'w') as f:
            f.write(f'Adversarial Mean Noise: {mean_adversarial_noise}\n')
            f.write(f'Adversarial Median Noise: {median_adversarial_noise}\n')
            f.write(f'Denosed Mean Noise: {mean_denosed_noise}\n')
            f.write(f'Denosed Median Noise: {median_denosed_noise}\n')

        return output

    def run_video(self, videofilepath, optional_box=None, debug=None, visdom_info=None, save_results=False):
        """Run the tracker with the vieofile.
        args:
            debug: Debug level.
        """

        params = self.get_parameters()

        debug_ = debug
        if debug is None:
            debug_ = getattr(params, 'debug', 0)
        params.debug = debug_

        params.tracker_name = self.name
        params.param_name = self.parameter_name
        # self._init_visdom(visdom_info, debug_)

        multiobj_mode = getattr(params, 'multiobj_mode', getattr(self.tracker_class, 'multiobj_mode', 'default'))

        if multiobj_mode == 'default':
            tracker = self.create_tracker(params)

        elif multiobj_mode == 'parallel':
            tracker = MultiObjectWrapper(self.tracker_class, params, self.visdom, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

        assert os.path.isfile(videofilepath), "Invalid param {}".format(videofilepath)
        ", videofilepath must be a valid videofile"

        output_boxes = []

        cap = cv.VideoCapture(videofilepath)
        display_name = 'Display: ' + tracker.params.tracker_name
        cv.namedWindow(display_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
        cv.resizeWindow(display_name, 960, 720)
        success, frame = cap.read()
        cv.imshow(display_name, frame)

        def _build_init_info(box):
            return {'init_bbox': box}

        if success is not True:
            print("Read frame from {} failed.".format(videofilepath))
            exit(-1)
        if optional_box is not None:
            assert isinstance(optional_box, (list, tuple))
            assert len(optional_box) == 4, "valid box's foramt is [x,y,w,h]"
            tracker.initialize(frame, _build_init_info(optional_box))
            output_boxes.append(optional_box)
        else:
            while True:
                # cv.waitKey()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)
                break

        while True:
            ret, frame = cap.read()

            if frame is None:
                break

            frame_disp = frame.copy()

            # Draw box
            out = tracker.track(frame)
            state = [int(s) for s in out['target_bbox']]
            output_boxes.append(state)

            cv.rectangle(frame_disp, (state[0], state[1]), (state[2] + state[0], state[3] + state[1]),
                         (0, 255, 0), 5)

            font_color = (0, 0, 0)
            cv.putText(frame_disp, 'Tracking!', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press r to reset', (20, 55), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)
            cv.putText(frame_disp, 'Press q to quit', (20, 80), cv.FONT_HERSHEY_COMPLEX_SMALL, 1,
                       font_color, 1)

            # Display the resulting frame
            cv.imshow(display_name, frame_disp)
            key = cv.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('r'):
                ret, frame = cap.read()
                frame_disp = frame.copy()

                cv.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5,
                           (0, 0, 0), 1)

                cv.imshow(display_name, frame_disp)
                x, y, w, h = cv.selectROI(display_name, frame_disp, fromCenter=False)
                init_state = [x, y, w, h]
                tracker.initialize(frame, _build_init_info(init_state))
                output_boxes.append(init_state)

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        if save_results:
            if not os.path.exists(self.results_dir):
                os.makedirs(self.results_dir)
            video_name = Path(videofilepath).stem
            base_results_path = os.path.join(self.results_dir, 'video_{}'.format(video_name))

            tracked_bb = np.array(output_boxes).astype(int)
            bbox_file = '{}.txt'.format(base_results_path)
            np.savetxt(bbox_file, tracked_bb, delimiter='\t', fmt='%d')

    def get_parameters(self):
        """Get parameters."""
        param_module = importlib.import_module('lib.test.parameter.{}'.format(self.name))
        params = param_module.parameters(self.parameter_name)
        return params

    def _read_image(self, image_file: str):
        if isinstance(image_file, str):
            im = cv.imread(image_file)
            return cv.cvtColor(im, cv.COLOR_BGR2RGB)
        elif isinstance(image_file, list) and len(image_file) == 2:
            return decode_img(image_file[0], image_file[1])
        else:
            raise ValueError("type of image_file should be str or list")


def overlap_ratio(rect1, rect2):
    '''
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or
            2d array of N x [x,y,w,h]
    '''
    rect1 = np.transpose(rect1)

    if rect1.ndim == 1:
        rect1 = rect1[None, :]
    if rect2.ndim == 1:
        rect2 = rect2[None, :]

    left = np.maximum(rect1[:, 0], rect2[:, 0])
    right = np.minimum(rect1[:, 0] + rect1[:, 2], rect2[:, 0] + rect2[:, 2])
    top = np.maximum(rect1[:, 1], rect2[:, 1])
    bottom = np.minimum(rect1[:, 1] + rect1[:, 3], rect2[:, 1] + rect2[:, 3])

    intersect = np.maximum(0, right - left) * np.maximum(0, bottom - top)
    union = rect1[:, 2] * rect1[:, 3] + rect2[:, 2] * rect2[:, 3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


def orthogonal_perturbation(delta, prev_sample, target_sample):
    size = int(max(prev_sample.shape[0] / 4, prev_sample.shape[1] / 4, 224))
    prev_sample_temp = np.resize(prev_sample, (size, size, 3))
    target_sample_temp = np.resize(target_sample, (size, size, 3))
    # Generate perturbation
    perturb = np.random.randn(size, size, 3)
    perturb /= get_diff(perturb, np.zeros_like(perturb))
    perturb *= delta * np.mean(get_diff(target_sample_temp, prev_sample_temp))
    # Project perturbation onto sphere around target
    diff = (target_sample_temp - prev_sample_temp).astype(np.float32)
    diff /= get_diff(target_sample_temp, prev_sample_temp)
    diff = diff.reshape(3, size, size)
    perturb = perturb.reshape(3, size, size)
    for i, channel in enumerate(diff):
        perturb[i] -= np.dot(perturb[i], channel) * channel
    perturb = perturb.reshape(size, size, 3)
    perturb_temp = np.resize(perturb, (prev_sample.shape[0], prev_sample.shape[1], 3))
    return perturb_temp


def forward_perturbation(epsilon, prev_sample, target_sample):
    perturb = (target_sample - prev_sample).astype(np.float32)
    perturb /= get_diff(target_sample, prev_sample)
    perturb *= epsilon
    return perturb


def get_diff(sample_1, sample_2):
    sample_1 = sample_1.reshape(3, sample_1.shape[0], sample_1.shape[1])
    sample_2 = sample_2.reshape(3, sample_2.shape[0], sample_2.shape[1])
    sample_1 = np.resize(sample_1, (3, 271, 271))
    sample_2 = np.resize(sample_2, (3, 271, 271))

    diff = []
    for i, channel in enumerate(sample_1):
        diff.append(np.linalg.norm((channel - sample_2[i]).astype(np.float32)))
    return np.array(diff)


#
class PatchAdversarialRemoval:
    def __init__(self, adversarial_image, clean_image, tracker, original_iou, iou_threshold=0.1, initial_patch_size=64,
                 min_patch_size=1):
        self.adversarial_image = adversarial_image
        self.clean_image = clean_image
        self.tracker = tracker
        self.iou_threshold = iou_threshold
        self.original_iou = original_iou
        self.patch_size = initial_patch_size
        self.min_patch_size = min_patch_size

        self.noise_image = self.adversarial_image - self.clean_image

        self.noise_magnitude_mask = self._initialize_noise_magnitude_mask()
        self.noise_sensitivity_mask = np.ones_like(self.noise_magnitude_mask)

    def _initialize_noise_magnitude_mask(self):
        """
        使用每个块的 L2 范数值初始化噪声幅度掩码。
        """
        h, w, _ = self.adversarial_image.shape

        mask_h = h // self.patch_size
        mask_w = w // self.patch_size

        magnitude_mask = np.zeros((mask_h, mask_w))

        for i in range(mask_h):
            for j in range(mask_w):
                i_start, j_start = i * self.patch_size, j * self.patch_size

                patch = self.noise_image[i_start:i_start + self.patch_size, j_start:j_start + self.patch_size]

                l2_norm = np.linalg.norm(patch)
                magnitude_mask[i, j] = l2_norm

        return magnitude_mask

    def _calculate_iou(self, original_bbox, current_bbox):
        """
        计算去噪后图像与原始跟踪结果之间的交并比（IoU）值。
        """
        current_bbox_array = np.array(current_bbox)
        current_iou = overlap_ratio(original_bbox, current_bbox_array)
        return current_iou

    def reduce_noise(self, original_bbox, max_iterations=1):
        """
        以块为单位逐步减少噪声，直到 IoU 值开始显著上升。
        """
        current_noise = self.noise_image.copy()
        iteration = 0

        original_distance = np.linalg.norm(self.adversarial_image - self.clean_image)

        ori_iou = self.original_iou

        while self.patch_size >= self.min_patch_size and iteration < max_iterations:

            iteration += 1
            h, w, _ = current_noise.shape

            mask_h = h // self.patch_size
            mask_w = w // self.patch_size

            for i in range(mask_h):
                for j in range(mask_w):
                    i_start, j_start = i * self.patch_size, j * self.patch_size

                    i_end = min(i_start + self.patch_size, h)
                    j_end = min(j_start + self.patch_size, w)

                    if i >= self.noise_sensitivity_mask.shape[0] or j >= self.noise_sensitivity_mask.shape[1]:
                        continue

                    if self.noise_sensitivity_mask[i, j] == 0:
                        continue

                    patch = current_noise[i_start:i_end, j_start:j_end].copy()
                    current_noise[i_start:i_end, j_start:j_end] = 0

                    denoised_image = self.clean_image + current_noise
                    tracking_result = self.tracker.track(denoised_image)
                    current_bbox = tracking_result['target_bbox']

                    current_iou = self._calculate_iou(original_bbox, current_bbox)

                    iou_increase = current_iou > self.original_iou
                    if iou_increase:

                        current_noise[i_start:i_end, j_start:j_end] = patch
                        self.noise_sensitivity_mask[i, j] = 0
                    else:

                        ori_iou = current_iou
                        self.noise_image[i_start:i_end, j_start:j_end] = current_noise[i_start:i_end, j_start:j_end]

            self.patch_size //= 2

        self.noise_image = current_noise

        final_noise_image = self.noise_image
        final_denoised_image = self.clean_image + final_noise_image
        final_distance = np.linalg.norm(final_denoised_image - self.clean_image)

        tracking_result_no_denoise = self.tracker.track(self.adversarial_image)
        iou_before = overlap_ratio(original_bbox, np.array(tracking_result_no_denoise['target_bbox']))
        tracking_result_denoised = self.tracker.track(final_denoised_image)
        iou_after = overlap_ratio(original_bbox, np.array(tracking_result_denoised['target_bbox']))
        return final_denoised_image


class PatchTrackingAttack:
    def __init__(self, adversarial_image, clean_image, tracker, original_bbox, iou_threshold,
                 add_noise_patch_size, denoise_initial_patch_size, denoise_iterations, min_patch_size=1):

        self.adversarial_image = torch.from_numpy(adversarial_image).float().cuda()
        self.clean_image = torch.from_numpy(clean_image).float().cuda()
        self.tracker = tracker
        self.original_bbox = torch.tensor(original_bbox).float().cuda()
        self.iou_threshold = torch.tensor(iou_threshold).float().cuda()
        self.add_noise_patch_size = add_noise_patch_size
        self.denoise_initial_patch_size = denoise_initial_patch_size
        self.denoise_iterations = denoise_iterations
        self.patch_size = add_noise_patch_size
        self.min_patch_size = min_patch_size

        self.noise_image = self.adversarial_image - self.clean_image
        self.perturbed_image = self.clean_image.clone()

    def attack(self):
        perturbed_image = self.perturbed_image.clone()
        perturbed_regions = torch.zeros_like(self.noise_image, dtype=torch.bool).cuda()
        tracking_results = []
        iou_scores = []

        mask_h, mask_w = self.noise_image.shape[0] // self.add_noise_patch_size, self.noise_image.shape[
            1] // self.add_noise_patch_size
        attack_successful = False

        for i in range(mask_h):
            for j in range(mask_w):

                i_start, j_start = i * self.add_noise_patch_size, j * self.add_noise_patch_size
                i_end = min(i_start + self.add_noise_patch_size, self.noise_image.shape[0])
                j_end = min(j_start + self.add_noise_patch_size, self.noise_image.shape[1])

                patch = self.noise_image[i_start:i_end, j_start:j_end]
                temp_image = perturbed_image.clone()
                temp_image[i_start:i_end, j_start:j_end] += patch
                temp_image = temp_image.clamp(0, 255).byte()

                temp_image_cpu = temp_image.cpu().numpy()
                tracking_result = self.tracker.track(temp_image_cpu)
                current_bbox = torch.tensor(tracking_result['target_bbox']).float().cuda()
                current_iou = self._calculate_iou(self.original_bbox, current_bbox)

                tracking_results.append(tracking_result)
                iou_scores.append(current_iou.item())

                if current_iou < self.iou_threshold.item():
                    perturbed_image[i_start:i_end, j_start:j_end] += patch
                    perturbed_regions[i_start:i_end, j_start:j_end] = True
                    attack_successful = True
                    break
                else:
                    perturbed_image[i_start:i_end, j_start:j_end] += patch
                    perturbed_regions[i_start:i_end, j_start:j_end] = True

            if attack_successful:
                break

        if attack_successful:
            patch_size = self.denoise_initial_patch_size
            for iteration in range(self.denoise_iterations):

                for i in range(0, self.noise_image.shape[0], patch_size):
                    for j in range(0, self.noise_image.shape[1], patch_size):
                        if not perturbed_regions[i:i + patch_size, j:j + patch_size].any():
                            continue

                        i_end = min(i + patch_size, self.noise_image.shape[0])
                        j_end = min(j + patch_size, self.noise_image.shape[1])

                        temp_image = perturbed_image.clone()
                        temp_image[i:i_end, j:j_end] -= self.noise_image[i:i_end, j:j_end]
                        temp_image = temp_image.clamp(0, 255).byte()

                        temp_image_cpu = temp_image.cpu().numpy()
                        tracking_result = self.tracker.track(temp_image_cpu)
                        current_bbox = torch.tensor(tracking_result['target_bbox']).float().cuda()
                        current_iou = self._calculate_iou(self.original_bbox, current_bbox)

                        tracking_results.append(tracking_result)
                        iou_scores.append(current_iou.item())

                        if current_iou < self.iou_threshold.item():
                            perturbed_image[i:i_end, j:j_end] -= self.noise_image[i:i_end, j:j_end]
                            perturbed_regions[i:i_end, j:j_end] = False

                patch_size //= 2
                if patch_size < self.min_patch_size:
                    break

        if tracking_results:
            min_iou_index = np.argmin(iou_scores)
            min_tracking_result = tracking_results[min_iou_index]
            min_tracking_image = perturbed_image.cpu().numpy()
        else:

            return None, self.adversarial_image.cpu().numpy()

        return min_tracking_result, min_tracking_image

    def _calculate_iou(self, original_bbox, current_bbox):

        left = torch.max(original_bbox[0], current_bbox[0])
        right = torch.min(original_bbox[0] + original_bbox[2], current_bbox[0] + current_bbox[2])
        top = torch.max(original_bbox[1], current_bbox[1])
        bottom = torch.min(original_bbox[1] + original_bbox[3], current_bbox[1] + current_bbox[3])

        intersect = torch.max(right - left, torch.tensor(0.0).cuda()) * torch.max(bottom - top,
                                                                                  torch.tensor(0.0).cuda())
        union = original_bbox[2] * original_bbox[3] + current_bbox[2] * current_bbox[3] - intersect
        iou = torch.clamp(intersect / union, 0, 1)
        return iou