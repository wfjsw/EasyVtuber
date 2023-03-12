import functools
import math
import numpy as np
from args import args
import tha2

ifm_converter = tha2.poser.modes.mode_20_wx.IFacialMocapPoseConverter20()

simplify_arr = [1000] * ifm_converter.pose_size
if args.simplify >= 1:
    simplify_arr = [200] * ifm_converter.pose_size
    simplify_arr[ifm_converter.eye_wink_left_index] = 50
    simplify_arr[ifm_converter.eye_wink_right_index] = 50
    simplify_arr[ifm_converter.eye_happy_wink_left_index] = 50
    simplify_arr[ifm_converter.eye_happy_wink_right_index] = 50
    simplify_arr[ifm_converter.eye_surprised_left_index] = 30
    simplify_arr[ifm_converter.eye_surprised_right_index] = 30
    simplify_arr[ifm_converter.iris_rotation_x_index] = 25
    simplify_arr[ifm_converter.iris_rotation_y_index] = 25
    simplify_arr[ifm_converter.eye_raised_lower_eyelid_left_index] = 10
    simplify_arr[ifm_converter.eye_raised_lower_eyelid_right_index] = 10
    simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 5
    simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 5
    simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 5
    simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 5
if args.simplify >= 2:
    simplify_arr[ifm_converter.head_x_index] = 100
    simplify_arr[ifm_converter.head_y_index] = 100
    simplify_arr[ifm_converter.eye_surprised_left_index] = 10
    simplify_arr[ifm_converter.eye_surprised_right_index] = 10
    simplify_arr[ifm_converter.mouth_lowered_corner_left_index] = 0
    simplify_arr[ifm_converter.mouth_lowered_corner_right_index] = 0
    simplify_arr[ifm_converter.mouth_raised_corner_left_index] = 0
    simplify_arr[ifm_converter.mouth_raised_corner_right_index] = 0
if args.simplify >= 3:
    simplify_arr[ifm_converter.iris_rotation_x_index] = 20
    simplify_arr[ifm_converter.iris_rotation_y_index] = 20
    simplify_arr[ifm_converter.eye_wink_left_index] = 32
    simplify_arr[ifm_converter.eye_wink_right_index] = 32
    simplify_arr[ifm_converter.eye_happy_wink_left_index] = 32
    simplify_arr[ifm_converter.eye_happy_wink_right_index] = 32
if args.simplify >= 4:
    simplify_arr[ifm_converter.head_x_index] = 50
    simplify_arr[ifm_converter.head_y_index] = 50
    simplify_arr[ifm_converter.neck_z_index] = 100
    simplify_arr[ifm_converter.iris_rotation_x_index] = 10
    simplify_arr[ifm_converter.iris_rotation_y_index] = 10
    simplify_arr[ifm_converter.eye_wink_left_index] = 24
    simplify_arr[ifm_converter.eye_wink_right_index] = 24
    simplify_arr[ifm_converter.eye_happy_wink_left_index] = 24
    simplify_arr[ifm_converter.eye_happy_wink_right_index] = 24
    simplify_arr[ifm_converter.eye_surprised_left_index] = 8
    simplify_arr[ifm_converter.eye_surprised_right_index] = 8

for i in range(4, args.simplify):
    simplify_arr = [max(math.ceil(x * 0.8), 5) for x in simplify_arr]

simplify_view = np.array(simplify_arr)
simplify_view.flags.writeable = False

@functools.cache
def simplify(model_input_arr):
    model_input_arr = np.asarray(model_input_arr)
    if args.simplify >= 2:
        model_input_arr[ifm_converter.eye_wink_left_index] += model_input_arr[
            ifm_converter.eye_happy_wink_left_index]
        model_input_arr[ifm_converter.eye_happy_wink_left_index] = model_input_arr[
                                                                       ifm_converter.eye_wink_left_index] / 2
        model_input_arr[ifm_converter.eye_wink_left_index] = model_input_arr[
                                                                 ifm_converter.eye_wink_left_index] / 2
        model_input_arr[ifm_converter.eye_wink_right_index] += model_input_arr[
            ifm_converter.eye_happy_wink_right_index]
        model_input_arr[ifm_converter.eye_happy_wink_right_index] = model_input_arr[
                                                                        ifm_converter.eye_wink_right_index] / 2
        model_input_arr[ifm_converter.eye_wink_right_index] = model_input_arr[
                                                                  ifm_converter.eye_wink_right_index] / 2

        uosum = model_input_arr[ifm_converter.mouth_uuu_index] + \
                model_input_arr[ifm_converter.mouth_ooo_index]
        model_input_arr[ifm_converter.mouth_ooo_index] = uosum
        model_input_arr[ifm_converter.mouth_uuu_index] = 0
        is_open = (model_input_arr[ifm_converter.mouth_aaa_index] + model_input_arr[
            ifm_converter.mouth_iii_index] + uosum) > 0
        model_input_arr[ifm_converter.mouth_lowered_corner_left_index] = 0
        model_input_arr[ifm_converter.mouth_lowered_corner_right_index] = 0
        model_input_arr[ifm_converter.mouth_raised_corner_left_index] = 0.5 if is_open else 0
        model_input_arr[ifm_converter.mouth_raised_corner_right_index] = 0.5 if is_open else 0

    if args.simplify >= 4:
        model_input_arr[ifm_converter.eye_raised_lower_eyelid_left_index] = 0
        model_input_arr[ifm_converter.eye_raised_lower_eyelid_right_index] = 0
        model_input_arr[ifm_converter.eye_wink_left_index] += model_input_arr[
            ifm_converter.eye_wink_right_index]
        model_input_arr[ifm_converter.eye_wink_right_index] = model_input_arr[
                                                                  ifm_converter.eye_wink_left_index] / 2
        model_input_arr[ifm_converter.eye_wink_left_index] = model_input_arr[
                                                                 ifm_converter.eye_wink_left_index] / 2

        model_input_arr[ifm_converter.eye_surprised_left_index] += model_input_arr[
            ifm_converter.eye_surprised_right_index]
        model_input_arr[ifm_converter.eye_surprised_right_index] = model_input_arr[
                                                                       ifm_converter.eye_surprised_left_index] / 2
        model_input_arr[ifm_converter.eye_surprised_left_index] = model_input_arr[
                                                                      ifm_converter.eye_surprised_left_index] / 2

        model_input_arr[ifm_converter.eye_happy_wink_left_index] += model_input_arr[
            ifm_converter.eye_happy_wink_right_index]
        model_input_arr[ifm_converter.eye_happy_wink_right_index] = model_input_arr[
                                                                        ifm_converter.eye_happy_wink_left_index] / 2
        model_input_arr[ifm_converter.eye_happy_wink_left_index] = model_input_arr[
                                                                       ifm_converter.eye_happy_wink_left_index] / 2
        model_input_arr[ifm_converter.mouth_aaa_index] = min(
            model_input_arr[ifm_converter.mouth_aaa_index] +
            model_input_arr[ifm_converter.mouth_ooo_index] / 2 +
            model_input_arr[ifm_converter.mouth_iii_index] / 2 +
            model_input_arr[ifm_converter.mouth_uuu_index] / 2, 1
        )
        model_input_arr[ifm_converter.mouth_ooo_index] = 0
        model_input_arr[ifm_converter.mouth_iii_index] = 0
        model_input_arr[ifm_converter.mouth_uuu_index] = 0

    for i in range(0, len(simplify_view)):
        if simplify_arr[i] > 0:
            model_input_arr[i] = round(model_input_arr[i] * simplify_arr[i]) / simplify_arr[i]
    return model_input_arr
