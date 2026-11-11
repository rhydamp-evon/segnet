# import the necessary packages
import numpy as np
import cv2
import math


def final_adjustment(img, org, tint_val, modified_val):
    """
    Returns final image after adjusting tint of the image and keeping fraction
    of modified value.
    """
    if tint_val != 1.0:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        BGR = cv2.split(img)
        BGR[0] = tint_val * BGR[0] + (1.0 - tint_val) * grey
        BGR[1] = tint_val * BGR[1] + (1.0 - tint_val) * grey
        BGR[2] = tint_val * BGR[2] + (1.0 - tint_val) * grey
        img = cv2.merge(BGR)
    if modified_val != 1.0:
        img = modified_val * img + (1.0 - modified_val) * org
    return img


def saturation_processing(target, original, sat_val=1.0):
    """
    Returns image after adjusting saturation value for the image.
    """
    if sat_val != 1.0:
        # // Colour saturation will be computed in accordance
        # // with the definition used for the HSV colour space.
        target = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        temp = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
        hsv = cv2.split(target)
        tmp_hsv = cv2.split(temp)

        if sat_val < 0:
            # //  'SatVal' is less than 0, then compute 'SatVal'
            # //  as the ratio of the largest saturation value
            # //  in the original image to the largest value in
            # //  processed image.
            # min_v, max_v, min_pt, max_pt = cv2.minMaxLoc(hsv[1])
            max1 = np.max(hsv[1])
            max2 = np.max(tmp_hsv[1])
            # min_v1, max_v1, min_pt1, max_pt1 = cv2.minMaxLoc(tmp_hsv[1])
            sat_val = max2 / max1

        # // Compute a weighted mix of the processed target
        # // saturation channel  and the original image
        # // saturation channel to define an initial
        # // reference saturation channel.
        tmp_hsv[1] = cv2.addWeighted(hsv[1], sat_val, tmp_hsv[1], 1 - sat_val, 0.0)

        # // The initial reference saturation channel values
        # // will apply only to those pixels where the
        # // saturation in the processed image exceeds that
        # // in the original target image.
        _, mask = cv2.threshold((hsv[1] - tmp_hsv[1]), 0, 1, cv2.THRESH_BINARY)
        mask = mask.astype("uint8")
        # // Create a new reference saturation channel which is taken
        # // from the initial reference saturation channel except
        # // where the mask function indicates that the saturation
        # // value of the original image should be used.
        # // This gives a modified reference saturation channel.
        mask = mask.astype("float32")
        temp1 = cv2.multiply(tmp_hsv[1], mask)
        temp2 = cv2.multiply(hsv[1], 1 - mask)
        tmp_hsv[1] = cv2.add(temp1, temp2)

        # // Now match the mean and standard deviation of the
        # // saturation channel for the processed image channel
        # // to the mean and standard deviation of the modified
        # // reference saturation channel. This give the final
        # // saturation channel which is the output from
        # // the saturation processing
        tmean, tdev = hsv[1].mean(), hsv[1].std()
        tmpmean, tmpdev = tmp_hsv[1].mean(), tmp_hsv[1].std()

        hsv[1] = (hsv[1] - tmean) / tdev
        hsv[1] = hsv[1] * tmpdev + tmpmean
        target = cv2.merge(hsv)
        target = cv2.cvtColor(target, cv2.COLOR_HSV2BGR)
    return target


def full_shading(target, original, source, extra_shading=False, shader_val=50.0):
    """
    Returns image after applying shading to the monochrome image.
    """
    if extra_shading:
        greyt = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        greyp = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        greys = source.copy()

        smean, sdev = greys.mean(), greys.std()
        tmean, tdev = greyt.mean(), greyt.std()
        greyt = (greyt - tmean) / tdev

        greyt = (
            greyt * (shader_val * sdev + (1.0 - shader_val) * tdev)
            + shader_val * smean
            + (1.0 - shader_val) * tmean
        )

        min_mat = np.full(greyp.shape, 1 / 255.0, dtype=np.float32)
        greyp = cv2.max(greyp, min_mat)
        min_mat_2 = np.full(greyt.shape, 0.0, dtype=np.float32)
        greyt = cv2.max(greyt, min_mat_2)
        chans = cv2.split(target)
        chans[0] = cv2.divide(chans[0], greyp)
        chans[0] = np.multiply(chans[0], greyt)
        chans[1] = cv2.divide(chans[1], greyp)
        chans[1] = np.multiply(chans[1], greyt)
        chans[2] = cv2.divide(chans[2], greyp)
        chans[2] = np.multiply(chans[2], greyt)
        target = cv2.merge(chans)
    return target


def channel_condition(chan, schan):
    """
    Returns image after stripping off colors from original image for the further adjustment.
    Function makes sure it keeps as much as channel information as possible
    """
    wval = 0.25
    _, mask = cv2.threshold(schan, 0, 1, cv2.THRESH_BINARY)
    mask = mask.astype("uint8")
    smeanu = cv2.mean(schan, mask)
    #  np.mean(np.array([schan, mask]), axis=0)
    WU = cv2.exp(-schan * wval / smeanu[0])
    WU = cv2.multiply((1 - WU), (1 - WU))
    wmean = cv2.mean(WU, mask)
    # wmean = np.mean(np.array([WU, mask]), axis=0)
    chanu = cv2.pow(schan, 4)
    temp = cv2.multiply(WU, chanu)
    # smeanu = np.mean(np.array([temp, mask]), axis=0) / wmean[0]
    # smeanu = cv2.mean(temp, mask) / wmean[0]
    smeanu = [i / wmean[0] for i in cv2.mean(temp, mask)]

    """ 
  Processing Lower channel (schan)
  """
    # smeanL = np.mean(np.array([schan, (1-mask)]), axis=0)
    smeanL = cv2.mean(schan, 1 - mask)
    WL = cv2.exp(-schan * wval / smeanL[0])
    WL = cv2.multiply((1 - WL), (1 - WL))
    # wmean = np.mean(np.array([WL, 1-mask]), axis=0)
    wmean = cv2.mean(WL, 1 - mask)
    chanL = cv2.pow(schan, 4)
    temp = cv2.multiply(WL, chanL)
    # smeanL = np.mean(np.array([temp, 1-mask]), axis=0) / wmean[0]
    smeanL = [i / wmean[0] for i in cv2.mean(temp, 1 - mask)]
    # smeanL = cv2.mean(temp,1- mask) / wmean[0]

    _, mask = cv2.threshold(chan, 0, 1, cv2.THRESH_BINARY)
    mask = mask.astype("uint8")
    # tmeanu = np.mean(np.array([chan, mask]), axis=0)
    tmeanu = cv2.mean(chan, mask)
    WU = cv2.exp(-chan * wval / tmeanu[0])
    WU = cv2.multiply((1 - WU), (1 - WU))
    # wmean = np.mean(np.array([WU, mask]), axis=0)
    wmean = cv2.mean(WU, mask)
    chanu = cv2.pow(chan, 4)
    temp = cv2.multiply(WU, chanu)
    # tmeanu = np.mean(np.array([temp, mask]), axis=0) / wmean[0]
    tmeanu = [i / wmean[0] for i in cv2.mean(temp, mask)]
    # tmeanu = cv2.mean(temp, mask) / wmean[0]

    # tmeanL = np.mean(np.array([chan, (1-mask)]), axis=0)
    tmeanL = cv2.mean(chan, 1 - mask)
    WL = cv2.exp(-chan * wval / tmeanL[0])
    WL = cv2.multiply((1 - WL), (1 - WL))
    # wmean = np.mean(np.array([WL, 1-mask]), axis=0)
    wmean = cv2.mean(WL, 1 - mask)
    chanL = cv2.pow(chan, 4)
    temp = cv2.multiply(WL, chanL)
    # tmeanL = np.mean(np.array([temp, 1-mask]), axis=0) / wmean[0]
    # tmeanL = cv2.mean(temp, 1-mask) / wmean[0]
    tmeanL = [i / wmean[0] for i in cv2.mean(temp, 1 - mask)]

    k = math.sqrt(math.sqrt(smeanu[0] / tmeanu[0]))
    temp = 1 + WU * (k - 1)
    # temp = 1 + temp
    chanu = cv2.multiply(temp, chan)
    # print('chanu: ', chanu[0])

    k = math.sqrt(math.sqrt(smeanL[0] / tmeanL[0]))
    temp = 1 + WL * (k - 1)
    # temp = 1 + temp
    chanL = cv2.multiply(temp, chan)
    # print('chanL: ', chanL[0])

    chan = np.zeros(chan.shape, dtype=np.float32)
    chanu = chanu.astype("float32")
    chanL = chanL.astype("float32")
    # plt.imshow(mask * 255)
    # plt.show()
    # cv2_imshow(mask*255)
    # chan = cv2.add(chan, chanu, mask)

    mask = mask.astype("float32")
    chan1 = cv2.multiply(chanu, mask)
    chan2 = cv2.multiply(chanL, 1 - mask)
    chan = cv2.add(chan1, chan2)

    tmean, tdev = cv2.meanStdDev(chan)
    chan = (chan - tmean[0]) / tdev[0]
    # print(chan[0])
    return chan


def adjust_covariance(lab, slab, cov_lim):
    if cov_lim != 0.0:
        temp1 = cv2.multiply(lab[1], lab[2])
        temp2 = cv2.mean(temp1)
        tcrosscorr = temp2[0]

        temp1 = cv2.multiply(slab[1], slab[2])
        temp2 = cv2.mean(temp1)
        scrosscorr = temp2[0]

        # left = 0.5 * math.sqrt((1+scrosscorr) / (1+tcrosscorr))
        # right = 0.5 * math.sqrt((1-scrosscorr)/(1-tcrosscorr))

        W1 = 0.5 * math.sqrt((1 + scrosscorr) / (1 + tcrosscorr)) + 0.5 * math.sqrt(
            (1 - scrosscorr) / (1 - tcrosscorr)
        )
        W2 = 0.5 * math.sqrt((1 + scrosscorr) / (1 + tcrosscorr)) - 0.5 * math.sqrt(
            (1 - scrosscorr) / (1 - tcrosscorr)
        )

        if abs(W2) > cov_lim * abs(W1):
            W2 = math.copysign(cov_lim * W1, W2)
            norm = 1.0 / math.sqrt(W1 * W1 + W2 * W2 + 2 * W1 * W2 * tcrosscorr)
            W1 = W1 * norm
            W2 = W2 * norm

        z1 = lab[1].copy()
        lab[1] = W1 * z1 + W2 * lab[2]
        lab[2] = W1 * lab[2] + W2 * z1

    temp1 = cv2.merge(lab)
    return temp1


RGB_to_LMS = np.array(
    [[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]]
)

i3 = 1 / math.sqrt(3)
i6 = 1 / math.sqrt(6)
i2 = 1 / math.sqrt(2)

LMS_to_lab = np.array([[i3, i3, i3], [i6, i6, -2 * i6], [i2, -i2, 0]])


def convertTolab(image):
    img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_lms = cv2.transform(img_RGB, RGB_to_LMS)
    eps = 1.0 / 255
    min_mat = np.full(image.shape, eps, dtype=np.float32)
    img_lms = cv2.max(img_lms, min_mat)
    img_lms = cv2.log(img_lms)
    img_lms = img_lms / math.log(10)
    img_lab = cv2.transform(img_lms, LMS_to_lab)
    return img_lab


def convertFromlab(image):
    _, temp = cv2.invert(LMS_to_lab)
    img_lms = cv2.transform(image, temp)
    img_lms = cv2.exp(img_lms)
    img_lms = cv2.pow(img_lms, math.log(10.0))
    _, temp = cv2.invert(RGB_to_LMS)
    img_RGB = cv2.transform(img_lms, temp)
    img_RGB = img_RGB.astype("float32")
    img_BGR = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)
    return img_BGR


def core_processing(target, source, cross_cov_lim, reshape_iter, shader_val):
    source = convertTolab(source)
    target = convertTolab(target)
    # compute color statistics for the target and source images
    # tmean = [0, 0, 0]
    # tdev = [0, 0, 0]
    tmean, tdev = cv2.meanStdDev(target)
    # tmean = [i[0] for i in tmean]
    # tdev = [i[0] for i in tdev]
    # tmean[0], tdev[0] = target[0].mean(axis=1), target[1].std(axis=0)
    # tmean[1], tdev[1] = target[1].mean(), target[1].std()
    # tmean[2], tdev[2] = target[2].mean(), target[2].std()
    lab = cv2.split(target)

    # smean = [0, 0, 0]
    # sdev = [0, 0, 0]
    smean, sdev = cv2.meanStdDev(source)
    # smean[0], sdev[0] = source[0].mean(), source[0].std()
    # smean[1], sdev[1] = source[1].mean(), source[1].std()
    # smean[2], sdev[2] = source[2].mean(), source[2].std()
    slab = cv2.split(source)

    lab[0] = (lab[0] - tmean[0]) / tdev[0]
    lab[1] = (lab[1] - tmean[1]) / tdev[1]
    lab[2] = (lab[2] - tmean[2]) / tdev[2]

    slab[0] = (slab[0] - smean[0]) / sdev[0]
    slab[1] = (slab[1] - smean[1]) / sdev[1]
    slab[2] = (slab[2] - smean[2]) / sdev[2]

    jcnt = reshape_iter
    while jcnt > math.ceil((reshape_iter + 1) // 2):
        lab[1] = channel_condition(lab[1], slab[1])
        lab[2] = channel_condition(lab[2], slab[2])
        jcnt = jcnt - 1

    target = adjust_covariance(lab, slab, cross_cov_lim)
    lab = cv2.split(target)

    while jcnt > 0:
        lab[1] = channel_condition(lab[1], slab[1])
        lab[2] = channel_condition(lab[2], slab[2])
        jcnt = jcnt - 1

    lab[1] = lab[1] * sdev[1] + smean[1]
    lab[2] = lab[2] * sdev[2] + smean[2]
    lab[0] = (
        lab[0] * (shader_val * sdev[0] + (1.0 - shader_val) * tdev[0])
        + shader_val * smean[0]
        + (1.0 - shader_val) * tmean[0]
    )
    resultant = cv2.merge(lab)
    resultant = convertFromlab(resultant)
    return resultant


def color_transfer(target, source, tint_val=100, mod_val=50):
    cross_cov_lim = 0.5  # p1
    per_shade_shift = 1.0  # p2
    per_sat_shift = 1.0  # p3
    reshape_iter = 2  # p4
    # tint_val = 100                  #p5
    # mod_val = 50                   #p6
    extra_shading = True

    # source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
    # target = cv2.cvtColor(target, cv2.COLOR_RGB2BGR)
    source = source.astype("float32") * (1.0 / 255.0)
    target = target.astype("float32") * (1.0 / 255.0)
    original = target.copy()
    target = core_processing(
        target, source, cross_cov_lim, reshape_iter, per_shade_shift / 100
    )
    source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    target = saturation_processing(target, original, per_sat_shift / 100)
    target = full_shading(
        target, original, source, extra_shading, per_shade_shift / 100
    )
    target = final_adjustment(target, original, tint_val / 100, mod_val / 100)
    target = target * 255
    # target = cv2.cvtColor(target, cv2.COLOR_LAB2BGR)
    return target[:, :, ::-1]
