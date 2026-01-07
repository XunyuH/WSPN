import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import torch
import random
import yaml


rec_header_dtd = \
    [
        ("nx", "i4"),  # Number of columns
        ("ny", "i4"),  # Number of rows
        ("nz", "i4"),  # Number of sections

        ("mode", "i4"),  # Types of pixels in the image. Values used by IMOD:
        #  0 = unsigned or signed bytes depending on flag in imodFlags
        #  1 = signed short integers (16 bits)
        #  2 = float (32 bits)
        #  3 = short * 2, (used for complex data)
        #  4 = float * 2, (used for complex data)
        #  6 = unsigned 16-bit integers (non-standard)
        # 16 = unsigned char * 3 (for rgb data, non-standard)

        ("nxstart", "i4"),  # Starting point of sub-image (not used in IMOD)
        ("nystart", "i4"),
        ("nzstart", "i4"),

        ("mx", "i4"),  # Grid size in X, Y and Z
        ("my", "i4"),
        ("mz", "i4"),

        ("xlen", "f4"),  # Cell size; pixel spacing = xlen/mx, ylen/my, zlen/mz
        ("ylen", "f4"),
        ("zlen", "f4"),

        ("alpha", "f4"),  # Cell angles - ignored by IMOD
        ("beta", "f4"),
        ("gamma", "f4"),

        # These need to be set to 1, 2, and 3 for pixel spacing to be interpreted correctly
        ("mapc", "i4"),  # map column  1=x,2=y,3=z.
        ("mapr", "i4"),  # map row     1=x,2=y,3=z.
        ("maps", "i4"),  # map section 1=x,2=y,3=z.

        # These need to be set for proper scaling of data
        ("amin", "f4"),  # Minimum pixel value
        ("amax", "f4"),  # Maximum pixel value
        ("amean", "f4"),  # Mean pixel value

        ("ispg", "i4"),  # space group number (ignored by IMOD)
        ("next", "i4"),  # number of bytes in extended header (called nsymbt in MRC standard)
        ("creatid", "i2"),  # used to be an ID number, is 0 as of IMOD 4.2.23
        ("extra_data", "V30"),  # (not used, first two bytes should be 0)

        # These two values specify the structure of data in the extended header; their meaning depend on whether the
        # extended header has the Agard format, a series of 4-byte integers then real numbers, or has data
        # produced by SerialEM, a series of short integers. SerialEM stores a float as two shorts, s1 and s2, by:
        # value = (sign of s1)*(|s1|*256 + (|s2| modulo 256)) * 2**((sign of s2) * (|s2|/256))
        ("nint", "i2"),
        # Number of integers per section (Agard format) or number of bytes per section (SerialEM format)
        ("nreal", "i2"),  # Number of reals per section (Agard format) or bit
        # Number of reals per section (Agard format) or bit
        # flags for which types of short data (SerialEM format):
        # 1 = tilt angle * 100  (2 bytes)
        # 2 = piece coordinates for montage  (6 bytes)
        # 4 = Stage position * 25    (4 bytes)
        # 8 = Magnification / 100 (2 bytes)
        # 16 = Intensity * 25000  (2 bytes)
        # 32 = Exposure dose in e-/A2, a float in 4 bytes
        # 128, 512: Reserved for 4-byte items
        # 64, 256, 1024: Reserved for 2-byte items
        # If the number of bytes implied by these flags does
        # not add up to the value in nint, then nint and nreal
        # are interpreted as ints and reals per section

        ("extra_data2", "V20"),  # extra data (not used)
        ("imodStamp", "i4"),  # 1146047817 indicates that file was created by IMOD
        ("imodFlags", "i4"),  # Bit flags: 1 = bytes are stored as signed

        # Explanation of type of data
        ("idtype", "i2"),  # ( 0 = mono, 1 = tilt, 2 = tilts, 3 = lina, 4 = lins)
        ("lens", "i2"),
        # ("nd1", "i2"),  # for idtype = 1, nd1 = axis (1, 2, or 3)
        # ("nd2", "i2"),
        ("nphase", "i4"),
        ("vd1", "i2"),  # vd1 = 100. * tilt increment
        ("vd2", "i2"),  # vd2 = 100. * starting angle

        # Current angles are used to rotate a model to match a new rotated image.  The three values in each set are
        # rotations about X, Y, and Z axes, applied in the order Z, Y, X.
        ("triangles", "f4", 6),  # 0,1,2 = original:  3,4,5 = current

        ("xorg", "f4"),  # Origin of image
        ("yorg", "f4"),
        ("zorg", "f4"),

        ("cmap", "S4"),  # Contains "MAP "
        ("stamp", "u1", 4),  # First two bytes have 17 and 17 for big-endian or 68 and 65 for little-endian

        ("rms", "f4"),  # RMS deviation of densities from mean density

        ("nlabl", "i4"),  # Number of labels with useful data
        ("labels", "S80", 10)  # 10 labels of 80 charactors
    ]


def read_mrc(filename, filetype='image'):
    fd = open(filename, 'rb')
    header = np.fromfile(fd, dtype=rec_header_dtd, count=1)

    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]

    if header[0][3] == 1:
        data_type = 'int16'
    elif header[0][3] == 2:
        data_type = 'float32'
    elif header[0][3] == 4:
        data_type = 'single'
        nx = nx * 2
    elif header[0][3] == 6:
        data_type = 'uint16'

    data = np.ndarray(shape=(nx, ny, nz))
    imgrawdata = np.fromfile(fd, data_type)
    fd.close()

    if filetype == 'image':
        for iz in range(nz):
            data_2d = imgrawdata[nx * ny * iz:nx * ny * (iz + 1)]
            data[:, :, iz] = data_2d.reshape(nx, ny, order='F')
    else:
        data = imgrawdata
    return header, data


def write_mrc(filename,
              img_data,
              header):

    if img_data.dtype == 'int16':
        header[0][3] = 1
    elif img_data.dtype == 'float32':
        header[0][3] = 2
    elif img_data.dtype == 'uint16':
        header[0][3] = 6

    fd = open(filename, 'wb')
    for i in range(len(rec_header_dtd)):
        header[rec_header_dtd[i][0]].tofile(fd)

    nx, ny, nz = header['nx'][0], header['ny'][0], header['nz'][0]
    imgrawdata = np.ndarray(shape=(nx * ny * nz), dtype='uint16')
    for iz in range(nz):
        imgrawdata[nx * ny * iz:nx * ny * (iz + 1)] = img_data[:, :, iz].reshape(nx * ny, order='F')
    imgrawdata.tofile(fd)

    fd.close()
    return


def norm(img):
    if type(img) == torch.Tensor:
        return (img - torch.min(img)) / (torch.max(img) - torch.min(img))
    else:
        return (img - np.min(img)) / (np.max(img) - np.min(img))


def mrc2wf(data_path):
    _, data = read_mrc(data_path)
    data = data.astype(np.float32)

    img_list = []
    for i in range(data.shape[-1]):
        img = norm(data[:, :, i]) * 255
        img_list.append(img)
    return np.mean(np.stack(img_list, axis=0), axis=0)


def mrc2gt(data_path):
    _, data = read_mrc(data_path)
    data = data.astype(np.float32)
    return norm(data) * 255


def convert_mrc(raw_data_dir):
    raw_data_base = Path(raw_data_dir)
    convert_base = Path.cwd() / 'data' / 'BioSR'

    specimens = ['CCPs', 'ER', 'Microtubules', 'F-actin']
    for specimen in specimens:
        search_base = raw_data_base / specimen
        for mode in ['WF', 'GT']:
            if mode == 'WF':
                pattern = 'RawSIMData_level_*.mrc'
                wf_files = list(search_base.rglob(pattern))
                with tqdm(total=len(wf_files)) as pbar:
                    pbar.set_description(f'Converting {specimen} {mode}')
                    for wf in wf_files:
                        if specimen == 'ER':
                            cell = wf.parts[-3]
                        else:
                            cell = wf.parts[-2]
                        level = wf.stem.split('_')[-1]
                        saved_dir = convert_base / specimen / cell / mode
                        if not saved_dir.exists():
                            saved_dir.mkdir(parents=True)
                        saved_path = saved_dir / f'{level}.tiff'
                        wf_np = mrc2wf(str(wf))
                        cv2.imwrite(str(saved_path), wf_np)
                        pbar.update(1)
            else:
                if specimen == 'ER':
                    pattern = 'GTSIM_level_*.mrc'
                else:
                    pattern = 'SIM_gt.mrc'
                gt_files = list(search_base.rglob(pattern))
                with tqdm(total=len(gt_files)) as pbar:
                    pbar.set_description(f'Converting {specimen} {mode}')
                    for gt in gt_files:
                        if specimen == 'ER':
                            cell = gt.parts[-3]
                        else:
                            cell = gt.parts[-2]
                        level = gt.stem.split('_')[-1]
                        saved_dir = convert_base / specimen / cell / mode
                        if not saved_dir.exists():
                            saved_dir.mkdir(parents=True)
                        saved_path = saved_dir / f'{level}.tiff'
                        gt_np = mrc2gt(str(gt))
                        cv2.imwrite(str(saved_path), gt_np)
                        pbar.update(1)


def random_partition(dataset,
                     specimen,
                     partition_size,
                     partition):

    partition_base = (Path.cwd() /
                      'partition' /
                      dataset /
                      specimen)
    total = sum(partition_size)
    random_idx = random.sample(list(map(str, range(total))), total)

    train_idx = random_idx[:partition_size[0]]
    train_partition_dir = partition_base / 'train'
    if not train_partition_dir.exists():
        train_partition_dir.mkdir(parents=True)
    train_partition = train_partition_dir / f'{partition}.yaml'
    with open(train_partition, 'w') as f:
        yaml.dump(train_idx, f)

    valid_idx = random_idx[partition_size[0]:partition_size[0]+partition_size[1]]
    valid_partition_dir = partition_base / 'validate'
    if not valid_partition_dir.exists():
        valid_partition_dir.mkdir(parents=True)
    valid_partition = valid_partition_dir / f'{partition}.yaml'
    with open(valid_partition, 'w') as f:
        yaml.dump(valid_idx, f)

    if len(partition_size) == 3:
        test_idx = random_idx[partition_size[0]+partition_size[1]:]
        test_partition_dir = partition_base / 'test'
        if not test_partition_dir.exists():
            test_partition_dir.mkdir(parents=True)
        test_partition = test_partition_dir / f'{partition}.yaml'
        with open(test_partition, 'w') as f:
            yaml.dump(test_idx, f)


def random_crop(dataset,
                specimen,
                num_data,
                num_crop,
                original_size,
                crop_size,
                crop):

    crop_dir = (Path.cwd() /
                'crop' /
                dataset /
                specimen)

    if not crop_dir.exists():
        crop_dir.mkdir(parents=True)
    crop_path = crop_dir / f'{crop}.yaml'
    with open(crop_path, 'w') as f:
        pairs = []
        for _ in range(num_data * num_crop):
            x_random_lr = random.randint(0, original_size - crop_size)
            y_random_lr = random.randint(0, original_size - crop_size)
            pairs.append([x_random_lr, y_random_lr])
        yaml.dump(pairs, f)


def read_cell_list(dataset,
                   specimen,
                   mode,
                   partition):

    partition_file = (Path.cwd() /
                      'partition' /
                      dataset /
                      specimen /
                      mode /
                      f'{partition}.yaml')

    full_list = sorted(list((Path.cwd() /
                             'data' /
                             dataset /
                             specimen).iterdir()))

    with open(partition_file, 'r') as f:
        idx = yaml.safe_load(f)
    cell_list = []
    for i in idx:
        cell_list.append(full_list[i].parts[-1])
    return cell_list


def read_crop_list(dataset,
                   specimen,
                   crop):

    crop_file = (Path.cwd() /
                 'crop' /
                 dataset /
                 specimen /
                 f'{crop}.yaml')

    with open(crop_file, 'r') as f:
        return yaml.safe_load(f)


def check_size(img, size):
    if img.shape[-1] > size:
        if len(img.shape) == 3:
            resize_channel = []
            for i in range(img.shape[0]):
                single_channel = img[i, :, :]
                resize_channel.append(cv2.resize(single_channel,
                                                 (size, size),
                                                 interpolation=cv2.INTER_AREA))
            return np.stack(resize_channel, axis=0)
        else:
            return cv2.resize(img,
                              (size, size),
                              interpolation=cv2.INTER_AREA)
    elif img.shape[-1] < size:
        if len(img.shape) == 3:
            resize_channel = []
            for i in range(img.shape[0]):
                single_channel = img[i, :, :]
                resize_channel.append(cv2.resize(single_channel,
                                                 (size, size),
                                                 interpolation=cv2.INTER_LANCZOS4))
            return np.stack(resize_channel, axis=0)
        else:
            return cv2.resize(img,
                              (size, size),
                              interpolation=cv2.INTER_LANCZOS4)
    else:
        return img


def gray2pseudo_green(img):
    expanded_channel = np.zeros_like(img)
    return np.stack([expanded_channel, img, expanded_channel], axis=-1).astype(np.uint8)


def pseudo_green2gray(img):
    return img[:, :, 1]
