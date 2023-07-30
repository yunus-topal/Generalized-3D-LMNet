import glob
import multiprocessing
import os
import pathlib

from mesh_to_sdf import sample_sdf_near_surface
import trimesh
import pyrender
import numpy as np
import pymeshlab


def ReadAllObjFiles(path):

    obj_files = []

    # glob all folders
    categories = glob.glob(path + '/*')
    for category in categories:

        # glob all model folders
        models = glob.glob(category + '/*')
        for model in models:
            # find the obj file in the Scan folder
            files = glob.glob(model + '/Scan/Scan.obj') + glob.glob(model + '/Scan/Scan_reduced.obj')

            if len(files) > 0:
                obj_files.append(files[0])
            else:
                print("No obj file found in folder: ", model)
    # sort
    obj_files.sort()

    # # todo
    # obj_files = obj_files[:1]

    return obj_files


def thread_work(obj_files):
    count = 0
    for obj_file in obj_files:
        count += 1
        print(obj_file, count)

        # get parent folder of obj file
        obj_file_path = pathlib.Path(obj_file)
        obj_file_parent = obj_file_path.parent

        # create path of smooth.obj file
        smooth_obj_file_path = obj_file_parent / 'smooth.obj'

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(obj_file)
        ms.apply_filter('apply_coord_laplacian_smoothing_scale_dependent', stepsmoothnum = 10)
        ms.save_current_mesh(str(smooth_obj_file_path))

                
        mesh = trimesh.load(str(smooth_obj_file_path))

        points, sdf = sample_sdf_near_surface(mesh, number_of_points=40000)

        # save with numpy
        np.save(str(smooth_obj_file_path)[:-4] + '_points.npy', points)
        np.save(str(smooth_obj_file_path)[:-4] + '_sdf.npy', sdf)


if __name__ == '__main__':
    obj_files = ReadAllObjFiles('data')

    thread_num = 10

    obj_files_per_thread = []

    for i in range(thread_num):
        obj_files_per_thread.append([])

    for i in range(len(obj_files)):
        obj_files_per_thread[i % thread_num].append(obj_files[i])

    threads = []

    # now create thread_num threads and send obj_files_per_thread[i] to thread[i]
    for i in range(thread_num):
        p = multiprocessing.Process(target=thread_work, args=(obj_files_per_thread[i],))
        p.start()
        threads.append(p)

    for thread in threads:
        thread.join()

    print('done')


