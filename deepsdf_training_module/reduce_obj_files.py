import glob
import multiprocessing
import os

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
            # add "/Scan/Scan.obj" to path
            obj_path = model + '/Scan/Scan.obj'

            obj_files.append(obj_path)

    return obj_files

# ms = pymeshlab.MeshSet()
# ms.load_new_mesh('Scan.obj')
# ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetperc=0.05)
# ms.save_current_mesh('Scan_reduced.obj')

def thread_work(obj_files):
    count = 0
    for obj_file in obj_files:
        count += 1
        print(obj_file, count)

        final_obj_file = None

        # if file is greater than 10MB, reduce it
        if os.path.getsize(obj_file) > 10000000:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(obj_file)
            ms.apply_filter('meshing_decimation_quadric_edge_collapse', targetperc=0.05)
            ms.save_current_mesh(obj_file[:-4] + '_reduced.obj')
            # delete original file
            os.remove(obj_file)

            final_obj_file = obj_file[:-4] + '_reduced.obj'
        else:
            
            final_obj_file = obj_file

                
        mesh = trimesh.load(final_obj_file)

        points, sdf = sample_sdf_near_surface(mesh, number_of_points=40000)

        # save with numpy
        np.save(final_obj_file[:-4] + '_points.npy', points)
        np.save(final_obj_file[:-4] + '_sdf.npy', sdf)


if __name__ == '__main__':
    obj_files = ReadAllObjFiles('data6')

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


