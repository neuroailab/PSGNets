import numpy as np
import json
import os
import sys
import tensorflow as tf

class Attribute(object):
    def __init__(self, name, shape, dtype):
        self.name = name
        self.shape = shape
        self.dtype = dtype

class PARTICLE_INFO(object):
    NUMBER = 0
    OFFSET = 1

class OBJECT(object):
    # Object properties
    NAME = 0
    ID = 1
    POSITION = 2
    ROTATION = 3
    IS_STATIC = 4
    ROUND_NUM = 5
    EXTENTS = 6
    CENTER = 7
    FLEX_ID = 8
    PARTICLE_INFO = 9
    LAST_ACTED_PARTICLE_INDEX = 10
    NUM_STATIC_COLLISIONS_ENTER = 11
    NUM_STATIC_COLLISIONS_STAY = 12
    NUM_STATIC_COLLISIONS_EXIT = 13
    NUM_DYNAMIC_COLLISIONS_ENTER = 14
    NUM_DYNAMIC_COLLISIONS_STAY = 15
    NUM_DYNAMIC_COLLISIONS_EXIT = 16
    FLEX_GRAVITY = 17
    SEND_PARTICLES_EVERY_FRAME = 18
    IS_SETTLED = 19
    CLUSTER_STIFFNESS = 20
    FORCE_INFO = 21
    SOFT_BODY_CLUSTER_INDICES = 22

class Scale(object):
    def __init__(self,
        mean = 1.0,
        var = 0.0,
        min = 1.0,
        max = 1.0,
        mode = 'do_not_scale',
        seed = 0,
        apply_to_instance = False,
        ):
        self.mean = mean
        self.var = var
        self.min = min
        self.max = max
        self.mode = mode
        self.seed = seed
        self.apply_to_instance = apply_to_instance

    def get_config(self):
        return {
                'option': self.mode,
                'scale': self.mean,
                'var': self.var,
                'seed': self.seed,
                'apply_to_inst': self.apply_to_instance
                }

class PhysicalObject(object):
    def __init__(self,
            path,
            name = None,
            mass = 1.0,
            scale = Scale(mean = 1.0, var = 0.0, min = 1.0, max = 1.0, \
                    mode = 'do_not_scale', seed = 0, apply_to_instance = False),
            static = False,
            host = 'local',
            physics = 'FleX',
            ):
        # Parameters
        self.path = path
        self.mass = mass
        self.static = static
        self.host = host
        self.physics = physics
        self.scale = scale
        if name is None:
            name = os.path.splitext(os.path.basename(self.path))[0]
        self.name = name

        # Fixed properties
        self.has_texture = True
        self.is_light = False
        self.num_items = 1
        self.is_observed = False

        # Frame by frame properties
        self.id = -1
        self.position = np.zeros(3)
        self.rotation = np.zeros(4)
        self.is_static = False
        self.round_num = -1
        self.extents = np.zeros(4)
        self.center = np.zeros(4)
        self.flex_id = -1
        self.particle_number = 0.0
        self.particle_offset = 0.0
        self.particles = []
        self.last_acted_particle_index = -1
        self.num_static_collisions_enter = 0
        self.num_static_collisions_stay = 0
        self.num_static_collisions_exit = 0
        self.num_dynamic_collisions_enter = 0
        self.num_dynamic_collisions_stay = 0
        self.num_dynamic_collisions_exit = 0
        self.flex_gravity = -9.81
        self.send_particles_every_frame = True
        self.is_settled = False
        self.cluster_stiffness = []
        self.force_info = []
        self.soft_body_cluster_indices = []

    def get_config(self):
        return {
                'host': self.host,
                'aws_address': self.path,
                'physics_type': self.physics.lower(),
                'type': self.physics.upper(),
                'scale': self.scale.get_config(),
                'mass': self.mass,
                'num_items': self.num_items,
                'has_texture': self.has_texture,
                'isLight': self.is_light,
                }

    def set_observed(self, obj):
        self.name = obj[OBJECT.NAME]
        self.id = int(obj[OBJECT.ID])
        self.position = np.array(obj[OBJECT.POSITION])
        self.rotation = np.array(obj[OBJECT.ROTATION])
        self.is_static = obj[OBJECT.IS_STATIC]
        self.round_num = int(obj[OBJECT.ROUND_NUM])
        self.extents = np.array(obj[OBJECT.EXTENTS])
        self.center = np.array(obj[OBJECT.CENTER])
        self.flex_id = int(obj[OBJECT.FLEX_ID])
        self.particle_number = \
                int(obj[OBJECT.PARTICLE_INFO][PARTICLE_INFO.NUMBER])
        self.particle_offset = \
                int(obj[OBJECT.PARTICLE_INFO][PARTICLE_INFO.OFFSET])
        self.last_acted_particle_index = \
                int(obj[OBJECT.LAST_ACTED_PARTICLE_INDEX])
        self.num_static_collisions_enter = \
                int(obj[OBJECT.NUM_STATIC_COLLISIONS_ENTER])
        self.num_static_collisions_stay = \
                int(obj[OBJECT.NUM_STATIC_COLLISIONS_STAY])
        self.num_static_collisions_exit = \
                int(obj[OBJECT.NUM_STATIC_COLLISIONS_EXIT])
        self.num_dynamic_collisions_enter = \
                int(obj[OBJECT.NUM_DYNAMIC_COLLISIONS_ENTER])
        self.num_dynamic_collisions_stay = \
                int(obj[OBJECT.NUM_DYNAMIC_COLLISIONS_STAY])
        self.num_dynamic_collisions_exit = \
                int(obj[OBJECT.NUM_STATIC_COLLISIONS_EXIT])
        self.num_dynamic_collisions_enter = \
                int(obj[OBJECT.NUM_DYNAMIC_COLLISIONS_ENTER])
        self.num_dynamic_collisions_stay = \
                int(obj[OBJECT.NUM_DYNAMIC_COLLISIONS_STAY])
        self.num_dynamic_collisions_exit = \
                int(obj[OBJECT.NUM_DYNAMIC_COLLISIONS_EXIT])
        self.flex_gravity = float(obj[OBJECT.FLEX_GRAVITY])
        self.send_particles_every_frame = \
                obj[OBJECT.SEND_PARTICLES_EVERY_FRAME]
        self.is_settled = obj[OBJECT.IS_SETTLED]
        self.cluster_stiffness = obj[OBJECT.CLUSTER_STIFFNESS]
        self.force_info = obj[OBJECT.FORCE_INFO]
        if len(obj) > OBJECT.SOFT_BODY_CLUSTER_INDICES:
            self.soft_body_cluster_indices = obj[OBJECT.SOFT_BODY_CLUSTER_INDICES]

        self.is_observed = True

    def set_particles(self, particles, particle_number, particle_offset):
        particles = np.reshape(particles, [-1, PARTICLE_STATE_DIM])
        self.particles = \
                particles[particle_offset : particle_offset + particle_number]


def get_objects_from_info(info):
    static_objects = []
    dynamic_objects = []
    for _, observed_object in enumerate(info['observed_objects']):
        if not observed_object[OBJECT.IS_STATIC]:
            new_object = PhysicalObject(observed_object[OBJECT.NAME])
            new_object.set_observed(observed_object)
            if new_object.particle_number == 0:
                new_object.static = True
                static_objects.append(new_object)
            else:
                dynamic_objects.append(new_object)
    return static_objects, dynamic_objects

def get_object_idxs(info):
    static_objects, dynamic_objects = \
            get_objects_from_info(info)
    static_object_idxs = [obj.id for obj in static_objects]
    dynamic_object_idxs = [obj.id for obj in dynamic_objects]
    return static_object_idxs, dynamic_object_idxs

def get_dataset_info_from_hdf5(hdf5_file, recount = False):
    assert all([k in hdf5_file.keys() for k in ['worldinfo']])
    max_steps = len(hdf5_file['worldinfo'])
    info = json.loads(hdf5_file['worldinfo'][-1])
    max_n_dynamic_objects = \
            int(info['max_num_dynamic_flex_objects'])
    max_n_particles = int(info['max_particles'])
    if 'images1' in hdf5_file.keys():
        height, width = hdf5_file['images1'].shape[1:3]
    else:
        height, width = (1, 1)

    if recount:
        max_n_particles = int(np.max([hdf5_file['particles'][i].shape[0] / 7 \
                for i in range(len(hdf5_file['particles']))]))

    static_obj_idx, dynamic_obj_idx = get_object_idxs(info)
    assert max_n_dynamic_objects >= len(dynamic_obj_idx), \
            (max_n_dynamic_objects, dynamic_obj_idx)
    return max_steps, height, width, max_n_particles, \
            max_n_dynamic_objects, dynamic_obj_idx
