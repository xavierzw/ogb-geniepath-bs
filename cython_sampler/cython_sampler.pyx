# cython: language_level=3
# distutils: language=c++
# distutils: extra_compile_args = -fopenmp -std=c++11
# distutils: extra_link_args = -fopenmp
cimport cython
from cython.parallel import prange,parallel
from cython.operator import dereference as deref, preincrement as inc
from cython cimport Py_buffer
from cpython cimport PyObject, Py_INCREF
from libcpp.vector cimport vector
from libcpp.algorithm cimport sort, unique, lower_bound
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map
from libcpp.unordered_set cimport unordered_set
from libcpp.utility cimport pair
import numpy as np
cimport numpy as np
from libc.stdio cimport printf
from libcpp cimport bool
import time,math
import random
from libc.math cimport exp, sqrt, log
from libc.stdlib cimport rand, malloc, free
cdef extern from "stdlib.h" nogil:
    int RAND_MAX
cdef extern from "<algorithm>" namespace "std" nogil:
    Iter find[Iter, T](Iter first, Iter last, const T& value)
cdef extern from "<utility>" namespace "std" nogil:
    T move[T](T)
from scipy.sparse import lil_matrix

cimport cython_utils as cutils
import cython_utils as cutils

import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

#srand48(123)

cdef class Sampler:
    # hyper parameters
    cdef int num_proc
    cdef int neighbor_limit
    cdef float max_reward

    # graph related members
    cdef int num_node
    cdef unordered_map[int,vector[double]] sample_weights
    cdef unordered_map[int,vector[double]] sample_probs
    cdef unordered_map[int, unordered_map[int, int]] sample_index
    cdef unordered_map[int, vector[int]] sample_set0
    cdef vector[int] degree
    cdef unordered_map[int,vector[int]] adj

    def __cinit__(self):
        self.num_proc = FLAGS.num_proc
        self.neighbor_limit = FLAGS.neighbor_limit
        self.max_reward = FLAGS.max_reward

    def init(self, adj):
        self.num_node = adj.shape[0]
        self.degree = vector[int](self.num_node)
        for src in range(self.num_node):
            if len(adj[src].rows[0]) == 0:
                self.degree[src] = 0
                continue
            dst_list = np.array(adj[src].rows[0], dtype=np.int32)
            self.c_init(src, dst_list)

    cdef void c_init(self, int src, np.ndarray[int,ndim=1,mode='c'] dst_list):
        cdef vector[int] dst_vec
        cutils.npy2vec_int(dst_list, dst_vec)

        cdef int degree = dst_vec.size()
        self.degree[src] = degree
        if degree == 0:
            return
        self.sample_weights[src].resize(degree)
        self.sample_probs[src].resize(degree)
        self.adj[src].resize(degree)

        cdef int dst
        cdef int idx
        idx = 0
        while idx < degree:
            dst = dst_vec[idx]
            self.adj[src][idx] = dst
            self.sample_weights[src][idx] = 1.
            self.sample_probs[src][idx] = 1./degree
            self.sample_index[src][dst] = idx
            idx += 1

    def get_degree(self, int src):
        return self.degree[src]

    def get_sample_probs(self, int src, int dst):
        cdef int idx = self.sample_index[src][dst]
        return self.sample_probs[src][idx]

    def get_sample_probs_list(self, int src):
        return self.sample_probs[src]

    def get_sample_weights(self, int src):
        return self.sample_weights[src]

    def update(self, np.ndarray[int,ndim=1,mode='c'] np_src_list,
               np.ndarray[int,ndim=1,mode='c'] np_dst_list,
               np.ndarray[float,ndim=1,mode='c'] np_att_list):
        raise NotImplementedError

    def sample(self, int node, int sample_size):
        degree = self.degree[node]
        neighbors = []
        if sample_size >= degree:
            return []
        else:
            probs = [x for x in self.sample_probs[node][:degree]]
            print(probs)
            neighbors = np.random.choice(probs, sample_size, p=probs, replace=False)
        return neighbors


cdef class BanditSampler(Sampler):
    def __cinit__(self):
        pass

    def update(self, np.ndarray[int,ndim=1,mode='c'] np_src_list,
               np.ndarray[int,ndim=1,mode='c'] np_dst_list,
               np.ndarray[float,ndim=1,mode='c'] np_att_list):
        cdef vector[int] src_list
        cdef vector[int] dst_list
        cdef vector[float] att_list
        cutils.npy2vec_int(np_src_list, src_list)
        cutils.npy2vec_int(np_dst_list, dst_list)
        cutils.npy2vec_float(np_att_list, att_list)

        # att indice map
        cdef int num_data = src_list.size()
        cdef unordered_map[int, unordered_map[int,int]] att_map
        cdef int i = 0
        while i < num_data:
            att_map[src_list[i]][dst_list[i]] = i
            i += 1

        cdef int p = 0
        cdef int neighbor_limit = FLAGS.neighbor_limit
        cdef float delta = FLAGS.delta
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc, schedule='dynamic'):
                self.update_sample_weights(
                        att_map, p, num_data, src_list, dst_list, att_list,
                        neighbor_limit, delta)

        cdef float eta = FLAGS.eta
        np_src_set = np.array(list(set(np_src_list)))
        cdef vector[int] src_set
        cutils.npy2vec_int(np_src_set, src_set)
        num_data = src_set.size()
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc, schedule='dynamic'):
                self.update_sample_probs(p, num_data, src_set, eta)

    # disable index bounds checking and negative indexing for speedups
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void update_sample_weights(
            self, unordered_map[int, unordered_map[int,int]]& att_map,
            int p, int num_data, vector[int]& src_list, vector[int]& dst_list,
            vector[float]& att_list, int neighbor_limit, float delta) nogil:
        cdef int i = 0
        cdef int src = 0
        cdef int dst = 0
        cdef int degree = 0
        cdef int idx = 0
        cdef float att_val = 0.
        cdef float reward = 0.
        while i < num_data:
            if i % self.num_proc != p:
                i += 1
                continue
            src = src_list[i]
            dst = dst_list[i]
            degree = self.degree[src]
            if degree <= neighbor_limit:
                i += 1
                continue
            delta = delta/degree**2
            idx = self.sample_index[src][dst]
            att_val = att_list[att_map[src][dst]]
            reward = delta*att_val**2/self.sample_probs[src][idx]**2
            if reward > self.max_reward:
                reward = self.max_reward
            self.sample_weights[src][idx] *= exp(reward)
            i += 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void update_sample_probs(
            self, int p, int num_data, vector[int] src_list, float eta) nogil:
        cdef int i = 0
        cdef int idx = 0
        cdef int dst = 0
        cdef int degree = 0
        cdef double unifom_prob = 0.
        cdef int src = 0
        while i < num_data:
            if i % self.num_proc != p:
                i += 1
                continue
            src = src_list[i]
            degree = self.degree[src]
            if degree <= self.neighbor_limit:
                i += 1
                continue
            weights_sum = sum_double(self.sample_weights[src])
            unifom_prob = 1./degree

            idx = 0
            while idx < degree:
                dst = self.adj[src][idx]
                self.sample_probs[src][idx] = (1-eta)*self.sample_weights[src][idx] / weights_sum \
                                              + eta*unifom_prob
                idx += 1
            i += 1


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def sample_graph(self, py_roots):
        cdef vector[int] roots
        cutils.npy2vec_int(py_roots, roots)

        cdef vector[int] edges
        cdef unordered_set[int] n_depth

        # 1st layer
        cdef int num_data = roots.size()
        cdef int p = 0
        cdef vector[vector[int]] edges_all
        cdef vector[unordered_set[int]] n_depth_all
        edges_all.resize(self.num_proc)
        n_depth_all.resize(self.num_proc)
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc, schedule='dynamic'):
                self.c_sample_graph_v1(p, num_data, roots, edges_all[p], n_depth_all[p])

        cdef unordered_set[int].iterator it
        cdef vector[int].iterator found

        cdef int i = 0
        cdef int edge_size = edges.size()
        while i < self.num_proc:
            edge_size += edges_all[i].size()
            i += 1
        edges.reserve(edge_size)

        i = 0
        cdef int k = 0
        while i < self.num_proc:
            edges.insert(edges.end(), edges_all[i].begin(), edges_all[i].end())
            it = n_depth_all[i].begin()
            while it != n_depth_all[i].end():
                found = find[vector[int].iterator, int](roots.begin(), roots.end(), deref(it))
                if found != roots.end():
                    inc(it)
                    continue
                n_depth.insert(deref(it))
                inc(it)
            i += 1

        # 2nd layer
        cdef vector[int] n_depth_vec = vector[int](n_depth.size())
        it = n_depth.begin()
        i = 0
        while it != n_depth.end():
            n_depth_vec[i] = deref(it)
            inc(it)
            i += 1
        num_data = n_depth_vec.size()

        edges_all.clear()
        edges_all.resize(self.num_proc)
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc, schedule='dynamic'):
                self.c_sample_graph_v2(p, num_data, n_depth_vec, edges_all[p])
        i = 0
        edge_size = edges.size()
        while i < self.num_proc:
            edge_size += edges_all[i].size()
            i += 1
        edges.reserve(edge_size)

        i = 0
        while i < self.num_proc:
            edges.insert(edges.end(), edges_all[i].begin(), edges_all[i].end())
            i += 1

        # sort edges
        #sort[vector[int].iterator, f_type](edges.begin(), edges.end(), compare)

        cdef cutils.array_wrapper_int w_edges = cutils.array_wrapper_int()
        w_edges.set_data(edges)
        np_edges = np.frombuffer(w_edges, dtype=np.int32)
        np_edges = np_edges.reshape([-1,2])
        return np_edges


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void c_sample_graph_v1(self, int p, int num_data, vector[int]& roots,
                                vector[int]& edges, unordered_set[int]& n_depth) nogil:
        cdef int i = 0
        cdef int sample_size
        cdef int node
        while i < num_data:
            if i % self.num_proc != p:
                i += 1
                continue
            node = roots[i]
            sample_size = self.sample_neighbors_v1(node, edges, n_depth)
            i += 1


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void c_sample_graph_v2(self, int p, int num_data, vector[int]& roots,
                                vector[int]& edges) nogil:
        cdef int i = 0
        cdef int node
        while i < num_data:
            if i % self.num_proc != p:
                i += 1
                continue
            node = roots[i]
            self.sample_neighbors_v2(node, edges)
            i += 1


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef int sample_neighbors_v1(self, int node, vector[int]& edges,
                                  unordered_set[int]& n_depth) nogil:
        cdef int sample_size = 0
        cdef int degree
        cdef int edge_size
        cdef vector[int]* neighbors = &self.adj[node]
        degree = self.degree[node]
        edge_size = edges.size()
        cdef vector[int].iterator it
        cdef int i
        cdef vector[int] samples
        cdef int sample_id
        cdef vector[double] probs
        cdef vector[double]* sample_probs = &self.sample_probs[node]
        if degree <= self.neighbor_limit:
            edges.resize(edge_size + degree*2)
            i = 0
            while i < degree:
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = deref(neighbors)[i]
                n_depth.insert(deref(neighbors)[i])
                inc(it)
                i += 1
            sample_size = degree
        else:
            edges.resize(edge_size + self.neighbor_limit*2)
            samples = random_choice(deref(neighbors), deref(sample_probs), self.neighbor_limit)

            i = 0
            while i < self.neighbor_limit:
                sample_id = samples[i]
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = sample_id
                n_depth.insert(sample_id)
                i += 1
                sample_size += 1
        return sample_size


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void sample_neighbors_v2(self, int node, vector[int]& edges) nogil:
        cdef int degree
        cdef int edge_size
        cdef vector[int]* neighbors = &self.adj[node]
        degree = self.degree[node]
        edge_size = edges.size()
        cdef vector[int].iterator it
        cdef int i
        cdef vector[int] samples
        cdef vector[double] probs
        cdef vector[double]* sample_probs = &self.sample_probs[node]
        if degree <= self.neighbor_limit:
            edges.resize(edge_size + degree*2)
            i = 0
            while i < degree:
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = deref(neighbors)[i]
                inc(it)
                i += 1
        else:
            edges.resize(edge_size + self.neighbor_limit*2)
            samples = random_choice(deref(neighbors), deref(sample_probs), self.neighbor_limit)

            i = 0
            while i < self.neighbor_limit:
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = samples[i]
                i += 1


cdef class BanditLinearSampler(Sampler):
    def __cinit__(self):
        pass

    def update(self, np.ndarray[int,ndim=1,mode='c'] np_src_list,
               np.ndarray[int,ndim=1,mode='c'] np_dst_list,
               np.ndarray[float,ndim=1,mode='c'] np_att_list):
        cdef vector[int] src_list
        cdef vector[int] dst_list
        cdef vector[float] att_list
        cutils.npy2vec_int(np_src_list, src_list)
        cutils.npy2vec_int(np_dst_list, dst_list)
        cutils.npy2vec_float(np_att_list, att_list)

        cdef int num_data = src_list.size()
        cdef int p = 0
        cdef int neighbor_limit = FLAGS.neighbor_limit
        cdef float delta = FLAGS.delta

        # att indice map
        cdef unordered_map[int, unordered_map[int,int]] att_map
        cdef int i = 0
        while i < num_data:
            att_map[src_list[i]][dst_list[i]] = i
            i += 1

        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc, schedule='dynamic'):
                self.update_sample_weights(
                        att_map, p, num_data, src_list, dst_list, att_list,
                        neighbor_limit, delta)

        cdef float eta = FLAGS.eta
        np_src_set = np.array(list(set(np_src_list)))
        cdef vector[int] src_set
        cutils.npy2vec_int(np_src_set, src_set)
        num_data = src_set.size()
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc, schedule='dynamic'):
                self.update_sample_probs(p, num_data, src_set, eta)

    # disable index bounds checking and negative indexing for speedups
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void update_sample_weights(
            self, unordered_map[int,unordered_map[int,int]]& att_map, int p, int num_data,
            vector[int]& src_list, vector[int]& dst_list, vector[float]& att_list,
            int neighbor_limit, float delta) nogil:
        cdef int i = 0
        cdef int src = 0
        cdef int dst = 0
        cdef int degree = 0
        cdef int idx = 0
        cdef float att = 0.
        cdef float reward = 0.
        while i < num_data:
            if i % self.num_proc != p:
                i += 1
                continue
            src = src_list[i]
            dst = dst_list[i]
            degree = self.degree[src]
            if degree <= neighbor_limit:
                i += 1
                continue
            delta = delta/degree**2
            idx = self.sample_index[src][dst]
            att = att_list[att_map[src][dst]]
            reward = delta*att**2/self.sample_probs[src][idx]**2
            #if reward > self.max_reward:
            #    reward = self.max_reward
            self.sample_weights[src][idx] += reward
            #self.sample_weights[src][idx] *= exp(reward)
            i += 1

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void update_sample_probs(
            self, int p, int num_data, vector[int] src_list, float eta) nogil:
        cdef int i = 0
        cdef int idx = 0
        cdef int dst = 0
        cdef int degree = 0
        cdef double unifom_prob = 0.
        cdef int src = 0
        while i < num_data:
            if i % self.num_proc != p:
                i += 1
                continue
            src = src_list[i]
            degree = self.degree[src]
            if degree <= self.neighbor_limit:
                i += 1
                continue
            weights_sum = sum_double(self.sample_weights[src])
            unifom_prob = 1./degree

            idx = 0
            while idx < degree:
                self.sample_probs[src][idx] = (1-eta)*self.sample_weights[src][idx] / weights_sum \
                                              + eta*unifom_prob
                idx += 1
            i += 1


    @cython.wraparound(False)
    @cython.boundscheck(False)
    def sample_graph(self, py_roots):
        cdef vector[int] roots
        cutils.npy2vec_int(py_roots, roots)

        cdef vector[int] edges
        cdef unordered_set[int] n_depth

        # 1st layer
        cdef int num_data = roots.size()
        cdef int p = 0
        cdef vector[vector[int]] edges_all
        cdef vector[unordered_set[int]] n_depth_all
        edges_all.resize(self.num_proc)
        n_depth_all.resize(self.num_proc)
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc, schedule='dynamic'):
                self.c_sample_graph_v1(p, num_data, roots, edges_all[p], n_depth_all[p])

        cdef unordered_set[int].iterator it
        cdef vector[int].iterator found

        cdef int i = 0
        cdef int edge_size = edges.size()
        while i < self.num_proc:
            edge_size += edges_all[i].size()
            i += 1
        edges.reserve(edge_size)

        i = 0
        cdef int k = 0
        while i < self.num_proc:
            edges.insert(edges.end(), edges_all[i].begin(), edges_all[i].end())
            it = n_depth_all[i].begin()
            while it != n_depth_all[i].end():
                found = find[vector[int].iterator, int](roots.begin(), roots.end(), deref(it))
                if found != roots.end():
                    inc(it)
                    continue
                n_depth.insert(deref(it))
                inc(it)
            i += 1

        # 2nd layer
        cdef vector[int] n_depth_vec = vector[int](n_depth.size())
        it = n_depth.begin()
        i = 0
        while it != n_depth.end():
            n_depth_vec[i] = deref(it)
            inc(it)
            i += 1
        num_data = n_depth_vec.size()

        edges_all.clear()
        edges_all.resize(self.num_proc)
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc, schedule='dynamic'):
                self.c_sample_graph_v2(p, num_data, n_depth_vec, edges_all[p])
        i = 0
        edge_size = edges.size()
        while i < self.num_proc:
            edge_size += edges_all[i].size()
            i += 1
        edges.reserve(edge_size)

        i = 0
        while i < self.num_proc:
            edges.insert(edges.end(), edges_all[i].begin(), edges_all[i].end())
            i += 1

        # sort edges
        #sort[vector[int].iterator, f_type](edges.begin(), edges.end(), compare)

        cdef cutils.array_wrapper_int w_edges = cutils.array_wrapper_int()
        w_edges.set_data(edges)
        np_edges = np.frombuffer(w_edges, dtype=np.int32)
        np_edges = np_edges.reshape([-1,2])
        return np_edges

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void c_sample_graph_v1(self, int p, int num_data, vector[int]& roots,
                                vector[int]& edges, unordered_set[int]& n_depth) nogil:
        cdef int i = 0
        cdef int sample_size
        cdef int node
        while i < num_data:
            if i % self.num_proc != p:
                i += 1
                continue
            node = roots[i]
            sample_size = self.sample_neighbors_v1(node, edges, n_depth)
            i += 1


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void c_sample_graph_v2(self, int p, int num_data, vector[int]& roots,
                                vector[int]& edges) nogil:
        cdef int i = 0
        cdef int node
        while i < num_data:
            if i % self.num_proc != p:
                i += 1
                continue
            node = roots[i]
            self.sample_neighbors_v2(node, edges)
            i += 1


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef int sample_neighbors_v1(self, int node, vector[int]& edges,
                                 unordered_set[int]& n_depth) nogil:
        cdef int sample_size = 0
        cdef int degree
        cdef int edge_size
        cdef vector[int]* neighbors = &self.adj[node]
        degree = self.degree[node]
        edge_size = edges.size()
        cdef vector[int].iterator it
        cdef int i
        cdef vector[int] samples
        cdef int sample_id
        cdef vector[double] probs
        cdef vector[double]* sample_probs = &self.sample_probs[node]
        if degree <= self.neighbor_limit:
            edges.resize(edge_size + degree*2)
            i = 0
            while i < degree:
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = deref(neighbors)[i]
                n_depth.insert(deref(neighbors)[i])
                inc(it)
                i += 1
            sample_size = degree
        else:
            edges.resize(edge_size + self.neighbor_limit*2)
            samples = random_choice(deref(neighbors), deref(sample_probs), self.neighbor_limit)

            i = 0
            while i < self.neighbor_limit:
                sample_id = samples[i]
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = sample_id
                n_depth.insert(sample_id)
                sample_size += 1
                i += 1
        return sample_size


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void sample_neighbors_v2(self, int node, vector[int]& edges) nogil:
        cdef int degree
        cdef int edge_size
        cdef vector[int]* neighbors = &self.adj[node]
        degree = self.degree[node]
        edge_size = edges.size()
        cdef vector[int].iterator it
        cdef int i
        cdef vector[int] samples
        cdef vector[double] probs
        cdef vector[double]* sample_probs = &self.sample_probs[node]
        if degree <= self.neighbor_limit:
            edges.resize(edge_size + degree*2)
            i = 0
            while i < degree:
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = deref(neighbors)[i]
                inc(it)
                i += 1
        else:
            edges.resize(edge_size + self.neighbor_limit*2)
            samples = random_choice(deref(neighbors), deref(sample_probs), self.neighbor_limit)

            i = 0
            while i < self.neighbor_limit:
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = samples[i]
                i += 1


cdef class BanditMPSampler(Sampler):
    def __cinit__(self):
        pass

    def update(self, np.ndarray[int,ndim=1,mode='c'] np_src_list,
               np.ndarray[int,ndim=1,mode='c'] np_dst_list,
               np.ndarray[float,ndim=1,mode='c'] np_att_list):
        cdef vector[int] src_list
        cdef vector[int] dst_list
        cdef vector[float] att_list
        cutils.npy2vec_int(np_src_list, src_list)
        cutils.npy2vec_int(np_dst_list, dst_list)
        cutils.npy2vec_float(np_att_list, att_list)

        cdef int num_data = src_list.size()
        cdef int p = 0
        cdef int neighbor_limit = FLAGS.neighbor_limit
        cdef float delta = FLAGS.delta
  
        # mark src start & end position
        cdef int i = 0
        cdef int num_src = 0
        cdef vector[int] src_set
        cdef unordered_map[int,int] src_start
        cdef unordered_map[int,int] src_end
        cdef int last_src = -1
        cdef int src
        while i < num_data:
            src = src_list[i]
            if src != last_src:
                if last_src > 0:
                    src_end[last_src] = i
                last_src = src
                src_start[src] = i
                src_set.push_back(src)
                num_src += 1
            i += 1
        src_end[last_src] = num_data

        # att indice map
        cdef unordered_map[int, unordered_map[int,int]] att_map
        i = 0
        while i < num_data:
            att_map[src_list[i]][dst_list[i]] = i
            i += 1

        cdef float eta = FLAGS.eta
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc, schedule='dynamic'):
                self.update_sample_weights(
                        att_map, src_start, src_end, p, num_src, src_set,
                        src_list, dst_list, att_list,
                        neighbor_limit, delta, eta)

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void update_sample_weights(
            self, unordered_map[int,unordered_map[int,int]]& att_map,
            unordered_map[int,int]& src_start, unordered_map[int,int]& src_end,
            int p, int num_src, vector[int]& src_set, vector[int]& src_list,
            vector[int]& dst_list, vector[float]& att_list,
            int neighbor_limit, float delta, float eta) nogil:
        cdef int src
        cdef int dst
        cdef int src_idx = 0
        cdef int i
        cdef int n_arm
        cdef int idx
        cdef float h_norm
        cdef vector[double] weight_list
        cdef double weight_sum
        cdef double alpha
        cdef double C_sum
        cdef double left_sum
        cdef double alpha_i
        cdef int N
        cdef vector[double] weight_prime
        cdef vector[double]* src_sample_weights
        cdef double weight_prime_sum = 0.
        cdef int T = 40
        while src_idx < num_src:
            src = src_set[src_idx]
            #if self.sample_set0.find(src) == self.sample_set0.end():
            #    self.sample_set0[src] = vector[int]()

            if src_idx % self.num_proc != p:
                src_idx += 1
                continue
            n_arm = self.degree[src]
            if n_arm <= self.neighbor_limit:
                src_idx += 1
                continue

            #delta = delta/n_arm**2
            delta = 1./n_arm

            # update weights
            p_s = 0.
            i = src_start[src]
            while i < src_end[src]:
                dst = dst_list[i]
                idx = self.sample_index[src][dst]
                p_s += self.sample_probs[src][idx]
                i += 1
            i = src_start[src]
            while i < src_end[src]:
                dst = dst_list[i]
                if vec_find(self.sample_set0[src], dst):
                    i += 1
                    continue
                idx = self.sample_index[src][dst]
                att_idx = att_map[src][dst]
                reward = att_list[att_idx] * p_s / self.neighbor_limit
                self.sample_weights[src][idx] += reward

                #reward = att_list[att_idx]**2 \
                #         / (self.sample_probs[src][idx]**3)
                #reward = reward / self.neighbor_limit \
                #         * sqrt(((1-eta)*eta**4*self.neighbor_limit**5*log(n_arm/self.neighbor_limit))/(4*T*n_arm**4))
                #reward = exp(reward)
                #self.sample_weights[src][idx] *= min([reward, 2.0])
                i += 1
            src_idx += 1

            weight_list.clear()
            weight_list.insert(
                    weight_list.begin(), self.sample_weights[src].begin(),
                    self.sample_weights[src].end())
            sort(weight_list.begin(), weight_list.end())
            weight_sum = sum_double(weight_list)

            # reset sample_set0
            self.sample_set0[src].clear()
            self.sample_set0[src] = vector[int]()

            # decide alpha
            alpha = 0.
            C = (1./self.neighbor_limit - eta/n_arm)/(1-eta)
            C_sum = C*weight_sum
            if vec_max(weight_list) >= C_sum:
                left_sum = 0.
                N = weight_list.size()
                i = 0
                while i < N-1:
                    left_sum += weight_list[i]
                    if double_abs(1 - C*(N-i-1)) < 1e-6:
                        i += 1
                        continue
                    alpha_i = C*left_sum / (1 - C*(N-i-1))
                    if alpha_i > weight_list[i] and alpha_i <= weight_list[i+1]:
                        alpha = alpha_i
                        break
                    i += 1
                if alpha == 0.:
                    printf("Error! Not find alpha!")

                weight_prime.clear()
                weight_prime_sum = 0.
                src_sample_weights = &self.sample_weights[src]
                i = 0
                while i < n_arm:
                    w = deref(src_sample_weights)[i]
                    if w < alpha:
                        weight_prime.push_back(w)
                        weight_prime_sum += w
                    else:
                        weight_prime.push_back(alpha)
                        weight_prime_sum += alpha
                        dst = self.adj[src][i]
                        self.sample_set0[src].push_back(dst)
                    i += 1

                # update sample probs
                i = 0
                while i < n_arm:
                    self.sample_probs[src][i] = \
                        (eta*1./n_arm + (1-eta)*weight_prime[i]/weight_prime_sum)\
                        *self.neighbor_limit
                    i += 1
            else:
                # update sample probs
                i = 0
                src_sample_weights = &self.sample_weights[src]
                while i < n_arm:
                    self.sample_probs[src][i] = \
                        (eta*1./n_arm + (1-eta)*deref(src_sample_weights)[i]/weight_sum)\
                        *self.neighbor_limit
                    i += 1


    @cython.wraparound(False)
    @cython.boundscheck(False)
    def sample_graph(self, py_roots):
        cdef vector[int] roots
        cutils.npy2vec_int(py_roots, roots)

        cdef vector[int] edges
        cdef unordered_set[int] n_depth

        # 1st layer
        cdef int num_data = roots.size()
        cdef int p = 0
        cdef vector[vector[int]] edges_all
        cdef vector[unordered_set[int]] n_depth_all
        edges_all.resize(self.num_proc)
        n_depth_all.resize(self.num_proc)
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc, schedule='dynamic'):
                self.c_sample_graph_v1(p, num_data, roots, edges_all[p], n_depth_all[p])

        cdef unordered_set[int].iterator it
        cdef vector[int].iterator found

        cdef int i = 0
        cdef int edge_size = edges.size()
        while i < self.num_proc:
            edge_size += edges_all[i].size()
            i += 1
        edges.reserve(edge_size)

        i = 0
        cdef int k = 0
        while i < self.num_proc:
            edges.insert(edges.end(), edges_all[i].begin(), edges_all[i].end())
            it = n_depth_all[i].begin()
            while it != n_depth_all[i].end():
                found = find[vector[int].iterator, int](roots.begin(), roots.end(), deref(it))
                if found != roots.end():
                    inc(it)
                    continue
                n_depth.insert(deref(it))
                inc(it)
            i += 1

        # 2nd layer
        cdef vector[int] n_depth_vec = vector[int](n_depth.size())
        it = n_depth.begin()
        i = 0
        while it != n_depth.end():
            n_depth_vec[i] = deref(it)
            inc(it)
            i += 1
        num_data = n_depth_vec.size()

        edges_all.clear()
        edges_all.resize(self.num_proc)
        with nogil, parallel(num_threads=self.num_proc):
            for p in prange(self.num_proc, schedule='dynamic'):
                self.c_sample_graph_v2(p, num_data, n_depth_vec, edges_all[p])
        i = 0
        edge_size = edges.size()
        while i < self.num_proc:
            edge_size += edges_all[i].size()
            i += 1
        edges.reserve(edge_size)

        i = 0
        while i < self.num_proc:
            edges.insert(edges.end(), edges_all[i].begin(), edges_all[i].end())
            i += 1

        # sort edges
        #sort[vector[int].iterator, f_type](edges.begin(), edges.end(), compare)

        cdef cutils.array_wrapper_int w_edges = cutils.array_wrapper_int()
        w_edges.set_data(edges)
        np_edges = np.frombuffer(w_edges, dtype=np.int32)
        np_edges = np_edges.reshape([-1,2])
        return np_edges

    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void c_sample_graph_v1(self, int p, int num_data, vector[int]& roots,
                                vector[int]& edges, unordered_set[int]& n_depth) nogil:
        cdef int i = 0
        cdef int sample_size
        cdef int node
        while i < num_data:
            if i % self.num_proc != p:
                i += 1
                continue
            node = roots[i]
            sample_size = self.sample_neighbors_v1(node, edges, n_depth)
            i += 1


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void c_sample_graph_v2(self, int p, int num_data, vector[int]& roots,
                                vector[int]& edges) nogil:
        cdef int i = 0
        cdef int node
        while i < num_data:
            if i % self.num_proc != p:
                i += 1
                continue
            node = roots[i]
            self.sample_neighbors_v2(node, edges)
            i += 1


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef int sample_neighbors_v1(self, int node, vector[int]& edges,
                                 unordered_set[int]& n_depth) nogil:
        cdef int sample_size = 0
        cdef int degree
        cdef int edge_size
        cdef vector[int]* neighbors = &self.adj[node]
        degree = self.degree[node]
        edge_size = edges.size()
        cdef vector[int].iterator it
        cdef int i
        cdef vector[int] samples
        cdef int sample_id
        cdef vector[double] probs
        cdef vector[double]* sample_probs = &self.sample_probs[node]
        if degree <= self.neighbor_limit:
            edges.resize(edge_size + degree*2)
            i = 0
            while i < degree:
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = deref(neighbors)[i]
                n_depth.insert(deref(neighbors)[i])
                inc(it)
                i += 1
            sample_size = degree
        else:
            edges.resize(edge_size + self.neighbor_limit*2)
            samples = random_choice(deref(neighbors), deref(sample_probs), self.neighbor_limit)

            i = 0
            while i < self.neighbor_limit:
                sample_id = samples[i]
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = sample_id
                n_depth.insert(sample_id)
                sample_size += 1
                i += 1
        return sample_size


    @cython.wraparound(False)
    @cython.boundscheck(False)
    cdef void sample_neighbors_v2(self, int node, vector[int]& edges) nogil:
        cdef int degree
        cdef int edge_size
        cdef vector[int]* neighbors = &self.adj[node]
        degree = self.degree[node]
        edge_size = edges.size()
        cdef vector[int].iterator it
        cdef int i
        cdef vector[int] samples
        cdef vector[double] probs
        cdef vector[double]* sample_probs = &self.sample_probs[node]
        if degree <= self.neighbor_limit:
            edges.resize(edge_size + degree*2)
            i = 0
            while i < degree:
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = deref(neighbors)[i]
                inc(it)
                i += 1
        else:
            edges.resize(edge_size + self.neighbor_limit*2)
            samples = random_choice(deref(neighbors), deref(sample_probs), self.neighbor_limit)

            i = 0
            while i < self.neighbor_limit:
                edges[edge_size+2*i] = node
                edges[edge_size+2*i+1] = samples[i]
                i += 1


#def test_random_choice():
#    cdef int degree = 50
#    cdef int* ids = <int *>malloc(degree*cython.sizeof(int))
#    cdef double* probs = <double *>malloc(degree*cython.sizeof(double))
#    cdef int i = 0
#    while i < degree:
#        ids[i] = i
#        probs[i] = 1./degree
#        i += 1
#    return random_choice(ids, probs, degree, 10)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[int] random_choice(
        vector[int]& ids, vector[double]& input_probs, int sample_size) nogil:
    cdef vector[int] samples = vector[int](sample_size)

    cdef int n = ids.size()
    cdef vector[double] p = vector[double](n)
    p.insert(p.begin(), input_probs.begin(), input_probs.end())

    cdef int i = 0
    cdef vector[int] found = vector[int](sample_size)
    cdef vector[double] x
    cdef vector[double] cdf
    cdef vector[int] new
    cdef vector[int] indices
    cdef int n_uniq = 0
    cdef int n_indices
    while n_uniq < sample_size:
        # random initial x
        x.clear()
        x.resize(sample_size - n_uniq)
        i = 0
        while i < sample_size - n_uniq:
            x[i] = (<double> rand()) / RAND_MAX
            i += 1

        # update probs
        if n_uniq > 0:
            i = 0
            while i < sample_size:
                p[found[i]] = 0.
                i += 1

        # compute cdf
        cdf = cumsum(p)
        i = 0
        while i < n:
            cdf[i] = cdf[i] / cdf[n-1]
            i += 1

        # search sorted
        new = searchsorted(cdf, x)
        indices = unique_index(new)

        # update found
        i = 0
        n_indices = indices.size()
        while i < n_indices:
            found[n_uniq+i] = new[indices[i]]
            i += 1
        n_uniq += n_indices

    # generate samples
    i = 0
    while i < sample_size:
        samples[i] = ids[found[i]]
        i += 1
    return samples


@cython.wraparound(False)
@cython.boundscheck(False)
cdef double sum_double(vector[double]& x) nogil:
    cdef double val = 0.
    cdef int i = 0
    cdef int n = x.size()
    while i < n:
        val += x[i]
        i += 1
    return val


#def test_random_choice():
#    cdef int degree = 50
#    cdef int* ids = <int *>malloc(degree*cython.sizeof(int))
#    cdef double* probs = <double *>malloc(degree*cython.sizeof(double))
#    cdef int i = 0
#    while i < degree:
#        ids[i] = i
#        probs[i] = 1./degree
#        i += 1
#    return random_choice(ids, probs, degree, 10)


#@cython.wraparound(False)
#@cython.boundscheck(False)
#cdef vector[int] random_choice(
#        vector[int]& ids, vector[double]& input_probs, int sample_size) nogil:
#    cdef vector[int] samples = vector[int](sample_size)
#
#    cdef int n = ids.size()
#    cdef vector[double] p = vector[double](n)
#    p.insert(p.begin(), input_probs.begin(), input_probs.end())
#
#    cdef int i = 0
#    cdef vector[int] found = vector[int](sample_size)
#    cdef vector[double] x
#    cdef vector[double] cdf
#    cdef vector[int] new
#    cdef vector[int] indices
#    cdef int n_uniq = 0
#    cdef int n_indices
#    while n_uniq < sample_size:
#        # random initial x
#        x.clear()
#        x.resize(sample_size - n_uniq)
#        i = 0
#        while i < sample_size - n_uniq:
#            x[i] = (<double> rand()) / RAND_MAX
#            i += 1
#
#        # update probs
#        if n_uniq > 0:
#            i = 0
#            while i < sample_size:
#                p[found[i]] = 0.
#                i += 1
#
#        # compute cdf
#        cdf = cumsum(p)
#        i = 0
#        while i < n:
#            cdf[i] = cdf[i] / cdf[n-1]
#            i += 1
#
#        # search sorted
#        new = searchsorted(cdf, x)
#        indices = unique_index(new)
#
#        # update found
#        i = 0
#        n_indices = indices.size()
#        while i < n_indices:
#            found[n_uniq+i] = new[indices[i]]
#            i += 1
#        n_uniq += n_indices
#
#    # generate samples
#    i = 0
#    while i < sample_size:
#        samples[i] = ids[found[i]]
#        i += 1
#    return samples


@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[double] cumsum(vector[double]& probs) nogil:
    cdef int n = probs.size()
    cdef vector[double] results = vector[double](n)
    results[0] = probs[0]
    cdef int i = 1
    while i < n:
        results[i] = probs[i] + results[i-1]
        i += 1
    return results


@cython.wraparound(False)
@cython.boundscheck(False)
cdef int searchsorted_one(vector[double]& cdf, double x) nogil:
    cdef int n = cdf.size()
    cdef int i = 0
    while i < n:
        if x < cdf[i]:
            break
        i += 1
    return i

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[int] searchsorted(vector[double]& cdf, vector[double]& x) nogil:
    cdef int n = x.size()
    cdef int i = 0
    cdef vector[int] results = vector[int](n)
    while i < n:
        results[i] = searchsorted_one(cdf, x[i])
        i += 1
    return results

@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[int] unique_index(vector[int]& new) nogil:
    cdef unordered_set[int] new_set
    cdef int i = 0
    cdef int n_new = new.size()
    while i < n_new:
        new_set.insert(new[i])
        i += 1

    cdef int n = new_set.size()
    cdef vector[int] results = vector[int](n)
    cdef vector[int].iterator found
    cdef unordered_set[int].iterator it = new_set.begin()
    i = 0
    while it != new_set.end():
        found = find[vector[int].iterator, int](new.begin(), new.end(), deref(it))
        results[i] = found - new.begin()
        inc(it)
        i += 1
    sort(results.begin(), results.end())
    return results

@cython.wraparound(False)
@cython.boundscheck(False)
cdef bool vec_find(vector[int]& vec, int x) nogil:
    cdef vector[int].iterator found
    found = find[vector[int].iterator, int](vec.begin(), vec.end(), x)
    return found != vec.end()

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double vec_max(vector[double]& vec) nogil:
    cdef double max_val = 0
    cdef vector[double].iterator it
    it = vec.begin()
    while it != vec.end():
        if deref(it) > max_val:
            max_val = deref(it)
        inc(it)
    return max_val

@cython.wraparound(False)
@cython.boundscheck(False)
cdef double double_abs(double x) nogil:
    if x >= 0:
        return x
    else:
        return -x


def depround_test(py_ids, py_probs, sample_size):
    cdef vector[int] ids
    cutils.npy2vec_int(py_ids, ids)
    cdef vector[double] probs
    cutils.npy2vec_double(py_probs, probs)

    return depround(ids, probs, sample_size)


@cython.wraparound(False)
@cython.boundscheck(False)
cdef vector[int] depround(
        vector[int]& ids, vector[double]& input_probs, int sample_size) nogil:
    #srand48(time(NULL))
    cdef vector[int] samples

    # copy input_probs
    cdef int p_size = input_probs.size()
    cdef vector[double] probs
    probs.reserve(p_size)
    probs.insert(probs.end(), input_probs.begin(), input_probs.end())

    cdef double epsilon = 1e-6
    cdef vector[int] possible_list
    possible_list.resize(p_size)
    cdef int count = 0
    cdef int k = 0
    cdef double abs_p
    cdef double abs_1_p

    # search possible list
    while k < p_size:
        abs_p = probs[k]
        if abs_p < 0:
            abs_p = -abs_p
        abs_1_p = 1 - probs[k]
        if abs_1_p < 0:
            abs_1_p = -abs_1_p
        if abs_p >= epsilon and abs_1_p >= epsilon:
            possible_list[count] = k
            count += 1
        k += 1

    # random pick distinct i, j
    cdef int idx1 = rand() % count
    cdef int i = possible_list[idx1]
    cdef int idx2 = rand() % (count - 1)
    if idx2 >= idx1:
        idx2 += 1
    cdef int j = possible_list[idx2]

    # update probs[i], probs[j]
    cdef double alpha
    cdef double beta
    while count > 1:
        alpha = 1 - probs[i]
        if alpha > probs[j]:
            alpha = probs[j]
        beta = probs[i]
        if beta > 1 - probs[j]:
            beta = 1 - probs[j]
        if (<float> rand())/RAND_MAX < beta/(alpha+beta) :
            probs[i] += alpha
            probs[j] -= alpha
        else:
            probs[i] -= beta
            probs[j] += beta

        count = 0
        k = 0
        while k < p_size:
            abs_p = probs[k]
            if abs_p < 0:
                abs_p = -abs_p
            abs_1_p = 1 - probs[k]
            if abs_1_p < 0:
                abs_1_p = -abs_1_p
            if abs_p >= epsilon and abs_1_p >= epsilon:
                possible_list[count] = k
                count += 1
            k += 1

        if count > 1:
            idx1 = rand() % count
            i = possible_list[idx1]
            idx2 = rand() % (count - 1)
            if idx2 >= idx1:
                idx2 += 1
            j = possible_list[idx2]

    # sample ids
    i = 0
    while i < p_size:
        if double_abs(1-probs[i]) < epsilon:
            samples.push_back(ids[i])
        i += 1
    if samples.size() != sample_size:
        printf("Error! DepRound samples wrong number")

    return samples
