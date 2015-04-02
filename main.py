
# Collections en itereren {{{
import itertools # {{{

count = itertools.count
cycle = itertools.cycle

takewhile = itertools.takewhile
dropwhile = itertools.dropwhile
compress = itertools.compress
filterfalse = itertools.filterfalse
islice = itertools.islice

def how_not_to_list_evens():
    small = lambda x: x < 10
    print(list(takewhile(small, compress(count(), cycle([True,
        False])))))
    print(list(islice(count(), 0, 10, 2)))

repeat = itertools.repeat
accumulate = itertools.accumulate

def my_count(n):
    return accumulate(repeat(1, n))

itertools.chain
from_iterable = itertools.chain.from_iterable
tee = itertools.tee

def counted_cycle(seq, n):
    return from_iterable(tee(seq, n))

groupby = itertools.groupby

def odds_and_evens(seq):
    return groupby(seq, key=lambda x: x % 2)

itertools.starmap
zipl = itertools.zip_longest

def my_starmap(func, seq1, seq2):
    return map(lambda p: func(*p), zip(seq1, seq2))

itertools.product
itertools.permutations
itertools.combinations
itertools.combinations_with_replacement

def show_combinatorics():
    print(list(itertools.product("ABCD", repeat=2)))
    print(list(itertools.permutations("ABCD", 2)))
    print(list(itertools.combinations_with_replacement("ABCD", 2)))
    print(list(itertools.combinations("ABCD", 2)))

# }}}

import collections # {{{

deque = collections.deque

def check_nesting(string):
    mapping = { '}': '{', ')': '(', ']': '[' }
    d = deque()
    for c in string:
        if c in mapping:
            if len(d) == 0: return False
            e = d.pop()
            if e != mapping[c]: return False
        else:
            d.append(c)
        print(list(d))
    return len(d) == 0

class BFSTree(object):
    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right
    def __iter__(self):
        q = deque([self])
        while len(q):
            current = q.popleft()
            yield current
            if current.left: q.append(current.left)
            if current.right: q.append(current.right)

class StuffTree(BFSTree):
    def __init__(self, elem, left=None, right=None):
        super().__init__(left, right)
        self.elem = elem

def show_bfs():
    tree = StuffTree(1,
            StuffTree(2,
                StuffTree(3),
                StuffTree(4)),
            StuffTree(5))
    print([t.elem for t in tree])

collections.namedtuple

def show_structs():
    Point = collections.namedtuple("Point", ['x', 'y'])
    p1 = Point(5, 4)
    p2 = Point(1, 2)
    print(p1, p2)

collections.OrderedDict

collections.defaultdict

def count_elements(seq):
    d = collections.defaultdict(int)
    for e in seq: d[e] = d[e] + 1
    return d

collections.Counter

better_count_elements = collections.Counter

collections.UserDict
collections.UserList
collections.UserString

# }}}

import array # {{{
# }}}

import bisect # {{{
# }}}

import heapq # {{{
# }}}

# Bytes {{{
# }}}

# }}}

# Multiprocessing en threading {{{

import threading # {{{
# Ongeveer gebaseerd op het Java threading model
# Verschil: locks en conditievariabelen zijn aparte objecten in Python

class CounterThread(threading.Thread):
    def run(self):
        for i in range(10):
            print(i)

def how_to_use_threads():
    thread = CounterThread()
    thread.start()
    for i in range(10):
        print(i)
    thread.join()
    print('That\'s all folks')


def how_to_use_threads2():
    def count():
        for i in range(10):
            print(i)

    thread = threading.Thread(target=count)
    thread.start()


def how_to_use_locks():
    lock = threading.Lock()
    def f():
        lock.acquire()
        try:
            # Do something with some resource
            for i in range(5):
                print(i)
        finally:
            lock.release()
    f()
    f()
    f()


threading.RLock  # Reentrant lock
threading.Condition
threading.Semaphore
threading.BoundedSemaphore
threading.Timer
threading.Barrier
threading.Event

# }}}

import queue # {{{
queue.Queue
#queue.LIFOQueue
queue.PriorityQueue
# }}}

# Maar merk op dat GIL

import multiprocessing # {{{
multiprocessing.Process
multiprocessing.Pipe
multiprocessing.Queue
multiprocessing.Pool
# ...
# }}}

import subprocess
# }}}

# Divers {{{

import pdb # {{{
# }}}

import functools # {{{
functools.partialmethod
functools.singledispatch
functools.update_wrapper
functools.wraps
# }}}

import asyncio # {{{
# }}}

import pickle # {{{


class Foo:
    x = 'blabla'

    def __init__(self, y):
        self.y = y


def how_to_pickle():
    print(pickle.dumps(True))
    print(pickle.dumps(5))
    print(pickle.dumps((None, [{'a': 1}])))
    print(pickle.dumps(Foo))
    print(pickle.dumps(Foo(42)))

    foo = Foo(42)
    with open('somefile', 'wb') as picklefile:
        pickle.dump(foo, picklefile)

    pickled_foo = pickle.dumps(foo)
    print(pickle.loads(pickled_foo))

    with open('somefile', 'rb') as picklefile:
        print(pickle.load(picklefile))

# }}}

import shelve
import pprint
import locale
import logging
import decimal
import requests
# }}}

# profiling
# C extensions en SWIG

# vim: foldmethod=marker
