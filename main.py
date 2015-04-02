import time

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
def show_array():
    print(array.array('i', range(1, 30)))
# }}}

import bisect # {{{
def show_bisect():
    l = [1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 6]
    bisect.insort_left(l, 5)
    print(l)
    print(bisect.bisect_left(l, 3), bisect.bisect_right(l, 3))
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

# Thread-safe, vooral voor communicatie tussen meerdere threads
# Niet te verwarren met een gewone queue (zoals collections.deque)

def how_to_use_queues():
    q = queue.Queue()

    def worker():
        while True:
            item = q.get()
            time.sleep(1)
            print(item)
            q.task_done()

    for i in range(3):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()

    for item in range(10):
        q.put(item)

    q.join()  # Block until all tasks are done


queue.LifoQueue  # Stack
queue.PriorityQueue

# }}}

# GIL {{{
# Global interpreter lock
# Zorgt ervoor dat code die niet thread-safe is niet tegelijk  wordt uitgevoerd
# Zorgt ervoor dat er maar maximaal één thread per keer kan uitgevoerd worden
# Gevolg: threading zorgt niet voor parallellisatie
# Oplossing: multiprocessing
# }}}

import multiprocessing as mp # {{{

# Zorgt wel voor parallellisatie
# Niet in meerdere threads, maar in meerdere processen

# Process werkt zoals Thread
def how_to_use_processes():
    def count():
        for i in range(10):
            print(i)

    process = mp.Process(target=count)
    process.start()

    for i in range(10):
        print(i)

    process.join()
    print('That\'s all folks')


# Omdat processen minder vaak worden gewisseld dan threads, valt de
# multiprocessing niet op. Dus we voeren wat sleeps in.
def how_to_use_processes2():
    def count():
        for i in range(5):
            print(i)
        time.sleep(1)
        for i in range(5,10):
            print(i)

    process = mp.Process(target=count)
    process.start()

    for i in range(5):
        print(i)
    time.sleep(1)
    for i in range(5,10):
        print(i)

    process.join()
    print('That\'s all folks')


mp.Queue
# Zelfde als queue.Queue, maar dan voor multiprocessing (ipv multithreading)

mp.Pipe
# Gelijkaardig aan mp.Queue, maar voor slechts twee processen


# Zelfde synchronisatie-primitieven als threading
mp.Lock
mp.RLock
mp.Condition
mp.Semaphore
mp.BoundedSemaphore
mp.Barrier
mp.Event


# Worker pools
def f(x):
    time.sleep(1)
    return x * x

def how_to_use_pools():
    with mp.Pool(3) as p:
        print(p.map(f, [1, 2, 3]))


# Nog veel meer...

# }}}

import subprocess as sp  # {{{

def how_to_use_subprocesses1():
    exit_code = sp.call(['ls', 'een_bestand'])
    print(exit_code)

def how_to_use_subprocesses2():
    exit_code = sp.call(['ls', 'een_bestand'], stderr=sp.DEVNULL)
    print(exit_code)

def how_to_use_subprocesses3():
    try:
        exit_code = sp.check_call(['ls', 'een_bestand'])
        print(exit_code)
    except sp.CalledProcessError:
        print('The command failed')

def how_to_use_subprocesses4():
    try:
        out = sp.check_output(['ls', 'een_bestand'])
        print(out)
    except sp.CalledProcessError:
        print('The command failed')

def how_to_use_subprocesses5():
    open('een_bestand', 'w').close()

    how_to_use_subprocesses1()
    how_to_use_subprocesses2()
    how_to_use_subprocesses3()
    how_to_use_subprocesses4()

# }}}

# }}}

# Divers {{{

import pdb  # {{{

def how_to_debug():
    def f(x):
        return g(x)

    def g(x):
        return x + 1

    pdb.set_trace()
    for i in range(4):
        print(f(i))

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

import pprint # {{{
def show_pprint():
    complex_object = {
            1: list(range(20)),
            2: { 2: (1, 3) },
            3: ( "hello", list(range(20)) ) }
    pprint.pprint(complex_object)

# }}}
import locale
import logging
import decimal
import requests
# }}}

# profiling
# C extensions en SWIG

# vim: foldmethod=marker
