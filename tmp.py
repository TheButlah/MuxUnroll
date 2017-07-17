from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import time

from functools import partial as part


def main():

    the_tensors = {}

    def gen_tensor(x):
        print("GenTensor", x)

        def gen():
            print("Gen", x)
            a = tf.fill((10,), x)
            a = tf.Print(a, [x])
            the_tensor = (a + gen_tensor(x-1)) if x > 0 else a
            the_tensors[x] = lambda: the_tensor
            return the_tensor
        result = the_tensors.get(x, gen)  # There can only be one!
        return result()

    x = tf.placeholder(tf.float32, shape=(), name='X')
    zero = part(gen_tensor, 0)
    one = part(gen_tensor, 1)
    two = part(gen_tensor, 2)
    three = part(gen_tensor, 3)

    final_output = tf.case([
        (tf.equal(x, 3), three),
        (tf.equal(x, 2), two),
        (tf.equal(x, 1), one)],
        zero
    )

    sess = tf.Session()
    time.sleep(2)
    with sess.as_default():
        print("Started session!")
        print(final_output.eval(feed_dict={x: 1}))


if __name__ == "__main__":
    main()
