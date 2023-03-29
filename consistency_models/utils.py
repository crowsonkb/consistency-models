import time

import jax


def ema_update(tree_ema, tree, decay):
    """Update the exponential moving average of a tree.

    Parameters
    ----------
    tree_ema : Any
        The current value of the exponential moving average.
    tree : Any
        The new value to update the exponential moving average with.
    decay : float
        The decay factor of the exponential moving average.

    Returns
    -------
    Any
        The updated exponential moving average.
    """
    return jax.tree_map(
        lambda x_ema, x: x_ema * decay + x * (1 - decay), tree_ema, tree
    )


class PerfCounter:
    """Tracks an exponential moving average of the time between events.

    Parameters
    ----------
    decay : float
        The decay factor of the exponential moving average.
    """

    def __init__(self, decay=0.99):
        self.decay = decay
        self.count = 0
        self.value = 0.0
        self.decay_accum = 1.0
        self.last_time = None
        self.pause_time = None

    def get(self):
        """Get the current average time between events.

        Returns
        -------
        float
            The current average time between events. If fewer than two events have been
            recorded, returns NaN.
        """
        try:
            return self.value / (1 - self.decay_accum)
        except ZeroDivisionError:
            return float("nan")

    def get_count(self):
        """Get the number of events that have been recorded.

        Returns
        -------
        int
            The number of events that have been recorded.
        """
        return self.count

    def update(self, time_value=None):
        """Record the occurrence of an event.

        Parameters
        ----------
        time_value : float
            The time of the event. If None, the current time is used.

        Returns
        -------
        float
            The current average time between events.
        """
        time_value = time_value or time.time()
        self.resume(time_value)
        if self.last_time is not None:
            self.decay_accum *= self.decay
            self.value *= self.decay
            self.value += (time_value - self.last_time) * (1 - self.decay)
        self.last_time = time_value
        self.count += 1
        return self.get()

    def pause(self, time_value=None):
        """Pause the timer. If the timer is already paused, this does nothing.

        Parameters
        ----------
        time_value : float
            The time to pause the timer at. If None, the current time is used.
        """
        time_value = time_value or time.time()
        if self.pause_time is None and self.last_time is not None:
            self.pause_time = time_value

    def resume(self, time_value=None):
        """Resume the timer.

        Parameters
        ----------
        time_value : float
            The time to resume the timer at. If None, the current time is used.

        Returns
        -------
        float
            The duration of the pause, or 0.0 if the timer was not paused.
        """
        time_value = time_value or time.time()
        if self.pause_time is not None:
            pause_duration = time_value - self.pause_time
            self.last_time += pause_duration
            self.pause_time = None
            return pause_duration
        return 0.0


def rb(x, y):
    """Prepare x for right broadcasting against y by inserting trailing axes.

    Ordinary JAX broadcasting is left broadcasting: if x has fewer axes than y, axes
    are inserted on the left (leading axes) to match the number of dimensions of y. This
    function inserts axes on the right (trailing axes) instead.

    Parameters
    ----------
    x : jax.Array
        The array to insert trailing axes into.
    y : jax.Array
        The array to prepare x to be right broadcast against.

    Returns
    -------
    jax.Array
        x, with trailing axes inserted to match the number of dimensions of y.

    Examples
    --------
    >>> x = jnp.zeros((32,))
    >>> y = jnp.zeros((32, 224, 224, 3))
    >>> rb(x, y).shape
    (32, 1, 1, 1)
    """
    axes_to_insert = y.ndim - x.ndim
    if axes_to_insert < 0:
        raise ValueError(f"x has {x.ndim} dims but y has {y.ndim}, which is fewer")
    return x[(...,) + (None,) * axes_to_insert]


def split_by_process(key):
    """Splits a PRNG key, returning a different key in each JAX process."""
    return jax.random.split(key, jax.process_count())[jax.process_index()]


def tree_size(tree):
    """Return the number of elements in a tree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(tree))
