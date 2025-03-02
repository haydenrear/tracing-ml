import unittest

import numpy as np
import scipy.integrate as spi
import scipy.special as sp
import sympy

def nth_prime(n):
    """Returns the nth prime number using sympy."""
    return sympy.prime(n)

def prime_mellin(n, upper_bound=10):
    """Approximates the prime characteristic function using the integral form."""

    def integrand(x):
        # Ensure (-1)^x is treated as a real-valued sine function
        coeff = np.sin(np.pi * (x - 1)) / sp.gamma(2*x)  # Replaces (-1)^(x-1)
        cosine_term = np.cos(np.pi * n * (2*x - 1) / 2)**2
        return np.real_if_close(coeff * cosine_term)

    integral_value = spi.quad(integrand, 1, upper_bound, limit=100)
    return integral_value

# Compute the approximation for the nth prime
num_terms = 20  # Compute for the first 20 primes
for n in range(1, num_terms + 1):
    prime_n = nth_prime(n)
    result = prime_mellin(prime_n)
    print(f"P({prime_n}) = {result}")
